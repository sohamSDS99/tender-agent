"""
Brazil Compras.gov.br — PNCP open data API integration.

Queries the Brazilian federal procurement portal (Compras.gov.br / PNCP)
for active tender notices relevant to EHS/SDS/chemical safety:
  https://dadosabertos.compras.gov.br/

The API may return data in OCDS format or in Brazil's own contracting
schema (with fields like ``objetoContratacao``, ``valorTotalEstimado``,
``dataLimite``).  This module handles both formats transparently —
OCDS releases are parsed via the shared ``parse_ocds_release`` helper,
while custom-format records are mapped manually to ``OcdsTenderLead``.

No authentication is required.

Usage:
    searcher = BrazilComprasSearcher()
    leads = searcher.search("seguranca quimica")
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import structlog

from src.discovery.ocds_base import (
    OcdsTenderLead,
    _parse_date,
    parse_ocds_release,
    score_relevance,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# API endpoints (tried in order until one responds)
# ---------------------------------------------------------------------------

# PNCP is Brazil's new national procurement portal (Portal Nacional de
# Contratações Públicas). The old dadosabertos.compras.gov.br endpoints
# returned 404 across the board as of May 2026 — every municipality and
# federal entity now publishes to PNCP centrally. Confirmed live via the
# PNCP Swagger at https://pncp.gov.br/api/consulta/v3/api-docs.
COMPRAS_BASE_URL = "https://pncp.gov.br"

# The /contratacoes/publicacao endpoint requires:
#   dataInicial, dataFinal: yyyyMMdd (single-day or range, max 365d apart)
#   codigoModalidadeContratacao: integer 1-14 (procurement method)
#   pagina: page number starting at 1
#   tamanhoPagina: min 10, max 50
_PNCP_ENDPOINT = "/api/consulta/v1/contratacoes/publicacao"

# PNCP responsiveness is intermittent — modalidade 4 (Concorrência) is
# the only one that responds within 12s reliably (verified May 2026).
# Pregão (6) and Dispensa (8) time out >45s. Diálogo competitivo (13)
# is hit-and-miss. Stick to 4 and accept the occasional Brazilian
# coverage gap rather than dragging the whole search wall time up.
_PNCP_MODALIDADES: list[int] = [
    4,   # Concorrência eletrônica — most reliable + most relevant
]

# Public web view for a single procurement notice. The PNCP URL pattern
# is /app/editais/{agency_cnpj}/{ano}/{sequencial}.
PNCP_NOTICE_URL_TEMPLATE = "https://pncp.gov.br/app/editais/{cnpj}/{ano}/{seq}"

# ---------------------------------------------------------------------------
# Country-specific keywords (Portuguese + English)
# ---------------------------------------------------------------------------

EXTRA_STRONG_KEYWORDS: list[str] = [
    "ficha de seguranca",
    "seguranca quimica",
    "substancias perigosas",
    "gestao de residuos",
    "produtos quimicos",
]

EXTRA_PARTIAL_KEYWORDS: list[str] = [
    "seguranca",
    "quimico",
    "ambiental",
    "saude ocupacional",
    "residuos",
]

# ---------------------------------------------------------------------------
# Brazil custom-format field mappings
# ---------------------------------------------------------------------------

_TITLE_FIELDS: list[str] = [
    "objetoCompra",           # PNCP primary field
    "objetoContratacao",
    "objeto",
    "descricao",
    "titulo",
    "nomeContratacao",
]

_DESCRIPTION_FIELDS: list[str] = [
    "informacaoComplementar",  # PNCP primary field
    "descricao",
    "informacoesComplementares",
    "objetoCompra",
    "objetoContratacao",
    "objeto",
]

_AGENCY_FIELDS: list[str] = [
    "nomeOrgao",
    "orgao",
    "nomeUnidade",
    "unidadeGestora",
    "nomeEntidade",
]

_DEADLINE_FIELDS: list[str] = [
    "dataEncerramentoProposta",  # PNCP primary field
    "dataLimite",
    "dataHoraLimitePropostas",
    "dataFimVigencia",
    "dataFimRecebimentoPropostas",
]

_POSTED_FIELDS: list[str] = [
    "dataPublicacaoPncp",        # PNCP primary field (correct casing)
    "dataPublicacao",
    "dataPublicacaoPNCP",
    "dataAbertura",
    "dataInclusao",
]

_VALUE_FIELDS: list[str] = [
    "valorTotalEstimado",
    "valorEstimado",
    "valorTotal",
    "valorContrato",
]

_CURRENCY_FIELDS: list[str] = [
    "moeda",
]

_ID_FIELDS: list[str] = [
    "id",
    "codigoContratacao",
    "numeroContratacao",
    "sequencialContratacao",
]


def _first_nonempty(row: dict[str, Any], fields: list[str]) -> str:
    """Return the first non-empty string value from *fields* in *row*."""
    for field_name in fields:
        value = row.get(field_name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _first_numeric(row: dict[str, Any], fields: list[str]) -> float:
    """Return the first parseable numeric value from *fields* in *row*."""
    for field_name in fields:
        value = row.get(field_name)
        if value is not None:
            try:
                return float(str(value).replace(",", ".").strip())
            except (ValueError, TypeError):
                continue
    return 0.0


def _build_pncp_url(release: dict[str, Any]) -> str:
    """Build a PNCP web URL from an OCDS release dict."""
    ocid = release.get("ocid", "")
    release_id = release.get("id", "")
    notice_id = release_id or ocid
    if notice_id:
        return PNCP_NOTICE_URL_TEMPLATE.format(notice_id=notice_id)
    return ""


class BrazilComprasSearcher:
    """Searches Brazil's Compras.gov.br / PNCP for EHS/SDS tenders.

    Probes multiple API paths to find a working endpoint, handles both
    OCDS and Brazil's custom response formats, scores each tender for
    EHS/SDS relevance using Portuguese and English keywords.

    Usage:
        searcher = BrazilComprasSearcher()
        leads = searcher.search("seguranca quimica")
    """

    def __init__(
        self,
        # PNCP latency is genuinely intermittent — sometimes <2s,
        # sometimes >45s for the same query. Cap at 12s so a slow PNCP
        # run doesn't monopolise the bridge's parallel-fan-out worker
        # budget. We'd rather miss this source on a slow day than have
        # every search block waiting for it.
        timeout: float = 12.0,
        min_relevance: float = 0.10,
    ) -> None:
        """Initialise the searcher.

        Args:
            timeout: HTTP request timeout in seconds.
            min_relevance: Minimum relevance score to include a lead.
        """
        self.timeout = timeout
        self.min_relevance = min_relevance
        self._working_path: str | None = None
        logger.info("brazil_compras_searcher_initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[OcdsTenderLead]:
        """Search Compras.gov.br for active EHS/SDS-relevant tenders.

        Args:
            user_query: User's free-text search (reserved for future use).
            max_results: Maximum leads to return.
            days_back: Number of days back to consider.

        Returns:
            List of ``OcdsTenderLead`` objects sorted by relevance.
        """
        logger.info(
            "brazil_compras_search_start",
            days_back=days_back,
            max_results=max_results,
        )

        try:
            raw_data = self._fetch_tenders(days_back=days_back)
        except Exception as exc:
            logger.error("brazil_compras_fetch_failed", error=str(exc))
            return []

        logger.info("brazil_compras_raw_results", total=len(raw_data))

        leads: list[OcdsTenderLead] = []
        seen_ids: set[str] = set()

        for record in raw_data:
            try:
                lead = self._parse_record(record)
                if lead is None:
                    continue

                # Deduplicate
                if lead.lead_id in seen_ids:
                    continue
                seen_ids.add(lead.lead_id)

                # Skip expired tenders
                if lead.submission_deadline:
                    try:
                        dl = datetime.strptime(
                            lead.submission_deadline, "%Y-%m-%d"
                        ).date()
                        if dl < datetime.now(timezone.utc).date():
                            continue
                    except (ValueError, TypeError):
                        pass

                # Score relevance
                score, keywords = score_relevance(
                    lead.title,
                    lead.description,
                    lead.cpv_code,
                    extra_strong=EXTRA_STRONG_KEYWORDS,
                    extra_partial=EXTRA_PARTIAL_KEYWORDS,
                )
                lead.relevance_score = score
                lead.relevance_keywords = keywords

                if score >= self.min_relevance:
                    leads.append(lead)

            except Exception as exc:
                logger.debug("brazil_compras_parse_error", error=str(exc))

        # Sort by relevance descending and cap
        leads.sort(key=lambda l: l.relevance_score, reverse=True)
        leads = leads[:max_results]

        logger.info(
            "brazil_compras_search_complete",
            total_results=len(leads),
            top_score=leads[0].relevance_score if leads else 0,
        )

        return leads

    # ------------------------------------------------------------------
    # Internal helpers — endpoint discovery
    # ------------------------------------------------------------------

    def _discover_endpoint(
        self,
        client: httpx.Client,
        params: dict[str, str],
    ) -> tuple[str, Any] | None:
        """Try candidate API paths and return the first that works.

        Args:
            client: An active ``httpx.Client``.
            params: Query parameters to send.

        Returns:
            ``(path, parsed_json)`` or ``None`` if every path fails.
        """
        # If we already found a working path, try it first
        paths_to_try = list(_CANDIDATE_PATHS)
        if self._working_path:
            paths_to_try.remove(self._working_path)
            paths_to_try.insert(0, self._working_path)

        for path in paths_to_try:
            url = f"{COMPRAS_BASE_URL}{path}"
            try:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

                # Verify we got something usable
                if isinstance(data, list) and len(data) > 0:
                    self._working_path = path
                    logger.debug(
                        "brazil_compras_endpoint_found",
                        path=path,
                        record_count=len(data),
                    )
                    return path, data

                if isinstance(data, dict):
                    # OCDS release package or paginated wrapper
                    releases = (
                        data.get("releases")
                        or data.get("data")
                        or data.get("resultado")
                        or data.get("items")
                        or data.get("contratacoes")
                    )
                    if isinstance(releases, list) and len(releases) > 0:
                        self._working_path = path
                        logger.debug(
                            "brazil_compras_endpoint_found",
                            path=path,
                            record_count=len(releases),
                        )
                        return path, releases

            except Exception as exc:
                logger.debug(
                    "brazil_compras_path_failed",
                    path=path,
                    error=str(exc),
                )
                continue

        return None

    # ------------------------------------------------------------------
    # Internal helpers — fetch
    # ------------------------------------------------------------------

    def _fetch_tenders(
        self,
        days_back: int = 60,
        page_size: int = 50,
        # One page per modalidade keeps total wall time bounded even if
        # PNCP is in a slow mode. The first 50 records per modalidade
        # is more than enough — relevance scoring caps results anyway.
        max_pages: int = 1,
    ) -> list[dict[str, Any]]:
        """Fetch tender records from PNCP (Portal Nacional de Contratações Públicas).

        PNCP requires dataInicial+dataFinal in yyyyMMdd and a specific
        codigoModalidadeContratacao. We loop over a small set of modalidades
        most likely to contain SDS/EHS-relevant contracts. tamanhoPagina has
        a server-side minimum of 10 and maximum of 50.

        Args:
            days_back: Number of days back to search.
            page_size: Records per page (clamped to PNCP's 10-50 range).
            max_pages: Max pages per modalidade to avoid runaway queries.

        Returns:
            Flat list of PNCP record dicts.
        """
        # PNCP date format is yyyyMMdd, single-day or range up to 365 days.
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=max(days_back, 1))
        data_inicial = start.strftime("%Y%m%d")
        data_final = now.strftime("%Y%m%d")

        # Server-side: 10 <= tamanhoPagina <= 50
        page_size = max(10, min(50, page_size))

        all_records: list[dict[str, Any]] = []
        headers: dict[str, str] = {"Accept": "application/json"}

        with httpx.Client(timeout=self.timeout, headers=headers, follow_redirects=True) as client:
            for modalidade in _PNCP_MODALIDADES:
                for page_num in range(1, max_pages + 1):
                    params: dict[str, str] = {
                        "dataInicial": data_inicial,
                        "dataFinal": data_final,
                        "codigoModalidadeContratacao": str(modalidade),
                        "pagina": str(page_num),
                        "tamanhoPagina": str(page_size),
                    }

                    try:
                        url = f"{COMPRAS_BASE_URL}{_PNCP_ENDPOINT}"
                        resp = client.get(url, params=params)
                        if resp.status_code == 204:
                            # No data for this modalidade — move on.
                            break
                        resp.raise_for_status()
                        data = resp.json()

                        records = data.get("data") if isinstance(data, dict) else None
                        if not isinstance(records, list) or not records:
                            break

                        all_records.extend(records)
                        logger.debug(
                            "brazil_compras_page_fetched",
                            modalidade=modalidade,
                            page=page_num,
                            count=len(records),
                        )

                        # If fewer than page_size, that was the last page
                        if len(records) < page_size:
                            break

                    except Exception as exc:
                        logger.debug(
                            "brazil_compras_page_error",
                            modalidade=modalidade,
                            page=page_num,
                            error=str(exc),
                        )
                        break

        return all_records

    # ------------------------------------------------------------------
    # Internal helpers — parsing
    # ------------------------------------------------------------------

    def _parse_record(self, record: dict[str, Any]) -> OcdsTenderLead | None:
        """Parse a single API record into an ``OcdsTenderLead``.

        Detects whether the record is an OCDS release (has ``tender``
        key) or a Brazil custom-format object, and delegates accordingly.

        Args:
            record: A single JSON object from the API response.

        Returns:
            An ``OcdsTenderLead`` or ``None`` if the record is
            unparseable or irrelevant.
        """
        # Detect OCDS format by the presence of the "tender" key
        if "tender" in record:
            return self._parse_ocds_record(record)

        return self._parse_custom_record(record)

    def _parse_ocds_record(
        self, release: dict[str, Any],
    ) -> OcdsTenderLead | None:
        """Parse an OCDS-format release using the shared helper.

        Args:
            release: An OCDS release dict.

        Returns:
            An ``OcdsTenderLead`` or ``None``.
        """
        try:
            lead = parse_ocds_release(
                release,
                source_portal="brazil_compras",
                build_url=_build_pncp_url,
            )
            return lead
        except Exception as exc:
            logger.debug("brazil_compras_ocds_parse_error", error=str(exc))
            return None

    def _parse_custom_record(
        self, record: dict[str, Any],
    ) -> OcdsTenderLead | None:
        """Parse a Brazil custom-format record into an ``OcdsTenderLead``.

        Handles both legacy Compras.gov.br objects and the current PNCP
        v1 shape (which nests the buying organisation under
        ``orgaoEntidade`` and identifies tenders by the compound key
        ``{cnpj}/{anoCompra}/{sequencialCompra}``).
        """
        title = _first_nonempty(record, _TITLE_FIELDS)
        if not title:
            return None

        description = _first_nonempty(record, _DESCRIPTION_FIELDS)
        if description == title:
            description = ""

        # PNCP shape: agency name nested under orgaoEntidade.razaoSocial.
        # Fall back to the flat-name lookup for legacy shapes.
        orgao = record.get("orgaoEntidade") if isinstance(record.get("orgaoEntidade"), dict) else None
        agency = (
            (orgao or {}).get("razaoSocial")
            or _first_nonempty(record, _AGENCY_FIELDS)
            or "Governo Federal do Brasil"
        )

        deadline_raw = _first_nonempty(record, _DEADLINE_FIELDS)
        posted_raw = _first_nonempty(record, _POSTED_FIELDS)

        value_amount = _first_numeric(record, _VALUE_FIELDS)
        currency = _first_nonempty(record, _CURRENCY_FIELDS) or "BRL"

        # Source URL: prefer PNCP's compound identifier when present.
        source_url = ""
        record_id = _first_nonempty(record, _ID_FIELDS)
        ano = record.get("anoCompra")
        seq = record.get("sequencialCompra")
        cnpj = (orgao or {}).get("cnpj")
        if ano and seq and cnpj:
            source_url = PNCP_NOTICE_URL_TEMPLATE.format(
                cnpj=cnpj, ano=ano, seq=seq,
            )
            # PNCP's natural unique key for the record
            if not record_id:
                record_id = f"{cnpj}-{ano}-{seq}"
        elif record_id and "{notice_id}" in PNCP_NOTICE_URL_TEMPLATE:
            # Legacy shape — keep working if the template ever reverts
            source_url = PNCP_NOTICE_URL_TEMPLATE.format(notice_id=record_id)

        lead_id = record_id or f"BRCMP-{uuid.uuid4().hex[:8].upper()}"

        # Parse dates
        deadline_iso = _parse_date(deadline_raw)
        posted_iso = _parse_date(posted_raw)

        return OcdsTenderLead(
            lead_id=lead_id,
            title=title,
            description=description[:500] if description else "",
            agency=agency,
            source_portal="brazil_compras",
            source_url=source_url,
            submission_deadline=deadline_iso,
            posted_date=posted_iso,
            value_amount=value_amount,
            value_currency=currency,
            cpv_code="",
            relevance_score=0.0,
            relevance_keywords=[],
            raw_data={
                "source_portal": "brazil_compras",
                "raw_id": record_id,
            },
        )
