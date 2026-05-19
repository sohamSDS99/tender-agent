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

COMPRAS_BASE_URL = "https://dadosabertos.compras.gov.br"

# Candidate paths — the first one that returns a valid JSON response wins
_CANDIDATE_PATHS: list[str] = [
    "/modulo-pncp/1_consultarContratacao",
    "/modulo-contratacao/1_consultarContratacao",
    "/ocds/releases",
    "/modulo-pncp/consultarContratacao",
    "/modulo-contratacao/consultarContratacao",
]

# Public web view for a single procurement notice
PNCP_NOTICE_URL_TEMPLATE = "https://pncp.gov.br/app/editais/{notice_id}"

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
    "objetoContratacao",
    "objeto",
    "descricao",
    "titulo",
    "nomeContratacao",
]

_DESCRIPTION_FIELDS: list[str] = [
    "descricao",
    "informacoesComplementares",
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
    "dataLimite",
    "dataHoraLimitePropostas",
    "dataFimVigencia",
    "dataEncerramentoProposta",
    "dataFimRecebimentoPropostas",
]

_POSTED_FIELDS: list[str] = [
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
        timeout: float = 30.0,
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
        page_size: int = 100,
        max_pages: int = 3,
    ) -> list[dict[str, Any]]:
        """Fetch tender records from the Compras.gov.br API.

        Probes multiple API paths and paginates up to *max_pages*.

        Args:
            days_back: Number of days back (unused in query but kept for
                consistency — filtering is done post-fetch).
            page_size: Number of records per page.
            max_pages: Maximum number of pages to fetch.

        Returns:
            Flat list of record dicts (OCDS releases or custom objects).
        """
        all_records: list[dict[str, Any]] = []

        headers: dict[str, str] = {
            "Accept": "application/json",
        }

        with httpx.Client(timeout=self.timeout, headers=headers, follow_redirects=True) as client:
            for page_num in range(1, max_pages + 1):
                params: dict[str, str] = {
                    "pagina": str(page_num),
                    "tamanhoPagina": str(page_size),
                }

                try:
                    result = self._discover_endpoint(client, params)
                    if result is None:
                        logger.warning(
                            "brazil_compras_no_working_endpoint",
                            page=page_num,
                        )
                        break

                    _path, records = result

                    if not isinstance(records, list):
                        break

                    all_records.extend(records)
                    logger.debug(
                        "brazil_compras_page_fetched",
                        page=page_num,
                        count=len(records),
                    )

                    # If fewer records than page size, no more pages
                    if len(records) < page_size:
                        break

                except Exception as exc:
                    logger.error(
                        "brazil_compras_page_error",
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

        Maps Portuguese field names (``objetoContratacao``,
        ``valorTotalEstimado``, ``dataLimite``, etc.) to the shared
        dataclass.

        Args:
            record: A single custom-format JSON object.

        Returns:
            An ``OcdsTenderLead`` or ``None`` if the record is
            unparseable.
        """
        title = _first_nonempty(record, _TITLE_FIELDS)
        if not title:
            return None

        description = _first_nonempty(record, _DESCRIPTION_FIELDS)
        # Avoid duplicating title as description
        if description == title:
            description = ""

        agency = (
            _first_nonempty(record, _AGENCY_FIELDS)
            or "Governo Federal do Brasil"
        )

        deadline_raw = _first_nonempty(record, _DEADLINE_FIELDS)
        posted_raw = _first_nonempty(record, _POSTED_FIELDS)

        value_amount = _first_numeric(record, _VALUE_FIELDS)
        currency = _first_nonempty(record, _CURRENCY_FIELDS) or "BRL"

        # Build source URL
        record_id = _first_nonempty(record, _ID_FIELDS)
        source_url = ""
        if record_id:
            source_url = PNCP_NOTICE_URL_TEMPLATE.format(
                notice_id=record_id,
            )

        # Build a stable lead ID
        lead_id = record_id
        if not lead_id:
            lead_id = f"BRCMP-{uuid.uuid4().hex[:8].upper()}"

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
