"""
Colombia SECOP — Socrata/SODA API integration.

Queries the Colombian government's SECOP II procurement data via the
Socrata Open Data API (SODA) hosted on datos.gov.co:
  https://www.datos.gov.co/resource/jbjy-vk9h.json

This is NOT an OCDS-compliant feed.  SECOP publishes procurement data as
flat JSON rows through the Socrata platform with Spanish field names
(``nombre_del_procedimiento``, ``nombre_entidad``, ``fecha_de_cierre``,
etc.).  We build ``OcdsTenderLead`` objects manually from those fields
and score them using the shared ``score_relevance`` helper augmented
with Spanish-language keywords.

No authentication is required for basic access.  An optional Socrata
application token can be passed for higher rate limits.

Usage:
    searcher = ColombiaSecopSearcher()
    leads = searcher.search("seguridad quimica")
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
    score_relevance,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

SECOP_API_URL = "https://www.datos.gov.co/resource/jbjy-vk9h.json"

# Public web view for a single procurement notice
SECOP_NOTICE_URL_TEMPLATE = (
    "https://community.secop.gov.co/Public/Tendering/"
    "OpportunityDetail/Index?noticeUID=CO1.NTC.{notice_id}"
)

# ---------------------------------------------------------------------------
# Country-specific keywords (Spanish + English)
# ---------------------------------------------------------------------------

EXTRA_STRONG_KEYWORDS: list[str] = [
    "seguridad quimica",
    "hoja de seguridad",
    "sustancias peligrosas",
    "manejo de quimicos",
    "residuos peligrosos",
]

EXTRA_PARTIAL_KEYWORDS: list[str] = [
    "seguridad",
    "quimico",
    "ambiental",
    "salud ocupacional",
    "riesgo",
]

# ---------------------------------------------------------------------------
# Socrata field mappings — the dataset uses Spanish column names
# ---------------------------------------------------------------------------

# Title fields (try in order of preference)
_TITLE_FIELDS: list[str] = [
    "nombre_del_procedimiento",
    "descripcion_del_procedimiento",
    "nombre_del_proceso",
]

# Description fields
_DESCRIPTION_FIELDS: list[str] = [
    "descripcion_del_procedimiento",
    "detalle_del_objeto_a_contratar",
    "descripcion_del_proceso",
    "objeto_del_proceso",
]

# Agency fields
_AGENCY_FIELDS: list[str] = [
    "nombre_entidad",
    "entidad",
    "nombre_de_la_entidad",
]

# Deadline fields (closing date)
_DEADLINE_FIELDS: list[str] = [
    "fecha_de_cierre",
    "fecha_cierre",
    "fecha_de_cierre_del_proceso",
]

# Posted date fields
_POSTED_FIELDS: list[str] = [
    "fecha_de_publicacion",
    "fecha_publicacion",
    "fecha_de_publicacion_del",
]

# Value fields
_VALUE_FIELDS: list[str] = [
    "valor_del_contrato",
    "cuantia_contrato",
    "cuantia_proceso",
    "valor_total_de_la_orden",
]

# Currency fields
_CURRENCY_FIELDS: list[str] = [
    "moneda",
]

# URL fields
_URL_FIELDS: list[str] = [
    "url_proceso",
    "urlproceso",
]

# ID fields (for building the notice URL fallback)
_ID_FIELDS: list[str] = [
    "id_del_portafolio",
    "id_del_proceso",
    "uid",
    "id",
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
                return float(str(value).replace(",", "").strip())
            except (ValueError, TypeError):
                continue
    return 0.0


class ColombiaSecopSearcher:
    """Searches Colombia's SECOP II portal via the Socrata SODA API.

    Queries ``datos.gov.co`` for recent procurement notices, parses the
    flat Socrata JSON rows into ``OcdsTenderLead`` objects, and scores
    each for EHS/SDS relevance using both English and Spanish keywords.

    Usage:
        searcher = ColombiaSecopSearcher()
        leads = searcher.search("seguridad quimica")
    """

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.10,
        app_token: str | None = None,
    ) -> None:
        """Initialise the searcher.

        Args:
            timeout: HTTP request timeout in seconds.
            min_relevance: Minimum relevance score to include a lead.
            app_token: Optional Socrata application token for higher
                rate limits.
        """
        self.timeout = timeout
        self.min_relevance = min_relevance
        self.app_token = app_token
        logger.info("colombia_secop_searcher_initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[OcdsTenderLead]:
        """Search SECOP for active EHS/SDS-relevant tenders.

        Args:
            user_query: User's free-text search (reserved for future use).
            max_results: Maximum leads to return.
            days_back: Only include tenders with a closing date within
                this many days from today.

        Returns:
            List of ``OcdsTenderLead`` objects sorted by relevance.
        """
        logger.info(
            "colombia_secop_search_start",
            days_back=days_back,
            max_results=max_results,
        )

        try:
            raw_rows = self._fetch_tenders(days_back=days_back)
        except Exception as exc:
            logger.error("colombia_secop_fetch_failed", error=str(exc))
            return []

        logger.info("colombia_secop_raw_results", total=len(raw_rows))

        leads: list[OcdsTenderLead] = []
        seen_ids: set[str] = set()

        for row in raw_rows:
            try:
                lead = self._parse_row(row)
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
                logger.debug("colombia_secop_parse_error", error=str(exc))

        # Sort by relevance descending and cap
        leads.sort(key=lambda l: l.relevance_score, reverse=True)
        leads = leads[:max_results]

        logger.info(
            "colombia_secop_search_complete",
            total_results=len(leads),
            top_score=leads[0].relevance_score if leads else 0,
        )

        return leads

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_tenders(
        self,
        days_back: int = 60,
        limit: int = 100,
        pages: int = 3,
    ) -> list[dict[str, Any]]:
        """Fetch tender rows from the Socrata SODA API.

        Paginates up to *pages* pages of *limit* rows each, filtering
        for tenders whose closing date is in the future (relative to
        ``days_back`` days ago, to catch recently posted ones).

        Args:
            days_back: Only include tenders with closing date after this
                many days ago.
            limit: Number of rows per page (Socrata ``$limit``).
            pages: Maximum number of pages to fetch.

        Returns:
            Flat list of row dicts from the Socrata response.
        """
        cutoff_date = (
            datetime.now(timezone.utc) - timedelta(days=days_back)
        ).strftime("%Y-%m-%dT00:00:00.000")

        all_rows: list[dict[str, Any]] = []

        headers: dict[str, str] = {
            "Accept": "application/json",
        }
        if self.app_token:
            headers["X-App-Token"] = self.app_token

        with httpx.Client(timeout=self.timeout, headers=headers, follow_redirects=True) as client:
            for page in range(pages):
                offset = page * limit

                params: dict[str, str] = {
                    "$limit": str(limit),
                    "$offset": str(offset),
                    "$order": "fecha_de_cierre DESC",
                    "$where": f"fecha_de_cierre > '{cutoff_date}'",
                }

                try:
                    resp = client.get(SECOP_API_URL, params=params)
                    resp.raise_for_status()
                    rows = resp.json()
                except Exception as exc:
                    logger.error(
                        "colombia_secop_page_error",
                        page=page,
                        error=str(exc),
                    )
                    break

                if not isinstance(rows, list) or not rows:
                    break

                all_rows.extend(rows)
                logger.debug(
                    "colombia_secop_page_fetched",
                    page=page,
                    count=len(rows),
                )

                # If fewer rows than limit, no more pages
                if len(rows) < limit:
                    break

        return all_rows

    def _parse_row(self, row: dict[str, Any]) -> OcdsTenderLead | None:
        """Parse a single Socrata row into an ``OcdsTenderLead``.

        SECOP data is flat (not OCDS), so we map Spanish field names to
        the shared dataclass manually.

        Args:
            row: A single JSON object from the Socrata response.

        Returns:
            An ``OcdsTenderLead`` or ``None`` if the row is unparseable.
        """
        title = _first_nonempty(row, _TITLE_FIELDS)
        if not title:
            return None

        description = _first_nonempty(row, _DESCRIPTION_FIELDS)
        # Avoid duplicating title as description
        if description == title:
            description = ""

        agency = _first_nonempty(row, _AGENCY_FIELDS) or "Gobierno de Colombia"

        deadline_raw = _first_nonempty(row, _DEADLINE_FIELDS)
        posted_raw = _first_nonempty(row, _POSTED_FIELDS)

        value_amount = _first_numeric(row, _VALUE_FIELDS)
        currency = _first_nonempty(row, _CURRENCY_FIELDS) or "COP"

        # Build source URL — prefer the url_proceso field, fall back to
        # constructing from the portfolio/process ID
        source_url = _first_nonempty(row, _URL_FIELDS)
        if not source_url:
            notice_id = _first_nonempty(row, _ID_FIELDS)
            if notice_id:
                source_url = SECOP_NOTICE_URL_TEMPLATE.format(
                    notice_id=notice_id,
                )

        # Build a stable lead ID
        lead_id = _first_nonempty(row, _ID_FIELDS)
        if not lead_id:
            lead_id = f"SECOP-{uuid.uuid4().hex[:8].upper()}"

        # Parse dates
        deadline_iso = _parse_date(deadline_raw)
        posted_iso = _parse_date(posted_raw)

        return OcdsTenderLead(
            lead_id=lead_id,
            title=title,
            description=description[:500] if description else "",
            agency=agency,
            source_portal="colombia_secop",
            source_url=source_url,
            submission_deadline=deadline_iso,
            posted_date=posted_iso,
            value_amount=value_amount,
            value_currency=currency,
            cpv_code="",  # SECOP does not use CPV codes
            relevance_score=0.0,
            relevance_keywords=[],
            raw_data={
                "source_portal": "colombia_secop",
                "raw_id": _first_nonempty(row, _ID_FIELDS),
            },
        )
