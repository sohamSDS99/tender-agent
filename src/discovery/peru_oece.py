"""
Peru OECE — Organismo Especializado en Contrataciones del Estado OCDS API.

Queries Peru's open contracting data API (formerly OSCE, now OECE) for
active tender notices relevant to EHS/SDS/chemical safety:
  https://contratacionesabiertas.oece.gob.pe/api

Bulk downloads available at:
  https://contratacionesabiertas.oece.gob.pe/descargas

Free, public, no authentication required.  Returns OCDS release packages
containing standard OCDS 1.1 objects.  Peru's dataset is massive (2.7M+
tenders) so date filtering is critical.

This module is a thin wrapper around ``ocds_base`` — it only defines the
OECE-specific API endpoints, URL builder, country-specific keywords, and
searcher class.  All keyword scoring and OCDS release parsing is delegated
to shared helpers.

Usage:
    searcher = PeruOeceSearcher()
    leads = searcher.search("chemical safety")
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import structlog

from src.discovery.ocds_base import (
    OcdsTenderLead,
    parse_ocds_release,
    score_relevance,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# API endpoints — tried in order until one succeeds
# ---------------------------------------------------------------------------

API_ENDPOINTS: list[str] = [
    "https://contratacionesabiertas.oece.gob.pe/api/ocds/releases",
    "https://contratacionesabiertas.oece.gob.pe/api/releases",
    "https://contratacionesabiertas.oece.gob.pe/api/record-package",
]

# Date filter parameter names to try (appended as query params)
DATE_FILTER_PARAMS: list[str] = ["publishedFrom", "since", "from_date"]

# Public web view for individual processes
OECE_WEB_BASE = "https://contratacionesabiertas.oece.gob.pe/proceso"

# Fallback portal URL
OECE_PORTAL_URL = "https://contratacionesabiertas.oece.gob.pe/"

# ---------------------------------------------------------------------------
# Country-specific keywords (Spanish)
# ---------------------------------------------------------------------------

EXTRA_STRONG_KEYWORDS: list[str] = [
    "hoja de seguridad",
    "seguridad quimica",
    "sustancias peligrosas",
    "residuos peligrosos",
    "manejo de quimicos",
]

EXTRA_PARTIAL_KEYWORDS: list[str] = [
    "seguridad",
    "quimico",
    "ambiental",
    "salud ocupacional",
    "residuos",
    "peligroso",
]


# ---------------------------------------------------------------------------
# URL builder for OECE
# ---------------------------------------------------------------------------

def _build_oece_url(release: dict) -> str:
    """Build a public OECE web URL from an OCDS release.

    Peru's open contracting portal uses the OCID as the process
    identifier in the URL.  We try ``ocid`` first, then ``id``,
    then ``tender.id``.  If none yields a usable ID, we fall back
    to the portal homepage.
    """
    # Prefer OCID — this is the canonical process identifier
    ocid = release.get("ocid", "")
    if ocid:
        return f"{OECE_WEB_BASE}/{ocid}"

    # Try release-level id
    release_id = release.get("id", "")
    if release_id:
        return f"{OECE_WEB_BASE}/{release_id}"

    # Try tender.id inside the release
    tender = release.get("tender", {})
    tender_id = tender.get("id", "") if isinstance(tender, dict) else ""
    if tender_id:
        return f"{OECE_WEB_BASE}/{tender_id}"

    # Fallback — portal homepage
    return OECE_PORTAL_URL


# ---------------------------------------------------------------------------
# PeruOeceSearcher class
# ---------------------------------------------------------------------------

class PeruOeceSearcher:
    """Searches Peru's OECE OCDS API for active EHS/SDS tenders.

    Fetches OCDS release packages from the OECE API (trying multiple
    endpoint paths), parses each release with the shared
    ``parse_ocds_release`` helper, scores for EHS/SDS relevance using
    both global and Spanish-language keywords, and returns the top matches.

    Note: Peru's dataset is massive (2.7M+ tenders with daily ETL), so
    the default ``days_back`` is 90 to ensure adequate coverage while
    keeping request sizes manageable via the ``limit`` parameter.

    Usage:
        searcher = PeruOeceSearcher()
        leads = searcher.search("chemical safety")
    """

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.10,
    ) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        logger.info("peru_oece_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 90,
    ) -> list[OcdsTenderLead]:
        """Search OECE for active EHS/SDS tenders.

        Args:
            user_query: User's search text (used as an additional keyword
                        signal during relevance scoring).
            max_results: Maximum results to return.
            days_back: How many days back to search.  Default is 90
                       (larger than other portals) because Peru's dataset
                       uses daily ETL and we want adequate coverage.

        Returns:
            List of OcdsTenderLead objects, filtered by relevance and
            sorted descending by relevance_score.
        """
        logger.info(
            "peru_oece_search_start",
            days_back=days_back,
            max_results=max_results,
        )

        try:
            releases = self._fetch_releases(days_back)
        except Exception as exc:
            logger.error(
                "peru_oece_fetch_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return []

        if not releases:
            logger.warning(
                "peru_oece_no_releases",
                msg="All API endpoints returned no releases; returning empty list",
            )
            return []

        # Parse, score, and filter
        leads = self._parse_and_filter(releases, user_query)

        # Sort by relevance (highest first) and cap
        leads.sort(key=lambda lead: lead.relevance_score, reverse=True)
        leads = leads[:max_results]

        logger.info(
            "peru_oece_search_complete",
            raw_count=len(releases),
            filtered_count=len(leads),
            top_score=leads[0].relevance_score if leads else 0,
        )

        return leads

    # ------------------------------------------------------------------
    # API fetch — tries multiple endpoints in order
    # ------------------------------------------------------------------

    def _fetch_releases(self, days_back: int) -> list[dict]:
        """Fetch OCDS releases from the OECE API.

        Tries each endpoint in ``API_ENDPOINTS`` with each date filter
        parameter in ``DATE_FILTER_PARAMS``.  Returns releases from the
        first combination that succeeds.

        The ``record-package`` endpoint may return records instead of
        releases; in that case, we extract releases from each record's
        ``releases`` array.

        Returns a list of raw OCDS release dicts, or an empty list on
        any failure.
        """
        now = datetime.now(timezone.utc)
        from_date = now - timedelta(days=days_back)
        from_str = from_date.strftime("%Y-%m-%d")

        for endpoint in API_ENDPOINTS:
            # Try with each date filter parameter
            for date_param in DATE_FILTER_PARAMS:
                params = {"limit": "100", date_param: from_str}
                releases = self._fetch_from_url(endpoint, params)
                if releases:
                    return releases

            # Try without date filter as last resort for this endpoint
            releases = self._fetch_from_url(endpoint, {"limit": "100"})
            if releases:
                return releases

        return []

    def _fetch_from_url(
        self,
        url: str,
        params: dict[str, str],
    ) -> list[dict]:
        """Fetch and extract releases from a single OECE OCDS URL.

        Handles:
          - Standard OCDS release package: ``{"releases": [...]}``
          - OCDS record package: ``{"records": [{"releases": [...]}]}``
          - Flat array of releases
          - Alternative top-level keys

        Returns a list of release dicts or an empty list on failure.
        """
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            # Standard OCDS release package
            if isinstance(data, dict) and "releases" in data:
                releases = data["releases"]
                if isinstance(releases, list):
                    logger.info(
                        "peru_oece_releases_fetched",
                        url=url[:120],
                        count=len(releases),
                    )
                    return releases

            # OCDS record package — extract releases from each record
            if isinstance(data, dict) and "records" in data:
                records = data["records"]
                if isinstance(records, list):
                    all_releases: list[dict] = []
                    for record in records:
                        if isinstance(record, dict):
                            rec_releases = record.get("releases", [])
                            if isinstance(rec_releases, list):
                                all_releases.extend(rec_releases)
                            # Also check compiledRelease (single latest release)
                            compiled = record.get("compiledRelease")
                            if isinstance(compiled, dict) and not rec_releases:
                                all_releases.append(compiled)
                    if all_releases:
                        logger.info(
                            "peru_oece_record_package_extracted",
                            url=url[:120],
                            record_count=len(records),
                            release_count=len(all_releases),
                        )
                        return all_releases

            # Flat array of releases
            if isinstance(data, list):
                logger.info(
                    "peru_oece_releases_flat_array",
                    url=url[:120],
                    count=len(data),
                )
                return data

            # Try common alternative keys
            for key in ("data", "results", "tenders"):
                if isinstance(data, dict) and key in data:
                    items = data[key]
                    if isinstance(items, list):
                        logger.info(
                            "peru_oece_releases_alt_key",
                            url=url[:120],
                            key=key,
                            count=len(items),
                        )
                        return items

            logger.debug(
                "peru_oece_unexpected_shape",
                url=url[:120],
                keys=list(data.keys()) if isinstance(data, dict) else type(data).__name__,
            )
            return []

        except httpx.HTTPStatusError as exc:
            logger.debug(
                "peru_oece_http_error",
                status_code=exc.response.status_code,
                url=url[:120],
            )
            return []

        except httpx.TimeoutException:
            logger.debug("peru_oece_timeout", url=url[:120])
            return []

        except Exception as exc:
            logger.debug(
                "peru_oece_fetch_error",
                error=str(exc),
                error_type=type(exc).__name__,
                url=url[:120],
            )
            return []

    # ------------------------------------------------------------------
    # Parse and filter releases into leads
    # ------------------------------------------------------------------

    def _parse_and_filter(
        self,
        releases: list[dict],
        user_query: str,
    ) -> list[OcdsTenderLead]:
        """Parse raw OCDS releases into scored, filtered leads."""
        leads: list[OcdsTenderLead] = []
        seen_ids: set[str] = set()

        for release in releases:
            try:
                lead = parse_ocds_release(
                    release,
                    source_portal="peru_oece",
                    build_url=_build_oece_url,
                )
                if not lead:
                    continue

                # Deduplicate by lead_id
                if lead.lead_id in seen_ids:
                    continue
                seen_ids.add(lead.lead_id)

                # Score relevance using shared scorer with Peru-specific keywords
                score, keywords = score_relevance(
                    lead.title,
                    lead.description,
                    lead.cpv_code,
                    extra_strong=EXTRA_STRONG_KEYWORDS,
                    extra_partial=EXTRA_PARTIAL_KEYWORDS,
                )

                # Bonus if user_query terms appear in title or description
                if user_query:
                    query_lower = user_query.lower()
                    text = f"{lead.title} {lead.description}".lower()
                    query_words = query_lower.split()
                    query_hits = sum(1 for w in query_words if w in text)
                    if query_hits > 0:
                        query_bonus = min(query_hits * 0.05, 0.15)
                        score = min(score + query_bonus, 1.0)
                        score = round(score, 2)

                lead.relevance_score = score
                lead.relevance_keywords = keywords

                if score >= self.min_relevance:
                    leads.append(lead)

            except Exception as exc:
                logger.debug(
                    "peru_oece_release_parse_error",
                    error=str(exc),
                )

        return leads
