"""
UK Find a Tender Service (FTS) — official post-Brexit UK procurement OCDS API.

Find a Tender Service replaced OJEU for UK contract notices in Jan 2021.
It carries every public-sector tender published in the UK including NHS
Supply Chain, MoD, Cabinet Office, devolved administrations, etc.
Particularly rich for chemical-safety / lab / hazmat work via NHS and
Public Health England buyers.

  API base: https://www.find-tender.service.gov.uk
  OCDS docs: https://www.find-tender.service.gov.uk/Help/Article/27

Free, public, no authentication required. The OCDS Release Package
endpoint is well-documented and stable.

Usage:
    searcher = UkFtsSearcher()
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
# API endpoints — primary first, fallbacks if FTS reshuffles
# ---------------------------------------------------------------------------

UK_FTS_ENDPOINTS: list[str] = [
    "https://www.find-tender.service.gov.uk/api/1.0/ocdsReleasePackages",
    # Legacy/alternate paths kept in case FTS rotates URL forms
    "https://www.find-tender.service.gov.uk/api/ocdsReleasePackages",
    "https://www.find-tender.service.gov.uk/api/1.0/ocds/releases",
]

UK_FTS_WEB_BASE = "https://www.find-tender.service.gov.uk/Notice"

# ---------------------------------------------------------------------------
# UK-specific extras (FTS-flavoured phrasing)
# ---------------------------------------------------------------------------

UK_EXTRA_STRONG: list[str] = [
    "coshh",                     # UK regulation for hazardous substances
    "control of substances",     # COSHH expanded
    "dsear",                     # Dangerous Substances regulation
    "nhs supply chain",
]

UK_EXTRA_PARTIAL: list[str] = [
    "framework agreement",
    "lot",
    "ojeu",
]


def _build_uk_fts_url(release: dict) -> str:
    """Build a public Notice URL from an OCDS release.

    FTS notice URLs follow:
        https://www.find-tender.service.gov.uk/Notice/<noticeId>
    The noticeId is usually the OCDS release id or ocid suffix.
    """
    for key in ("id", "ocid"):
        value = release.get(key, "")
        if value:
            # Strip the OCDS prefix (e.g. "ocds-h6vhtk-001234" → "001234")
            clean = str(value).split("-")[-1] if "-" in str(value) else str(value)
            return f"{UK_FTS_WEB_BASE}/{clean}"

    tender = release.get("tender", {})
    tender_id = tender.get("id", "") if isinstance(tender, dict) else ""
    if tender_id:
        return f"{UK_FTS_WEB_BASE}/{tender_id}"

    return "https://www.find-tender.service.gov.uk/Search"


class UkFtsSearcher:
    """Searches UK Find a Tender Service for active EHS/SDS tenders.

    Probes each documented OCDS endpoint until one returns valid data,
    caches the working URL, and parses every release with the shared
    ``parse_ocds_release`` helper. Returns the top relevance-scored
    leads.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.10,
    ) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        self._working_endpoint: str | None = None
        logger.info("uk_fts_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[OcdsTenderLead]:
        logger.info(
            "uk_fts_search_start",
            days_back=days_back,
            max_results=max_results,
        )

        releases = self._fetch_releases(days_back)
        if not releases:
            logger.warning("uk_fts_no_releases")
            return []

        leads = self._parse_and_filter(releases, user_query)
        leads.sort(key=lambda l: l.relevance_score, reverse=True)
        leads = leads[:max_results]

        logger.info(
            "uk_fts_search_complete",
            raw_count=len(releases),
            filtered_count=len(leads),
            top_score=leads[0].relevance_score if leads else 0,
        )
        return leads

    # ------------------------------------------------------------------
    # API fetch
    # ------------------------------------------------------------------

    def _fetch_releases(self, days_back: int) -> list[dict]:
        now = datetime.now(timezone.utc)
        from_iso = (now - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%S")

        # FTS doc-confirmed query param is `updatedFrom`. Older paths
        # historically used `publishedFrom`. Send both — extras ignored.
        params: dict[str, str] = {
            "updatedFrom": from_iso,
            "publishedFrom": from_iso,
            "limit": "100",
        }

        if self._working_endpoint:
            releases = self._fetch_from_url(self._working_endpoint, params)
            if releases:
                return releases
            self._working_endpoint = None

        for endpoint in UK_FTS_ENDPOINTS:
            releases = self._fetch_from_url(endpoint, params)
            if releases:
                self._working_endpoint = endpoint
                logger.info("uk_fts_endpoint_found", endpoint=endpoint, count=len(releases))
                return releases

        logger.error("uk_fts_all_endpoints_failed", endpoints_tried=len(UK_FTS_ENDPOINTS))
        return []

    def _fetch_from_url(self, url: str, params: dict[str, str]) -> list[dict]:
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            # OCDS release package: { releases: [...] }
            if isinstance(data, dict) and "releases" in data and isinstance(data["releases"], list):
                return data["releases"]

            # OCDS release-package envelope: { packages: [{ releases: [...] }, ...] }
            if isinstance(data, dict) and "packages" in data:
                out: list[dict] = []
                for pkg in data["packages"]:
                    if isinstance(pkg, dict) and isinstance(pkg.get("releases"), list):
                        out.extend(pkg["releases"])
                if out:
                    return out

            # FTS sometimes returns { results: [...] } with release-shape objects
            for key in ("results", "items", "records", "data"):
                if isinstance(data, dict) and isinstance(data.get(key), list):
                    return data[key]

            return []

        except httpx.HTTPStatusError as exc:
            logger.debug("uk_fts_http_error", status_code=exc.response.status_code, url=url[:120])
            return []
        except httpx.TimeoutException:
            logger.debug("uk_fts_timeout", url=url[:120])
            return []
        except Exception as exc:
            logger.debug(
                "uk_fts_fetch_error",
                error=str(exc),
                error_type=type(exc).__name__,
                url=url[:120],
            )
            return []

    # ------------------------------------------------------------------
    # Parse + filter
    # ------------------------------------------------------------------

    def _parse_and_filter(self, releases: list[dict], user_query: str) -> list[OcdsTenderLead]:
        leads: list[OcdsTenderLead] = []
        seen_ids: set[str] = set()

        for release in releases:
            try:
                lead = parse_ocds_release(
                    release,
                    source_portal="uk_fts",
                    build_url=_build_uk_fts_url,
                )
                if not lead:
                    continue
                if lead.lead_id in seen_ids:
                    continue
                seen_ids.add(lead.lead_id)

                score, keywords = score_relevance(
                    lead.title,
                    lead.description,
                    lead.cpv_code,
                    extra_strong=UK_EXTRA_STRONG,
                    extra_partial=UK_EXTRA_PARTIAL,
                )

                if user_query:
                    query_lower = user_query.lower()
                    text = f"{lead.title} {lead.description}".lower()
                    hits = sum(1 for w in query_lower.split() if w in text)
                    if hits:
                        score = round(min(score + min(hits * 0.05, 0.15), 1.0), 2)

                lead.relevance_score = score
                lead.relevance_keywords = keywords
                if score >= self.min_relevance:
                    leads.append(lead)

            except Exception as exc:
                logger.debug("uk_fts_release_parse_error", error=str(exc))

        return leads
