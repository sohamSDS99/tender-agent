"""
AusTender — Australia's federal procurement OCDS API.

Queries AusTender's public OCDS API for active tender notices relevant
to EHS/SDS/chemical safety:
  https://api.tenders.gov.au/ocds/findByDates/contractPublished/{from}/{to}

Free, public, no authentication required.  Returns an OCDS release
package (``releases`` array) containing standard OCDS 1.1 objects.

This module is a thin wrapper around ``ocds_base`` — it only defines the
AusTender-specific API endpoint, URL builder, and searcher class.  All
keyword scoring and OCDS release parsing is delegated to shared helpers.

Usage:
    searcher = AusTenderSearcher()
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
# API endpoints
# ---------------------------------------------------------------------------

# Primary endpoint — search by contract publication date range
AUSTENDER_OCDS_URL = (
    "https://api.tenders.gov.au/ocds/findByDates/contractPublished"
)

# Alternative — search by last-modified date range
AUSTENDER_OCDS_MODIFIED_URL = (
    "https://api.tenders.gov.au/ocds/findByDates/lastModified"
)

# Public web view base
AUSTENDER_WEB_BASE = "https://www.tenders.gov.au/"


# ---------------------------------------------------------------------------
# URL builder for AusTender
# ---------------------------------------------------------------------------

def _build_austender_url(release: dict) -> str:
    """Build a public AusTender web URL from an OCDS release.

    AusTender's web interface uses a CNUUID parameter.  We try to extract
    a usable identifier from the release's ``id``, ``ocid``, or the
    ``tender.id`` field.  If none yields a plausible UUID, we fall back
    to the portal homepage.
    """
    # Try release-level identifiers first
    for key in ("id", "ocid"):
        value = release.get(key, "")
        if value:
            # Strip common OCDS prefixes (e.g. "ocds-abc123-" prefix)
            # to get the raw UUID/CN identifier
            parts = str(value).rsplit("-", 1)
            candidate = parts[-1] if len(parts) > 1 else str(value)
            if candidate:
                return (
                    f"https://www.tenders.gov.au/"
                    f"?event=public.cn.view&CNUUID={candidate}"
                )

    # Try tender.id inside the release
    tender = release.get("tender", {})
    tender_id = tender.get("id", "") if isinstance(tender, dict) else ""
    if tender_id:
        return (
            f"https://www.tenders.gov.au/"
            f"?event=public.cn.view&CNUUID={tender_id}"
        )

    # Fallback — portal homepage
    return AUSTENDER_WEB_BASE


# ---------------------------------------------------------------------------
# AusTender searcher class
# ---------------------------------------------------------------------------

class AusTenderSearcher:
    """Searches Australia's AusTender OCDS API for active EHS/SDS tenders.

    Fetches OCDS release packages from AusTender, parses each release
    with the shared ``parse_ocds_release`` helper, scores for EHS/SDS
    relevance, and returns the top matches.

    Usage:
        searcher = AusTenderSearcher()
        leads = searcher.search("chemical safety")
    """

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.10,
    ) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        logger.info("austender_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[OcdsTenderLead]:
        """Search AusTender for active EHS/SDS tenders.

        Args:
            user_query: User's search text (used as an additional keyword
                        signal during relevance scoring).
            max_results: Maximum results to return.
            days_back: How many days back to search.

        Returns:
            List of OcdsTenderLead objects, filtered by relevance and
            sorted descending by relevance_score.
        """
        logger.info(
            "austender_search_start",
            days_back=days_back,
            max_results=max_results,
        )

        releases = self._fetch_releases(days_back)

        if not releases:
            logger.warning(
                "austender_no_releases",
                msg="API returned no releases; returning empty list",
            )
            return []

        # Parse, score, and filter
        leads = self._parse_and_filter(releases, user_query)

        # Sort by relevance (highest first) and cap
        leads.sort(key=lambda lead: lead.relevance_score, reverse=True)
        leads = leads[:max_results]

        logger.info(
            "austender_search_complete",
            raw_count=len(releases),
            filtered_count=len(leads),
            top_score=leads[0].relevance_score if leads else 0,
        )

        return leads

    # ------------------------------------------------------------------
    # API fetch
    # ------------------------------------------------------------------

    def _fetch_releases(self, days_back: int) -> list[dict]:
        """Fetch OCDS releases from the AusTender API.

        Tries the primary ``contractPublished`` endpoint first; if that
        fails (HTTP error, empty results) falls back to the
        ``lastModified`` endpoint.

        Date parameters use ISO 8601 with Z suffix, e.g.
        ``2026-04-01T00:00:00Z``.

        Returns a list of raw OCDS release dicts, or an empty list on
        any failure.
        """
        now = datetime.now(timezone.utc)
        from_date = now - timedelta(days=days_back)

        from_str = from_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        to_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Try primary endpoint
        primary_url = f"{AUSTENDER_OCDS_URL}/{from_str}/{to_str}"
        releases = self._fetch_from_url(primary_url)

        if releases:
            return releases

        # Fallback to lastModified endpoint
        logger.info(
            "austender_fallback_to_last_modified",
            msg="Primary endpoint returned no results; trying lastModified",
        )
        fallback_url = f"{AUSTENDER_OCDS_MODIFIED_URL}/{from_str}/{to_str}"
        return self._fetch_from_url(fallback_url)

    def _fetch_from_url(self, url: str) -> list[dict]:
        """Fetch and extract releases from a single AusTender OCDS URL.

        Handles both a top-level ``releases`` array (standard OCDS release
        package) and paginated / alternative response shapes.

        Returns a list of release dicts or an empty list on failure.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()

            # Standard OCDS release package
            if isinstance(data, dict) and "releases" in data:
                releases = data["releases"]
                if isinstance(releases, list):
                    logger.info(
                        "austender_releases_fetched",
                        url=url[:120],
                        count=len(releases),
                    )
                    return releases

            # Flat array of releases
            if isinstance(data, list):
                logger.info(
                    "austender_releases_flat_array",
                    url=url[:120],
                    count=len(data),
                )
                return data

            # Try common alternative keys
            for key in ("records", "data", "results"):
                if isinstance(data, dict) and key in data:
                    records = data[key]
                    if isinstance(records, list):
                        logger.info(
                            "austender_releases_alt_key",
                            url=url[:120],
                            key=key,
                            count=len(records),
                        )
                        return records

            logger.warning(
                "austender_unexpected_shape",
                url=url[:120],
                keys=list(data.keys()) if isinstance(data, dict) else type(data).__name__,
            )
            return []

        except httpx.HTTPStatusError as exc:
            logger.error(
                "austender_http_error",
                status_code=exc.response.status_code,
                url=url[:120],
            )
            return []

        except httpx.TimeoutException:
            logger.error("austender_timeout", url=url[:120])
            return []

        except Exception as exc:
            logger.error(
                "austender_fetch_error",
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
                    source_portal="austender",
                    build_url=_build_austender_url,
                )
                if not lead:
                    continue

                # Deduplicate by lead_id
                if lead.lead_id in seen_ids:
                    continue
                seen_ids.add(lead.lead_id)

                # Score relevance using shared scorer (no extra AU-specific keywords)
                score, keywords = score_relevance(
                    lead.title,
                    lead.description,
                    lead.cpv_code,
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
                    "austender_release_parse_error",
                    error=str(exc),
                )

        return leads
