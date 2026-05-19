"""
SA eTender — South Africa's government procurement OCDS API.

Queries South Africa's eTender OCDS API for active tender notices
relevant to EHS/SDS/chemical safety:
  https://ocds-api.etenders.gov.za/

Free, public, no authentication required.  The exact endpoint structure
is probed at runtime — we try several known patterns and use the first
that returns a valid response.

This module is a thin wrapper around ``ocds_base`` — it only defines the
SA-specific API endpoint, URL builder, extra keywords, and searcher
class.  All keyword scoring and OCDS release parsing is delegated to
shared helpers.

Usage:
    searcher = SaEtenderSearcher()
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
# API endpoints — tried in order until one returns HTTP 200
# ---------------------------------------------------------------------------

SA_ETENDER_ENDPOINTS: list[str] = [
    "https://ocds-api.etenders.gov.za/api/ocds/releases",
    "https://ocds-api.etenders.gov.za/releases",
    "https://ocds-api.etenders.gov.za/api/releases",
    "https://ocds-api.etenders.gov.za/api/ocds/release-packages",
]

# Public web view base
SA_ETENDER_WEB_BASE = "https://www.etenders.gov.za/content/advertised-tenders"

# ---------------------------------------------------------------------------
# South Africa-specific extra keywords
# ---------------------------------------------------------------------------

SA_EXTRA_STRONG: list[str] = [
    "broad-based black economic empowerment",
    "b-bbee",
]

SA_EXTRA_PARTIAL: list[str] = []


# ---------------------------------------------------------------------------
# URL builder for SA eTender
# ---------------------------------------------------------------------------

def _build_sa_etender_url(release: dict) -> str:
    """Build a public SA eTender web URL from an OCDS release.

    If we can extract a usable tender identifier we build a specific URL;
    otherwise we fall back to the advertised-tenders listing page.
    """
    # Try release-level identifiers
    for key in ("id", "ocid"):
        value = release.get(key, "")
        if value:
            return f"{SA_ETENDER_WEB_BASE}?id={value}"

    # Try tender.id inside the release
    tender = release.get("tender", {})
    tender_id = tender.get("id", "") if isinstance(tender, dict) else ""
    if tender_id:
        return f"{SA_ETENDER_WEB_BASE}?id={tender_id}"

    # Fallback — listing page
    return SA_ETENDER_WEB_BASE


# ---------------------------------------------------------------------------
# SA eTender searcher class
# ---------------------------------------------------------------------------

class SaEtenderSearcher:
    """Searches South Africa's eTender OCDS API for active EHS/SDS tenders.

    Probes multiple possible endpoint patterns at runtime to find the
    one that returns data, fetches OCDS releases, parses each with the
    shared ``parse_ocds_release`` helper, scores for EHS/SDS relevance
    (including B-BBEE keywords), and returns the top matches.

    Usage:
        searcher = SaEtenderSearcher()
        leads = searcher.search("chemical safety")
    """

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.10,
    ) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        self._working_endpoint: str | None = None
        logger.info("sa_etender_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[OcdsTenderLead]:
        """Search SA eTender for active EHS/SDS tenders.

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
            "sa_etender_search_start",
            days_back=days_back,
            max_results=max_results,
        )

        releases = self._fetch_releases(days_back)

        if not releases:
            logger.warning(
                "sa_etender_no_releases",
                msg="API returned no releases; returning empty list",
            )
            return []

        # Parse, score, and filter
        leads = self._parse_and_filter(releases, user_query)

        # Sort by relevance (highest first) and cap
        leads.sort(key=lambda lead: lead.relevance_score, reverse=True)
        leads = leads[:max_results]

        logger.info(
            "sa_etender_search_complete",
            raw_count=len(releases),
            filtered_count=len(leads),
            top_score=leads[0].relevance_score if leads else 0,
        )

        return leads

    # ------------------------------------------------------------------
    # API fetch — probe multiple endpoints
    # ------------------------------------------------------------------

    def _fetch_releases(self, days_back: int) -> list[dict]:
        """Fetch OCDS releases from the SA eTender API.

        Tries each candidate endpoint in order.  Once a working endpoint
        is found it is cached for subsequent calls within the same
        searcher instance.

        Returns a list of raw OCDS release dicts, or an empty list on
        any failure.
        """
        now = datetime.now(timezone.utc)
        from_date = now - timedelta(days=days_back)

        from_str = from_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        to_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Common query parameters for date filtering and pagination
        params: dict[str, str] = {
            "since": from_str,
            "from": from_str,
            "to": to_str,
            "publishedFrom": from_date.strftime("%Y-%m-%d"),
            "publishedTo": now.strftime("%Y-%m-%d"),
            "page": "1",
            "limit": "200",
        }

        # If we already know which endpoint works, use it directly
        if self._working_endpoint:
            releases = self._fetch_from_url(self._working_endpoint, params)
            if releases:
                return releases
            # Cached endpoint stopped working — clear and re-probe
            self._working_endpoint = None

        # Probe each candidate endpoint
        for endpoint in SA_ETENDER_ENDPOINTS:
            releases = self._fetch_from_url(endpoint, params)
            if releases:
                self._working_endpoint = endpoint
                logger.info(
                    "sa_etender_endpoint_found",
                    endpoint=endpoint,
                    count=len(releases),
                )
                return releases

        # All endpoints failed — try base URL as last resort
        base_url = "https://ocds-api.etenders.gov.za/"
        releases = self._fetch_from_url(base_url, params)
        if releases:
            self._working_endpoint = base_url
            return releases

        logger.error(
            "sa_etender_all_endpoints_failed",
            endpoints_tried=len(SA_ETENDER_ENDPOINTS) + 1,
        )
        return []

    def _fetch_from_url(
        self,
        url: str,
        params: dict[str, str],
    ) -> list[dict]:
        """Fetch and extract releases from a single SA eTender URL.

        Handles standard OCDS release packages, paginated responses,
        and various alternative JSON shapes.

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
                        "sa_etender_releases_fetched",
                        url=url[:120],
                        count=len(releases),
                    )
                    return releases

            # Flat array of releases
            if isinstance(data, list):
                logger.info(
                    "sa_etender_releases_flat_array",
                    url=url[:120],
                    count=len(data),
                )
                return data

            # Paginated response — try common wrapper keys
            for key in ("records", "data", "results", "items", "content"):
                if isinstance(data, dict) and key in data:
                    records = data[key]
                    if isinstance(records, list):
                        logger.info(
                            "sa_etender_releases_alt_key",
                            url=url[:120],
                            key=key,
                            count=len(records),
                        )
                        return records

            logger.debug(
                "sa_etender_unexpected_shape",
                url=url[:120],
                keys=list(data.keys()) if isinstance(data, dict) else type(data).__name__,
            )
            return []

        except httpx.HTTPStatusError as exc:
            logger.debug(
                "sa_etender_http_error",
                status_code=exc.response.status_code,
                url=url[:120],
            )
            return []

        except httpx.TimeoutException:
            logger.debug("sa_etender_timeout", url=url[:120])
            return []

        except Exception as exc:
            logger.debug(
                "sa_etender_fetch_error",
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
                    source_portal="sa_etender",
                    build_url=_build_sa_etender_url,
                )
                if not lead:
                    continue

                # Deduplicate by lead_id
                if lead.lead_id in seen_ids:
                    continue
                seen_ids.add(lead.lead_id)

                # Score relevance with SA-specific extra keywords
                score, keywords = score_relevance(
                    lead.title,
                    lead.description,
                    lead.cpv_code,
                    extra_strong=SA_EXTRA_STRONG,
                    extra_partial=SA_EXTRA_PARTIAL,
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
                    "sa_etender_release_parse_error",
                    error=str(exc),
                )

        return leads
