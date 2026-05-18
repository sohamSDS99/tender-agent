"""
Germany oeffentlichevergabe.de (Bekanntmachungsservice) — OCDS procurement API.

Queries Germany's national procurement announcement service for active tender
notices relevant to EHS / SDS / chemical safety.

Endpoint candidates (tried in order until one responds with HTTP 200):
  1. https://www.oeffentlichevergabe.de/api/opendata/v1/releases?limit=100
  2. https://www.oeffentlichevergabe.de/api/ocds-api/releases?limit=100
  3. https://www.oeffentlichevergabe.de/api/v1/ocds/releases?limit=100

Swagger docs:
  https://www.oeffentlichevergabe.de/documentation/swagger-ui/opendata/index.html

Free, public, no authentication required.  Returns OCDS release packages
(eForms-compatible).

Usage:
    searcher = GermanyBkmsSearcher()
    leads = searcher.search("chemical safety")
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import structlog

from src.discovery.ocds_base import (
    OcdsTenderLead,
    parse_ocds_release,
    score_relevance,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# API endpoints — tried in order; first 200 wins
# ---------------------------------------------------------------------------

ENDPOINT_CANDIDATES: list[str] = [
    "https://www.oeffentlichevergabe.de/api/opendata/v1/releases",
    "https://www.oeffentlichevergabe.de/api/ocds-api/releases",
    "https://www.oeffentlichevergabe.de/api/v1/ocds/releases",
]

SOURCE_PORTAL = "germany_bkms"

# ---------------------------------------------------------------------------
# Country-specific keywords (German + English EHS/SDS terms)
# ---------------------------------------------------------------------------

EXTRA_STRONG_KEYWORDS: list[str] = [
    "sicherheitsdatenblatt",
    "chemikaliensicherheit",
    "gefahrstoff",
    "gefahrstoffmanagement",
    "arbeitsschutz software",
    "umweltschutz software",
]

EXTRA_PARTIAL_KEYWORDS: list[str] = [
    "sicherheit",
    "chemikalien",
    "umwelt",
    "gefahrstoff",
    "arbeitsschutz",
]


# ---------------------------------------------------------------------------
# URL builder for individual notices
# ---------------------------------------------------------------------------

def _build_notice_url(release: dict) -> str:
    """Build a public-facing URL for a German BKMS notice.

    Tries to extract the notice ID from the release, falling back to the
    OCID or release ID.  The public UI is available at:
        https://www.oeffentlichevergabe.de/ui/en/publication/{noticeId}
    """
    # Prefer an explicit noticeId if embedded in the release
    notice_id = release.get("id", "") or release.get("ocid", "")

    # Some releases nest a noticeId inside tender or tag
    if not notice_id:
        tender = release.get("tender", {})
        notice_id = tender.get("id", "")

    if notice_id:
        return f"https://www.oeffentlichevergabe.de/ui/en/publication/{notice_id}"

    return ""


# ---------------------------------------------------------------------------
# Searcher class
# ---------------------------------------------------------------------------

class GermanyBkmsSearcher:
    """Searches Germany's oeffentlichevergabe.de for active EHS/SDS tenders.

    Iterates over candidate API endpoints until one returns HTTP 200, then
    parses the OCDS release package, scores each release for EHS/SDS
    relevance, and returns the top matches.

    Usage:
        searcher = GermanyBkmsSearcher()
        leads = searcher.search("chemical safety")
    """

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.10,
    ) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        logger.info("germany_bkms_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[OcdsTenderLead]:
        """Search oeffentlichevergabe.de for active EHS/SDS tenders.

        Args:
            user_query: User's search text (reserved for future boost logic).
            max_results: Maximum results to return.
            days_back: How many days back to consider.

        Returns:
            List of OcdsTenderLead objects, filtered and sorted by relevance.
        """
        logger.info(
            "germany_bkms_search_start",
            days_back=days_back,
            max_results=max_results,
        )

        try:
            releases = self._fetch_releases(days_back)
        except Exception as exc:
            logger.error("germany_bkms_fetch_error", error=str(exc))
            return []

        if not releases:
            logger.warning("germany_bkms_no_releases")
            return []

        logger.info("germany_bkms_raw_releases", total=len(releases))

        # Parse, score, filter
        leads: list[OcdsTenderLead] = []
        seen_ids: set[str] = set()

        for release in releases:
            try:
                lead = parse_ocds_release(
                    release,
                    source_portal=SOURCE_PORTAL,
                    build_url=_build_notice_url,
                )
                if not lead:
                    continue

                # Deduplicate
                if lead.lead_id in seen_ids:
                    continue
                seen_ids.add(lead.lead_id)

                # Score relevance using shared scorer with German extras
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
                logger.debug("germany_bkms_parse_error", error=str(exc))

        # Sort by relevance descending, then cap
        leads.sort(key=lambda l: l.relevance_score, reverse=True)
        leads = leads[:max_results]

        logger.info(
            "germany_bkms_search_complete",
            total_results=len(leads),
            top_score=leads[0].relevance_score if leads else 0,
        )

        return leads

    # ------------------------------------------------------------------
    # API fetch — try endpoints in order
    # ------------------------------------------------------------------

    def _fetch_releases(self, days_back: int) -> list[dict]:
        """Try each candidate endpoint in order; return releases from the
        first one that responds with HTTP 200.

        Appends date-filtering query parameters where supported.

        Returns:
            List of raw OCDS release dicts, or empty list on failure.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days_back)
        ).strftime("%Y-%m-%dT00:00:00Z")

        cutoff_date_only = (
            datetime.now(timezone.utc) - timedelta(days=days_back)
        ).strftime("%Y-%m-%d")

        for endpoint_url in ENDPOINT_CANDIDATES:
            # Try multiple date-parameter names; the API may accept any one
            param_sets: list[dict[str, Any]] = [
                {"limit": 100, "publishedFrom": cutoff},
                {"limit": 100, "updatedFrom": cutoff},
                {"limit": 100, "since": cutoff_date_only},
                {"limit": 100},
            ]

            for params in param_sets:
                try:
                    releases = self._try_endpoint(endpoint_url, params)
                    if releases is not None:
                        logger.info(
                            "germany_bkms_endpoint_success",
                            url=endpoint_url,
                            params=list(params.keys()),
                            count=len(releases),
                        )
                        return releases
                except Exception as exc:
                    logger.debug(
                        "germany_bkms_endpoint_attempt_failed",
                        url=endpoint_url,
                        params=list(params.keys()),
                        error=str(exc),
                    )

        logger.warning("germany_bkms_all_endpoints_failed")
        return []

    def _try_endpoint(
        self,
        url: str,
        params: dict[str, Any],
    ) -> list[dict] | None:
        """Attempt a single GET request to the given endpoint.

        Returns:
            List of release dicts if HTTP 200 and parseable, else None.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url, params=params)

            if resp.status_code != 200:
                logger.debug(
                    "germany_bkms_non_200",
                    url=url,
                    status=resp.status_code,
                )
                return None

            data = resp.json()

            # OCDS release package: {"releases": [...]}
            if isinstance(data, dict) and "releases" in data:
                releases = data["releases"]
                if isinstance(releases, list):
                    return releases

            # Flat array of releases
            if isinstance(data, list):
                return data

            # Try common alternative top-level keys
            for key in ("records", "data", "results", "notices"):
                if isinstance(data, dict) and key in data:
                    items = data[key]
                    if isinstance(items, list):
                        # records may wrap releases: [{"releases": [...]}]
                        if items and isinstance(items[0], dict) and "releases" in items[0]:
                            all_releases: list[dict] = []
                            for record in items:
                                all_releases.extend(record.get("releases", []))
                            return all_releases
                        return items

            logger.debug(
                "germany_bkms_unexpected_shape",
                url=url,
                keys=list(data.keys()) if isinstance(data, dict) else type(data).__name__,
            )
            return None

        except httpx.HTTPStatusError as exc:
            logger.debug(
                "germany_bkms_http_error",
                url=url,
                status=exc.response.status_code,
            )
            return None

        except httpx.RequestError as exc:
            logger.debug("germany_bkms_request_error", url=url, error=str(exc))
            return None

        except Exception as exc:
            logger.debug("germany_bkms_try_error", url=url, error=str(exc))
            return None
