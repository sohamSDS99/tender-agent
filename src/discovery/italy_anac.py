"""
Italy ANAC (Autorità Nazionale Anticorruzione) — OCDS procurement API.

Queries the Italian National Anti-Corruption Authority's open-data portal
for active tender notices relevant to EHS / SDS / chemical safety.

Endpoint candidates (tried in order until one responds with HTTP 200):
  1. https://dati.anticorruzione.it/opendata/ocds/api/contracts?limit=100
  2. https://dati.anticorruzione.it/opendata/ocds/api/releases?limit=100
  3. https://dati.anticorruzione.it/opendata/api/3/action/package_search?q=ocds&rows=100

Swagger docs:
  https://dati.anticorruzione.it/opendata/ocds/api/ui

Free, public, no authentication required.  Returns OCDS release packages
covering contracts above EUR 40,000.

Usage:
    searcher = ItalyAnacSearcher()
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
    "https://dati.anticorruzione.it/opendata/ocds/api/contracts",
    "https://dati.anticorruzione.it/opendata/ocds/api/releases",
    "https://dati.anticorruzione.it/opendata/api/3/action/package_search",
]

SOURCE_PORTAL = "italy_anac"

# ---------------------------------------------------------------------------
# Country-specific keywords (Italian + English EHS/SDS terms)
# ---------------------------------------------------------------------------

EXTRA_STRONG_KEYWORDS: list[str] = [
    "scheda dati di sicurezza",
    "sicurezza chimica",
    "sostanze pericolose",
    "gestione rifiuti pericolosi",
    "salute e sicurezza sul lavoro",
]

EXTRA_PARTIAL_KEYWORDS: list[str] = [
    "sicurezza",
    "chimico",
    "ambientale",
    "rifiuti",
    "pericoloso",
]


# ---------------------------------------------------------------------------
# URL builder for individual notices
# ---------------------------------------------------------------------------

def _build_notice_url(release: dict) -> str:
    """Build a public-facing URL for an Italian ANAC notice.

    Tries to extract a CIG (Codice Identificativo Gara) from the release
    for the canonical API URL.  Falls back to the tender ID or OCID.
    """
    # Check for CIG in tender.id or release.id
    tender = release.get("tender", {})
    cig = tender.get("id", "") or release.get("id", "") or release.get("ocid", "")

    if cig:
        return f"https://dati.anticorruzione.it/opendata/ocds/api/contracts/{cig}"

    return ""


# ---------------------------------------------------------------------------
# Searcher class
# ---------------------------------------------------------------------------

class ItalyAnacSearcher:
    """Searches Italy's ANAC open-data portal for active EHS/SDS tenders.

    Iterates over candidate API endpoints until one returns HTTP 200, then
    parses the OCDS release package, scores each release for EHS/SDS
    relevance, and returns the top matches.

    Usage:
        searcher = ItalyAnacSearcher()
        leads = searcher.search("chemical safety")
    """

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.10,
    ) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        logger.info("italy_anac_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[OcdsTenderLead]:
        """Search ANAC for active EHS/SDS tenders.

        Args:
            user_query: User's search text (reserved for future boost logic).
            max_results: Maximum results to return.
            days_back: How many days back to consider.

        Returns:
            List of OcdsTenderLead objects, filtered and sorted by relevance.
        """
        logger.info(
            "italy_anac_search_start",
            days_back=days_back,
            max_results=max_results,
        )

        try:
            releases = self._fetch_releases(days_back)
        except Exception as exc:
            logger.error("italy_anac_fetch_error", error=str(exc))
            return []

        if not releases:
            logger.warning("italy_anac_no_releases")
            return []

        logger.info("italy_anac_raw_releases", total=len(releases))

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

                # Score relevance using shared scorer with Italian extras
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
                logger.debug("italy_anac_parse_error", error=str(exc))

        # Sort by relevance descending, then cap
        leads.sort(key=lambda l: l.relevance_score, reverse=True)
        leads = leads[:max_results]

        logger.info(
            "italy_anac_search_complete",
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
        cutoff_iso = (
            datetime.now(timezone.utc) - timedelta(days=days_back)
        ).strftime("%Y-%m-%dT00:00:00Z")

        cutoff_date_only = (
            datetime.now(timezone.utc) - timedelta(days=days_back)
        ).strftime("%Y-%m-%d")

        for endpoint_url in ENDPOINT_CANDIDATES:
            is_ckan = "action/package_search" in endpoint_url

            if is_ckan:
                # CKAN-style endpoint uses different parameter names
                param_sets: list[dict[str, Any]] = [
                    {"q": "ocds", "rows": 100},
                ]
            else:
                # OCDS-style endpoint — try several date parameter names
                param_sets = [
                    {"limit": 100, "publishedFrom": cutoff_iso},
                    {"limit": 100, "updatedFrom": cutoff_iso},
                    {"limit": 100, "since": cutoff_date_only},
                    {"limit": 100},
                ]

            for params in param_sets:
                try:
                    releases = self._try_endpoint(endpoint_url, params, is_ckan)
                    if releases is not None:
                        logger.info(
                            "italy_anac_endpoint_success",
                            url=endpoint_url,
                            params=list(params.keys()),
                            count=len(releases),
                        )
                        return releases
                except Exception as exc:
                    logger.debug(
                        "italy_anac_endpoint_attempt_failed",
                        url=endpoint_url,
                        params=list(params.keys()),
                        error=str(exc),
                    )

        logger.warning("italy_anac_all_endpoints_failed")
        return []

    def _try_endpoint(
        self,
        url: str,
        params: dict[str, Any],
        is_ckan: bool = False,
    ) -> list[dict] | None:
        """Attempt a single GET request to the given endpoint.

        Args:
            url: The API endpoint URL.
            params: Query parameters to send.
            is_ckan: Whether this is a CKAN-style endpoint (package_search).

        Returns:
            List of release dicts if HTTP 200 and parseable, else None.
        """
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                resp = client.get(url, params=params)

            if resp.status_code != 200:
                logger.debug(
                    "italy_anac_non_200",
                    url=url,
                    status=resp.status_code,
                )
                return None

            data = resp.json()

            # Handle CKAN package_search response shape
            if is_ckan:
                return self._parse_ckan_response(data)

            # Standard OCDS release package: {"releases": [...]}
            if isinstance(data, dict) and "releases" in data:
                releases = data["releases"]
                if isinstance(releases, list):
                    return releases

            # Flat array of releases
            if isinstance(data, list):
                return data

            # Try common alternative top-level keys
            for key in ("records", "data", "results", "notices", "contracts"):
                if isinstance(data, dict) and key in data:
                    items = data[key]
                    if isinstance(items, list):
                        # Records may wrap releases: [{"releases": [...]}]
                        if items and isinstance(items[0], dict) and "releases" in items[0]:
                            all_releases: list[dict] = []
                            for record in items:
                                all_releases.extend(record.get("releases", []))
                            return all_releases
                        return items

            logger.debug(
                "italy_anac_unexpected_shape",
                url=url,
                keys=list(data.keys()) if isinstance(data, dict) else type(data).__name__,
            )
            return None

        except httpx.HTTPStatusError as exc:
            logger.debug(
                "italy_anac_http_error",
                url=url,
                status=exc.response.status_code,
            )
            return None

        except httpx.RequestError as exc:
            logger.debug("italy_anac_request_error", url=url, error=str(exc))
            return None

        except Exception as exc:
            logger.debug("italy_anac_try_error", url=url, error=str(exc))
            return None

    def _parse_ckan_response(self, data: dict) -> list[dict] | None:
        """Parse a CKAN package_search response into OCDS releases.

        CKAN wraps results in:
            {"success": true, "result": {"results": [...]}}

        Each result may be a dataset with resources containing OCDS data,
        or it may directly embed release-like structures.

        Returns:
            List of release dicts, or None if unparseable.
        """
        try:
            if not data.get("success"):
                return None

            result = data.get("result", {})

            # CKAN search results are datasets, not individual releases.
            # Each dataset may have resources with OCDS JSON URLs.
            results_list = result.get("results", [])
            if not results_list:
                return None

            all_releases: list[dict] = []

            for dataset in results_list:
                # Check if the dataset itself looks like a release
                if "tender" in dataset:
                    all_releases.append(dataset)
                    continue

                # Check embedded releases
                if "releases" in dataset:
                    releases = dataset["releases"]
                    if isinstance(releases, list):
                        all_releases.extend(releases)
                    continue

                # Try to fetch resources that contain OCDS JSON
                resources = dataset.get("resources", [])
                for resource in resources:
                    resource_format = (resource.get("format", "") or "").upper()
                    resource_url = resource.get("url", "")

                    if resource_format == "JSON" and resource_url:
                        try:
                            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                                res_resp = client.get(resource_url)

                            if res_resp.status_code == 200:
                                res_data = res_resp.json()
                                if isinstance(res_data, dict) and "releases" in res_data:
                                    all_releases.extend(res_data["releases"])
                                elif isinstance(res_data, list):
                                    all_releases.extend(res_data)
                        except Exception as exc:
                            logger.debug(
                                "italy_anac_ckan_resource_fetch_error",
                                url=resource_url[:120],
                                error=str(exc),
                            )

            if all_releases:
                return all_releases

            return None

        except Exception as exc:
            logger.debug("italy_anac_ckan_parse_error", error=str(exc))
            return None
