"""
Nigeria NOCOPO (National Open Contracting Portal) — OCDS integration.

Queries Nigeria's Bureau of Public Procurement (BPP) open data portal
for tenders from 700+ federal agencies.

API endpoint:
  GET https://nocopo.bpp.gov.ng/api/ocds/releases
  Alternative: https://nocopo.bpp.gov.ng/ocds/api/releases.json

OCDS compliant — uses shared ocds_base.py for parsing.
No authentication required — completely free and public.

Usage:
    searcher = NigeriaNocopoSearcher()
    leads = searcher.search("chemical safety")
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import structlog

from .ocds_base import (
    OcdsTenderLead,
    parse_ocds_release,
    score_relevance,
)

logger = structlog.get_logger(__name__)

# Nigeria-specific strong keywords (Pidgin/local terms + English)
NIGERIA_STRONG: list[str] = [
    "chemical handling", "laboratory safety",
    "oil spill", "gas flaring", "environmental remediation",
    "nesrea",  # National Environmental Standards Regulatory Agency
    "son",     # Standards Organisation of Nigeria
    "nafdac",  # National Agency for Food and Drug Admin
]

NIGERIA_PARTIAL: list[str] = [
    "petroleum", "refinery", "pipeline safety",
    "environmental impact", "waste disposal",
    "industrial safety", "fire safety",
]

# Multiple endpoint attempts (OCDS portal URLs vary)
ENDPOINTS: list[str] = [
    "https://nocopo.bpp.gov.ng/api/ocds/releases",
    "https://nocopo.bpp.gov.ng/ocds/api/releases.json",
    "https://nocopo.bpp.gov.ng/api/releases",
    "https://nocopo.bpp.gov.ng/opendata/api/3/action/package_search",
]


class NigeriaNocopoSearcher:
    """Searches Nigeria's NOCOPO portal for EHS/SDS-related tenders.

    Tries multiple endpoint variants since the portal has been
    recently modernized and URL patterns may vary.

    Usage:
        searcher = NigeriaNocopoSearcher()
        leads = searcher.search("hazardous waste management")
    """

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.10,
    ) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        logger.info("nigeria_nocopo_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[OcdsTenderLead]:
        """Search NOCOPO for active EHS/SDS tenders.

        Args:
            user_query: User search text (for logging).
            max_results: Maximum results to return.
            days_back: How far back to look for tenders.

        Returns:
            List of OcdsTenderLead sorted by relevance.
        """
        logger.info("nigeria_search_start", user_query=user_query[:80] if user_query else "")

        releases = self._fetch_releases(days_back)
        logger.info("nigeria_releases_fetched", count=len(releases))

        leads: list[OcdsTenderLead] = []
        for release in releases:
            try:
                lead = parse_ocds_release(
                    release,
                    source_portal="nigeria_nocopo",
                    build_url=lambda r: f"https://nocopo.bpp.gov.ng/tender/{r.get('ocid', '')}",
                )
                if lead is None:
                    continue

                score, keywords = score_relevance(
                    lead.title, lead.description,
                    lead.cpv_code,
                    extra_strong=NIGERIA_STRONG,
                    extra_partial=NIGERIA_PARTIAL,
                )
                lead.relevance_score = score
                lead.relevance_keywords = keywords

                if score >= self.min_relevance:
                    leads.append(lead)
            except Exception as exc:
                logger.debug("nigeria_parse_error", error=str(exc))

        leads.sort(key=lambda l: l.relevance_score, reverse=True)
        result = leads[:max_results]

        logger.info("nigeria_search_complete", total=len(result))
        return result

    def _fetch_releases(self, days_back: int) -> list[dict]:
        """Try multiple NOCOPO endpoints to fetch OCDS releases."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        cutoff_str = cutoff.strftime("%Y-%m-%dT00:00:00Z")

        for endpoint in ENDPOINTS:
            releases = self._try_endpoint(endpoint, cutoff_str)
            if releases:
                logger.info("nigeria_endpoint_success", endpoint=endpoint, count=len(releases))
                return releases

        logger.warning("nigeria_all_endpoints_failed")
        return []

    def _try_endpoint(self, endpoint: str, since: str) -> list[dict]:
        """Try a single endpoint URL."""
        params: dict[str, Any] = {}

        if "package_search" in endpoint:
            # CKAN-style endpoint
            params = {"q": "tender", "rows": 100}
        else:
            # OCDS-style endpoint
            params = {
                "releaseDate[gte]": since,
                "limit": 100,
                "offset": 0,
            }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(endpoint, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.debug("nigeria_endpoint_error", endpoint=endpoint, error=str(exc))
            return []

        return self._extract_releases(data)

    def _extract_releases(self, data: Any) -> list[dict]:
        """Extract OCDS releases from various response formats."""
        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            # Standard OCDS release package
            if "releases" in data:
                releases = data["releases"]
                if isinstance(releases, list):
                    return releases

            # OCDS record package
            if "records" in data:
                releases = []
                for record in data["records"]:
                    if isinstance(record, dict):
                        compiled = record.get("compiledRelease")
                        if compiled:
                            releases.append(compiled)
                        else:
                            rel_list = record.get("releases", [])
                            if rel_list:
                                releases.append(rel_list[-1])
                return releases

            # Paginated wrapper
            for key in ("data", "results", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]

            # CKAN response
            if "result" in data:
                result = data["result"]
                if isinstance(result, dict) and "results" in result:
                    return result["results"]
                if isinstance(result, list):
                    return result

        return []
