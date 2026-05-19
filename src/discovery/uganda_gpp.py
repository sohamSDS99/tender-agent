"""
Uganda GPP (Government Procurement Portal) — OCDS integration.

Queries Uganda's Public Procurement and Disposal of Public Assets
Authority (PPDA) open data for tenders from 210+ government entities.

API endpoint:
  GET https://gpp.ppda.go.ug/public/open-data/ocds/releases
  Alternative: https://gpp.ppda.go.ug/api/ocds/releases.json

OCDS compliant — uses shared ocds_base.py for parsing.
No authentication required — completely free and public.

Usage:
    searcher = UgandaGppSearcher()
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

# Uganda-specific keywords
UGANDA_STRONG: list[str] = [
    "nema uganda",  # National Environment Management Authority
    "unbs",         # Uganda National Bureau of Standards
    "chemical handling", "laboratory safety",
    "pesticide management",
]

UGANDA_PARTIAL: list[str] = [
    "environmental impact", "waste disposal",
    "industrial safety", "water treatment",
    "pollution control", "fumigation",
    "petroleum", "mining safety",
]

ENDPOINTS: list[str] = [
    "https://gpp.ppda.go.ug/public/open-data/ocds/releases",
    "https://gpp.ppda.go.ug/api/ocds/releases.json",
    "https://gpp.ppda.go.ug/api/releases",
    "https://gpp.ppda.go.ug/adminapi/public/api/open-data/v1/releases.json",
]


class UgandaGppSearcher:
    """Searches Uganda's GPP portal for EHS/SDS-related tenders.

    Uganda's GPP covers central government, local government,
    and statutory bodies — 210+ procuring entities.

    Usage:
        searcher = UgandaGppSearcher()
        leads = searcher.search("hazardous waste")
    """

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.10,
    ) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        logger.info("uganda_gpp_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[OcdsTenderLead]:
        """Search Uganda GPP for active EHS/SDS tenders.

        Args:
            user_query: User search text (for logging).
            max_results: Maximum results to return.
            days_back: How far back to look.

        Returns:
            List of OcdsTenderLead sorted by relevance.
        """
        logger.info("uganda_search_start", user_query=user_query[:80] if user_query else "")

        releases = self._fetch_releases(days_back)
        logger.info("uganda_releases_fetched", count=len(releases))

        leads: list[OcdsTenderLead] = []
        for release in releases:
            try:
                lead = parse_ocds_release(
                    release,
                    source_portal="uganda_gpp",
                    build_url=lambda r: f"https://gpp.ppda.go.ug/public/open-data/ocds/tender/{r.get('ocid', '')}",
                )
                if lead is None:
                    continue

                score, keywords = score_relevance(
                    lead.title, lead.description,
                    lead.cpv_code,
                    extra_strong=UGANDA_STRONG,
                    extra_partial=UGANDA_PARTIAL,
                )
                lead.relevance_score = score
                lead.relevance_keywords = keywords

                if score >= self.min_relevance:
                    leads.append(lead)
            except Exception as exc:
                logger.debug("uganda_parse_error", error=str(exc))

        leads.sort(key=lambda l: l.relevance_score, reverse=True)
        result = leads[:max_results]

        logger.info("uganda_search_complete", total=len(result))
        return result

    def _fetch_releases(self, days_back: int) -> list[dict]:
        """Try multiple Uganda GPP endpoints."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        cutoff_str = cutoff.strftime("%Y-%m-%dT00:00:00Z")

        for endpoint in ENDPOINTS:
            releases = self._try_endpoint(endpoint, cutoff_str)
            if releases:
                logger.info("uganda_endpoint_success", endpoint=endpoint, count=len(releases))
                return releases

        logger.warning("uganda_all_endpoints_failed")
        return []

    def _try_endpoint(self, endpoint: str, since: str) -> list[dict]:
        """Try a single endpoint."""
        params: dict[str, Any] = {
            "releaseDate[gte]": since,
            "limit": 100,
            "offset": 0,
        }

        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                resp = client.get(endpoint, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.debug("uganda_endpoint_error", endpoint=endpoint, error=str(exc))
            return []

        return self._extract_releases(data)

    def _extract_releases(self, data: Any) -> list[dict]:
        """Extract OCDS releases from various response shapes."""
        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            if "releases" in data:
                releases = data["releases"]
                if isinstance(releases, list):
                    return releases

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

            for key in ("data", "results", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]

        return []
