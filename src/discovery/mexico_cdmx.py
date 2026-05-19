"""
Mexico City CDMX Open Contracting — OCDS integration.

Queries Mexico City's Tianguis Digital open data portal for
procurement notices from city government agencies.

API endpoint:
  GET https://datosabiertostianguisdigital.cdmx.gob.mx/api/v1/plannings
  Alternative: https://datosabiertostianguisdigital.cdmx.gob.mx/api/v1/releases

OCDS compliant — uses shared ocds_base.py for parsing.
No authentication required — completely free and public.

Note: This covers Mexico City (CDMX) procurement only, not all of Mexico.
For federal Mexican procurement, CompraNet would be needed (separate system).

Usage:
    searcher = MexicoCdmxSearcher()
    leads = searcher.search("quimico seguridad")
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

# Mexico/Spanish-specific strong keywords
MEXICO_STRONG: list[str] = [
    # Spanish terms
    "hoja de seguridad", "hoja de datos de seguridad",
    "seguridad quimica", "manejo de quimicos",
    "sustancias peligrosas", "materiales peligrosos",
    "residuos peligrosos", "seguridad industrial",
    "seguridad ocupacional", "salud ocupacional",
    "medio ambiente", "impacto ambiental",
    # Regulatory bodies
    "semarnat", "cofepris", "stps",
    "profepa", "conagua",
    # English
    "chemical safety", "hazardous material",
    "safety data sheet", "environmental health",
]

MEXICO_PARTIAL: list[str] = [
    # Spanish
    "seguridad", "quimico", "peligroso",
    "ambiental", "residuos", "contaminacion",
    "toxico", "fumigacion", "plaguicida",
    # English
    "safety", "chemical", "hazard",
    "environmental", "waste", "pollution",
]

ENDPOINTS: list[str] = [
    "https://datosabiertostianguisdigital.cdmx.gob.mx/api/v1/releases",
    "https://datosabiertostianguisdigital.cdmx.gob.mx/api/v1/plannings",
    "https://tianguisdigital.cdmx.gob.mx/api/v1/releases",
    "https://tianguisdigital.cdmx.gob.mx/api/ocds/releases",
]


class MexicoCdmxSearcher:
    """Searches Mexico City CDMX Tianguis Digital for EHS/SDS tenders.

    Mexico City's open contracting portal publishes OCDS data for
    all city government procurement. Covers environment, health,
    public works, and service contracts.

    Usage:
        searcher = MexicoCdmxSearcher()
        leads = searcher.search("residuos peligrosos")
    """

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.10,
    ) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        logger.info("mexico_cdmx_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[OcdsTenderLead]:
        """Search Mexico City CDMX for active EHS/SDS tenders.

        Args:
            user_query: User search text (for logging).
            max_results: Maximum results to return.
            days_back: How far back to look.

        Returns:
            List of OcdsTenderLead sorted by relevance.
        """
        logger.info("mexico_search_start", user_query=user_query[:80] if user_query else "")

        releases = self._fetch_releases(days_back)
        logger.info("mexico_releases_fetched", count=len(releases))

        leads: list[OcdsTenderLead] = []
        for release in releases:
            try:
                lead = parse_ocds_release(
                    release,
                    source_portal="mexico_cdmx",
                    build_url=lambda r: f"https://tianguisdigital.cdmx.gob.mx/contrataciones/{r.get('ocid', '')}",
                )
                if lead is None:
                    continue

                score, keywords = score_relevance(
                    lead.title, lead.description,
                    lead.cpv_code,
                    extra_strong=MEXICO_STRONG,
                    extra_partial=MEXICO_PARTIAL,
                )
                lead.relevance_score = score
                lead.relevance_keywords = keywords

                if score >= self.min_relevance:
                    leads.append(lead)
            except Exception as exc:
                logger.debug("mexico_parse_error", error=str(exc))

        leads.sort(key=lambda l: l.relevance_score, reverse=True)
        result = leads[:max_results]

        logger.info("mexico_search_complete", total=len(result))
        return result

    def _fetch_releases(self, days_back: int) -> list[dict]:
        """Try multiple CDMX endpoints."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        cutoff_str = cutoff.strftime("%Y-%m-%dT00:00:00Z")

        for endpoint in ENDPOINTS:
            releases = self._try_endpoint(endpoint, cutoff_str)
            if releases:
                logger.info("mexico_endpoint_success", endpoint=endpoint, count=len(releases))
                return releases

        logger.warning("mexico_all_endpoints_failed")
        return []

    def _try_endpoint(self, endpoint: str, since: str) -> list[dict]:
        """Try a single endpoint."""
        params: dict[str, Any] = {}

        if "plannings" in endpoint:
            params = {"page": 1, "pageSize": 100}
        else:
            params = {
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
            logger.debug("mexico_endpoint_error", endpoint=endpoint, error=str(exc))
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

            for key in ("data", "results", "items", "plannings"):
                if key in data and isinstance(data[key], list):
                    return data[key]

        return []
