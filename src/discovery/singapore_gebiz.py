"""
Singapore GeBIZ — Singapore government procurement (via data.gov.sg).

Singapore's Government eBusiness portal (GeBIZ) publishes every public
tender. The official open-data mirror at data.gov.sg exposes the same
notices through a CKAN-style datastore API.

  Portal:    https://www.gebiz.gov.sg
  Open data: https://data.gov.sg (search "GeBIZ government procurement")
  CKAN API:  https://data.gov.sg/api/3/action/datastore_search

Free, public, no authentication required. The dataset is partitioned
by financial year; we try the current and previous FY resource IDs in
order and fall through to the v2 unified API if both miss.

Usage:
    searcher = SingaporeGebizSearcher()
    leads = searcher.search("chemical safety")
"""

from __future__ import annotations

from datetime import datetime
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
# CKAN resource IDs for the "Government Procurement via GeBIZ" dataset.
# data.gov.sg publishes a new resource per financial year. These IDs are
# stable but new ones get added each FY — module probes the most likely
# current-year IDs and falls back to the newer v2 API discovery query.
# ---------------------------------------------------------------------------

# Known historical resource IDs (oldest → newest). The most recent
# one usually carries the active tenders. Tried in reverse order.
GEBIZ_RESOURCE_IDS: list[str] = [
    "d_69b3380ad7e51aff385ac3450de3cb24",   # FY2024 (assumed; verify after first run)
    "d_3f960c10fed6145404ca7b821f263b87",   # FY2023
    "d_acff7c41cd5a5e8a4d80b87cbfba8af2",   # FY2022
]

CKAN_BASES: list[str] = [
    "https://data.gov.sg/api/3/action/datastore_search",
    "https://api-production.data.gov.sg/v2/public/api/datasets/<resource_id>/records",
]

# Direct search across the data.gov.sg catalogue (last-resort)
PACKAGE_SEARCH_URL = "https://data.gov.sg/api/3/action/package_search"

SG_EXTRA_STRONG: list[str] = [
    "chemical safety",
    "hazardous chemical",
    "wsh",                      # Workplace Safety and Health Act
    "wshc",
    "national environment agency",
    "nea",
]
SG_EXTRA_PARTIAL: list[str] = [
    "itq",                      # Invitation to Quote
    "itt",                      # Invitation to Tender
    "open quotation",
    "open tender",
]


def _build_gebiz_url(release: dict) -> str:
    """Build a public GeBIZ notice URL.

    GeBIZ notice URLs use ?docid=, but per-record source URLs aren't
    always embedded in the CKAN response — fall back to the search page.
    """
    tender_no = (
        release.get("tender_no")
        or release.get("tender_number")
        or release.get("id")
        or ""
    )
    if tender_no:
        return f"https://www.gebiz.gov.sg/ptn/opportunity/BOListing.xhtml?searchValue={tender_no}"
    return "https://www.gebiz.gov.sg/ptn/opportunity/BOListing.xhtml"


class SingaporeGebizSearcher:
    """Searches Singapore GeBIZ via data.gov.sg CKAN datastore."""

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.10,
    ) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        self._working_resource_id: str | None = None
        logger.info("gebiz_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,  # CKAN doesn't filter by date — handled downstream
    ) -> list[OcdsTenderLead]:
        logger.info("gebiz_search_start")
        records = self._fetch_records(user_query)
        if not records:
            logger.warning("gebiz_no_records")
            return []
        leads = self._parse_and_filter(records, user_query)
        leads.sort(key=lambda l: l.relevance_score, reverse=True)
        return leads[:max_results]

    def _fetch_records(self, user_query: str) -> list[dict]:
        # Try known resource IDs first (newest first per insertion order)
        if self._working_resource_id:
            r = self._fetch_resource(self._working_resource_id, user_query)
            if r:
                return r
            self._working_resource_id = None

        for rid in GEBIZ_RESOURCE_IDS:
            r = self._fetch_resource(rid, user_query)
            if r:
                self._working_resource_id = rid
                logger.info("gebiz_resource_found", resource_id=rid, count=len(r))
                return r

        # Fallback: query the catalogue for any "GeBIZ" dataset and
        # use its latest resource. Catches the case where data.gov.sg
        # has rotated all our known IDs.
        records = self._discover_resource_via_catalogue(user_query)
        if records:
            return records

        logger.error("gebiz_all_paths_failed")
        return []

    def _fetch_resource(self, resource_id: str, user_query: str) -> list[dict]:
        params: dict[str, Any] = {"resource_id": resource_id, "limit": "500"}
        if user_query:
            params["q"] = user_query[:120]
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                resp = client.get(CKAN_BASES[0], params=params)
                resp.raise_for_status()
                data = resp.json()
            if not isinstance(data, dict):
                return []
            if not data.get("success"):
                return []
            result = data.get("result", {})
            records = result.get("records", [])
            if isinstance(records, list):
                return records
            return []
        except httpx.HTTPStatusError as exc:
            logger.debug("gebiz_http_error", status=exc.response.status_code, resource=resource_id)
            return []
        except httpx.TimeoutException:
            logger.debug("gebiz_timeout", resource=resource_id)
            return []
        except Exception as exc:
            logger.debug("gebiz_fetch_error", error=str(exc), resource=resource_id)
            return []

    def _discover_resource_via_catalogue(self, user_query: str) -> list[dict]:
        """Last-resort: search the data.gov.sg catalogue for any
        'GeBIZ' dataset, take the most-recently-modified resource,
        and pull its records."""
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                resp = client.get(PACKAGE_SEARCH_URL, params={"q": "gebiz government procurement", "rows": 5})
                resp.raise_for_status()
                data = resp.json()
            if not (isinstance(data, dict) and data.get("success")):
                return []
            results = data.get("result", {}).get("results", [])
            for pkg in results:
                for res in pkg.get("resources", []):
                    rid = res.get("id")
                    if rid:
                        records = self._fetch_resource(rid, user_query)
                        if records:
                            logger.info(
                                "gebiz_resource_discovered",
                                package=pkg.get("name", ""),
                                resource_id=rid,
                                count=len(records),
                            )
                            self._working_resource_id = rid
                            return records
            return []
        except Exception as exc:
            logger.debug("gebiz_catalogue_error", error=str(exc))
            return []

    def _parse_and_filter(self, records: list[dict], user_query: str) -> list[OcdsTenderLead]:
        leads: list[OcdsTenderLead] = []
        seen: set[str] = set()
        for record in records:
            try:
                # CKAN records are flat — normalise into the OCDS shape
                # parse_ocds_release expects.
                title = (
                    record.get("tender_description")
                    or record.get("tender_title")
                    or record.get("Tender Description")
                    or record.get("title")
                    or ""
                )
                if not title:
                    continue
                buyer = (
                    record.get("agency")
                    or record.get("Agency")
                    or record.get("agency_name")
                    or "Singapore Government"
                )
                # GeBIZ uses tender_close_date / award_date
                deadline = (
                    record.get("tender_close_date")
                    or record.get("Tender Close Date")
                    or record.get("submission_deadline")
                    or ""
                )
                shaped = {
                    "id": record.get("tender_no") or record.get("_id") or record.get("Tender No.") or "",
                    "tender": {
                        "title": title,
                        "description": record.get("description", ""),
                        "tenderPeriod": {"endDate": deadline},
                    },
                    "buyer": {"name": buyer},
                    "date": record.get("publish_date") or record.get("Publish Date", ""),
                }
                lead = parse_ocds_release(shaped, "singapore_gebiz", build_url=_build_gebiz_url)
                if not lead or lead.lead_id in seen:
                    continue
                seen.add(lead.lead_id)
                # GeBIZ doesn't include CPV — score on title/desc only
                score, kws = score_relevance(
                    lead.title,
                    lead.description,
                    "",
                    extra_strong=SG_EXTRA_STRONG,
                    extra_partial=SG_EXTRA_PARTIAL,
                )
                if user_query:
                    text = f"{lead.title} {lead.description}".lower()
                    hits = sum(1 for w in user_query.lower().split() if w in text)
                    if hits:
                        score = round(min(score + min(hits * 0.05, 0.15), 1.0), 2)
                lead.relevance_score = score
                lead.relevance_keywords = kws
                if score >= self.min_relevance:
                    leads.append(lead)
            except Exception as exc:
                logger.debug("gebiz_parse_error", error=str(exc))
        return leads
