"""
UK Government Tender APIs — Contracts Finder + Find a Tender.

Queries BOTH UK procurement portals for active tender notices:
  1. Contracts Finder — lower-value public sector contracts
     https://www.contractsfinder.service.gov.uk/apidocumentation
  2. Find a Tender — higher-value contracts (post-Brexit TED replacement)
     https://www.find-tender.service.gov.uk/Developer/Documentation

Both are free, public, no authentication required, and return
Open Contracting Data Standard (OCDS) JSON with structured fields:
  - Title, description, buyer name, deadline, value, CPV codes

No API key required for either service.

Usage:
    searcher = UkTenderSearcher()
    leads = searcher.search("chemical safety")
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

# Contracts Finder — OCDS search (lower-value contracts)
CF_API_URL = "https://www.contractsfinder.service.gov.uk/Published/Notices/OCDS/Search"

# Find a Tender — OCDS release packages (higher-value contracts)
FT_API_URL = "https://www.find-tender.service.gov.uk/api/1.0/ocdsReleasePackages"

# ---------------------------------------------------------------------------
# Keyword filtering — same domain as SAM.gov / TED
# ---------------------------------------------------------------------------

# Strong keywords: if ANY of these appear, it's almost certainly relevant
STRONG_KEYWORDS: list[str] = [
    "safety data sheet", "sds management", "sds authoring",
    "msds", "chemical safety", "chemical management",
    "chemical inventory", "chemical compliance",
    "ghs", "globally harmonized",
    "ehs software", "ehs management", "ehs compliance",
    "hazardous material", "hazardous chemical", "hazardous substance",
    "hazard communication", "coshh",  # UK-specific: Control of Substances Hazardous to Health
    "reach regulation", "clp regulation",
    "workplace safety software", "occupational safety software",
    "environmental health and safety",
]

# Partial keywords: need 2+ to count
PARTIAL_KEYWORDS: list[str] = [
    "safety", "compliance", "environmental", "regulation",
    "chemical", "hazard", "risk management", "software platform",
    "occupational health", "dangerous goods", "toxic",
]

# CPV codes relevant to SDS/EHS (same as TED integration)
RELEVANT_CPV_PREFIXES: list[str] = [
    "905",      # Environmental services
    "713172",   # Health and safety services
    "480000",   # Software packages
    "720000",   # IT services
    "334212",   # Safety equipment
    "331410",   # Industrial chemicals
    "905240",   # Hazardous waste management
]


@dataclass
class UkTenderLead:
    """A tender discovered from UK government APIs."""
    lead_id: str
    title: str
    description: str
    agency: str
    source_portal: str = "uk_gov"
    source_url: str = ""
    submission_deadline: str = ""
    posted_date: str = ""
    value_amount: float = 0.0
    value_currency: str = "GBP"
    cpv_code: str = ""
    relevance_score: float = 0.0
    relevance_keywords: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)


def score_relevance_uk(title: str, description: str, cpv_code: str = "") -> tuple[float, list[str]]:
    """Score how relevant a UK tender is to our EHS/SDS domain.

    Checks title + description against keyword lists, plus CPV code matching.
    """
    text = f"{title} {description}".lower()
    matched: list[str] = []

    # Strong keyword matches (0.20 each, capped at 0.80)
    strong_count = 0
    for kw in STRONG_KEYWORDS:
        if kw in text:
            matched.append(kw)
            strong_count += 1

    # Partial keyword matches (0.05 each, capped at 0.20)
    partial_count = 0
    for kw in PARTIAL_KEYWORDS:
        if kw in text and kw not in matched:
            matched.append(kw)
            partial_count += 1

    # CPV code bonus (0.15 if relevant CPV code)
    cpv_bonus = 0.0
    if cpv_code:
        for prefix in RELEVANT_CPV_PREFIXES:
            if cpv_code.startswith(prefix):
                cpv_bonus = 0.15
                matched.append(f"cpv:{cpv_code}")
                break

    strong_score = min(strong_count * 0.20, 0.80)
    partial_score = min(partial_count * 0.05, 0.20)
    total = min(strong_score + partial_score + cpv_bonus, 1.0)

    return round(total, 2), matched


class UkTenderSearcher:
    """Searches UK government procurement portals for active tenders.

    Queries both Contracts Finder and Find a Tender, then filters
    results by EHS/SDS relevance keywords and CPV codes.

    Usage:
        searcher = UkTenderSearcher()
        leads = searcher.search("chemical safety")
    """

    def __init__(self, timeout: float = 30.0, min_relevance: float = 0.10) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        logger.info("uk_tender_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[UkTenderLead]:
        """Search both UK procurement APIs for active EHS/SDS tenders.

        Args:
            user_query: User's search text (used to boost relevance scoring).
            max_results: Maximum results to return.
            days_back: How many days back to search.

        Returns:
            List of UkTenderLead objects, filtered and sorted by relevance.
        """
        logger.info("uk_search_start", days_back=days_back, max_results=max_results)

        all_releases: list[dict] = []

        # Date range for the query
        now = datetime.now(timezone.utc)
        date_from = (now - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00")
        date_to = now.strftime("%Y-%m-%dT23:59:59")

        # Query both APIs in sequence (they're fast)
        cf_releases = self._fetch_contracts_finder(date_from, date_to)
        ft_releases = self._fetch_find_a_tender(date_from, date_to)

        all_releases.extend(cf_releases)
        all_releases.extend(ft_releases)

        logger.info(
            "uk_raw_results",
            contracts_finder=len(cf_releases),
            find_a_tender=len(ft_releases),
            total=len(all_releases),
        )

        # Parse, score, and filter
        leads = self._parse_and_filter(all_releases, user_query)

        # Sort by relevance and cap
        leads.sort(key=lambda l: l.relevance_score, reverse=True)
        leads = leads[:max_results]

        logger.info(
            "uk_search_complete",
            total_results=len(leads),
            top_score=leads[0].relevance_score if leads else 0,
        )

        return leads

    def _fetch_contracts_finder(
        self, date_from: str, date_to: str,
    ) -> list[dict]:
        """Fetch from Contracts Finder OCDS API."""
        releases: list[dict] = []
        cursor = None

        try:
            # Fetch up to 3 pages (300 results max)
            for page in range(3):
                params: dict[str, str | int] = {
                    "publishedFrom": date_from,
                    "publishedTo": date_to,
                    "stages": "tender",
                    "limit": 100,
                }
                if cursor:
                    params["cursor"] = cursor

                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.get(CF_API_URL, params=params)
                    resp.raise_for_status()
                    data = resp.json()

                page_releases = data.get("releases", [])
                for r in page_releases:
                    r["_source_api"] = "contracts_finder"
                releases.extend(page_releases)

                # Check if there are more pages
                # Contracts Finder uses a cursor in the URI for pagination
                uri = data.get("uri", "")
                if "cursor=" in uri:
                    import urllib.parse
                    parsed = urllib.parse.urlparse(uri)
                    qs = urllib.parse.parse_qs(parsed.query)
                    cursor = qs.get("cursor", [None])[0]
                    if not cursor:
                        break
                else:
                    break

                if len(page_releases) < 100:
                    break  # Last page

            logger.info("contracts_finder_fetched", count=len(releases))

        except Exception as exc:
            logger.error("contracts_finder_error", error=str(exc))

        return releases

    def _fetch_find_a_tender(
        self, date_from: str, date_to: str,
    ) -> list[dict]:
        """Fetch from Find a Tender OCDS API."""
        releases: list[dict] = []
        cursor = None

        try:
            for page in range(3):
                params: dict[str, str | int] = {
                    "updatedFrom": date_from,
                    "updatedTo": date_to,
                    "stages": "tender",
                    "limit": 100,
                }
                if cursor:
                    params["cursor"] = cursor

                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.get(FT_API_URL, params=params)
                    resp.raise_for_status()
                    data = resp.json()

                page_releases = data.get("releases", [])
                for r in page_releases:
                    r["_source_api"] = "find_a_tender"
                releases.extend(page_releases)

                # Check for more pages
                if len(page_releases) < 100:
                    break

                # Find a Tender doesn't put cursor in URI the same way,
                # but if response has exactly 100 items, there may be more
                # We'd need to extract the cursor somehow — for now,
                # stop at 100 results per API (sufficient for keyword filtering)
                break

            logger.info("find_a_tender_fetched", count=len(releases))

        except Exception as exc:
            logger.error("find_a_tender_error", error=str(exc))

        return releases

    def _parse_and_filter(
        self, releases: list[dict], user_query: str,
    ) -> list[UkTenderLead]:
        """Parse OCDS releases and filter for EHS/SDS relevance."""
        leads: list[UkTenderLead] = []
        seen_ids: set[str] = set()

        for release in releases:
            try:
                lead = self._parse_release(release)
                if not lead:
                    continue

                # Deduplicate by ID
                if lead.lead_id in seen_ids:
                    continue
                seen_ids.add(lead.lead_id)

                # Score relevance
                score, keywords = score_relevance_uk(
                    lead.title, lead.description, lead.cpv_code,
                )
                lead.relevance_score = score
                lead.relevance_keywords = keywords

                # Filter: must meet minimum relevance
                if score >= self.min_relevance:
                    leads.append(lead)

            except Exception as exc:
                logger.debug("uk_parse_error", error=str(exc))

        return leads

    def _parse_release(self, release: dict) -> UkTenderLead | None:
        """Parse a single OCDS release into a UkTenderLead."""
        source_api = release.get("_source_api", "unknown")
        ocid = release.get("ocid", "")
        notice_id = release.get("id", "")

        tender = release.get("tender", {})
        if not tender:
            return None

        title = tender.get("title", "")
        description = tender.get("description", "")
        if not title:
            return None

        # Extract buyer name
        buyer = release.get("buyer", {})
        buyer_name = buyer.get("name", "")
        if not buyer_name:
            # Try parties
            parties = release.get("parties", [])
            for party in parties:
                if "buyer" in party.get("roles", []):
                    buyer_name = party.get("name", "")
                    break
        if not buyer_name:
            buyer_name = "UK Government"

        # Extract deadline (tenderPeriod.endDate)
        tender_period = tender.get("tenderPeriod", {})
        deadline = tender_period.get("endDate", "")

        # Extract published/posted date
        posted_date = release.get("date", "")

        # Extract value
        value = tender.get("value", {})
        value_amount = value.get("amount", 0) or 0
        value_currency = value.get("currency", "GBP")

        # Extract CPV code
        classification = tender.get("classification", {})
        cpv_code = classification.get("id", "")

        # Build source URL
        if source_api == "contracts_finder":
            source_url = f"https://www.contractsfinder.service.gov.uk/Notice/{notice_id}"
        elif source_api == "find_a_tender":
            source_url = f"https://www.find-tender.service.gov.uk/Notice/{notice_id}"
        else:
            source_url = ""

        # Parse dates to ISO format
        deadline_iso = self._parse_date(deadline)
        posted_iso = self._parse_date(posted_date)

        # Skip if deadline has already passed
        if deadline_iso:
            try:
                from dateutil import parser as dateparser
                dl = dateparser.parse(deadline_iso).date()
                if dl < datetime.now(timezone.utc).date():
                    return None  # Expired
            except Exception:
                pass

        return UkTenderLead(
            lead_id=notice_id or ocid or f"UK-{uuid.uuid4().hex[:8].upper()}",
            title=title,
            description=description[:500] if description else "",
            agency=buyer_name,
            source_portal="uk_gov",
            source_url=source_url,
            submission_deadline=deadline_iso,
            posted_date=posted_iso,
            value_amount=float(value_amount),
            value_currency=value_currency,
            cpv_code=cpv_code,
            relevance_keywords=[],
            raw_data={
                "ocid": ocid,
                "source_api": source_api,
                "has_strong_match": True,   # Will be set properly after scoring
                "has_tender_signal": True,  # Government API = always a tender
                "is_empty_page": False,
            },
        )

    @staticmethod
    def _parse_date(date_str: str) -> str:
        """Parse OCDS date formats to ISO YYYY-MM-DD."""
        if not date_str:
            return ""

        # OCDS uses ISO 8601: "2026-05-15T10:00:00Z" or "2026-05-15T10:00:00+01:00"
        date_str = date_str.strip()

        # Already ISO date
        if re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
            return date_str[:10]

        try:
            from dateutil import parser as dateparser
            parsed = dateparser.parse(date_str)
            if parsed:
                return parsed.strftime("%Y-%m-%d")
        except Exception:
            pass

        return ""
