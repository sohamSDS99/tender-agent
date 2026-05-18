"""
World Bank Procurement API Integration — International tender discovery.

Queries the World Bank's Procurement notices dataset for active tenders
relevant to EHS/SDS/chemical safety.  The World Bank publishes procurement
notices for projects it finances across all member countries, covering
sectors like environment, health, water, and infrastructure.

API endpoint:
  GET https://datacatalogapi.worldbank.org/dexapps/fone/api/apiservice
  Parameters: datasetId=DS00979, resourceId=RS00909, top=N, skip=N, type=json

No authentication required — the Data Catalog API is free and public.

Response columns (13 per record):
  bid_description, country_code, country_name, deadline_date, id,
  notice_type, procurement_category, procurement_method, project_id,
  publication_date, region, sector, url

Usage:
    searcher = WorldBankSearcher()
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
# Constants
# ---------------------------------------------------------------------------

WB_API_URL = (
    "https://datacatalogapi.worldbank.org/dexapps/fone/api/apiservice"
)

# Strong keywords: if ANY of these appear, almost certainly relevant
STRONG_KEYWORDS: list[str] = [
    "safety data sheet", "sds management", "sds authoring",
    "msds", "chemical safety", "chemical management",
    "chemical inventory", "chemical compliance",
    "ghs", "globally harmonized",
    "ehs software", "ehs management", "ehs compliance",
    "hazardous material", "hazardous chemical", "hazardous substance",
    "hazard communication", "hazardous waste",
    "reach regulation", "clp regulation",
    "workplace safety software", "occupational safety software",
    "environmental health and safety",
]

# Partial keywords: individually weak, but 2+ suggest relevance
PARTIAL_KEYWORDS: list[str] = [
    "safety", "compliance", "environmental", "regulation",
    "chemical", "hazard", "risk management", "software platform",
    "occupational health", "dangerous goods", "toxic",
    "pollution", "contamination", "pesticide",
]


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class WorldBankTenderLead:
    """A tender discovered from the World Bank Procurement API.

    Attributes:
        lead_id: Unique identifier (World Bank notice ID or generated UUID).
        title: Bid description / opportunity title.
        description: Full bid description text.
        agency: Issuing country name (country_name from API).
        source_portal: Always "world_bank" for this integration.
        source_url: Direct URL to the procurement notice.
        submission_deadline: Deadline as YYYY-MM-DD.
        posted_date: Publication date as YYYY-MM-DD.
        country_code: ISO country code from the API.
        region: World Bank region (e.g. "Africa", "East Asia and Pacific").
        sector: Project sector (e.g. "Environment", "Health").
        procurement_category: Category (e.g. "Goods", "Works", "Consulting").
        relevance_score: 0.0–1.0 based on keyword matching.
        relevance_keywords: Which keywords matched.
        raw_data: Full API record for debugging.
    """
    lead_id: str
    title: str
    description: str
    agency: str
    source_portal: str = "world_bank"
    source_url: str = ""
    submission_deadline: str = ""
    posted_date: str = ""
    country_code: str = ""
    region: str = ""
    sector: str = ""
    procurement_category: str = ""
    relevance_score: float = 0.0
    relevance_keywords: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lead_id": self.lead_id,
            "title": self.title,
            "description": self.description,
            "agency": self.agency,
            "source_portal": self.source_portal,
            "source_url": self.source_url,
            "submission_deadline": self.submission_deadline,
            "posted_date": self.posted_date,
            "country_code": self.country_code,
            "region": self.region,
            "sector": self.sector,
            "procurement_category": self.procurement_category,
            "relevance_score": self.relevance_score,
            "relevance_keywords": self.relevance_keywords,
        }


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

def score_relevance_wb(
    title: str,
    description: str,
    sector: str = "",
) -> tuple[float, list[str]]:
    """Score how relevant a World Bank tender is to our EHS/SDS domain.

    Checks title + description against keyword lists.  Adds a sector
    bonus when the project sector contains "environment" or "health".

    Returns:
        Tuple of (score 0.0–1.0, list of matched keywords).
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

    # Sector bonus: environment or health sectors are more likely relevant
    sector_bonus = 0.0
    if sector:
        sector_lower = sector.lower()
        if "environment" in sector_lower or "health" in sector_lower:
            sector_bonus = 0.10
            matched.append(f"sector:{sector}")

    strong_score = min(strong_count * 0.20, 0.80)
    partial_score = min(partial_count * 0.05, 0.20)
    total = min(strong_score + partial_score + sector_bonus, 1.0)

    return round(total, 2), matched


# ---------------------------------------------------------------------------
# World Bank searcher class
# ---------------------------------------------------------------------------

class WorldBankSearcher:
    """Searches World Bank procurement notices for EHS/SDS tenders.

    Queries the World Bank Data Catalog API for recent procurement notices,
    then filters and scores them for relevance to the EHS/SDS/chemical
    safety domain.

    Usage:
        searcher = WorldBankSearcher()
        leads = searcher.search("chemical safety")
    """

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.10,
    ) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        logger.info(
            "world_bank_searcher_initialized",
            min_relevance=self.min_relevance,
        )

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 90,
    ) -> list[WorldBankTenderLead]:
        """Search World Bank procurement notices for active EHS/SDS tenders.

        Fetches up to 200 records from the API, filters for notices posted
        within ``days_back`` days, scores them for relevance, and returns
        the top results.

        Args:
            user_query: User's search text (currently used for logging;
                keyword filtering handles relevance).
            max_results: Maximum number of results to return.
            days_back: How many days back to include (by publication_date).

        Returns:
            List of WorldBankTenderLead objects, filtered and sorted by
            relevance_score descending.
        """
        logger.info(
            "wb_search_start",
            user_query=user_query[:80] if user_query else "",
            max_results=max_results,
            days_back=days_back,
        )

        # Fetch raw records from the API (up to 200)
        raw_records: list[dict] = []
        for skip in (0, 100):
            batch = self._fetch_records(top=100, skip=skip)
            raw_records.extend(batch)
            if len(batch) < 100:
                break  # No more pages

        logger.info("wb_raw_records_fetched", count=len(raw_records))

        # Determine the cutoff date for "recent" notices
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=days_back)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        # Parse, filter by date, score, and collect
        leads: list[WorldBankTenderLead] = []
        for record in raw_records:
            try:
                lead = self._parse_record(record)
                if lead is None:
                    continue

                # Skip if posted before the cutoff
                if lead.posted_date and lead.posted_date < cutoff_str:
                    continue

                # Skip expired tenders (deadline already passed)
                if lead.submission_deadline:
                    today_str = now.strftime("%Y-%m-%d")
                    if lead.submission_deadline < today_str:
                        continue

                # Score relevance
                score, keywords = score_relevance_wb(
                    lead.title, lead.description, lead.sector,
                )
                lead.relevance_score = score
                lead.relevance_keywords = keywords

                if score >= self.min_relevance:
                    leads.append(lead)

            except Exception as exc:
                logger.debug(
                    "wb_parse_error",
                    error=str(exc),
                    record_id=record.get("id", "unknown"),
                )

        # Sort by relevance (highest first) and cap
        leads.sort(key=lambda l: l.relevance_score, reverse=True)
        leads = leads[:max_results]

        logger.info(
            "wb_search_complete",
            total_results=len(leads),
            top_score=leads[0].relevance_score if leads else 0.0,
        )

        return leads

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _fetch_records(self, top: int = 100, skip: int = 0) -> list[dict]:
        """Fetch procurement records from the World Bank Data Catalog API.

        Args:
            top: Number of records to fetch (max per request).
            skip: Offset for pagination.

        Returns:
            List of record dicts from the API response.
        """
        params = {
            "datasetId": "DS00979",
            "resourceId": "RS00909",
            "top": top,
            "skip": skip,
            "type": "json",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(WB_API_URL, params=params)
                resp.raise_for_status()
                data = resp.json()

            # The API returns a list of records directly, or nested under a key
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # Try common wrapper keys
                for key in ("data", "records", "value", "results"):
                    if key in data and isinstance(data[key], list):
                        return data[key]
                # Might be the record itself (unlikely for a list endpoint)
                return [data] if data else []

            return []

        except httpx.HTTPStatusError as exc:
            logger.error(
                "wb_api_http_error",
                status=exc.response.status_code,
                body=exc.response.text[:500],
            )
            return []
        except Exception as exc:
            logger.error("wb_api_error", error=str(exc))
            return []

    def _parse_record(self, record: dict) -> WorldBankTenderLead | None:
        """Parse a single World Bank API record into a WorldBankTenderLead.

        Field mapping:
          - bid_description  -> title + description
          - deadline_date    -> submission_deadline
          - publication_date -> posted_date
          - url              -> source_url
          - country_name     -> agency
          - country_code     -> country_code
          - region           -> region
          - sector           -> sector
          - procurement_category -> procurement_category
          - id               -> lead_id
        """
        bid_desc = record.get("bid_description", "") or ""
        if not bid_desc:
            return None

        notice_id = record.get("id", "")
        lead_id = str(notice_id) if notice_id else f"WB-{uuid.uuid4().hex[:8].upper()}"

        # Use bid_description for both title and description.
        # Title: first 200 chars (or first sentence); description: full text.
        title = bid_desc[:200].strip()
        description = bid_desc.strip()

        country_name = record.get("country_name", "") or ""
        country_code = record.get("country_code", "") or ""
        region = record.get("region", "") or ""
        sector = record.get("sector", "") or ""
        procurement_category = record.get("procurement_category", "") or ""
        source_url = record.get("url", "") or ""

        # Parse dates
        deadline_raw = record.get("deadline_date", "") or ""
        posted_raw = record.get("publication_date", "") or ""

        submission_deadline = self._parse_date(deadline_raw)
        posted_date = self._parse_date(posted_raw)

        return WorldBankTenderLead(
            lead_id=lead_id,
            title=title,
            description=description[:2000],
            agency=country_name or "World Bank Project",
            source_url=source_url,
            submission_deadline=submission_deadline,
            posted_date=posted_date,
            country_code=country_code,
            region=region,
            sector=sector,
            procurement_category=procurement_category,
            raw_data=record,
        )

    @staticmethod
    def _parse_date(date_str: str) -> str:
        """Parse World Bank date strings to ISO YYYY-MM-DD.

        Handles various formats:
          - "2026-05-15T00:00:00"  (ISO with time)
          - "2026-05-15"           (ISO date only)
          - "05/15/2026"           (US month/day/year)
          - "15-May-2026"          (day-month_name-year)
          - "20260515"             (compact)
        """
        if not date_str:
            return ""

        date_str = date_str.strip()

        # ISO format: "2026-05-15" or "2026-05-15T..."
        if re.match(r"^\d{4}-\d{2}-\d{2}", date_str):
            return date_str[:10]

        # Compact: "20260515"
        if re.match(r"^\d{8}$", date_str):
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        # US format: "05/15/2026" or "5/15/2026"
        m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", date_str)
        if m:
            month, day, year = m.group(1), m.group(2), m.group(3)
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        # Named month: "15-May-2026"
        try:
            parsed = datetime.strptime(date_str, "%d-%b-%Y")
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            pass

        # Fallback: try dateutil
        try:
            from dateutil import parser as dateparser
            parsed_dt = dateparser.parse(date_str)
            if parsed_dt:
                return parsed_dt.strftime("%Y-%m-%d")
        except Exception:
            pass

        return ""
