"""
World Bank Procurement Search API v2 — Keyword-based tender discovery.

This is a MORE POWERFUL endpoint than the Data Catalog API (world_bank.py).
The Search API supports:
  - Keyword search (qterm parameter)
  - Deadline filtering (submission_deadline_date)
  - Country filtering (countryshortname)
  - Result sorting (srt parameter)
  - Pagination (os=offset, rows=count)

API endpoint:
  GET https://search.worldbank.org/api/v2/procnotices
  Parameters: format=json, qterm=..., rows=50, os=0,
              submission_deadline_date=YYYY-MM-DD (future deadlines)

No authentication required — completely free and public.

Fields per record:
  id, noticetext, project_name, countryshortname, regionname,
  sector, submission_deadline_date, notice_posted_date,
  notice_type, procurement_method, borrower, url, ...

Usage:
    searcher = WorldBankV2Searcher()
    leads = searcher.search("chemical safety management")
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WB_SEARCH_URL = "https://search.worldbank.org/api/v2/procnotices"

# EHS/SDS search terms to cycle through when user query is generic
DEFAULT_SEARCH_TERMS: list[str] = [
    "chemical safety",
    "hazardous material management",
    "environmental health safety",
    "safety data sheet",
    "occupational safety",
    "waste management chemical",
]

# Strong keywords for relevance scoring
STRONG_KEYWORDS: list[str] = [
    "safety data sheet", "sds management", "sds authoring",
    "msds", "chemical safety", "chemical management",
    "chemical inventory", "chemical compliance",
    "ghs", "globally harmonized",
    "ehs software", "ehs management", "ehs compliance",
    "hazardous material", "hazardous chemical", "hazardous substance",
    "hazard communication", "hazardous waste",
    "reach regulation", "clp regulation",
    "workplace safety", "occupational safety",
    "environmental health and safety",
]

# Partial keywords
PARTIAL_KEYWORDS: list[str] = [
    "safety", "compliance", "environmental", "regulation",
    "chemical", "hazard", "risk management", "software",
    "occupational health", "dangerous goods", "toxic",
    "pollution", "contamination", "pesticide",
]


@dataclass
class WorldBankV2TenderLead:
    """A tender from the World Bank Search API v2."""
    lead_id: str
    title: str
    description: str
    agency: str
    source_portal: str = "world_bank_v2"
    source_url: str = ""
    submission_deadline: str = ""
    posted_date: str = ""
    country_name: str = ""
    region: str = ""
    sector: str = ""
    procurement_method: str = ""
    notice_type: str = ""
    project_name: str = ""
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
            "country_name": self.country_name,
            "region": self.region,
            "sector": self.sector,
            "procurement_method": self.procurement_method,
            "notice_type": self.notice_type,
            "project_name": self.project_name,
            "relevance_score": self.relevance_score,
            "relevance_keywords": self.relevance_keywords,
        }


def score_relevance_wb2(
    title: str,
    description: str,
    sector: str = "",
) -> tuple[float, list[str]]:
    """Score relevance to EHS/SDS domain."""
    text = f"{title} {description}".lower()
    matched: list[str] = []

    strong_count = 0
    for kw in STRONG_KEYWORDS:
        if kw in text:
            matched.append(kw)
            strong_count += 1

    partial_count = 0
    for kw in PARTIAL_KEYWORDS:
        if kw in text and kw not in matched:
            matched.append(kw)
            partial_count += 1

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


class WorldBankV2Searcher:
    """Searches World Bank procurement via the Search API v2.

    Unlike the Data Catalog API, this endpoint supports keyword search
    (qterm), so results are pre-filtered by the API before local scoring.
    This means higher quality results with fewer false positives.

    Usage:
        searcher = WorldBankV2Searcher()
        leads = searcher.search("chemical safety management")
    """

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.05,
    ) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        logger.info("world_bank_v2_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 20,
    ) -> list[WorldBankV2TenderLead]:
        """Search World Bank procurement notices with keyword matching.

        The v2 API supports server-side keyword search, so we send
        EHS-related terms directly and get pre-filtered results.

        Args:
            user_query: User's search query (used alongside default terms).
            max_results: Maximum leads to return.

        Returns:
            Sorted list of WorldBankV2TenderLead by relevance.
        """
        logger.info("wb_v2_search_start", user_query=user_query[:80] if user_query else "")

        # Build search terms: user query + default EHS terms
        search_terms = []
        if user_query:
            search_terms.append(user_query)
        search_terms.extend(DEFAULT_SEARCH_TERMS[:3])

        seen_ids: set[str] = set()
        all_leads: list[WorldBankV2TenderLead] = []

        for term in search_terms:
            batch = self._search_term(term)
            for lead in batch:
                if lead.lead_id not in seen_ids:
                    seen_ids.add(lead.lead_id)
                    all_leads.append(lead)

        # Sort by relevance and cap
        all_leads.sort(key=lambda l: l.relevance_score, reverse=True)
        result = all_leads[:max_results]

        logger.info(
            "wb_v2_search_complete",
            total=len(result),
            top_score=result[0].relevance_score if result else 0.0,
        )
        return result

    def _search_term(self, qterm: str) -> list[WorldBankV2TenderLead]:
        """Query the API for a single search term."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        params = {
            "format": "json",
            "qterm": qterm,
            "rows": 50,
            "os": 0,
            "submission_deadline_date": today,
            "srt": "submission_deadline_date",
            "order": "asc",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(WB_SEARCH_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error("wb_v2_http_error", status=exc.response.status_code, term=qterm)
            return []
        except Exception as exc:
            logger.error("wb_v2_api_error", error=str(exc), term=qterm)
            return []

        # Parse response — could be {"procnotices": {...}} or {"total": N, ...}
        records = []
        if isinstance(data, dict):
            proc = data.get("procnotices", data)
            if isinstance(proc, dict):
                # Records might be under numbered keys or a list
                for key, val in proc.items():
                    if isinstance(val, dict) and ("id" in val or "noticetext" in val):
                        records.append(val)
            elif isinstance(proc, list):
                records = proc
        elif isinstance(data, list):
            records = data

        leads: list[WorldBankV2TenderLead] = []
        for record in records:
            try:
                lead = self._parse_record(record)
                if lead and lead.relevance_score >= self.min_relevance:
                    leads.append(lead)
            except Exception as exc:
                logger.debug("wb_v2_parse_error", error=str(exc))

        return leads

    def _parse_record(self, record: dict) -> WorldBankV2TenderLead | None:
        """Parse a single v2 API record."""
        notice_text = record.get("noticetext", "") or record.get("bid_description", "") or ""
        project_name = record.get("project_name", "") or ""
        title_text = project_name or notice_text[:200]
        if not title_text.strip():
            return None

        notice_id = record.get("id", "")
        lead_id = str(notice_id) if notice_id else f"WBv2-{uuid.uuid4().hex[:8].upper()}"

        country = record.get("countryshortname", "") or ""
        region = record.get("regionname", "") or ""
        sector = record.get("sector", "") or record.get("majorsector_percent", "") or ""
        procurement_method = record.get("procurement_method", "") or ""
        notice_type = record.get("notice_type", "") or ""
        borrower = record.get("borrower", "") or ""
        url = record.get("url", "") or ""

        deadline_raw = record.get("submission_deadline_date", "") or ""
        posted_raw = record.get("notice_posted_date", "") or record.get("docdt", "") or ""

        deadline = self._parse_date(deadline_raw)
        posted = self._parse_date(posted_raw)

        # Skip expired
        if deadline:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if deadline < today:
                return None

        # Score
        full_text = f"{title_text} {notice_text}"
        score, keywords = score_relevance_wb2(full_text, notice_text, sector)

        agency = borrower or country or "World Bank Project"

        return WorldBankV2TenderLead(
            lead_id=lead_id,
            title=title_text[:300].strip(),
            description=notice_text[:2000].strip(),
            agency=agency,
            source_url=url,
            submission_deadline=deadline,
            posted_date=posted,
            country_name=country,
            region=region,
            sector=sector,
            procurement_method=procurement_method,
            notice_type=notice_type,
            project_name=project_name,
            relevance_score=score,
            relevance_keywords=keywords,
            raw_data={"id": notice_id, "source_portal": "world_bank_v2"},
        )

    @staticmethod
    def _parse_date(date_str: str) -> str:
        """Parse date strings to ISO YYYY-MM-DD."""
        if not date_str:
            return ""
        date_str = str(date_str).strip()

        if re.match(r"^\d{4}-\d{2}-\d{2}", date_str):
            return date_str[:10]

        if re.match(r"^\d{8}$", date_str):
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", date_str)
        if m:
            month, day, year = m.group(1), m.group(2), m.group(3)
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        try:
            from dateutil import parser as dateparser
            parsed_dt = dateparser.parse(date_str)
            if parsed_dt:
                return parsed_dt.strftime("%Y-%m-%d")
        except Exception:
            pass

        return ""
