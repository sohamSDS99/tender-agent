"""
Prozorro — Ukraine's public procurement API.

Queries the Prozorro public procurement platform for active tender notices
relevant to EHS/SDS/chemical safety:
  https://public-api.prozorro.gov.ua/api/2.5/tenders

Free, public, no authentication required.  Returns JSON with tender stubs
(id, tenderID, title, status, etc.).  Full tender details are available at
GET /api/2.5/tenders/{id} but the list endpoint already includes the fields
we need for relevance filtering.

Pagination uses an ``offset`` token returned with each page.

Usage:
    searcher = ProzorroSearcher()
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
# API endpoint
# ---------------------------------------------------------------------------

PROZORRO_API_URL = "https://public-api.prozorro.gov.ua/api/2.5/tenders"

# ---------------------------------------------------------------------------
# Keyword filtering — EHS / SDS domain (English + Ukrainian)
# ---------------------------------------------------------------------------

# Strong keywords: if ANY of these appear, it's almost certainly relevant
STRONG_KEYWORDS: list[str] = [
    "safety data sheet", "sds management", "sds authoring",
    "msds", "chemical safety", "chemical management",
    "chemical inventory", "chemical compliance",
    "ghs", "globally harmonized",
    "ehs software", "ehs management", "ehs compliance",
    "hazardous material", "hazardous chemical", "hazardous substance",
    "hazard communication",
    "reach regulation", "clp regulation",
    "workplace safety software", "occupational safety software",
    "environmental health and safety",
    # Ukrainian equivalents
    "безпека",        # safety
    "хімічний",       # chemical
    "паспорт безпеки",  # safety data sheet
    "небезпечні речовини",  # hazardous substances
]

# Partial keywords: need 2+ to count
PARTIAL_KEYWORDS: list[str] = [
    "safety", "compliance", "environmental", "regulation",
    "chemical", "hazard", "risk management", "software platform",
    "occupational health", "dangerous goods", "toxic",
    # Ukrainian partial
    "хімія",          # chemistry
    "небезпечний",    # hazardous
    "екологічний",    # ecological / environmental
]

# CPV codes relevant to SDS/EHS (same as UK / TED integrations)
RELEVANT_CPV_PREFIXES: list[str] = [
    "905",      # Environmental services
    "713172",   # Health and safety services
    "480000",   # Software packages
    "720000",   # IT services
    "334212",   # Safety equipment
    "331410",   # Industrial chemicals
    "905240",   # Hazardous waste management
]

# Acceptable tender statuses for active opportunities
ACTIVE_STATUSES: set[str] = {"active.tendering", "active.enquiries"}


@dataclass
class ProzorroTenderLead:
    """A tender discovered from the Prozorro procurement portal."""
    lead_id: str
    title: str
    description: str
    agency: str
    source_portal: str = "prozorro"
    source_url: str = ""
    submission_deadline: str = ""
    posted_date: str = ""
    value_amount: float = 0.0
    value_currency: str = "UAH"
    cpv_code: str = ""
    relevance_score: float = 0.0
    relevance_keywords: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)


def score_relevance_prozorro(
    title: str, description: str, cpv_code: str = "",
) -> tuple[float, list[str]]:
    """Score how relevant a Prozorro tender is to our EHS/SDS domain.

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


class ProzorroSearcher:
    """Searches Ukraine's Prozorro portal for active EHS/SDS tenders.

    Queries the public Prozorro API, filters for active tendering status,
    scores each tender for EHS/SDS relevance, and returns the top matches.

    Usage:
        searcher = ProzorroSearcher()
        leads = searcher.search("chemical safety")
    """

    def __init__(self, timeout: float = 30.0, min_relevance: float = 0.10) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        logger.info("prozorro_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[ProzorroTenderLead]:
        """Search Prozorro for active EHS/SDS tenders.

        Args:
            user_query: User's search text (reserved for future boost logic).
            max_results: Maximum results to return.
            days_back: How many days back to consider (filters by dateModified).

        Returns:
            List of ProzorroTenderLead objects, filtered and sorted by relevance.
        """
        logger.info("prozorro_search_start", days_back=days_back, max_results=max_results)

        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        all_tenders: list[dict] = []
        offset: str | None = None

        # Paginate up to 3 pages
        for page in range(3):
            try:
                page_tenders, next_offset = self._fetch_tenders_page(offset)
            except Exception as exc:
                logger.error("prozorro_fetch_error", page=page, error=str(exc))
                break

            if not page_tenders:
                break

            all_tenders.extend(page_tenders)
            logger.debug("prozorro_page_fetched", page=page, count=len(page_tenders))

            if not next_offset:
                break
            offset = next_offset

        logger.info("prozorro_raw_results", total=len(all_tenders))

        # Parse, filter status, score, and filter relevance
        leads: list[ProzorroTenderLead] = []
        seen_ids: set[str] = set()

        for tender in all_tenders:
            try:
                # Only active tenders
                status = tender.get("status", "")
                if status not in ACTIVE_STATUSES:
                    continue

                lead = self._parse_tender(tender)
                if not lead:
                    continue

                # Deduplicate
                if lead.lead_id in seen_ids:
                    continue
                seen_ids.add(lead.lead_id)

                # Skip if dateModified is before our cutoff
                date_modified = tender.get("dateModified", "")
                if date_modified:
                    try:
                        modified_dt = datetime.fromisoformat(
                            date_modified.replace("Z", "+00:00"),
                        )
                        if modified_dt < cutoff:
                            continue
                    except (ValueError, TypeError):
                        pass

                # Skip expired tenders
                if lead.submission_deadline:
                    try:
                        dl = datetime.strptime(lead.submission_deadline, "%Y-%m-%d").date()
                        if dl < datetime.now(timezone.utc).date():
                            continue
                    except (ValueError, TypeError):
                        pass

                # Score relevance
                score, keywords = score_relevance_prozorro(
                    lead.title, lead.description, lead.cpv_code,
                )
                lead.relevance_score = score
                lead.relevance_keywords = keywords

                if score >= self.min_relevance:
                    leads.append(lead)

            except Exception as exc:
                logger.debug("prozorro_parse_error", error=str(exc))

        # Sort by relevance descending
        leads.sort(key=lambda l: l.relevance_score, reverse=True)
        leads = leads[:max_results]

        logger.info(
            "prozorro_search_complete",
            total_results=len(leads),
            top_score=leads[0].relevance_score if leads else 0,
        )

        return leads

    def _fetch_tenders_page(
        self, offset: str | None = None,
    ) -> tuple[list[dict], str | None]:
        """Fetch a single page of tenders from the Prozorro API.

        Args:
            offset: Pagination offset token from a previous response.

        Returns:
            Tuple of (list of tender dicts, next offset or None).
        """
        params: dict[str, str] = {}
        if offset:
            params["offset"] = offset

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(PROZORRO_API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        tenders = data.get("data", [])
        next_offset = data.get("next_page", {}).get("offset")
        if next_offset is None:
            next_offset = data.get("offset")

        return tenders, next_offset

    def _parse_tender(self, tender: dict) -> ProzorroTenderLead | None:
        """Parse a single Prozorro tender dict into a ProzorroTenderLead."""
        tender_id = tender.get("tenderID", "")
        internal_id = tender.get("id", "")

        title = tender.get("title", "")
        description = tender.get("description", "")
        if not title:
            return None

        # Procuring entity name
        procuring_entity = tender.get("procuringEntity", {})
        agency = procuring_entity.get("name", "")
        if not agency:
            agency = "Ukraine Government"

        # Submission deadline from tenderPeriod.endDate
        tender_period = tender.get("tenderPeriod", {})
        deadline = tender_period.get("endDate", "")

        # Posted / modified date
        date_modified = tender.get("dateModified", "")

        # Value
        value_data = tender.get("value", {})
        value_amount = value_data.get("amount", 0) or 0
        value_currency = value_data.get("currency", "UAH")

        # CPV code from first item's classification
        cpv_code = ""
        items = tender.get("items", [])
        if items and isinstance(items, list):
            first_item = items[0]
            classification = first_item.get("classification", {})
            cpv_code = classification.get("id", "")

        # Source URL — public web view
        source_url = f"https://prozorro.gov.ua/tender/{tender_id}" if tender_id else ""

        # Parse dates
        deadline_iso = self._parse_date(deadline)
        posted_iso = self._parse_date(date_modified)

        return ProzorroTenderLead(
            lead_id=tender_id or internal_id or f"PZ-{uuid.uuid4().hex[:8].upper()}",
            title=title,
            description=description[:500] if description else "",
            agency=agency,
            source_portal="prozorro",
            source_url=source_url,
            submission_deadline=deadline_iso,
            posted_date=posted_iso,
            value_amount=float(value_amount),
            value_currency=value_currency,
            cpv_code=cpv_code,
            relevance_keywords=[],
            raw_data={
                "internal_id": internal_id,
                "tender_id": tender_id,
                "status": tender.get("status", ""),
            },
        )

    @staticmethod
    def _parse_date(date_str: str) -> str:
        """Parse Prozorro date formats to ISO YYYY-MM-DD.

        Prozorro uses ISO 8601 timestamps, e.g.
        ``2026-05-15T10:00:00+03:00`` or ``2026-05-15T10:00:00Z``.
        """
        if not date_str:
            return ""

        date_str = date_str.strip()

        # Fast path: already starts with YYYY-MM-DD
        if re.match(r"^\d{4}-\d{2}-\d{2}", date_str):
            return date_str[:10]

        # Fallback: try fromisoformat
        try:
            parsed = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return parsed.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass

        return ""
