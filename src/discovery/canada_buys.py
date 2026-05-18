"""
CanadaBuys Procurement Searcher — Discovers Canadian federal tenders.

Queries Canada's open procurement data for active tender notices relevant
to EHS/SDS/chemical safety.

HOW WE USE IT:
Canada publishes procurement data as Open Contracting Data Standard (OCDS)
JSON packages via the Open Canada portal (CKAN). We use a two-tier strategy:

  1. PRIMARY — CKAN API: Query the Open Canada CKAN API to retrieve the
     dataset metadata for the CanadaBuys procurement dataset
     (ID: 6abd20d4-7a1c-4b38-baa2-9525d0bb2fd2), extract the latest
     JSON/CSV resource URL, fetch it, and parse for relevant tenders.

  2. FALLBACK — If CKAN fails (endpoint change, timeout, unexpected
     format), we degrade gracefully and return an empty list with a
     warning log. The caller (coordinator) will still have results from
     other portals.

No API key or authentication required.

NOTE: This integration may need endpoint adjustment after live testing.
The CKAN dataset structure and resource formats can change between fiscal
years. The parsing logic is intentionally defensive — unknown fields are
skipped, date parsing failures are tolerated, and the searcher never
crashes on malformed data.

Usage:
    searcher = CanadaBuysSearcher()
    leads = searcher.search("chemical safety")
"""

from __future__ import annotations

import csv
import io
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

# CKAN API — Open Canada portal dataset metadata
CKAN_API_URL = (
    "https://open.canada.ca/data/api/action/package_show"
    "?id=6abd20d4-7a1c-4b38-baa2-9525d0bb2fd2"
)

# ---------------------------------------------------------------------------
# Keyword filtering — same domain as SAM.gov / TED / UK, plus Canada-specific
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
    "whmis",  # Workplace Hazardous Materials Information System (Canada-specific)
    "workplace safety software", "occupational safety software",
    "environmental health and safety",
    "controlled products", "hazardous products act",
    "hazardous products regulation",
    "reach regulation", "clp regulation",
]

# Partial keywords: need 2+ to count
PARTIAL_KEYWORDS: list[str] = [
    "safety", "compliance", "environmental", "regulation",
    "chemical", "hazard", "risk management", "software platform",
    "occupational health", "dangerous goods", "toxic",
    "saas", "cloud-based",
]


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class CanadaBuysTenderLead:
    """A tender discovered from CanadaBuys / Open Canada procurement data."""
    lead_id: str
    title: str
    description: str
    agency: str
    source_portal: str = "canada_buys"
    source_url: str = ""
    submission_deadline: str = ""
    posted_date: str = ""
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
            "relevance_score": self.relevance_score,
            "relevance_keywords": self.relevance_keywords,
        }


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

def score_relevance_canada(
    title: str, description: str,
) -> tuple[float, list[str]]:
    """Score how relevant a Canadian tender is to our EHS/SDS domain.

    Checks title + description against keyword lists. Strong keywords
    (direct EHS/SDS match) contribute more than partial keywords.

    Returns:
        Tuple of (score 0.0-1.0, list of matched keywords).
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

    strong_score = min(strong_count * 0.20, 0.80)
    partial_score = min(partial_count * 0.05, 0.20)
    total = min(strong_score + partial_score, 1.0)

    return round(total, 2), matched


# ---------------------------------------------------------------------------
# CanadaBuys searcher class
# ---------------------------------------------------------------------------

class CanadaBuysSearcher:
    """Searches Canadian federal procurement data for active EHS/SDS tenders.

    Uses the Open Canada CKAN API to discover and fetch the latest
    CanadaBuys procurement dataset, then parses and filters results
    for EHS/chemical safety relevance.

    Designed for graceful degradation: if the CKAN API is unreachable,
    returns its data in an unexpected format, or any resource fetch
    fails, the searcher logs a warning and returns an empty list
    rather than raising an exception.

    Usage:
        searcher = CanadaBuysSearcher()
        leads = searcher.search("chemical safety")
    """

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.10,
    ) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        logger.info("canada_buys_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[CanadaBuysTenderLead]:
        """Search Canadian procurement data for active EHS/SDS tenders.

        Args:
            user_query: User's search text (used as an additional keyword
                        signal during relevance scoring).
            max_results: Maximum results to return.
            days_back: How many days back to include.

        Returns:
            List of CanadaBuysTenderLead objects, filtered by relevance
            and sorted descending by relevance_score.
        """
        logger.info(
            "canada_buys_search_start",
            days_back=days_back,
            max_results=max_results,
        )

        raw_records = self._fetch_from_ckan(days_back)

        if not raw_records:
            logger.warning(
                "canada_buys_no_records",
                msg="CKAN fetch returned no records; returning empty list",
            )
            return []

        # Parse, score, and filter
        leads = self._parse_and_filter(raw_records, user_query, days_back)

        # Sort by relevance (highest first) and cap
        leads.sort(key=lambda lead: lead.relevance_score, reverse=True)
        leads = leads[:max_results]

        logger.info(
            "canada_buys_search_complete",
            raw_count=len(raw_records),
            filtered_count=len(leads),
            top_score=leads[0].relevance_score if leads else 0,
        )

        return leads

    # ------------------------------------------------------------------
    # CKAN API fetch
    # ------------------------------------------------------------------

    def _fetch_from_ckan(self, days_back: int) -> list[dict]:
        """Fetch the CanadaBuys dataset metadata from CKAN, then download
        and parse the latest JSON or CSV resource.

        Returns a list of raw record dicts, or an empty list on any failure.
        """
        try:
            # Step 1: Get dataset metadata (lists all resources / files)
            with httpx.Client(timeout=self.timeout) as client:
                meta_resp = client.get(CKAN_API_URL)
                meta_resp.raise_for_status()
                meta_data = meta_resp.json()

            if not meta_data.get("success"):
                logger.warning(
                    "canada_buys_ckan_not_success",
                    response_keys=list(meta_data.keys()),
                )
                return []

            result = meta_data.get("result", {})
            resources = result.get("resources", [])

            if not resources:
                logger.warning("canada_buys_no_resources")
                return []

            logger.info(
                "canada_buys_ckan_resources_found",
                count=len(resources),
            )

            # Step 2: Find the best resource to fetch.
            # Prefer JSON over CSV; prefer most recently modified.
            json_resources = [
                r for r in resources
                if r.get("format", "").upper() == "JSON"
                or r.get("url", "").lower().endswith(".json")
            ]
            csv_resources = [
                r for r in resources
                if r.get("format", "").upper() == "CSV"
                or r.get("url", "").lower().endswith(".csv")
            ]

            target_resource = None
            if json_resources:
                # Pick the most recently modified JSON resource
                json_resources.sort(
                    key=lambda r: r.get("last_modified", "") or r.get("created", ""),
                    reverse=True,
                )
                target_resource = json_resources[0]
            elif csv_resources:
                csv_resources.sort(
                    key=lambda r: r.get("last_modified", "") or r.get("created", ""),
                    reverse=True,
                )
                target_resource = csv_resources[0]
            else:
                # Try the first resource regardless of format
                target_resource = resources[0]

            resource_url = target_resource.get("url", "")
            resource_format = target_resource.get("format", "").upper()

            if not resource_url:
                logger.warning("canada_buys_resource_no_url")
                return []

            logger.info(
                "canada_buys_fetching_resource",
                url=resource_url[:120],
                format=resource_format,
                name=target_resource.get("name", ""),
            )

            # Step 3: Download and parse the resource.
            # Use a longer timeout for potentially large files.
            with httpx.Client(timeout=max(self.timeout, 60.0)) as client:
                data_resp = client.get(resource_url)
                data_resp.raise_for_status()

            content_type = data_resp.headers.get("content-type", "")

            # Attempt JSON parsing first
            if resource_format == "JSON" or "json" in content_type:
                return self._parse_json_resource(data_resp.text)

            # Attempt CSV parsing
            if resource_format == "CSV" or "csv" in content_type:
                return self._parse_csv_resource(data_resp.text)

            # Unknown format — try JSON, then CSV, then give up
            records = self._parse_json_resource(data_resp.text)
            if records:
                return records
            records = self._parse_csv_resource(data_resp.text)
            if records:
                return records

            logger.warning(
                "canada_buys_unknown_format",
                format=resource_format,
                content_type=content_type,
            )
            return []

        except httpx.HTTPStatusError as exc:
            logger.error(
                "canada_buys_http_error",
                status_code=exc.response.status_code,
                url=str(exc.request.url)[:120],
            )
            return []

        except httpx.TimeoutException:
            logger.error("canada_buys_timeout")
            return []

        except Exception as exc:
            logger.error(
                "canada_buys_fetch_error",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return []

    # ------------------------------------------------------------------
    # Resource parsers
    # ------------------------------------------------------------------

    def _parse_json_resource(self, text: str) -> list[dict]:
        """Parse a JSON resource, handling both OCDS packages and flat arrays.

        CanadaBuys OCDS JSON may be structured as:
          - An OCDS release package: {"releases": [...]}
          - A flat array of records: [{"tender": ...}, ...]
          - A single OCDS release: {"releases": [...], "uri": ..., ...}

        We handle all three shapes defensively.
        """
        import json

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.debug("canada_buys_json_parse_failed", error=str(exc))
            return []

        # Shape 1: OCDS release package with "releases" key
        if isinstance(data, dict) and "releases" in data:
            releases = data["releases"]
            if isinstance(releases, list):
                logger.info("canada_buys_json_ocds_package", count=len(releases))
                return releases

        # Shape 2: Flat array of records
        if isinstance(data, list):
            logger.info("canada_buys_json_flat_array", count=len(data))
            return data

        # Shape 3: Dict with some other top-level key containing records
        # Try common keys
        for key in ("records", "data", "results", "notices", "tenders"):
            if isinstance(data, dict) and key in data:
                records = data[key]
                if isinstance(records, list):
                    logger.info(
                        "canada_buys_json_alt_key",
                        key=key,
                        count=len(records),
                    )
                    return records

        # Single record wrapped in a dict (unlikely but defensive)
        if isinstance(data, dict) and ("tender" in data or "title" in data):
            return [data]

        logger.warning(
            "canada_buys_json_unknown_shape",
            keys=list(data.keys()) if isinstance(data, dict) else type(data).__name__,
        )
        return []

    def _parse_csv_resource(self, text: str) -> list[dict]:
        """Parse a CSV resource into a list of dicts."""
        try:
            reader = csv.DictReader(io.StringIO(text))
            records = list(reader)
            logger.info("canada_buys_csv_parsed", count=len(records))
            return records
        except Exception as exc:
            logger.debug("canada_buys_csv_parse_failed", error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Parse and filter records into leads
    # ------------------------------------------------------------------

    def _parse_and_filter(
        self,
        raw_records: list[dict],
        user_query: str,
        days_back: int,
    ) -> list[CanadaBuysTenderLead]:
        """Parse raw records (OCDS or CSV) into scored, filtered leads."""
        leads: list[CanadaBuysTenderLead] = []
        seen_ids: set[str] = set()
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)

        for record in raw_records:
            try:
                lead = self._parse_record(record, cutoff_date)
                if not lead:
                    continue

                # Deduplicate by lead_id
                if lead.lead_id in seen_ids:
                    continue
                seen_ids.add(lead.lead_id)

                # Score relevance
                score, keywords = score_relevance_canada(
                    lead.title,
                    lead.description,
                )

                # Bonus if user_query terms appear in title or description
                if user_query:
                    query_lower = user_query.lower()
                    text = f"{lead.title} {lead.description}".lower()
                    query_words = query_lower.split()
                    query_hits = sum(1 for w in query_words if w in text)
                    if query_hits > 0:
                        query_bonus = min(query_hits * 0.05, 0.15)
                        score = min(score + query_bonus, 1.0)
                        score = round(score, 2)

                lead.relevance_score = score
                lead.relevance_keywords = keywords

                if score >= self.min_relevance:
                    leads.append(lead)

            except Exception as exc:
                logger.debug(
                    "canada_buys_record_parse_error",
                    error=str(exc),
                )

        return leads

    def _parse_record(
        self,
        record: dict,
        cutoff_date: datetime,
    ) -> CanadaBuysTenderLead | None:
        """Parse a single record (OCDS release or CSV row) into a lead.

        Handles both OCDS-shaped records (with nested "tender", "buyer",
        "parties" keys) and flat CSV-shaped records (with column names
        like "title", "description", "buyer_name", etc.).
        """
        # --- Try OCDS format first ---
        tender = record.get("tender", {})
        if isinstance(tender, dict) and tender:
            return self._parse_ocds_record(record, tender, cutoff_date)

        # --- Try flat/CSV format ---
        return self._parse_flat_record(record, cutoff_date)

    def _parse_ocds_record(
        self,
        record: dict,
        tender: dict,
        cutoff_date: datetime,
    ) -> CanadaBuysTenderLead | None:
        """Parse an OCDS-shaped record."""
        ocid = record.get("ocid", "")
        notice_id = record.get("id", "")

        title = tender.get("title", "")
        description = tender.get("description", "")
        if not title:
            return None

        # Buyer / agency
        buyer = record.get("buyer", {})
        buyer_name = buyer.get("name", "") if isinstance(buyer, dict) else ""
        if not buyer_name:
            parties = record.get("parties", [])
            if isinstance(parties, list):
                for party in parties:
                    if isinstance(party, dict) and "buyer" in party.get("roles", []):
                        buyer_name = party.get("name", "")
                        break
        if not buyer_name:
            buyer_name = "Government of Canada"

        # Dates
        tender_period = tender.get("tenderPeriod", {})
        deadline = ""
        if isinstance(tender_period, dict):
            deadline = tender_period.get("endDate", "")

        posted_date = record.get("date", "")

        # Check if posted within our date range
        if posted_date:
            posted_dt = self._try_parse_datetime(posted_date)
            if posted_dt and posted_dt < cutoff_date:
                return None  # Too old

        # Check if deadline has already passed
        if deadline:
            deadline_dt = self._try_parse_datetime(deadline)
            if deadline_dt and deadline_dt < datetime.now(timezone.utc):
                return None  # Expired

        deadline_iso = self._parse_date(deadline)
        posted_iso = self._parse_date(posted_date)

        # Build source URL — CanadaBuys notice URL pattern
        lead_id = notice_id or ocid or f"CA-{uuid.uuid4().hex[:8].upper()}"
        source_url = ""
        if notice_id:
            source_url = (
                f"https://canadabuys.canada.ca/en/tender-opportunities/"
                f"{notice_id}"
            )
        elif ocid:
            source_url = (
                f"https://canadabuys.canada.ca/en/tender-opportunities/"
                f"{ocid}"
            )

        return CanadaBuysTenderLead(
            lead_id=lead_id,
            title=title,
            description=description[:500] if description else "",
            agency=buyer_name,
            source_portal="canada_buys",
            source_url=source_url,
            submission_deadline=deadline_iso,
            posted_date=posted_iso,
            relevance_score=0.0,
            relevance_keywords=[],
            raw_data={
                "ocid": ocid,
                "format": "ocds",
            },
        )

    def _parse_flat_record(
        self,
        record: dict,
        cutoff_date: datetime,
    ) -> CanadaBuysTenderLead | None:
        """Parse a flat/CSV-shaped record.

        CSV column names vary across datasets; we check multiple possible
        column names for each field.
        """
        # Title — try multiple possible column names
        title = (
            record.get("title", "")
            or record.get("Title", "")
            or record.get("tender_title", "")
            or record.get("solicitation_title", "")
            or record.get("notice_title", "")
            or ""
        )
        if not title:
            return None

        # Description
        description = (
            record.get("description", "")
            or record.get("Description", "")
            or record.get("tender_description", "")
            or record.get("solicitation_description", "")
            or ""
        )

        # Agency / buyer
        agency = (
            record.get("buyer_name", "")
            or record.get("Buyer", "")
            or record.get("buyer", "")
            or record.get("organization", "")
            or record.get("Organization", "")
            or record.get("department", "")
            or record.get("Department", "")
            or "Government of Canada"
        )

        # Deadline
        deadline_str = (
            record.get("closing_date", "")
            or record.get("Closing Date", "")
            or record.get("deadline", "")
            or record.get("submission_deadline", "")
            or record.get("tender_end_date", "")
            or ""
        )

        # Posted date
        posted_str = (
            record.get("date", "")
            or record.get("Date", "")
            or record.get("published_date", "")
            or record.get("publication_date", "")
            or record.get("posting_date", "")
            or record.get("tender_start_date", "")
            or ""
        )

        # Check dates against cutoff
        if posted_str:
            posted_dt = self._try_parse_datetime(posted_str)
            if posted_dt and posted_dt < cutoff_date:
                return None

        if deadline_str:
            deadline_dt = self._try_parse_datetime(deadline_str)
            if deadline_dt and deadline_dt < datetime.now(timezone.utc):
                return None

        deadline_iso = self._parse_date(deadline_str)
        posted_iso = self._parse_date(posted_str)

        # ID
        lead_id = (
            record.get("id", "")
            or record.get("Id", "")
            or record.get("notice_id", "")
            or record.get("solicitation_number", "")
            or record.get("reference_number", "")
            or f"CA-{uuid.uuid4().hex[:8].upper()}"
        )

        # Source URL
        source_url = (
            record.get("url", "")
            or record.get("URL", "")
            or record.get("link", "")
            or record.get("notice_url", "")
            or ""
        )
        if not source_url and lead_id and not lead_id.startswith("CA-"):
            source_url = (
                f"https://canadabuys.canada.ca/en/tender-opportunities/"
                f"{lead_id}"
            )

        return CanadaBuysTenderLead(
            lead_id=str(lead_id),
            title=title,
            description=description[:500] if description else "",
            agency=agency,
            source_portal="canada_buys",
            source_url=source_url,
            submission_deadline=deadline_iso,
            posted_date=posted_iso,
            relevance_score=0.0,
            relevance_keywords=[],
            raw_data={
                "format": "flat",
            },
        )

    # ------------------------------------------------------------------
    # Date helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(date_str: str) -> str:
        """Parse various date formats to ISO YYYY-MM-DD.

        Handles:
          - ISO 8601: "2026-05-15T10:00:00Z", "2026-05-15"
          - Slash-separated: "05/15/2026", "15/05/2026"
          - Other formats via dateutil fallback
        """
        if not date_str:
            return ""

        date_str = str(date_str).strip()

        # Already ISO date
        if re.match(r"^\d{4}-\d{2}-\d{2}", date_str):
            return date_str[:10]

        # Try dateutil for anything else
        try:
            from dateutil import parser as dateparser
            parsed = dateparser.parse(date_str)
            if parsed:
                return parsed.strftime("%Y-%m-%d")
        except Exception:
            pass

        return ""

    @staticmethod
    def _try_parse_datetime(date_str: str) -> datetime | None:
        """Try to parse a date string into a timezone-aware datetime.

        Returns None if parsing fails.
        """
        if not date_str:
            return None

        date_str = str(date_str).strip()

        try:
            from dateutil import parser as dateparser
            parsed = dateparser.parse(date_str)
            if parsed:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
        except Exception:
            pass

        return None
