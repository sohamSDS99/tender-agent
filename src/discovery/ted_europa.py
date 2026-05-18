"""
TED Europa API Integration — Structured EU tender search.

Queries the EU's official Tenders Electronic Daily (TED) API for
active procurement notices.  Unlike Google SERP, this returns
STRUCTURED data with real deadlines, statuses, and agencies.

Key advantages over SERP:
  - scope=ACTIVE guarantees only live, non-expired notices
  - Real deadline dates from the source of truth
  - Proper agency/buyer names
  - No news articles, blog posts, or empty portal pages

API docs: https://docs.ted.europa.eu/api/latest/search.html
No API key required for the Search API.

Usage:
    searcher = TedEuropaSearcher()
    leads = searcher.search("chemical safety")
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# TED API config
# ---------------------------------------------------------------------------

# Primary (new) and fallback (old) API endpoints
TED_API_URL = "https://api.ted.europa.eu/v3/notices/search"
TED_API_URL_FALLBACK = "https://ted.europa.eu/api/v3.0/notices/search"

# CPV codes relevant to SDS/EHS/chemical safety software
# See: https://simap.ted.europa.eu/cpv
RELEVANT_CPV_CODES: list[str] = [
    "905*",     # Environmental services (broad)
    "7131720*", # Health and safety services
    "4800000*", # Software package and information systems
    "7200000*", # IT services: consulting, development, support
    "3342120*", # Safety equipment
    "3314100*", # Industrial chemicals
    "9052400*", # Hazardous waste management
]

# Full-text search terms for SDS/EHS domain
SDS_SEARCH_TERMS: list[str] = [
    '"safety data sheet"',
    '"chemical safety"',
    '"SDS management"',
    '"hazardous material"',
    '"EHS"',
    '"GHS"',
    '"chemical compliance"',
    '"MSDS"',
    '"hazard communication"',
]

# Fields to request from the TED API
REQUESTED_FIELDS: list[str] = [
    "publication-number",
    "notice-title",
    "buyer-name",
    "notice-type",
    "publication-date",
    "deadline-receipt-tenders",
    "notice-url",
    "place-of-performance",
    "short-description",
]


@dataclass
class TedTenderLead:
    """A tender discovered from the TED Europa API."""
    lead_id: str
    title: str
    description: str
    agency: str
    source_portal: str = "ted.europa.eu"
    source_url: str = ""
    submission_deadline: str = ""
    posted_date: str = ""
    relevance_score: float = 0.80  # High default — TED results are pre-qualified
    relevance_keywords: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)


class TedEuropaSearcher:
    """Searches EU tenders via the TED Europa API.

    Returns structured, verified tender data — no guessing from Google snippets.
    Only returns ACTIVE notices (expired tenders excluded at the API level).
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout
        logger.info("ted_europa_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 20,
    ) -> list[TedTenderLead]:
        """Search TED for active SDS/EHS tenders.

        Args:
            user_query: User's search text — used to enhance the API query.
            max_results: Maximum number of results to return.

        Returns:
            List of TedTenderLead objects with real, structured data.
        """
        # Build the expert query for TED's search syntax
        query = self._build_query(user_query)

        logger.info("ted_search_start", query=query[:120], max_results=max_results)

        request_body = {
            "query": query,
            "fields": REQUESTED_FIELDS,
            "limit": min(max_results, 100),
            "scope": "ACTIVE",
            "paginationMode": "PAGE_NUMBER",
            "page": 1,
        }

        # Try primary endpoint, fall back to legacy
        results = None
        for api_url in [TED_API_URL, TED_API_URL_FALLBACK]:
            try:
                results = self._call_api(api_url, request_body)
                if results is not None:
                    break
            except Exception as exc:
                logger.warning("ted_api_attempt_failed", url=api_url, error=str(exc))
                continue

        if results is None:
            logger.error("ted_search_failed", msg="All API endpoints failed")
            return []

        leads = self._parse_results(results)

        logger.info(
            "ted_search_complete",
            total_results=len(leads),
            query=query[:80],
        )

        return leads

    def _build_query(self, user_query: str) -> str:
        """Build a TED expert search query from the user's text.

        TED expert query syntax:
          - FT=[term] for full-text search
          - TD=[CN] for contract notices
          - Multiple conditions joined with AND/OR
        """
        # Start with SDS/EHS full-text search terms
        ft_terms = " OR ".join(SDS_SEARCH_TERMS)

        # If user provided extra terms, add them
        if user_query:
            # Extract meaningful words (skip common ones)
            skip_words = {
                "find", "search", "tender", "tenders", "rfp", "procurement",
                "for", "in", "the", "a", "an", "and", "or", "europe",
                "european", "eu", "global", "globally",
            }
            user_words = [
                w for w in user_query.lower().split()
                if w not in skip_words and len(w) > 2
            ]
            if user_words:
                user_ft = " OR ".join(f'"{w}"' for w in user_words[:5])
                ft_terms = f"({ft_terms}) AND ({user_ft})"

        # Only active contract notices (CN) and prior information notices (PIN)
        query = f"FT=[{ft_terms}] AND TD=[CN OR PIN]"

        return query

    def _call_api(
        self, url: str, body: dict,
    ) -> dict | None:
        """Call the TED search API and return the JSON response."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        client = httpx.Client(timeout=self.timeout)
        try:
            response = client.post(url, json=body, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "ted_api_http_error",
                status=exc.response.status_code,
                body=exc.response.text[:500],
            )
            return None
        except Exception as exc:
            logger.error("ted_api_error", error=str(exc))
            return None
        finally:
            client.close()

    def _parse_results(self, response: dict) -> list[TedTenderLead]:
        """Parse TED API response into TedTenderLead objects."""
        leads: list[TedTenderLead] = []

        # The response structure has "notices" or "results" depending on version
        notices = response.get("notices", response.get("results", []))
        if not notices:
            # Try alternative response structure
            notices = response.get("data", [])

        for notice in notices:
            try:
                lead = self._parse_notice(notice)
                if lead:
                    leads.append(lead)
            except Exception as exc:
                logger.debug("ted_parse_error", error=str(exc), notice=str(notice)[:200])

        return leads

    def _parse_notice(self, notice: dict) -> TedTenderLead | None:
        """Parse a single TED notice into a TedTenderLead."""
        # TED returns fields as key-value pairs — extract what we need
        # The structure varies by API version; handle both formats

        pub_number = self._get_field(notice, "publication-number", "")
        title = self._get_field(notice, "notice-title", "")
        buyer = self._get_field(notice, "buyer-name", "Unknown")
        pub_date = self._get_field(notice, "publication-date", "")
        deadline = self._get_field(notice, "deadline-receipt-tenders", "")
        description = self._get_field(notice, "short-description", "")
        notice_url = self._get_field(notice, "notice-url", "")

        if not title and not pub_number:
            return None

        # Build source URL
        if not notice_url and pub_number:
            notice_url = f"https://ted.europa.eu/en/notice/-/detail/{pub_number}"

        # Parse deadline to ISO format
        deadline_iso = self._parse_ted_date(deadline)
        posted_iso = self._parse_ted_date(pub_date)

        return TedTenderLead(
            lead_id=pub_number or f"TED-{uuid.uuid4().hex[:8].upper()}",
            title=title or f"TED Notice {pub_number}",
            description=description[:500] if description else "",
            agency=buyer,
            source_url=notice_url,
            submission_deadline=deadline_iso,
            posted_date=posted_iso,
            relevance_keywords=["ted.europa.eu", "EU procurement"],
            raw_data={
                **notice,
                "has_strong_match": True,    # TED results are pre-qualified
                "has_tender_signal": True,   # It's literally a tender portal
                "is_empty_page": False,
            },
        )

    @staticmethod
    def _get_field(notice: dict, field_name: str, default: str = "") -> str:
        """Extract a field value from a TED notice.

        TED API returns fields in various formats depending on version:
          - Direct key-value: notice["notice-title"]
          - Nested in "fields": notice["fields"]["notice-title"]
          - Array of {name, value}: [{"name": "notice-title", "value": "..."}]
          - Multilingual: {"eng": "English title", "fra": "French title"}
        """
        # Try direct access
        val = notice.get(field_name)
        if val:
            if isinstance(val, dict):
                # Multilingual — prefer English
                return val.get("eng", val.get("en", next(iter(val.values()), default)))
            if isinstance(val, list):
                return val[0] if val else default
            return str(val)

        # Try nested in "fields"
        fields = notice.get("fields", {})
        if isinstance(fields, dict):
            val = fields.get(field_name)
            if val:
                if isinstance(val, dict):
                    return val.get("eng", val.get("en", next(iter(val.values()), default)))
                if isinstance(val, list):
                    return val[0] if val else default
                return str(val)

        # Try array-of-dicts format
        if isinstance(fields, list):
            for f in fields:
                if isinstance(f, dict) and f.get("name") == field_name:
                    return str(f.get("value", default))

        # Try content blob (some API versions embed all data here)
        content = notice.get("content", notice.get("CONTENT", ""))
        if content and isinstance(content, str) and field_name == "short-description":
            # Extract first meaningful text chunk
            text = re.sub(r'<[^>]+>', ' ', content)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:500] if text else default

        return default

    @staticmethod
    def _parse_ted_date(date_str: str) -> str:
        """Parse TED date formats into ISO YYYY-MM-DD."""
        if not date_str:
            return ""

        # TED uses various formats: "20260515", "2026-05-15", "2026/05/15",
        # "15/05/2026", "2026-05-15T10:00:00+02:00"
        date_str = date_str.strip()

        # ISO format already
        if re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
            return date_str[:10]

        # Compact: "20260515"
        if re.match(r'^\d{8}$', date_str):
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        # Slash format: "15/05/2026" or "2026/05/15"
        if re.match(r'^\d{2}/\d{2}/\d{4}$', date_str):
            parts = date_str.split("/")
            if int(parts[0]) > 12:  # day-first
                return f"{parts[2]}-{parts[1]}-{parts[0]}"
            return f"{parts[2]}-{parts[0]}-{parts[1]}"

        if re.match(r'^\d{4}/\d{2}/\d{2}$', date_str):
            return date_str.replace("/", "-")

        # Fallback: try dateutil
        try:
            from dateutil import parser as dateparser
            parsed = dateparser.parse(date_str)
            if parsed:
                return parsed.strftime("%Y-%m-%d")
        except Exception:
            pass

        return ""
