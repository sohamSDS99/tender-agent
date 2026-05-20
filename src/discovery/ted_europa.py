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
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# TED API config
# ---------------------------------------------------------------------------

# TED v3 search API. v2 was retired in 2025; v3 (eForms) is the only live
# endpoint as of May 2026.  No fallback — there's nothing valid to fall
# back TO; if v3 is down the searcher just returns [] and the parallel
# fan-out keeps moving.
TED_API_URL = "https://api.ted.europa.eu/v3/notices/search"

# CPV codes relevant to SDS/EHS/chemical safety software
# See: https://simap.ted.europa.eu/cpv
RELEVANT_CPV_CODES: list[str] = [
    "905",      # Environmental services (broad)
    "7131720",  # Health and safety services
    "4800000",  # Software package and information systems
    "7200000",  # IT services: consulting, development, support
    "3342120",  # Safety equipment
    "3314100",  # Industrial chemicals
    "9052400",  # Hazardous waste management
]

# Full-text search terms for SDS/EHS domain
SDS_SEARCH_TERMS: list[str] = [
    "safety data sheet",
    "chemical safety",
    "SDS management",
    "hazardous material",
    "EHS",
    "GHS",
    "chemical compliance",
    "MSDS",
    "hazard communication",
]

# Fields to request from the TED v3 API.  Each one was live-probed against
# api.ted.europa.eu/v3/notices/search and returns HTTP 200; sending any
# OTHER eForm field code (e.g. "buyer-name", "deadline-receipt-tenders",
# "place-of-performance") now returns 400 and kills the whole request.
#
# The v3 API exposes ~1830 eForm field codes; we only need the ones that
# carry user-visible content for the AMS card view and relevance scoring.
REQUESTED_FIELDS: list[str] = [
    "publication-number",
    "notice-title",
    "description-glo",
    "deadline-receipt-tender-date-lot",
    "organisation-name-buyer",
    "classification-cpv",
    "tender-value",
    "tender-value-cur",
    "publication-date",
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
        # Build the expert query for TED's v3 search syntax
        query = self._build_query(user_query)

        logger.info("ted_search_start", query=query[:120], max_results=max_results)

        request_body = {
            "query": query,
            "fields": REQUESTED_FIELDS,
            "limit": min(max_results, 100),
            "scope": "ACTIVE",
            # onlyLatestVersions=true dedupes corrigenda — without it a
            # single tender that's been amended N times surfaces N rows.
            "onlyLatestVersions": True,
            "paginationMode": "PAGE_NUMBER",
            "page": 1,
        }

        results = self._call_api(TED_API_URL, request_body)
        if results is None:
            logger.error("ted_search_failed", msg="v3 search API returned no result")
            return []

        leads = self._parse_results(results)

        logger.info(
            "ted_search_complete",
            total_results=len(leads),
            query=query[:80],
        )

        return leads

    def _build_query(self, user_query: str) -> str:
        """Build a TED v3 expert search query.

        v3 syntax is BARE values joined with the operator, e.g.
            FT="safety data sheet" OR FT="chemical safety"
        The old v2 bracket form (FT=[a OR b]) returns HTTP 400 on v3.
        Field codes (FT, PD, CY, TD, etc.) come from the eForms SDK.
        """
        # SDS/EHS full-text branch — joined by OR-of-FT clauses
        sds_branch = " OR ".join(f'FT="{t}"' for t in SDS_SEARCH_TERMS)

        # User branch — narrow the SDS set by intersecting user keywords
        user_branch = ""
        if user_query:
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
                user_branch = " OR ".join(f'FT="{w}"' for w in user_words[:5])

        if user_branch:
            query = f"({sds_branch}) AND ({user_branch})"
        else:
            query = sds_branch

        # Recency guard: TED scope=ACTIVE already filters out expired
        # notices, but we add a publication-date lower bound to keep the
        # candidate set small and fast.  Default: last 120 days.
        pd_floor = (datetime.now(timezone.utc) - timedelta(days=120)).strftime("%Y%m%d")
        query = f"({query}) AND PD>={pd_floor}"

        return query

    def _call_api(
        self, url: str, body: dict,
    ) -> dict | None:
        """Call the TED search API and return the JSON response."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        client = httpx.Client(timeout=self.timeout, follow_redirects=True)
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
        """Parse a single TED v3 notice into a TedTenderLead.

        v3 returns each requested field as either a scalar, a list, or a
        multilingual dict like ``{"ENG": "...", "FRA": "..."}``.  The
        helper ``_get_field`` collapses all three shapes.  The public
        notice URL lives in ``links.html.ENG`` — that's the only reliable
        web URL TED returns.
        """
        pub_number = self._get_field(notice, "publication-number", "")
        title = self._get_field(notice, "notice-title", "")
        buyer = self._get_field(notice, "organisation-name-buyer", "Unknown")
        pub_date = self._get_field(notice, "publication-date", "")
        deadline = self._get_field(notice, "deadline-receipt-tender-date-lot", "")
        description = self._get_field(notice, "description-glo", "")
        cpv_code = self._get_field(notice, "classification-cpv", "")

        # links.html.ENG is the only reliable public URL TED v3 returns.
        notice_url = ""
        links = notice.get("links", {}) or {}
        html_links = links.get("html", {}) or {}
        if isinstance(html_links, dict):
            notice_url = html_links.get("ENG") or html_links.get("ENG-GB") or ""
            if not notice_url and html_links:
                # fall back to whatever language came first
                notice_url = next(iter(html_links.values()), "")
        if not notice_url and pub_number:
            notice_url = f"https://ted.europa.eu/en/notice/-/detail/{pub_number}"

        if not title and not pub_number:
            return None

        deadline_iso = self._parse_ted_date(deadline)
        posted_iso = self._parse_ted_date(pub_date)

        keywords = ["ted.europa.eu", "EU procurement"]
        if cpv_code and any(cpv_code.startswith(p) for p in RELEVANT_CPV_CODES):
            keywords.append(f"cpv:{cpv_code}")

        return TedTenderLead(
            lead_id=pub_number or f"TED-{uuid.uuid4().hex[:8].upper()}",
            title=title or f"TED Notice {pub_number}",
            description=description[:500] if description else "",
            agency=buyer or "Unknown",
            source_url=notice_url,
            submission_deadline=deadline_iso,
            posted_date=posted_iso,
            relevance_keywords=keywords,
            raw_data={
                "publication-number": pub_number,
                "cpv": cpv_code,
                "has_strong_match": True,    # TED results are pre-qualified
                "has_tender_signal": True,   # It's literally a tender portal
                "is_empty_page": False,
            },
        )

    @staticmethod
    def _get_field(notice: dict, field_name: str, default: str = "") -> str:
        """Extract a field value from a TED v3 notice.

        TED v3 returns each requested field as one of:
          - scalar string: ``"...
          - list of scalars: ``["...
          - list of multilingual dicts: ``[{"ENG": "...
          - bare multilingual dict: ``{"ENG": "...
          - list of lists (when the eForm has multiple lots): ``[["...
        """
        def _unwrap(v):  # recursively collapse list/dict shells
            if v is None:
                return default
            if isinstance(v, str):
                return v
            if isinstance(v, (int, float)):
                return str(v)
            if isinstance(v, list):
                if not v:
                    return default
                # Pick first non-empty entry, then unwrap
                for item in v:
                    out = _unwrap(item)
                    if out:
                        return out
                return default
            if isinstance(v, dict):
                # Multilingual — prefer English. eForm code is "ENG" / "ENG-GB";
                # legacy two-letter "eng"/"en" kept for backward compat.
                for k in ("ENG", "ENG-GB", "eng", "en"):
                    if k in v and v[k]:
                        return _unwrap(v[k])
                if v:
                    return _unwrap(next(iter(v.values())))
                return default
            return str(v)

        # Try direct access first (v3 shape)
        if field_name in notice:
            out = _unwrap(notice.get(field_name))
            if out:
                return out

        # Legacy v2 shape, kept for resilience if TED ever back-ports v2
        # responses to the v3 endpoint.  None of these branches fire on
        # current production responses but they're cheap.
        fields = notice.get("fields", {})
        if isinstance(fields, dict) and field_name in fields:
            out = _unwrap(fields[field_name])
            if out:
                return out
        if isinstance(fields, list):
            for f in fields:
                if isinstance(f, dict) and f.get("name") == field_name:
                    out = _unwrap(f.get("value"))
                    if out:
                        return out

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
