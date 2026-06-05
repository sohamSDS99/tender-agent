"""
HigherGov — paid aggregator API for US federal contract opportunities.

Why we use it on top of sam_gov.py:
  - HigherGov reindexes SAM.gov AND adds DLA, SLED (state/local/education),
    pre-RFP opportunities, plus AI summaries. Same underlying SAM data on the
    federal slice, but extra coverage on SLED + DLA + forward-looking pursuits
    that SAM alone doesn't surface.
  - We dedupe by source+sourceId fingerprint downstream, so the federal
    overlap with sam_gov is harmless (each lead gets its own fingerprint via
    source_portal="highergov" — operators see both surfaced if both APIs
    found it; analytics will tell us whether to keep sam_gov on long-term).

Quota awareness:
  HigherGov subscription = 10,000 records/month total across all calls. Each
  API call returns up to 100 records (page_size cap). Our searcher fires ONE
  call per NAICS code per search (NAICS_CODES has 6 entries), so a single
  user-triggered search consumes up to ~600 records. At 10K/month that's
  ~16 searches/month before we'd need a higher tier — fine for the human-
  triggered discovery cadence we have today.

Auth:
  `?api_key=<key>` query parameter (NOT a header — HigherGov's choice).
  We use httpx's `params=` so the key never ends up in URL-logging middleware
  (the URL gets reassembled internally; logs only see the path).

API quirks confirmed by direct probing of the live endpoint:
  - The `q` / `search_text` parameters are SILENTLY IGNORED — they do not
    filter results. All keyword filtering must happen client-side.
  - `naics_code` accepts a single code only. Repeated `naics_code=X&naics_code=Y`
    overrides (last wins, can't OR). Hence the per-code loop in this searcher.
  - One of {posted_date, captured_date, source_id, opp_key, version_key,
    agency_key, search_id} is REQUIRED — the API 400s without it.
  - Many records have `due_date: null` — those are Award Notices (already
    awarded, no submission window). We drop them client-side.

Reference: https://docs.highergov.com/import-and-export/api
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import structlog

from src.discovery.sam_gov import score_relevance

logger = structlog.get_logger(__name__)


HIGHERGOV_API_BASE = "https://www.highergov.com/api-external/opportunity/"

# Per-request timeout. The fan-out pool ceiling is 90s, so we keep this well
# under 1/4 of that so a slow API doesn't poison the whole search.
HTTP_TIMEOUT_SECONDS = 12.0

# NAICS codes targeted at EHS / SDS / chemical-safety / environmental work.
# Same shortlist sam_gov.py uses for its strong-match set — keeping them
# aligned means the same opportunity scored consistently by both modules.
# Tight list (6 codes) keeps the per-search quota footprint at ~600 records.
NAICS_CODES: list[str] = [
    "541620",  # Environmental Consulting Services           (most direct fit)
    "541690",  # Other Scientific & Technical Consulting
    "562910",  # Environmental Remediation Services
    "541380",  # Testing Laboratories
    "325180",  # Other Basic Inorganic Chemical Manufacturing
    "511210",  # Software Publishers                          (SDS / EHS SaaS)
]

# opp_type.description values that indicate a CLOSED / awarded opportunity —
# operator can't submit to these. Filter early so they never reach the
# downstream deadline filter (which would drop them anyway, but noisily).
CLOSED_OPP_TYPES = {
    "Award Notice",
    "Justification",
    "Justification and Approval (J&A)",
    "Intent to Bundle Requirements (DoD- Funded)",
    "Fair Opportunity / Limited Sources Justification",
    "Sale of Surplus Property",
}


@dataclass
class TenderLead:
    """A single tender opportunity returned by HigherGov.

    Shape mirrors src.discovery.sam_gov.TenderLead so the bridge's fan-out
    treats HigherGov leads identically to native SAM.gov leads (deadline
    filter, relevance rescore, fingerprint dedup, attachment fetcher).
    """
    lead_id: str
    title: str
    description: str
    agency: str
    source_portal: str = "highergov"
    source_url: str = ""
    naics_code: str = ""
    submission_deadline: str = ""
    posted_date: str = ""
    relevance_score: float = 0.0
    relevance_keywords: list[str] = field(default_factory=list)
    attachment_urls: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lead_id": self.lead_id,
            "title": self.title,
            "description": self.description,
            "agency": self.agency,
            "source_portal": self.source_portal,
            "source_url": self.source_url,
            "naics_code": self.naics_code,
            "submission_deadline": self.submission_deadline,
            "posted_date": self.posted_date,
            "relevance_score": self.relevance_score,
            "relevance_keywords": self.relevance_keywords,
            "attachment_urls": self.attachment_urls,
        }


class HigherGovSearcher:
    """Search HigherGov for SDS / EHS / chemical-safety federal opportunities.

    Usage:
        searcher = HigherGovSearcher()
        leads = searcher.search("chemical safety", max_results=15, days_back=60)
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("HIGHERGOV_API_KEY", "").strip()

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[TenderLead]:
        if not self.api_key:
            logger.warning("highergov_skipped_no_key")
            return []

        # `posted_date` accepts a single ISO date and returns everything posted
        # ON that day. Searching back `days_back` days = walking dates back.
        # We don't actually loop dates here — for the first cut we anchor to
        # "yesterday" (the freshest fully-published day) and rely on the
        # downstream deadline filter to keep only future-due tenders. If the
        # operator needs deeper history we'll add a date-walk in a follow-up.
        anchor_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

        collected: list[TenderLead] = []
        seen_keys: set[str] = set()

        try:
            with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client:
                for naics in NAICS_CODES:
                    page_results = self._fetch_naics_page(
                        client, naics=naics, posted_date=anchor_date
                    )
                    for record in page_results:
                        opp_key = record.get("opp_key", "")
                        if opp_key and opp_key in seen_keys:
                            continue
                        if opp_key:
                            seen_keys.add(opp_key)
                        lead = self._record_to_lead(record)
                        if lead is None:
                            continue
                        collected.append(lead)
        except httpx.HTTPError as exc:
            logger.warning("highergov_http_error", error=str(exc))
            return collected  # return partial — fan-out is resilient

        # Sort by relevance descending and return top N.
        collected.sort(key=lambda l: l.relevance_score, reverse=True)
        result = collected[:max_results]

        logger.info(
            "highergov_search_complete",
            anchor_date=anchor_date,
            scanned=len(collected),
            returned=len(result),
            top_score=result[0].relevance_score if result else 0.0,
        )
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_naics_page(
        self,
        client: httpx.Client,
        *,
        naics: str,
        posted_date: str,
    ) -> list[dict[str, Any]]:
        """One paged API call for a single NAICS code on a given posted date."""
        params = {
            "api_key": self.api_key,
            "naics_code": naics,
            "posted_date": posted_date,
            "page_size": 100,
        }
        try:
            resp = client.get(HIGHERGOV_API_BASE, params=params)
        except httpx.HTTPError as exc:
            logger.warning("highergov_call_failed", naics=naics, error=str(exc))
            return []
        if resp.status_code != 200:
            logger.warning(
                "highergov_non_200",
                naics=naics,
                status=resp.status_code,
                body=resp.text[:200],
            )
            return []
        try:
            body = resp.json()
        except ValueError:
            logger.warning("highergov_bad_json", naics=naics)
            return []
        return list(body.get("results") or [])

    def _record_to_lead(self, record: dict[str, Any]) -> TenderLead | None:
        """Convert one HigherGov record to a TenderLead, or None to drop it."""
        # Drop Award Notices and other closed types immediately.
        opp_type = ((record.get("opp_type") or {}).get("description") or "").strip()
        if opp_type in CLOSED_OPP_TYPES:
            return None

        # Drop anything without a submission deadline — the downstream
        # deadline filter would block it anyway. Saves audit-log noise.
        due_date = record.get("due_date")
        if not due_date:
            return None

        title = (record.get("title") or "").strip()
        if not title:
            return None

        description_text = (record.get("description_text") or "").strip()
        ai_summary = (record.get("ai_summary") or "").strip()
        # Prefer the AI summary for relevance scoring since it's cleaner —
        # most description_text fields are amendment boilerplate. Pass both
        # to score_relevance so it can find keywords in either.
        scoring_blob = f"{ai_summary}\n\n{description_text}".strip()

        relevance, matched = score_relevance(title, scoring_blob)

        agency_obj = record.get("agency") or {}
        agency_name = agency_obj.get("agency_name") or agency_obj.get("agency_abbreviation") or ""

        naics_obj = record.get("naics_code") or {}
        naics_str = naics_obj.get("naics_code") or ""

        opp_key = record.get("opp_key") or ""
        lead_id = opp_key or str(uuid.uuid4())

        # Prefer the HigherGov path (cleaner URL) but fall back to source_path.
        url = record.get("path") or record.get("source_path") or ""

        # `document_path` is an API URL — operators can't open it in a browser
        # without the key. Skip embedding it; the bridge's attachment fetcher
        # uses SAM.gov's direct resourceLinks via the existing sam_gov module
        # if the same opportunity also surfaced there.
        attachment_urls: list[str] = []

        # Description preference for the operator-visible card: AI summary
        # if present (HigherGov's value-add), else the raw description.
        display_description = ai_summary if ai_summary else description_text

        return TenderLead(
            lead_id=lead_id,
            title=title,
            description=display_description,
            agency=agency_name,
            source_portal="highergov",
            source_url=url,
            naics_code=naics_str,
            submission_deadline=due_date,
            posted_date=record.get("posted_date") or "",
            relevance_score=relevance,
            relevance_keywords=matched,
            attachment_urls=attachment_urls,
            raw_data=record,
        )
