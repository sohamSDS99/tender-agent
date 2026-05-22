"""
EU Funding & Tenders Portal — calls for tenders from EU institutions.

The Funding & Tenders Portal is the single entry point for EU
institutional procurement (DG ENV, DG SANTE, DG GROW, ECHA, EFSA,
JRC, EEA, ECDC, etc.). For SDS / EHS / chemical-safety work this is
high-value — DG ENV runs REACH oversight, DG SANTE runs chemical
food-contact regs, ECHA literally regulates chemicals, JRC runs
chemical-safety research.

  Portal:    https://ec.europa.eu/info/funding-tenders/opportunities/portal/
  Search:    SEDIA public search API
  API docs:  https://webgate.ec.europa.eu/funding-tenders-opportunities/

The SEDIA search API uses a literal public token `apiKey=SEDIA` — it
is anonymous, no registration required. Has been live and stable
since 2017.

Usage:
    searcher = EuFundingTendersSearcher()
    leads = searcher.search("chemical safety")
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import structlog

from src.discovery.ocds_base import (
    OcdsTenderLead,
    score_relevance,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Endpoints — primary first, fallbacks for graceful degradation
# ---------------------------------------------------------------------------

SEDIA_SEARCH_API = "https://api.tech.ec.europa.eu/search-api/prod/rest/search"

# Fallback bulk-reference paths (used if SEDIA returns 0 records)
EU_FT_FALLBACK_ENDPOINTS: list[str] = [
    "https://ec.europa.eu/info/funding-tenders/opportunities/data/referenceData/grants.json",
    "https://ec.europa.eu/info/funding-tenders/opportunities/data/topicSearchResults.json",
]

EU_FT_WEB_BASE = (
    "https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/topic-details"
)

# ---------------------------------------------------------------------------
# EU institutional / DG-specific vocabulary
# ---------------------------------------------------------------------------

EU_EXTRA_STRONG: list[str] = [
    "reach regulation",
    "clp regulation",
    "biocidal products",
    "echa",
    "chemicals strategy",
    "food contact materials",
    "chemical safety assessment",
    "endocrine disruptor",
    "pesticide residue",
]

EU_EXTRA_PARTIAL: list[str] = [
    "horizon",
    "framework programme",
    "call for tenders",
    "dg env",
    "dg sante",
    "joint research centre",
    "jrc",
    "european chemicals agency",
]


def _build_eu_ft_url(topic_id: str) -> str:
    """Build a public portal URL for an EU F&T opportunity."""
    if not topic_id:
        return EU_FT_WEB_BASE
    return f"{EU_FT_WEB_BASE};code={topic_id}"


class EuFundingTendersSearcher:
    """Searches the EU Funding & Tenders Portal for chemical-safety
    relevant calls for tenders.

    Strategy:
      1. Try the SEDIA public search API first (most precise,
         supports text query).
      2. If SEDIA returns nothing, try the bulk reference-data dumps
         and filter client-side.
      3. Return [] on any failure — never crash the bridge.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.10,
    ) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        self._sedia_working: bool = True
        logger.info("eu_funding_tenders_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[OcdsTenderLead]:
        logger.info("eu_ft_search_start", days_back=days_back)

        records = self._fetch_via_sedia(user_query)
        if not records:
            logger.info("eu_ft_sedia_empty_falling_back")
            records = self._fetch_via_fallback()

        if not records:
            logger.warning("eu_ft_no_records")
            return []

        leads = self._parse_and_filter(records, user_query, days_back)
        leads.sort(key=lambda l: l.relevance_score, reverse=True)
        leads = leads[:max_results]
        logger.info(
            "eu_ft_search_complete",
            raw_count=len(records),
            filtered_count=len(leads),
            top_score=leads[0].relevance_score if leads else 0,
        )
        return leads

    # ------------------------------------------------------------------
    # SEDIA search API (primary)
    # ------------------------------------------------------------------

    def _fetch_via_sedia(self, user_query: str) -> list[dict]:
        """Query the SEDIA search API for tender-type opportunities.

        The API expects:
          - apiKey=SEDIA               (literal public token)
          - text=<query>               (full-text search; * = everything)
          - pageSize / pageNumber      (pagination)
          - The result body is a JSON envelope with `results: [...]`.

        We POST a JSON body for the type+status filters so the result
        set is pre-narrowed to active tenders (not grants/prizes).
        """
        if not self._sedia_working:
            return []

        # SEDIA's `text` param can be `*` to get everything, or a query.
        # We construct a chemical-safety-weighted query when the user
        # didn't supply one, so the default returns useful leads.
        query_text = user_query.strip() if user_query else (
            "chemical OR safety OR REACH OR CLP OR hazardous OR SDS OR EHS"
        )
        params = {
            "apiKey": "SEDIA",
            "text": query_text,
            "pageSize": "50",
            "pageNumber": "1",
        }
        # Body filters: tender type, status = OPEN ONLY (no forthcoming).
        # SEDIA status codes:
        #   31094501 = Forthcoming  — opening date in the future, no
        #                              bids yet. Operator can't act on
        #                              these so we DELIBERATELY exclude.
        #   31094502 = Open         — currently accepting submissions.
        #   31094503 = Closed       — past deadline, useless to surface.
        # Locking to Open prevents the agent from reporting calls that
        # haven't actually opened (red flag in operator workflows).
        body = {
            "languages": ["en"],
            "type": ["1", "2"],          # 1 = call for tender, 2 = call for proposal
            "status": ["31094502"],      # OPEN only — never forthcoming
        }
        try:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (compatible; TenderAgent/2.0)",
                },
            ) as client:
                resp = client.post(SEDIA_SEARCH_API, params=params, json=body)
                # SEDIA sometimes returns 200 with an error envelope;
                # treat anything non-200 as transient and degrade.
                resp.raise_for_status()
                data = resp.json()

            if not isinstance(data, dict):
                logger.debug("eu_ft_sedia_unexpected_shape", type=type(data).__name__)
                return []

            results = data.get("results") or data.get("Results") or []
            if not isinstance(results, list):
                return []
            logger.info("eu_ft_sedia_fetched", count=len(results))
            return results

        except httpx.HTTPStatusError as exc:
            logger.debug(
                "eu_ft_sedia_http_error",
                status=exc.response.status_code,
                url=SEDIA_SEARCH_API,
            )
            # 4xx on SEDIA likely means the API contract changed — flip
            # the flag so we don't waste round-trips on subsequent
            # searches in this process lifetime.
            if 400 <= exc.response.status_code < 500:
                self._sedia_working = False
            return []
        except httpx.TimeoutException:
            logger.debug("eu_ft_sedia_timeout")
            return []
        except Exception as exc:
            logger.debug(
                "eu_ft_sedia_fetch_error",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return []

    # ------------------------------------------------------------------
    # Bulk reference-data fallback
    # ------------------------------------------------------------------

    def _fetch_via_fallback(self) -> list[dict]:
        for url in EU_FT_FALLBACK_ENDPOINTS:
            try:
                with httpx.Client(
                    timeout=self.timeout,
                    follow_redirects=True,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "Mozilla/5.0 (compatible; TenderAgent/2.0)",
                    },
                ) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                    data = resp.json()
                if isinstance(data, dict):
                    for key in ("topicResults", "results", "data", "items", "records"):
                        if isinstance(data.get(key), list):
                            logger.info(
                                "eu_ft_fallback_fetched",
                                url=url[:120],
                                key=key,
                                count=len(data[key]),
                            )
                            return data[key]
                if isinstance(data, list):
                    return data
            except httpx.HTTPStatusError as exc:
                logger.debug(
                    "eu_ft_fallback_http_error",
                    status=exc.response.status_code,
                    url=url[:120],
                )
            except Exception as exc:
                logger.debug(
                    "eu_ft_fallback_fetch_error",
                    error=str(exc),
                    url=url[:120],
                )
        return []

    # ------------------------------------------------------------------
    # Parse + filter
    # ------------------------------------------------------------------

    def _parse_and_filter(
        self,
        records: list[dict],
        user_query: str,
        days_back: int,
    ) -> list[OcdsTenderLead]:
        leads: list[OcdsTenderLead] = []
        seen: set[str] = set()
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=days_back)

        for record in records:
            try:
                lead = self._record_to_lead(record)
                if not lead:
                    continue
                if lead.lead_id in seen:
                    continue
                seen.add(lead.lead_id)

                # Skip if deadline already passed
                if lead.submission_deadline:
                    try:
                        dl = datetime.strptime(lead.submission_deadline, "%Y-%m-%d").date()
                        if dl < now.date():
                            continue
                    except Exception:
                        pass

                # Skip if posted before our look-back window
                # AND skip if posted in the future (defensive: would
                # mean a Forthcoming-status tender slipped past the
                # SEDIA query filter — should never happen but cheap
                # to double-check). Operators should never see a
                # tender that hasn't actually opened yet.
                if lead.posted_date:
                    try:
                        pd = datetime.strptime(lead.posted_date, "%Y-%m-%d").date()
                        if pd < cutoff.date():
                            continue
                        if pd > now.date():
                            logger.info(
                                "eu_ft_dropped_forthcoming",
                                lead_id=lead.lead_id,
                                posted_date=lead.posted_date,
                            )
                            continue
                    except Exception:
                        pass

                score, kws = score_relevance(
                    lead.title,
                    lead.description,
                    lead.cpv_code,
                    extra_strong=EU_EXTRA_STRONG,
                    extra_partial=EU_EXTRA_PARTIAL,
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
                logger.debug("eu_ft_parse_error", error=str(exc))
        return leads

    def _record_to_lead(self, record: dict) -> OcdsTenderLead | None:
        """Normalise an EU F&T record into an OcdsTenderLead.

        Records from SEDIA and from the bulk fallback have different
        shapes — we accept either by checking each common field name
        in order. The metadata field naming has shifted over the years
        (callIdentifier, identifier, topicIdentifier, etc.) so we
        treat them as alternative spellings of the same thing.
        """
        # Metadata wrapper varies between endpoints
        meta = record.get("metadata") or record
        if not isinstance(meta, dict):
            return None

        def _first(*keys: str) -> str:
            for k in keys:
                v = meta.get(k)
                if isinstance(v, list) and v:
                    v = v[0]
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return ""

        topic_id = _first(
            "identifier", "topicIdentifier", "callIdentifier", "code", "id"
        )
        title = _first("title", "topicTitle", "callTitle")
        if not title:
            return None
        description = _first(
            "description", "topicDescription", "summary", "callSummary"
        )
        agency = _first("buyer", "publisher", "callingDg", "topicLeader") or "European Commission"

        deadline_raw = _first(
            "deadlineDate", "deadlineDates", "deadlineModel", "submissionDeadline", "endDate"
        )
        posted_raw = _first(
            "startDate", "publicationDate", "callStartDate"
        )

        deadline_iso = _parse_eu_date(deadline_raw)
        posted_iso = _parse_eu_date(posted_raw)

        return OcdsTenderLead(
            lead_id=topic_id or f"eu_ft-{uuid.uuid4().hex[:8].upper()}",
            title=title[:500],
            description=description[:500] if description else "",
            agency=agency,
            source_portal="eu_funding_tenders",
            source_url=_build_eu_ft_url(topic_id),
            submission_deadline=deadline_iso,
            posted_date=posted_iso,
        )


def _parse_eu_date(value: str) -> str:
    """Parse the EU portal's various date string formats to ISO YYYY-MM-DD."""
    if not value:
        return ""
    value = str(value).strip()

    # Already ISO
    if re.match(r"^\d{4}-\d{2}-\d{2}", value):
        return value[:10]

    # EU often serialises as DD/MM/YYYY or DD-MM-YYYY
    m = re.match(r"^(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", value)
    if m:
        d, mo, y = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"

    try:
        from dateutil import parser as dp
        return dp.parse(value).strftime("%Y-%m-%d")
    except Exception:
        return ""
