"""
UN Global Marketplace (UNGM) — UN procurement notices.

UNGM aggregates tender notices from UN agencies (UNICEF, WHO, UNDP,
UNHCR, FAO, WFP, IAEA, etc.). Very rich source for:
  • Lab equipment + chemical reagents
  • Hazardous material handling contracts (especially humanitarian)
  • Health & safety training in developing countries
  • Environmental remediation

  Portal:    https://www.ungm.org
  API docs:  https://www.ungm.org/Public/Notice (REST endpoint exposed
             behind vendor auth)

REQUIRES vendor registration. Free to sign up at
https://www.ungm.org/Account/Account/SignUp — once approved, generate
an API token from your vendor account and paste it into:

    tender-agent/.env
        UNGM_API_KEY=<your_token>

While UNGM_API_KEY is unset, this module self-disables — the searcher
returns [] immediately and the bridge logs a one-line skip notice.
NO HTTP calls are made.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import httpx
import structlog

from src.discovery.ocds_base import (
    OcdsTenderLead,
    parse_ocds_release,
    score_relevance,
)

logger = structlog.get_logger(__name__)

UNGM_ENDPOINTS: list[str] = [
    "https://www.ungm.org/Public/Notice",
    "https://www.ungm.org/api/v1/notice/search",
]

UNGM_WEB_BASE = "https://www.ungm.org/Public/Notice"

UN_EXTRA_STRONG: list[str] = [
    "laboratory reagent",
    "laboratory chemical",
    "hazardous waste",
    "pesticide",
    "medical waste",
    "ipc",                       # Infection Prevention & Control
]
UN_EXTRA_PARTIAL: list[str] = [
    "procurement notice",
    "request for proposal",
    "rfp",
    "rfq",
    "expression of interest",
    "eoi",
]


def _build_ungm_url(release: dict) -> str:
    for key in ("id", "noticeId", "ocid"):
        v = release.get(key, "")
        if v:
            return f"{UNGM_WEB_BASE}/{v}"
    return UNGM_WEB_BASE


class UngmSearcher:
    """Searches UN Global Marketplace for active EHS/SDS tenders.

    Self-disables when UNGM_API_KEY env var is not set — no HTTP calls
    are made and the bridge gracefully reports the skip in its log.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.10,
    ) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        self._working_endpoint: str | None = None
        self.api_key = (os.getenv("UNGM_API_KEY") or "").strip()
        if self.api_key:
            logger.info("ungm_searcher_initialized", auth="key_present")
        else:
            logger.info("ungm_searcher_initialized", auth="disabled_no_key")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[OcdsTenderLead]:
        if not self.api_key:
            # Hard-disable until the key arrives. Never make a request.
            logger.info("ungm_skipped_no_key")
            return []

        logger.info("ungm_search_start", days_back=days_back)
        notices = self._fetch_notices(days_back)
        if not notices:
            logger.warning("ungm_no_notices")
            return []

        leads = self._parse_and_filter(notices, user_query)
        leads.sort(key=lambda l: l.relevance_score, reverse=True)
        return leads[:max_results]

    def _fetch_notices(self, days_back: int) -> list[dict]:
        now = datetime.now(timezone.utc)
        from_iso = (now - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%S")
        params = {
            "publishedFrom": from_iso,
            "publishedTo": now.strftime("%Y-%m-%dT%H:%M:%S"),
            "limit": "100",
            "pageSize": "100",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-API-Key": self.api_key,
            "Accept": "application/json",
            "User-Agent": "TenderAgent/2.0",
        }

        if self._working_endpoint:
            r = self._fetch_from_url(self._working_endpoint, params, headers)
            if r:
                return r
            self._working_endpoint = None

        for endpoint in UNGM_ENDPOINTS:
            r = self._fetch_from_url(endpoint, params, headers)
            if r:
                self._working_endpoint = endpoint
                logger.info("ungm_endpoint_found", endpoint=endpoint, count=len(r))
                return r

        logger.error("ungm_all_endpoints_failed")
        return []

    def _fetch_from_url(
        self,
        url: str,
        params: dict[str, str],
        headers: dict[str, str],
    ) -> list[dict]:
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True, headers=headers) as client:
                resp = client.get(url, params=params)
                if resp.status_code == 401:
                    logger.error("ungm_unauthorized", msg="UNGM_API_KEY rejected — check registration")
                    return []
                resp.raise_for_status()
                data = resp.json()
            if isinstance(data, dict):
                for key in ("releases", "notices", "results", "items", "records", "data"):
                    if isinstance(data.get(key), list):
                        return data[key]
            if isinstance(data, list):
                return data
            return []
        except httpx.HTTPStatusError as exc:
            logger.debug("ungm_http_error", status=exc.response.status_code, url=url[:120])
            return []
        except httpx.TimeoutException:
            logger.debug("ungm_timeout", url=url[:120])
            return []
        except Exception as exc:
            logger.debug("ungm_fetch_error", error=str(exc), url=url[:120])
            return []

    def _parse_and_filter(self, notices: list[dict], user_query: str) -> list[OcdsTenderLead]:
        leads: list[OcdsTenderLead] = []
        seen: set[str] = set()
        for notice in notices:
            try:
                # UNGM may return native UN notice shape OR OCDS. Normalise.
                if "tender" not in notice and "title" in notice:
                    notice = {
                        "id": notice.get("noticeId") or notice.get("id", ""),
                        "tender": {
                            "title": notice.get("title", ""),
                            "description": notice.get("description", ""),
                            "tenderPeriod": {
                                "endDate": notice.get("deadline")
                                or notice.get("submissionDeadline", ""),
                            },
                        },
                        "buyer": {"name": notice.get("agency") or notice.get("buyer", "UN Agency")},
                        "date": notice.get("publishedDate", ""),
                    }

                lead = parse_ocds_release(notice, "ungm", build_url=_build_ungm_url)
                if not lead or lead.lead_id in seen:
                    continue
                seen.add(lead.lead_id)
                score, kws = score_relevance(
                    lead.title,
                    lead.description,
                    lead.cpv_code,
                    extra_strong=UN_EXTRA_STRONG,
                    extra_partial=UN_EXTRA_PARTIAL,
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
                logger.debug("ungm_parse_error", error=str(exc))
        return leads
