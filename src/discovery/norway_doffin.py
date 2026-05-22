"""
Norway Doffin — Norwegian government procurement OCDS API.

Doffin (Database for offentlige innkjøp) is Norway's official tender
publication service operated by DFØ. Publishes both Norwegian
(below-EU-threshold) and EU-level (TED-cross-published) notices.

  Portal:    https://doffin.no
  API base:  https://doffin.no/api/external
  OCDS docs: https://doffin.no/Help (in Norwegian; OCDS endpoint
             confirmed via Norwegian Open Government Data catalogue)

Free, public, no authentication required.

Usage:
    searcher = NorwayDoffinSearcher()
    leads = searcher.search("kjemikalier")  # "chemicals" in Norwegian
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import structlog

from src.discovery.ocds_base import (
    OcdsTenderLead,
    parse_ocds_release,
    score_relevance,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Candidate endpoints — Doffin has reshuffled its external API a few times
# ---------------------------------------------------------------------------

DOFFIN_ENDPOINTS: list[str] = [
    "https://doffin.no/api/external/notices/search",
    "https://doffin.no/api/external/Notices",
    "https://doffin.no/api/external/notice/search",
    "https://doffin.no/api/v1/notices",
]

DOFFIN_WEB_BASE = "https://doffin.no/notices"

# ---------------------------------------------------------------------------
# Norwegian SDS/EHS vocabulary
# ---------------------------------------------------------------------------

NO_EXTRA_STRONG: list[str] = [
    "sikkerhetsdatablad",   # safety data sheet
    "kjemikalier",          # chemicals
    "kjemisk sikkerhet",    # chemical safety
    "farlige stoffer",      # dangerous substances
    "hms",                  # HMS = Helse, Miljø, Sikkerhet (HSE)
    "arbeidsmiljø",         # work environment
]

NO_EXTRA_PARTIAL: list[str] = [
    "innkjøp",              # procurement
    "anskaffelse",          # acquisition
    "rammeavtale",          # framework agreement
]


def _build_doffin_url(release: dict) -> str:
    """Build a public Doffin notice URL.

    Notice URLs are doffin.no/notices/<id>. We try release.id, ocid,
    and tender.id in order.
    """
    for key in ("id", "ocid"):
        v = release.get(key, "")
        if v:
            return f"{DOFFIN_WEB_BASE}/{v}"
    tender = release.get("tender", {})
    if isinstance(tender, dict) and tender.get("id"):
        return f"{DOFFIN_WEB_BASE}/{tender['id']}"
    return "https://doffin.no"


class NorwayDoffinSearcher:
    """Searches Norway's Doffin OCDS API for active EHS/SDS tenders."""

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.10,
    ) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        self._working_endpoint: str | None = None
        logger.info("doffin_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[OcdsTenderLead]:
        logger.info("doffin_search_start", days_back=days_back)

        releases = self._fetch_releases(days_back)
        if not releases:
            logger.warning("doffin_no_releases")
            return []

        leads = self._parse_and_filter(releases, user_query)
        leads.sort(key=lambda l: l.relevance_score, reverse=True)
        leads = leads[:max_results]

        logger.info(
            "doffin_search_complete",
            raw_count=len(releases),
            filtered_count=len(leads),
            top_score=leads[0].relevance_score if leads else 0,
        )
        return leads

    def _fetch_releases(self, days_back: int) -> list[dict]:
        now = datetime.now(timezone.utc)
        from_iso = (now - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "publishedFrom": from_iso,
            "publishedTo": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dateFrom": from_iso[:10],
            "dateTo": now.strftime("%Y-%m-%d"),
            "size": "100",
            "limit": "100",
            "page": "1",
            "noticeStatus": "ACTIVE",
        }

        if self._working_endpoint:
            r = self._fetch_from_url(self._working_endpoint, params)
            if r:
                return r
            self._working_endpoint = None

        for endpoint in DOFFIN_ENDPOINTS:
            r = self._fetch_from_url(endpoint, params)
            if r:
                self._working_endpoint = endpoint
                logger.info("doffin_endpoint_found", endpoint=endpoint, count=len(r))
                return r

        logger.error("doffin_all_endpoints_failed")
        return []

    def _fetch_from_url(self, url: str, params: dict[str, str]) -> list[dict]:
        try:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; TenderAgent/2.0)",
                    "Accept": "application/json",
                },
            ) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            if isinstance(data, dict):
                for key in ("releases", "results", "items", "records", "notices", "data"):
                    if isinstance(data.get(key), list):
                        return data[key]
            if isinstance(data, list):
                return data
            return []

        except httpx.HTTPStatusError as exc:
            logger.debug("doffin_http_error", status=exc.response.status_code, url=url[:120])
            return []
        except httpx.TimeoutException:
            logger.debug("doffin_timeout", url=url[:120])
            return []
        except Exception as exc:
            logger.debug("doffin_fetch_error", error=str(exc), url=url[:120])
            return []

    def _parse_and_filter(self, releases: list[dict], user_query: str) -> list[OcdsTenderLead]:
        leads: list[OcdsTenderLead] = []
        seen: set[str] = set()
        for release in releases:
            try:
                # Doffin returns OCDS releases AND occasionally bare
                # "notice" objects. parse_ocds_release handles the
                # OCDS shape; for bare-notice shape we wrap it minimally.
                if "tender" not in release and "title" in release:
                    release = {
                        "id": release.get("id") or release.get("noticeId", ""),
                        "ocid": release.get("ocid", ""),
                        "tender": {
                            "title": release.get("title", ""),
                            "description": release.get("description", ""),
                            "tenderPeriod": {
                                "endDate": release.get("submissionDeadline")
                                or release.get("deadline", ""),
                            },
                        },
                        "buyer": {"name": release.get("buyerName", "")},
                        "date": release.get("publishedDate", ""),
                    }

                lead = parse_ocds_release(
                    release,
                    source_portal="norway_doffin",
                    build_url=_build_doffin_url,
                )
                if not lead or lead.lead_id in seen:
                    continue
                seen.add(lead.lead_id)

                score, kws = score_relevance(
                    lead.title,
                    lead.description,
                    lead.cpv_code,
                    extra_strong=NO_EXTRA_STRONG,
                    extra_partial=NO_EXTRA_PARTIAL,
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
                logger.debug("doffin_parse_error", error=str(exc))
        return leads
