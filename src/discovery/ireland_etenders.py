"""
Ireland eTenders — Irish government procurement OCDS API.

Operated by the Office of Government Procurement (OGP). Publishes
every Irish public-sector tender notice; also cross-publishes
above-threshold contracts to TED so a slight overlap with the
ted_europa source is expected (deduped downstream by fingerprint).

  Portal:    https://www.etenders.gov.ie
  Open data: https://data.gov.ie (search "etenders" / "OCDS")
  OCDS docs: https://www.etenders.gov.ie/api

Free, public, no authentication required.

Usage:
    searcher = IrelandETendersSearcher()
    leads = searcher.search("chemical safety")
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
# Endpoints — eTenders Ireland has both a portal API and a data.gov.ie mirror
# ---------------------------------------------------------------------------

IRELAND_ENDPOINTS: list[str] = [
    "https://www.etenders.gov.ie/api/ocds/releases",
    "https://www.etenders.gov.ie/api/v1/releases",
    "https://etenders.gov.ie/api/ocds/releases",
    "https://data.gov.ie/api/3/action/datastore_search?resource_id=ocds_etenders",
]

IRELAND_WEB_BASE = "https://www.etenders.gov.ie/epps/cft/listContractDocuments.do"

IE_EXTRA_STRONG: list[str] = [
    "hsa",                       # Health and Safety Authority Ireland
    "biological agents",
    "carcinogens regulation",
]

IE_EXTRA_PARTIAL: list[str] = [
    "framework",
    "request for tenders",
    "rft",
]


def _build_ireland_url(release: dict) -> str:
    for key in ("id", "ocid"):
        v = release.get(key, "")
        if v:
            return f"{IRELAND_WEB_BASE}?resourceId={v}"
    return "https://www.etenders.gov.ie"


class IrelandETendersSearcher:
    """Searches Ireland eTenders OCDS API for active EHS/SDS tenders."""

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.10,
    ) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        self._working_endpoint: str | None = None
        logger.info("ireland_etenders_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[OcdsTenderLead]:
        logger.info("ireland_etenders_search_start", days_back=days_back)
        releases = self._fetch_releases(days_back)
        if not releases:
            logger.warning("ireland_etenders_no_releases")
            return []

        leads = self._parse_and_filter(releases, user_query)
        leads.sort(key=lambda l: l.relevance_score, reverse=True)
        leads = leads[:max_results]
        logger.info(
            "ireland_etenders_search_complete",
            raw_count=len(releases),
            filtered_count=len(leads),
        )
        return leads

    def _fetch_releases(self, days_back: int) -> list[dict]:
        now = datetime.now(timezone.utc)
        from_date = now - timedelta(days=days_back)
        params = {
            "publishedFrom": from_date.strftime("%Y-%m-%d"),
            "publishedTo": now.strftime("%Y-%m-%d"),
            "dateFrom": from_date.strftime("%Y-%m-%d"),
            "dateTo": now.strftime("%Y-%m-%d"),
            "limit": "100",
            "size": "100",
        }

        if self._working_endpoint:
            r = self._fetch_from_url(self._working_endpoint, params)
            if r:
                return r
            self._working_endpoint = None

        for endpoint in IRELAND_ENDPOINTS:
            r = self._fetch_from_url(endpoint, params)
            if r:
                self._working_endpoint = endpoint
                logger.info("ireland_etenders_endpoint_found", endpoint=endpoint, count=len(r))
                return r

        logger.error("ireland_etenders_all_endpoints_failed")
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
                # CKAN-style data.gov.ie shape: { result: { records: [...] } }
                if "result" in data and isinstance(data["result"], dict):
                    inner = data["result"]
                    if isinstance(inner.get("records"), list):
                        return inner["records"]
                for key in ("releases", "results", "items", "records", "data"):
                    if isinstance(data.get(key), list):
                        return data[key]
            if isinstance(data, list):
                return data
            return []

        except httpx.HTTPStatusError as exc:
            logger.debug("ireland_etenders_http_error", status=exc.response.status_code, url=url[:120])
            return []
        except httpx.TimeoutException:
            logger.debug("ireland_etenders_timeout", url=url[:120])
            return []
        except Exception as exc:
            logger.debug("ireland_etenders_fetch_error", error=str(exc), url=url[:120])
            return []

    def _parse_and_filter(self, releases: list[dict], user_query: str) -> list[OcdsTenderLead]:
        leads: list[OcdsTenderLead] = []
        seen: set[str] = set()
        for release in releases:
            try:
                # data.gov.ie CKAN records have a different shape — normalise
                if "tender" not in release and ("title" in release or "Title" in release):
                    release = _normalise_ckan_record(release)

                lead = parse_ocds_release(
                    release,
                    source_portal="ireland_etenders",
                    build_url=_build_ireland_url,
                )
                if not lead or lead.lead_id in seen:
                    continue
                seen.add(lead.lead_id)

                score, kws = score_relevance(
                    lead.title,
                    lead.description,
                    lead.cpv_code,
                    extra_strong=IE_EXTRA_STRONG,
                    extra_partial=IE_EXTRA_PARTIAL,
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
                logger.debug("ireland_etenders_parse_error", error=str(exc))
        return leads


def _normalise_ckan_record(record: dict) -> dict:
    """Reshape a CKAN datastore record into an OCDS-ish release."""
    title = record.get("title") or record.get("Title") or ""
    description = record.get("description") or record.get("Description") or ""
    buyer = (
        record.get("buyerName")
        or record.get("BuyerName")
        or record.get("contracting_authority")
        or ""
    )
    deadline = (
        record.get("submissionDeadline")
        or record.get("SubmissionDeadline")
        or record.get("closingDate")
        or ""
    )
    return {
        "id": record.get("id") or record.get("_id") or record.get("noticeId") or "",
        "tender": {
            "title": title,
            "description": description,
            "tenderPeriod": {"endDate": deadline},
        },
        "buyer": {"name": buyer},
        "date": record.get("publishedDate", ""),
    }
