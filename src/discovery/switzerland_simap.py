"""
Switzerland SIMAP — Swiss federal procurement platform.

SIMAP (Système d'information sur les marchés publics) is the joint
federal/cantonal/communal publication system for public procurement
in Switzerland. Multi-language (DE/FR/IT/EN) tender notices.

  Portal:    https://www.simap.ch
  Open data: https://opendata.swiss (search "SIMAP")
  Beschaffungsstelle Portal: https://www.beschaffungswesen.ch

Free, public, no authentication required. SIMAP's OCDS feed was
introduced as part of the BöB/IVöB revision. Older XML-RPC feeds
are still served as a fallback.

Usage:
    searcher = SwitzerlandSimapSearcher()
    leads = searcher.search("chemische sicherheit")
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

SIMAP_ENDPOINTS: list[str] = [
    "https://www.simap.ch/api/ocds/releases",
    "https://www.simap.ch/api/v1/notices",
    "https://www.simap.ch/services/external/notices",
    # opendata.swiss CKAN mirror (kept as fallback)
    "https://opendata.swiss/api/3/action/datastore_search?resource_id=simap-notices",
]

SIMAP_WEB_BASE = "https://www.simap.ch/shabforms/servlet/Search"

# Trilingual EHS vocabulary so we score Swiss notices regardless of
# which official language they were published in.
CH_EXTRA_STRONG: list[str] = [
    # German
    "chemikaliensicherheit",
    "sicherheitsdatenblatt",
    "gefahrstoffe",
    "arbeitssicherheit",
    # French
    "sécurité chimique",
    "fiche de données de sécurité",
    "substances dangereuses",
    "santé sécurité travail",
    # Italian
    "sicurezza chimica",
    "scheda di sicurezza",
    "sostanze pericolose",
]

CH_EXTRA_PARTIAL: list[str] = [
    "rahmenvertrag",
    "accord-cadre",
    "accordo quadro",
    "ausschreibung",
    "appel d'offres",
    "bando di gara",
]


def _build_simap_url(release: dict) -> str:
    for key in ("id", "ocid"):
        v = release.get(key, "")
        if v:
            return f"{SIMAP_WEB_BASE}?NOTICE_NR={v}"
    return "https://www.simap.ch"


class SwitzerlandSimapSearcher:
    """Searches Switzerland SIMAP for active EHS/SDS tenders."""

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.10,
    ) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        self._working_endpoint: str | None = None
        logger.info("simap_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[OcdsTenderLead]:
        logger.info("simap_search_start", days_back=days_back)
        releases = self._fetch_releases(days_back)
        if not releases:
            logger.warning("simap_no_releases")
            return []
        leads = self._parse_and_filter(releases, user_query)
        leads.sort(key=lambda l: l.relevance_score, reverse=True)
        return leads[:max_results]

    def _fetch_releases(self, days_back: int) -> list[dict]:
        now = datetime.now(timezone.utc)
        from_d = now - timedelta(days=days_back)
        params = {
            "publishedFrom": from_d.strftime("%Y-%m-%d"),
            "publishedTo": now.strftime("%Y-%m-%d"),
            "dateFrom": from_d.strftime("%Y-%m-%d"),
            "dateTo": now.strftime("%Y-%m-%d"),
            "limit": "100",
            "size": "100",
        }
        if self._working_endpoint:
            r = self._fetch_from_url(self._working_endpoint, params)
            if r:
                return r
            self._working_endpoint = None
        for endpoint in SIMAP_ENDPOINTS:
            r = self._fetch_from_url(endpoint, params)
            if r:
                self._working_endpoint = endpoint
                logger.info("simap_endpoint_found", endpoint=endpoint, count=len(r))
                return r
        logger.error("simap_all_endpoints_failed")
        return []

    def _fetch_from_url(self, url: str, params: dict[str, str]) -> list[dict]:
        try:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; TenderAgent/2.0)",
                    "Accept": "application/json",
                    "Accept-Language": "en, de;q=0.8, fr;q=0.7",
                },
            ) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
            if isinstance(data, dict):
                if "result" in data and isinstance(data["result"], dict):
                    inner = data["result"]
                    if isinstance(inner.get("records"), list):
                        return inner["records"]
                for key in ("releases", "results", "items", "records", "notices", "data"):
                    if isinstance(data.get(key), list):
                        return data[key]
            if isinstance(data, list):
                return data
            return []
        except httpx.HTTPStatusError as exc:
            logger.debug("simap_http_error", status=exc.response.status_code, url=url[:120])
            return []
        except httpx.TimeoutException:
            logger.debug("simap_timeout", url=url[:120])
            return []
        except Exception as exc:
            logger.debug("simap_fetch_error", error=str(exc), url=url[:120])
            return []

    def _parse_and_filter(self, releases: list[dict], user_query: str) -> list[OcdsTenderLead]:
        leads: list[OcdsTenderLead] = []
        seen: set[str] = set()
        for release in releases:
            try:
                if "tender" not in release and ("title" in release or "Titel" in release or "Bezeichnung" in release):
                    title = (
                        release.get("title")
                        or release.get("Titel")
                        or release.get("Bezeichnung")
                        or ""
                    )
                    release = {
                        "id": release.get("id") or release.get("_id") or release.get("noticeId", ""),
                        "tender": {
                            "title": title,
                            "description": release.get("description") or release.get("Beschreibung", ""),
                            "tenderPeriod": {
                                "endDate": release.get("submissionDeadline")
                                or release.get("Eingabefrist", ""),
                            },
                        },
                        "buyer": {"name": release.get("buyerName") or release.get("Auftraggeber", "")},
                        "date": release.get("publishedDate", ""),
                    }
                lead = parse_ocds_release(
                    release,
                    source_portal="switzerland_simap",
                    build_url=_build_simap_url,
                )
                if not lead or lead.lead_id in seen:
                    continue
                seen.add(lead.lead_id)
                score, kws = score_relevance(
                    lead.title,
                    lead.description,
                    lead.cpv_code,
                    extra_strong=CH_EXTRA_STRONG,
                    extra_partial=CH_EXTRA_PARTIAL,
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
                logger.debug("simap_parse_error", error=str(exc))
        return leads
