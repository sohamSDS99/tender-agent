"""
New Zealand GETS — Government Electronic Tenders Service.

GETS is New Zealand's central tender notification service operated
by MBIE (Ministry of Business, Innovation & Employment). Publishes
every public-sector open tender in NZ.

  Portal:    https://www.gets.govt.nz
  RSS feed:  https://www.gets.govt.nz/ExternalIndex.htm?action=rss
  Help:      https://www.procurement.govt.nz

Free, public, no authentication required. The RSS feed is the most
reliable interface — XML/Atom output, no rate limits in practice,
stable for a decade.

Usage:
    searcher = NzGetsSearcher()
    leads = searcher.search("chemical safety")
"""

from __future__ import annotations

import re
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from src.discovery.ocds_base import (
    OcdsTenderLead,
    score_relevance,
)

logger = structlog.get_logger(__name__)

# RSS endpoint — multiple URL variants tried so a URL-rotation doesn't
# silently break this module.
NZ_GETS_FEEDS: list[str] = [
    "https://www.gets.govt.nz/ExternalIndex.htm?action=rss",
    "https://www.gets.govt.nz/ExternalIndex.rss",
    "https://www.gets.govt.nz/feed/rss",
]

NZ_EXTRA_STRONG: list[str] = [
    "hazardous substances",
    "hsno",                  # Hazardous Substances and New Organisms Act
    "worksafe",
    "ehs",
]
NZ_EXTRA_PARTIAL: list[str] = [
    "rfp", "rfx", "rft", "rfi",
    "panel arrangement",
    "all-of-government",
]


class NzGetsSearcher:
    """Searches NZ GETS RSS feed for active EHS/SDS tenders."""

    def __init__(
        self,
        timeout: float = 30.0,
        min_relevance: float = 0.10,
    ) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        self._working_feed: str | None = None
        logger.info("nz_gets_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[OcdsTenderLead]:
        logger.info("nz_gets_search_start")
        items = self._fetch_feed()
        if not items:
            logger.warning("nz_gets_no_items")
            return []
        leads = self._parse_and_filter(items, user_query)
        leads.sort(key=lambda l: l.relevance_score, reverse=True)
        return leads[:max_results]

    def _fetch_feed(self) -> list[dict]:
        if self._working_feed:
            items = self._fetch_url(self._working_feed)
            if items:
                return items
            self._working_feed = None

        for url in NZ_GETS_FEEDS:
            items = self._fetch_url(url)
            if items:
                self._working_feed = url
                logger.info("nz_gets_feed_found", url=url, count=len(items))
                return items

        logger.error("nz_gets_all_feeds_failed")
        return []

    def _fetch_url(self, url: str) -> list[dict]:
        try:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; TenderAgent/2.0)",
                    "Accept": "application/rss+xml, application/xml, text/xml",
                },
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()
                xml_text = resp.text
            return self._parse_rss(xml_text)
        except httpx.HTTPStatusError as exc:
            logger.debug("nz_gets_http_error", status=exc.response.status_code, url=url[:120])
            return []
        except httpx.TimeoutException:
            logger.debug("nz_gets_timeout", url=url[:120])
            return []
        except Exception as exc:
            logger.debug("nz_gets_fetch_error", error=str(exc), url=url[:120])
            return []

    def _parse_rss(self, xml_text: str) -> list[dict]:
        """Parse RSS 2.0 / Atom into dicts with the fields we need."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.debug("nz_gets_xml_parse_error", error=str(exc))
            return []

        items: list[dict] = []
        # RSS 2.0: rss/channel/item
        for item in root.iter():
            tag = item.tag.lower().split("}")[-1]
            if tag != "item" and tag != "entry":
                continue
            record: dict[str, Any] = {}
            for child in list(item):
                ctag = child.tag.lower().split("}")[-1]
                text = (child.text or "").strip()
                if ctag in ("title", "description", "link", "pubdate", "guid", "summary"):
                    record[ctag] = text
                # Atom <link href="..."/>
                if ctag == "link" and not text:
                    href = child.attrib.get("href")
                    if href:
                        record["link"] = href
            if record.get("title"):
                items.append(record)
        return items

    def _parse_and_filter(self, items: list[dict], user_query: str) -> list[OcdsTenderLead]:
        leads: list[OcdsTenderLead] = []
        seen: set[str] = set()
        now = datetime.now(timezone.utc)
        for item in items:
            try:
                title = (item.get("title") or "").strip()
                desc = self._clean_html(item.get("description") or item.get("summary") or "")
                link = (item.get("link") or "").strip()
                guid = (item.get("guid") or link or f"gets-{uuid.uuid4().hex[:8]}").strip()
                if not title:
                    continue
                if guid in seen:
                    continue
                seen.add(guid)

                # Extract deadline from description (GETS embeds closing
                # date in the item body as "Close: dd-mmm-yyyy ..." or
                # "Closing Date: ...")
                deadline = self._extract_deadline(f"{title} {desc}")
                if deadline:
                    try:
                        dl = datetime.strptime(deadline, "%Y-%m-%d").date()
                        if dl < now.date():
                            continue
                    except Exception:
                        pass

                lead = OcdsTenderLead(
                    lead_id=guid,
                    title=title,
                    description=desc[:500],
                    agency="New Zealand Government",
                    source_portal="nz_gets",
                    source_url=link or "https://www.gets.govt.nz",
                    submission_deadline=deadline,
                    posted_date=self._parse_pubdate(item.get("pubdate", "")),
                )
                score, kws = score_relevance(
                    title,
                    desc,
                    "",
                    extra_strong=NZ_EXTRA_STRONG,
                    extra_partial=NZ_EXTRA_PARTIAL,
                )
                if user_query:
                    text = f"{title} {desc}".lower()
                    hits = sum(1 for w in user_query.lower().split() if w in text)
                    if hits:
                        score = round(min(score + min(hits * 0.05, 0.15), 1.0), 2)
                lead.relevance_score = score
                lead.relevance_keywords = kws
                if score >= self.min_relevance:
                    leads.append(lead)
            except Exception as exc:
                logger.debug("nz_gets_parse_error", error=str(exc))
        return leads

    @staticmethod
    def _clean_html(text: str) -> str:
        # GETS RSS embeds basic HTML in <description>. Strip tags & entities.
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _extract_deadline(text: str) -> str:
        """Find ISO YYYY-MM-DD closing date inside a free-text blob."""
        # Look for "Close ... <date>" patterns
        m = re.search(
            r"(?:clos\w+\s+(?:date|time)?[:\s]+)([\d]{1,2}[-/\s][A-Za-z]{3,9}[-/\s][\d]{4})",
            text,
            re.IGNORECASE,
        )
        if not m:
            return ""
        candidate = m.group(1)
        try:
            from dateutil import parser as dp
            dt = dp.parse(candidate, dayfirst=True)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return ""

    @staticmethod
    def _parse_pubdate(text: str) -> str:
        if not text:
            return ""
        try:
            from dateutil import parser as dp
            return dp.parse(text).strftime("%Y-%m-%d")
        except Exception:
            return ""
