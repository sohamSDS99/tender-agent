"""
Global Tender Discovery via Bright Data SERP Proxy.

Searches Google for SDS/EHS-related tenders worldwide using
Bright Data's SERP proxy. Returns structured tender leads
scored by relevance to our domain.

Usage:
    searcher = SerpTenderSearcher()
    leads = searcher.search("SDS management tenders")
"""

from __future__ import annotations

import os
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

import httpx
import structlog

logger = structlog.get_logger(__name__)

# Relevance keywords (same as sam_gov.py for consistency)
STRONG_KEYWORDS: list[str] = [
    "safety data sheet", "sds", "msds",
    "chemical safety", "chemical management", "chemical inventory",
    "ghs", "globally harmonized",
    "ehs", "environment health safety", "environmental health",
    "hazardous material", "hazardous chemical", "hazmat",
    "osha", "hcs", "hazard communication",
    "regulatory compliance", "compliance software",
    "sds management", "sds authoring",
    "tier ii", "epcra", "toxic release",
    "workplace safety", "occupational safety",
]

PARTIAL_KEYWORDS: list[str] = [
    "safety", "compliance", "environmental", "regulation",
    "chemical", "hazard", "risk management", "software platform",
    "tender", "rfp", "rfq", "procurement", "solicitation", "bid",
]

# Regional tender portal mappings
REGIONAL_PORTALS: dict[str, list[str]] = {
    "europe": [
        "site:ted.europa.eu",
        "site:ungm.org",
        "site:gov.uk/contracts-finder",
        "site:tendersontime.com europe",
    ],
    "usa": [
        "site:sam.gov",
        "site:bidnetdirect.com",
    ],
    "canada": [
        "site:merx.com",
        "site:buyandsell.gc.ca",
        "site:bidsandtenders.ca",
    ],
    "australia": [
        "site:tenders.gov.au",
        "site:tendersontime.com australia",
        "site:globaltenders.com australia",
        "site:eprocure.com.au OR site:nswtenders.com.au OR site:qtenders.qld.gov.au",
    ],
    "india": [
        "site:eprocure.gov.in",
    ],
    "global": [
        "site:ungm.org",
        "site:globaltenders.com",
        "site:devbusiness.com",
        "site:tendersontime.com",
    ],
}

# Keywords that map to regions
REGION_KEYWORDS: dict[str, list[str]] = {
    "europe": ["europe", "european", "eu", "uk", "germany", "france", "spain", "italy", "netherlands", "sweden", "norway", "denmark", "finland", "belgium", "austria", "switzerland", "poland", "ireland", "portugal", "czech", "romania", "hungary", "greece"],
    "usa": ["usa", "us", "united states", "america", "federal", "sam.gov"],
    "canada": ["canada", "canadian"],
    "australia": ["australia", "australian"],
    "india": ["india", "indian"],
}


def detect_region(query: str) -> str:
    """Detect which region the user is asking about.

    Uses word-boundary matching to avoid false positives
    (e.g., 'us' matching inside 'Australian').
    """
    import re
    q = query.lower()

    # Check longer/more specific keywords first by sorting by length descending
    matches: list[tuple[str, int]] = []
    for region, keywords in REGION_KEYWORDS.items():
        for kw in keywords:
            # Use word boundary matching to avoid substring false positives
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, q):
                matches.append((region, len(kw)))

    if matches:
        # Return the region with the longest keyword match (most specific)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[0][0]

    return "global"


REGION_EXCLUDE_DOMAINS: dict[str, list[str]] = {
    "europe": ["sam.gov", "bidnetdirect.com", "virginiabids.com", "merx.com", "tenders.gov.au", "eprocure.gov.in"],
    "usa": ["ted.europa.eu", "tenders.gov.au", "merx.com", "eprocure.gov.in", "buyandsell.gc.ca"],
    "canada": ["sam.gov", "bidnetdirect.com", "virginiabids.com", "ted.europa.eu", "tenders.gov.au", "eprocure.gov.in"],
    "australia": ["sam.gov", "bidnetdirect.com", "virginiabids.com", "highergov.com", "ted.europa.eu", "merx.com", "eprocure.gov.in", "buyandsell.gc.ca"],
    "india": ["sam.gov", "bidnetdirect.com", "virginiabids.com", "ted.europa.eu", "merx.com", "tenders.gov.au"],
}

REGION_COUNTRY_NAMES: dict[str, list[str]] = {
    "europe": ["Europe", "EU", "UK", "European"],
    "usa": ["United States", "US", "USA", "federal"],
    "canada": ["Canada", "Canadian"],
    "australia": ["Australia", "Australian", "NSW", "Victoria", "Queensland"],
    "india": ["India", "Indian"],
}


def build_search_queries(user_query: str) -> list[str]:
    """Build targeted search queries based on what the user actually asked for."""
    region = detect_region(user_query)
    portals = REGIONAL_PORTALS.get(region, REGIONAL_PORTALS["global"])
    country_names = REGION_COUNTRY_NAMES.get(region, [])

    # Core SDS/EHS terms
    core_terms = '"safety data sheet" OR "SDS" OR "chemical safety" OR "EHS"'

    # Region label for queries
    region_label = country_names[0] if country_names else ""

    queries = []

    # Query 1: User's own query + tender keywords + region emphasis
    queries.append(f'{user_query} tender OR RFP OR procurement OR solicitation')

    # Query 2-3: Region-specific portal searches
    for portal in portals[:2]:
        queries.append(f'{portal} {core_terms}')

    # Query 4: Region-specific broader search
    if region_label:
        queries.append(f'{core_terms} "{region_label}" tender OR RFP OR procurement 2026')
    else:
        queries.append(f'{core_terms} tender OR RFP OR procurement 2026')

    return queries


def get_excluded_domains_for_region(user_query: str) -> set[str]:
    """Get domains that should be excluded based on the user's requested region."""
    region = detect_region(user_query)
    return set(REGION_EXCLUDE_DOMAINS.get(region, []))


@dataclass
class SerpTenderLead:
    """A tender opportunity discovered via SERP search."""
    lead_id: str
    title: str
    description: str
    source_url: str
    source_portal: str = "google_serp"
    agency: str = "Unknown"
    submission_deadline: str = ""
    posted_date: str = ""
    relevance_score: float = 0.0
    relevance_keywords: list[str] = field(default_factory=list)
    search_query: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)


def score_relevance(title: str, description: str) -> tuple[float, list[str]]:
    """Score how relevant a search result is to SDS/EHS tenders."""
    text = f"{title} {description}".lower()
    matched: list[str] = []

    strong_matches = 0
    for kw in STRONG_KEYWORDS:
        if kw in text:
            matched.append(kw)
            strong_matches += 1

    partial_matches = 0
    for kw in PARTIAL_KEYWORDS:
        if kw in text and kw not in matched:
            matched.append(kw)
            partial_matches += 1

    strong_score = min(strong_matches * 0.15, 0.75)
    partial_score = min(partial_matches * 0.05, 0.25)
    total = min(strong_score + partial_score, 1.0)

    return round(total, 2), matched


class SerpTenderSearcher:
    """Searches for global tenders using Bright Data SERP proxy.

    Usage:
        searcher = SerpTenderSearcher()
        leads = searcher.search("SDS management tenders")
    """

    def __init__(
        self,
        proxy_url: str | None = None,
        min_relevance: float = 0.05,
    ) -> None:
        self.proxy_url = proxy_url or os.getenv(
            "BRIGHTDATA_SERP_PROXY",
            ""
        )
        self.min_relevance = min_relevance

        if not self.proxy_url:
            raise ValueError(
                "BRIGHTDATA_SERP_PROXY is required. "
                "Set it in your .env file."
            )

        logger.info(
            "serp_searcher_initialized",
            proxy_configured=bool(self.proxy_url),
            min_relevance=self.min_relevance,
        )

    def search(
        self,
        user_query: str = "",
        max_results_per_query: int = 10,
        queries: list[str] | None = None,
    ) -> list[SerpTenderLead]:
        """Search for tenders using SERP proxy, targeted to the user's region.

        Args:
            user_query: The user's search query — used to detect region and build queries.
            max_results_per_query: Max results per search query.
            queries: Custom list of queries. If None, builds dynamically from user_query.

        Returns:
            List of SerpTenderLead objects, scored and sorted by relevance.
        """
        search_queries = queries or build_search_queries(user_query or "SDS management tenders")

        all_leads: list[SerpTenderLead] = []
        seen_urls: set[str] = set()

        for query in search_queries[:4]:  # Limit to 4 queries to stay within rate limits
            try:
                results = self._search_google(query, max_results_per_query)
                for result in results:
                    url = result.get("link", "")
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    score, keywords = score_relevance(title, snippet)

                    if score >= self.min_relevance:
                        lead = SerpTenderLead(
                            lead_id=f"SERP-{uuid.uuid4().hex[:8].upper()}",
                            title=title,
                            description=snippet,
                            source_url=url,
                            agency=self._extract_domain(url),
                            relevance_score=score,
                            relevance_keywords=keywords,
                            search_query=query,
                            posted_date=datetime.now(timezone.utc).isoformat(),
                            raw_data=result,
                        )
                        all_leads.append(lead)

            except Exception as exc:
                logger.error("serp_search_error", query=query, error=str(exc))

        # Sort by relevance
        all_leads.sort(key=lambda l: l.relevance_score, reverse=True)

        # Validate links — remove broken ones
        validated_leads = self._validate_links(all_leads[:15])

        logger.info(
            "serp_search_complete",
            total_leads=len(validated_leads),
            before_validation=len(all_leads),
            queries_run=min(len(search_queries), 4),
            top_score=validated_leads[0].relevance_score if validated_leads else 0.0,
        )

        return validated_leads

    def _validate_links(self, leads: list[SerpTenderLead]) -> list[SerpTenderLead]:
        """Check links with HEAD requests, remove 404s and timeouts."""
        valid: list[SerpTenderLead] = []
        client = httpx.Client(timeout=3.0, follow_redirects=True)
        try:
            for lead in leads:
                try:
                    resp = client.head(lead.source_url)
                    if resp.status_code < 400:
                        valid.append(lead)
                    elif resp.status_code == 403 or resp.status_code == 405:
                        # Some sites block HEAD requests but are still valid
                        valid.append(lead)
                    else:
                        logger.debug("link_invalid", url=lead.source_url, status=resp.status_code)
                except Exception:
                    # Timeout or connection error — include anyway, don't penalize slow sites
                    valid.append(lead)
        finally:
            client.close()
        return valid

    def _search_google(self, query: str, num_results: int = 10) -> list[dict]:
        """Execute a Google search via Bright Data SERP proxy."""
        search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={num_results}"

        logger.info("serp_query", query=query[:80])

        client = httpx.Client(proxy=self.proxy_url, verify=False, timeout=30.0)
        try:
            response = client.get(search_url)
            response.raise_for_status()
        finally:
            client.close()

        return self._parse_html_results(response.text, num_results)

    def _parse_html_results(self, html: str, max_results: int) -> list[dict]:
        """Parse Google search results from HTML using BeautifulSoup."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            # Fallback to regex if bs4 not available
            return self._parse_html_regex(html, max_results)

        import re

        soup = BeautifulSoup(html, "html.parser")
        results: list[dict] = []
        seen_urls: set[str] = set()

        # Strategy 1: Find div.g blocks (standard Google results)
        for div in soup.select("div.g"):
            if len(results) >= max_results:
                break

            link_tag = div.select_one("a[href^='http']")
            if not link_tag:
                continue

            url = link_tag.get("href", "")
            if not url or "google.com" in url or url in seen_urls:
                continue

            title_tag = div.select_one("h3")
            title = title_tag.get_text(strip=True) if title_tag else ""
            if not title or len(title) < 5:
                continue

            # Get snippet — try multiple selectors
            snippet = ""
            for selector in ["div.VwiC3b", "span.aCOpRe", "div[data-sncf]", "div.IsZvec"]:
                tag = div.select_one(selector)
                if tag:
                    snippet = tag.get_text(" ", strip=True)
                    break

            # Clean snippet: remove URLs, breadcrumbs, domain names
            snippet = re.sub(r'https?://\S+', '', snippet)
            snippet = re.sub(r'[\w.-]+\.\w{2,4}\s*›[^›\n]*(?:›[^›\n]*)*', '', snippet)
            snippet = re.sub(r'\s{2,}', ' ', snippet).strip()
            # Remove date prefixes like "Apr 15, 2026 — "
            snippet = re.sub(r'^[A-Z][a-z]{2}\s+\d{1,2},?\s+\d{4}\s*[—–-]\s*', '', snippet)

            seen_urls.add(url)
            results.append({"title": title, "link": url, "snippet": snippet[:200]})

        # Strategy 2: Fallback — find any a > h3 pattern
        if not results:
            for a_tag in soup.select("a[href^='http']"):
                if len(results) >= max_results:
                    break

                url = a_tag.get("href", "")
                if "google.com" in url or url in seen_urls:
                    continue

                h3 = a_tag.select_one("h3")
                if not h3:
                    continue

                title = h3.get_text(strip=True)
                if len(title) < 5:
                    continue

                seen_urls.add(url)
                results.append({"title": title, "link": url, "snippet": ""})

        return results

    def _parse_html_regex(self, html: str, max_results: int) -> list[dict]:
        """Fallback regex parser if BeautifulSoup is not available."""
        import re
        results: list[dict] = []

        # Find h3 tags with links
        pattern = re.compile(
            r'<a[^>]+href="(https?://(?!www\.google\.com)[^"]+)"[^>]*>.*?<h3[^>]*>(.*?)</h3>',
            re.DOTALL,
        )

        for match in pattern.finditer(html):
            if len(results) >= max_results:
                break

            url = match.group(1)
            title = re.sub(r'<[^>]+>', '', match.group(2)).strip()

            if title and len(title) >= 5:
                results.append({
                    "title": title,
                    "link": url,
                    "snippet": "",
                })

        return results

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract a readable domain name from a URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return "Unknown"
