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
import re
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from urllib.parse import quote_plus

import httpx
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Deadline extraction from snippet text
# ---------------------------------------------------------------------------

# Regex patterns to find deadline/due/closing dates in Google snippets
# Date-capture fragment shared by all deadline patterns
_DATE_CAPTURE = (
    r'(\d{1,2}[\s/\-]\w{3,9}[\s/\-]\d{4}'     # 15 April 2026 / 15-Mar-2026
    r'|\w{3,9}\s+\d{1,2},?\s+\d{4}'             # April 15, 2026
    r'|\d{4}[\-/]\d{2}[\-/]\d{2}'                # 2026-04-15
    r'|\d{1,2}[\-/]\d{1,2}[\-/]\d{4}'            # 04/15/2026
    r')'
)

_DEADLINE_PATTERNS: list[re.Pattern[str]] = [
    # "Deadline: April 15, 2026" / "Deadline on: 13-Mar-2026" / "Due: 15 April 2026"
    # The (?:on\s*)? handles UNGM's "Deadline on:" format
    re.compile(
        r'(?:deadline|due\s*date|closing\s*date|closes|close\s*date|'
        r'submissions?\s*(?:due|deadline)|response\s*(?:due|deadline)|'
        r'proposals?\s*due|bids?\s*due|offers?\s*due|submit\s*by)'
        r'\s*(?:on\s*)?[:\-–—]?\s*'
        + _DATE_CAPTURE,
        re.IGNORECASE,
    ),
    # "Open until..." / "Valid through..." / "Expires..."
    re.compile(
        r'(?:open\s+until|valid\s+(?:until|through|till)|'
        r'expir(?:es?|ation|y)\s*(?:date)?)\s*'
        r'[:\-–—]?\s*'
        + _DATE_CAPTURE,
        re.IGNORECASE,
    ),
    # "Published on: 18-Feb-2026" — not a deadline, but captured so we can
    # at least determine freshness (the bridge's staleness check uses this)
    re.compile(
        r'(?:published|posted|released|listed)\s*(?:on\s*)?[:\-–—]?\s*'
        + _DATE_CAPTURE,
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Empty page / no-results detection
# ---------------------------------------------------------------------------
_EMPTY_PAGE_INDICATORS: list[str] = [
    "no tenders found",
    "no results found",
    "sorry, no",
    "0 results",
    "no matching",
    "no opportunities found",
    "check back later",
    "no records found",
    "currently no",
]


def is_empty_page_snippet(snippet: str) -> bool:
    """Detect if a Google snippet suggests the page has zero actual content."""
    text = snippet.lower()
    return any(indicator in text for indicator in _EMPTY_PAGE_INDICATORS)


# ---------------------------------------------------------------------------
# Posted-date extraction — detects when a tender was published/released
# ---------------------------------------------------------------------------

_POSTED_DATE_PATTERNS: list[re.Pattern[str]] = [
    # "released on Nov 28, 2024" / "posted March 15, 2025" / "published Jan 2025"
    re.compile(
        r'(?:released\s+on|posted\s+on|posted|published\s+on|published|'
        r'date\s+(?:of\s+)?(?:release|publication|posting)|'
        r'(?:was|were)\s+released|listed\s+on|announced\s+on)\s*'
        r'[:\-–—]?\s*'
        r'(\w{3,9}\s+\d{1,2},?\s+\d{4}'             # Nov 28, 2024
        r'|\d{1,2}\s+\w{3,9}\s+\d{4}'                 # 28 Nov 2024
        r'|\d{4}[\-/]\d{2}[\-/]\d{2}'                  # 2024-11-28
        r'|\d{1,2}[\-/]\d{1,2}[\-/]\d{4}'              # 11/28/2024
        r'|\w{3,9}\s+\d{4}'                            # November 2024 (month+year)
        r')',
        re.IGNORECASE,
    ),
    # Google snippet date prefix: "Nov 28, 2024 — ..." (already stripped from
    # snippet by parser, but may survive in raw_data or title)
    re.compile(
        r'^(\w{3,9}\s+\d{1,2},?\s+\d{4})\s*[—–\-]',
        re.IGNORECASE,
    ),
]


def extract_posted_date_from_text(text: str) -> str:
    """Extract a posted/released/published date from snippet text.

    Returns an ISO-format date string (YYYY-MM-DD) or empty string.
    """
    if not text:
        return ""

    for pattern in _POSTED_DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            raw_date = match.group(1).strip()
            parsed = _parse_date_flexible(raw_date)
            if parsed:
                return parsed.strftime("%Y-%m-%d")

    return ""


def extract_deadline_from_text(text: str) -> str:
    """Extract a submission deadline from snippet/description text.

    Tries multiple regex patterns, then parses and validates the date.
    Returns an ISO-format date string (YYYY-MM-DD) or empty string.
    """
    if not text:
        return ""

    for pattern in _DEADLINE_PATTERNS:
        match = pattern.search(text)
        if match:
            raw_date = match.group(1).strip()
            parsed = _parse_date_flexible(raw_date)
            if parsed:
                return parsed.strftime("%Y-%m-%d")

    return ""


def _parse_date_flexible(raw: str) -> datetime | None:
    """Parse a date string in various formats. Returns None on failure.

    Uses a wide lookback window (3 years) so that expired tender dates
    are still parsed and can be classified as 'expired' downstream,
    rather than being treated as 'missing deadline'.
    """
    try:
        from dateutil import parser as dateparser
        parsed = dateparser.parse(raw, dayfirst=False)
        if parsed:
            # Sanity check: date should be within a reasonable range
            # (not in the distant past or far future)
            now = datetime.now(timezone.utc)
            # Allow dates up to 3 years in the past (to detect expired tenders)
            # and up to 2 years in the future
            min_date = now - timedelta(days=1095)
            max_date = now + timedelta(days=730)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            if min_date <= parsed <= max_date:
                return parsed
        return None
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Tender signal detection — is this actually a procurement page?
# ---------------------------------------------------------------------------
# A result must have at least ONE of these signals to be considered a tender.
# Without this check, news articles and blog posts about chemical safety
# score high on SDS keywords but are completely useless.
# ---------------------------------------------------------------------------

# Regex patterns that indicate procurement/tender language in snippets
_TENDER_SNIPPET_SIGNALS: list[re.Pattern[str]] = [
    re.compile(r'\brfp\b', re.IGNORECASE),
    re.compile(r'\brfq\b', re.IGNORECASE),
    re.compile(r'\brfi\b', re.IGNORECASE),
    re.compile(r'\bsolicitation\b', re.IGNORECASE),
    re.compile(r'\bprocurement\s+opportunit', re.IGNORECASE),
    re.compile(r'\bcontract\s+(?:opportunit|notice|award)', re.IGNORECASE),
    re.compile(r'\bbid\s*(?:ding|submission|s\b)', re.IGNORECASE),
    re.compile(r'\bsubmit\s+(?:by|before|proposal|your|a\s+)', re.IGNORECASE),
    re.compile(r'\bclosing\s+date\b', re.IGNORECASE),
    re.compile(r'\bdeadline\b', re.IGNORECASE),
    re.compile(r'\bdue\s+date\b', re.IGNORECASE),
    re.compile(r'\bnotice\s+(?:number|no|id)\b', re.IGNORECASE),
    re.compile(r'\breference\s*(?:number|no|id|:)\s*\S', re.IGNORECASE),
    re.compile(r'\bexpress\s+interest\b', re.IGNORECASE),
    re.compile(r'\binvitation\s+to\s+bid\b', re.IGNORECASE),
    re.compile(r'\bcall\s+for\s+(?:proposals?|tenders?|expressions?)', re.IGNORECASE),
    re.compile(r'\bopen\s+tender\b', re.IGNORECASE),
    re.compile(r'\bpublic\s+tender\b', re.IGNORECASE),
    re.compile(r'\btender\s+(?:notice|ref|id|number)', re.IGNORECASE),
    re.compile(r'\bpurchas(?:ing|e)\s+(?:office|department|agent)', re.IGNORECASE),
    re.compile(r'\bregistration\s+(?:level|required)\b', re.IGNORECASE),
]

# Domains that are definitively tender/procurement portals.
# A result from one of these domains automatically has a "tender signal"
# even if the snippet doesn't contain procurement language.
KNOWN_TENDER_PORTALS: set[str] = {
    "sam.gov", "ted.europa.eu", "ungm.org", "merx.com",
    "bidnetdirect.com", "virginiabids.com", "highergov.com",
    "tendersontime.com", "buyandsell.gc.ca", "bidsandtenders.ca",
    "tenders.gov.au", "eprocure.gov.in", "etenders.gov.za",
    "globaltenders.com", "dgmarket.com", "devbusiness.com",
    "contracts.gov.sg", "etimaden.gov.tr",
    "nswtenders.com.au", "qtenders.qld.gov.au", "eprocure.com.au",
    "bhutantenders.com", "tendersinfo.com", "tendertiger.com",
    "publictendering.com", "tenderlink.com",
}

# URL path patterns that indicate a specific tender/solicitation page
TENDER_URL_SIGNALS: list[str] = [
    "/opp/", "/notice/", "/solicitation", "/bid/", "/rfp/",
    "/tender/", "/procurement/", "/contract-opportunity/",
    "/open-bids/", "/opportunity/", "/search-rfp/",
    "/contract/", "/award/", "/announcement/",
]


def has_tender_signal(title: str, snippet: str, url: str) -> bool:
    """Check if a search result looks like an actual procurement/tender page.

    Returns True if ANY of these are true:
      - Snippet contains procurement-specific language (RFP, deadline, bid, etc.)
      - URL domain is a known tender portal
      - URL path contains tender-like patterns
    """
    from urllib.parse import urlparse

    # Check 1: Snippet/title contains procurement language
    text = f"{title} {snippet}"
    for pattern in _TENDER_SNIPPET_SIGNALS:
        if pattern.search(text):
            return True

    # Check 2: Domain is a known tender portal
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
        for portal in KNOWN_TENDER_PORTALS:
            if portal in domain:
                return True

        # Check 3: URL path has tender-like patterns
        path = urlparse(url).path.lower()
        for sig in TENDER_URL_SIGNALS:
            if sig in path:
                return True
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Relevance scoring keywords
# ---------------------------------------------------------------------------
# STRONG = high-confidence SDS/EHS terms (each worth 0.15)
# PARTIAL = general tender/compliance terms (each worth 0.05)
# CONTEXT_DEPENDENT = ambiguous terms that only count when a CONTEXT_ANCHOR
#   word also appears (prevents "Transport Sds" false positives)
# ---------------------------------------------------------------------------

STRONG_KEYWORDS: list[str] = [
    "safety data sheet", "msds",
    "chemical safety", "chemical management", "chemical inventory",
    "ghs", "globally harmonized",
    "environment health safety", "environmental health",
    "hazardous material", "hazardous chemical", "hazmat",
    "osha", "hcs", "hazard communication",
    "regulatory compliance", "compliance software",
    "sds management", "sds authoring",
    "tier ii", "epcra", "toxic release",
    "workplace safety", "occupational safety",
]

# These terms are ambiguous on their own — "sds" can be a company name,
# "ehs" can be an abbreviation for many things.  They only count as strong
# matches when accompanied by a CONTEXT_ANCHOR word.
CONTEXT_DEPENDENT_KEYWORDS: list[str] = [
    "sds", "ehs",
]

CONTEXT_ANCHORS: list[str] = [
    "chemical", "safety", "hazard", "compliance", "management",
    "regulatory", "environment", "occupational", "ghs", "osha",
    "data sheet", "authoring", "inventory",
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
    "europe": ["europe", "european", "eu", "uk", "germany", "france", "spain", "italy", "netherlands", "sweden", "norway", "denmark", "finland", "belgium", "austria", "switzerland", "poland", "ireland", "portugal", "czech", "romania", "hungary", "greece", "ukraine", "moldova", "georgia", "kosovo"],
    "usa": ["usa", "us", "united states", "america", "federal", "sam.gov"],
    "canada": ["canada", "canadian"],
    "australia": ["australia", "australian"],
    "india": ["india", "indian"],
    "south_america": ["brazil", "brazilian", "colombia", "colombian", "peru", "peruvian", "ecuador", "chile", "argentina", "south america", "latin america", "dominican republic", "honduras", "guatemala", "caribbean"],
    "africa": ["south africa", "african", "africa", "nigeria", "kenya", "rwanda", "ghana"],
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
    "south_america": ["sam.gov", "ted.europa.eu", "tenders.gov.au", "eprocure.gov.in"],
    "africa": ["sam.gov", "ted.europa.eu", "tenders.gov.au", "eprocure.gov.in"],
}

REGION_COUNTRY_NAMES: dict[str, list[str]] = {
    "europe": ["Europe", "EU", "UK", "European"],
    "usa": ["United States", "US", "USA", "federal"],
    "canada": ["Canada", "Canadian"],
    "australia": ["Australia", "Australian", "NSW", "Victoria", "Queensland"],
    "india": ["India", "Indian"],
    "south_america": ["South America", "Latin America", "Brazil", "Colombia", "Peru"],
    "africa": ["Africa", "South Africa", "Nigeria", "Kenya"],
}


# Negative terms appended to general (non-portal) queries to tell Google
# to exclude news articles, blog posts, and informational content.
# Portal-specific queries (site:...) don't need these because the portal
# itself is a procurement site.
NOISE_EXCLUSION = (
    '-blog -article -magazine -webinar -"white paper" '
    '-"case study" -news -training -podcast -conference'
)


def build_search_queries(user_query: str) -> list[str]:
    """Build targeted search queries based on what the user actually asked for.

    General queries include noise exclusion terms to filter out news/articles.
    Portal-specific queries rely on the portal's own curation.
    """
    region = detect_region(user_query)
    portals = REGIONAL_PORTALS.get(region, REGIONAL_PORTALS["global"])
    country_names = REGION_COUNTRY_NAMES.get(region, [])

    # Core SDS/EHS terms
    core_terms = '"safety data sheet" OR "SDS" OR "chemical safety" OR "EHS"'

    # Region label for queries
    region_label = country_names[0] if country_names else ""

    # Current year for recency bias
    current_year = str(datetime.now(timezone.utc).year)

    queries = []

    # Query 1: User's own query + tender keywords + noise exclusion
    queries.append(
        f'{user_query} tender OR RFP OR procurement OR solicitation {NOISE_EXCLUSION}'
    )

    # Query 2-3: Portal searches — NO year, NO noise exclusion.
    # Portal sites curate active procurement listings.
    for portal in portals[:2]:
        queries.append(f'{portal} {core_terms}')

    # Query 4: Broad regional search with year + noise exclusion
    if region_label:
        queries.append(
            f'{core_terms} "{region_label}" tender OR RFP OR procurement {current_year} {NOISE_EXCLUSION}'
        )
    else:
        queries.append(
            f'{core_terms} tender OR RFP OR procurement {current_year} {NOISE_EXCLUSION}'
        )

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


def score_relevance(title: str, description: str) -> tuple[float, list[str], bool]:
    """Score how relevant a search result is to SDS/EHS tenders.

    Returns:
        (score, matched_keywords, has_strong_match)
        has_strong_match is True when at least one unambiguous SDS/EHS
        keyword was found — used downstream to gate low-quality results.
    """
    text = f"{title} {description}".lower()
    matched: list[str] = []

    # 1. Unambiguous strong keywords
    strong_matches = 0
    for kw in STRONG_KEYWORDS:
        if kw in text:
            matched.append(kw)
            strong_matches += 1

    # 2. Context-dependent keywords (e.g. "sds", "ehs") — only count when
    #    a context anchor word also appears in the text
    has_context = any(anchor in text for anchor in CONTEXT_ANCHORS)
    for kw in CONTEXT_DEPENDENT_KEYWORDS:
        if kw in text and kw not in matched:
            if has_context:
                matched.append(kw)
                strong_matches += 1
            # If no context anchor, skip — "Transport Sds" ≠ Safety Data Sheet

    # 3. Partial keywords
    partial_matches = 0
    for kw in PARTIAL_KEYWORDS:
        if kw in text and kw not in matched:
            matched.append(kw)
            partial_matches += 1

    strong_score = min(strong_matches * 0.15, 0.75)
    partial_score = min(partial_matches * 0.05, 0.25)
    total = min(strong_score + partial_score, 1.0)

    return round(total, 2), matched, strong_matches > 0


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
                # Portal-specific queries (site:...) skip the date filter —
                # the portal itself curates active listings.
                is_portal_query = query.strip().startswith("site:")
                results = self._search_google(
                    query, max_results_per_query,
                    use_date_filter=not is_portal_query,
                )
                for result in results:
                    url = result.get("link", "")
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    score, keywords, has_strong = score_relevance(title, snippet)

                    # Extract deadline and posted date from the snippet text
                    combined_text = f"{title} {snippet}"
                    deadline = extract_deadline_from_text(combined_text)
                    posted = extract_posted_date_from_text(combined_text)

                    if score >= self.min_relevance:
                        # Compute classification signals for downstream filtering
                        tender_signal = has_tender_signal(title, snippet, url)
                        empty_page = is_empty_page_snippet(snippet)

                        lead = SerpTenderLead(
                            lead_id=f"SERP-{uuid.uuid4().hex[:8].upper()}",
                            title=title,
                            description=snippet,
                            source_url=url,
                            agency=self._extract_domain(url),
                            submission_deadline=deadline,
                            relevance_score=score,
                            relevance_keywords=keywords,
                            search_query=query,
                            posted_date=posted or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                            raw_data={
                                **result,
                                "has_strong_match": has_strong,
                                "has_tender_signal": tender_signal,
                                "is_empty_page": empty_page,
                            },
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

    def _search_google(
        self, query: str, num_results: int = 10, *, use_date_filter: bool = True,
    ) -> list[dict]:
        """Execute a Google search via Bright Data SERP proxy.

        Args:
            use_date_filter: When True, appends tbs=qdr:m6 (past 6 months)
                to the URL.  Set to False for portal-specific ``site:``
                queries where the portal already curates active listings.
        """
        search_url = (
            f"https://www.google.com/search"
            f"?q={quote_plus(query)}"
            f"&num={num_results}"
        )
        if use_date_filter:
            search_url += "&tbs=qdr:m6"

        date_label = "past_6mo" if use_date_filter else "none"
        logger.info("serp_query", query=query[:80], date_filter=date_label)

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
