"""
Deadline Verifier — fetches tender pages to extract and validate deadlines.

When the SERP pipeline can't find a deadline in the Google snippet, this
module fetches the actual tender page URL, extracts dates from the full
HTML text, and determines whether the tender is expired.

Three extraction layers (each fallback triggers the next):
  1. Regex patterns on the page text (fast, no cost)
  2. LLM extraction via OpenRouter (handles multilingual dates, complex layouts)
  3. Conservative heuristic: if the page mentions "closed" or "expired", block it

This module is called ONLY for SERP results where no deadline was found in
the snippet. API results already have structured deadlines and skip this.

Usage:
    from src.discovery.deadline_verifier import verify_deadline
    result = verify_deadline("https://ungm.org/Public/Notice/12345")
    # result = {"deadline": "2026-03-13", "status": "expired", "source": "regex"}
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Date patterns for full-page extraction (more comprehensive than snippet regex)
# These cover formats found across government procurement portals worldwide.
# ---------------------------------------------------------------------------

_DATE_FRAGMENT = (
    r'(\d{1,2}[\s/\-\.]\w{3,9}[\s/\-\.]\d{4}'     # 13-Mar-2026 / 15 April 2026 / 13.Mar.2026
    r'|\w{3,9}\s+\d{1,2},?\s+\d{4}'                  # March 13, 2026
    r'|\d{4}[\-/]\d{2}[\-/]\d{2}'                     # 2026-03-13
    r'|\d{1,2}[\-/]\d{1,2}[\-/]\d{4}'                 # 03/13/2026 or 13/03/2026
    r'|\d{1,2}\.\d{1,2}\.\d{4}'                       # 30.06.2026 (European dot format)
    r'|\d{1,2}\s+\w{3,9}\s+\d{4}'                     # 13 March 2026
    r')'
)

# Ordered by specificity — most specific patterns first
DEADLINE_PAGE_PATTERNS: list[re.Pattern[str]] = [
    # "Deadline on: 13-Mar-2026 08:00 (GMT -5.00)" — UNGM format
    re.compile(
        r'deadline\s*(?:on\s*)?[:\-–—]?\s*' + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
    # "Closing date: 15 April 2026" / "Close date: ..."
    re.compile(
        r'clos(?:ing|e)\s*date\s*[:\-–—]?\s*' + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
    # "Due date: ..." / "Due: ..."
    re.compile(
        r'due\s*(?:date\s*)?[:\-–—]?\s*' + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
    # "Submission deadline: ..." / "Submissions due: ..."
    re.compile(
        r'submissions?\s*(?:deadline|due\s*(?:date)?)\s*[:\-–—]?\s*' + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
    # "Response deadline: ..." / "Response due: ..."
    re.compile(
        r'response\s*(?:deadline|due\s*(?:date)?)\s*[:\-–—]?\s*' + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
    # "Proposals due: ..." / "Bids due: ..." / "Offers due: ..."
    re.compile(
        r'(?:proposals?|bids?|offers?|tenders?)\s*(?:due|deadline)\s*[:\-–—]?\s*' + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
    # "Submit by: ..." / "Submit before: ..."
    re.compile(
        r'submit\s*(?:by|before|no\s*later\s*than)\s*[:\-–—]?\s*' + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
    # "Open until: ..." / "Valid until: ..."
    re.compile(
        r'(?:open|valid)\s*(?:until|through|till)\s*[:\-–—]?\s*' + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
    # "Expires: ..." / "Expiry date: ..."
    re.compile(
        r'expir(?:es?|y|ation)\s*(?:date\s*)?[:\-–—]?\s*' + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
    # "Fecha de cierre: ..." (Spanish)
    re.compile(
        r'fecha\s*de\s*(?:cierre|vencimiento|entrega)\s*[:\-–—]?\s*' + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
    # "Date limite: ..." (French)
    re.compile(
        r'date\s*limite\s*(?:de\s*(?:soumission|depot|remise))?\s*[:\-–—]?\s*' + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
    # "Frist: ..." / "Abgabefrist: ..." (German)
    re.compile(
        r'(?:abgabe)?frist\s*[:\-–—]?\s*' + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
    # "Scadenza: ..." / "Termine: ..." (Italian)
    re.compile(
        r'(?:scadenza|termine)\s*(?:di\s*presentazione)?\s*[:\-–—]?\s*' + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
    # "Prazo: ..." (Portuguese)
    re.compile(
        r'prazo\s*(?:de\s*(?:entrega|submiss[aã]o))?\s*[:\-–—]?\s*' + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
]

# Patterns that indicate a tender is closed/expired without needing a date
CLOSED_INDICATORS: list[re.Pattern[str]] = [
    re.compile(r'\bthis\s+(?:tender|opportunity|notice|solicitation)\s+(?:has\s+)?(?:closed|expired|ended)\b', re.IGNORECASE),
    re.compile(r'\b(?:tender|opportunity|notice|solicitation)\s+is\s+(?:now\s+)?closed\b', re.IGNORECASE),
    re.compile(r'\bstatus\s*[:\-]?\s*closed\b', re.IGNORECASE),
    re.compile(r'\bstatus\s*[:\-]?\s*expired\b', re.IGNORECASE),
    re.compile(r'\bstatus\s*[:\-]?\s*cancelled\b', re.IGNORECASE),
    re.compile(r'\bno\s+longer\s+accepting\s+(?:bids?|proposals?|submissions?|responses?)\b', re.IGNORECASE),
    re.compile(r'\bsubmission\s+period\s+(?:has\s+)?(?:ended|closed|expired)\b', re.IGNORECASE),
    re.compile(r'\bdeadline\s+has\s+passed\b', re.IGNORECASE),
    re.compile(r'\b(?:award|awarded)\s+to\b', re.IGNORECASE),
    re.compile(r'\bcontract\s+awarded\b', re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Date parser
# ---------------------------------------------------------------------------

def _parse_date_robust(raw: str) -> datetime | None:
    """Parse a date string with broad format support.

    More lenient than the snippet parser — handles international formats
    and doesn't reject dates in the past (we need to identify them as expired).
    """
    if not raw:
        return None

    raw = raw.strip().rstrip('.')

    # Remove timezone suffixes like "(GMT -5.00)" / "(EST)" / "UTC"
    raw = re.sub(r'\s*\(.*?\)\s*$', '', raw)
    raw = re.sub(r'\s+(?:UTC|GMT|EST|PST|CST|MST|CET|BST|IST|AEST)\s*$', '', raw, flags=re.IGNORECASE)

    try:
        from dateutil import parser as dateparser
        parsed = dateparser.parse(raw, dayfirst=True)
        if parsed:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            # Sanity: date should be within 5 years past to 3 years future
            now = datetime.now(timezone.utc)
            if (now - timedelta(days=1825)) <= parsed <= (now + timedelta(days=1095)):
                return parsed
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# HTML text extraction (lightweight — no BeautifulSoup dependency)
# ---------------------------------------------------------------------------

def _html_to_text(html: str) -> str:
    """Extract visible text from HTML. Simple regex-based approach."""
    # Remove script and style blocks
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode common HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&nbsp;', ' ').replace('&#39;', "'").replace('&quot;', '"')
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# LLM deadline extraction (fallback when regex fails)
# ---------------------------------------------------------------------------

def _llm_extract_deadline(page_text: str, url: str) -> str | None:
    """Use LLM to extract the submission deadline from page text.

    Sends a focused prompt to the configured LLM asking for the deadline.
    Returns ISO date string or None.
    """
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
    base_url = os.getenv("QWEN_BASE_URL", "https://openrouter.ai/api/v1")

    if not api_key:
        logger.debug("llm_deadline_skip", reason="no API key")
        return None

    # Use a cheap, fast model for this simple extraction task
    model = "qwen/qwen3-8b"

    # Truncate page text to ~2000 chars around deadline-related keywords
    focused_text = _extract_deadline_context(page_text, max_chars=2000)
    if not focused_text:
        focused_text = page_text[:3000]

    prompt = (
        "Extract the submission deadline date from this tender/procurement page text. "
        "Look for phrases like 'Deadline', 'Closing date', 'Due date', 'Submit by', "
        "'Fecha de cierre', 'Date limite', or similar in any language.\n\n"
        "Rules:\n"
        "- Return ONLY the date in YYYY-MM-DD format\n"
        "- If multiple dates exist, return the SUBMISSION DEADLINE (not publication date)\n"
        "- If no deadline is found, return exactly: NONE\n"
        "- Do not explain, just return the date or NONE\n\n"
        f"Page URL: {url}\n\n"
        f"Page text:\n{focused_text}"
    )

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 30,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        answer = data["choices"][0]["message"]["content"].strip()

        # Clean up LLM response — extract date if wrapped in text
        # Remove <think>...</think> tags (Qwen3 thinking mode)
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()

        if "NONE" in answer.upper():
            return None

        # Try to find a date in the response
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', answer)
        if date_match:
            return date_match.group(0)

        # Try parsing the entire answer as a date
        parsed = _parse_date_robust(answer)
        if parsed:
            return parsed.strftime("%Y-%m-%d")

        return None

    except Exception as exc:
        logger.debug("llm_deadline_error", error=str(exc))
        return None


def _extract_deadline_context(text: str, max_chars: int = 2000) -> str:
    """Extract text around deadline-related keywords for focused LLM analysis."""
    keywords = [
        "deadline", "closing date", "due date", "submit by", "submission",
        "fecha de cierre", "date limite", "scadenza", "frist", "prazo",
        "expires", "expiry", "valid until", "open until",
    ]

    text_lower = text.lower()
    positions: list[int] = []

    for kw in keywords:
        idx = text_lower.find(kw)
        while idx != -1:
            positions.append(idx)
            idx = text_lower.find(kw, idx + 1)

    if not positions:
        return ""

    # Sort and take context windows around each match
    positions.sort()
    chunks: list[str] = []
    window = max_chars // max(len(positions), 1)
    window = max(window, 300)

    for pos in positions[:5]:  # Max 5 context windows
        start = max(0, pos - 100)
        end = min(len(text), pos + window)
        chunks.append(text[start:end])

    return "\n---\n".join(chunks)[:max_chars]


# ---------------------------------------------------------------------------
# Main verification function
# ---------------------------------------------------------------------------

def verify_deadline(
    url: str,
    timeout: float = 12.0,
    use_llm: bool = True,
) -> dict[str, str]:
    """Fetch a tender page and verify whether it has an active deadline.

    Args:
        url: The tender page URL to fetch and analyze.
        timeout: HTTP request timeout in seconds.
        use_llm: Whether to use LLM as fallback for date extraction.

    Returns:
        Dict with keys:
            - "deadline": ISO date string or "" if not found
            - "status": "active" | "expired" | "closed" | "unknown"
            - "source": "regex" | "llm" | "closed_indicator" | "none"
    """
    result = {"deadline": "", "status": "unknown", "source": "none"}

    if not url:
        return result

    # Fetch the page
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            html = resp.text
    except Exception as exc:
        logger.debug("deadline_verifier_fetch_error", url=url[:100], error=str(exc))
        return result

    page_text = _html_to_text(html)
    if not page_text or len(page_text) < 50:
        return result

    today = datetime.now(timezone.utc).date()

    # -----------------------------------------------------------------------
    # Layer 1: Check for "closed/expired" indicators (cheapest check)
    # -----------------------------------------------------------------------
    for pattern in CLOSED_INDICATORS:
        if pattern.search(page_text):
            logger.info("deadline_verifier_closed", url=url[:80], pattern=pattern.pattern[:60])
            result["status"] = "closed"
            result["source"] = "closed_indicator"
            return result

    # -----------------------------------------------------------------------
    # Layer 2: Regex extraction from full page text
    # -----------------------------------------------------------------------
    for pattern in DEADLINE_PAGE_PATTERNS:
        match = pattern.search(page_text)
        if match:
            raw_date = match.group(1).strip()
            parsed = _parse_date_robust(raw_date)
            if parsed:
                deadline_date = parsed.date()
                deadline_iso = parsed.strftime("%Y-%m-%d")
                is_expired = deadline_date < today

                logger.info(
                    "deadline_verifier_regex_match",
                    url=url[:80],
                    deadline=deadline_iso,
                    expired=is_expired,
                    pattern_fragment=pattern.pattern[:40],
                )

                result["deadline"] = deadline_iso
                result["status"] = "expired" if is_expired else "active"
                result["source"] = "regex"
                return result

    # -----------------------------------------------------------------------
    # Layer 3: LLM extraction (fallback)
    # -----------------------------------------------------------------------
    if use_llm:
        llm_date = _llm_extract_deadline(page_text, url)
        if llm_date:
            parsed = _parse_date_robust(llm_date)
            if parsed:
                deadline_date = parsed.date()
                is_expired = deadline_date < today

                logger.info(
                    "deadline_verifier_llm_match",
                    url=url[:80],
                    deadline=llm_date,
                    expired=is_expired,
                )

                result["deadline"] = parsed.strftime("%Y-%m-%d")
                result["status"] = "expired" if is_expired else "active"
                result["source"] = "llm"
                return result

    logger.debug("deadline_verifier_no_deadline", url=url[:80])
    return result


def batch_verify_deadlines(
    leads: list,
    max_concurrent: int = 3,
    timeout: float = 12.0,
    use_llm: bool = True,
) -> list[dict[str, Any]]:
    """Verify deadlines for a batch of SERP leads that have no deadline.

    Fetches each lead's URL and extracts/validates the deadline.
    Returns a list of dicts: [{"lead_index": i, "result": verify_result}, ...]

    Only verifies leads where submission_deadline is empty/missing.
    """
    results: list[dict[str, Any]] = []

    for i, lead in enumerate(leads):
        deadline = getattr(lead, 'submission_deadline', '') or ''
        if deadline and len(deadline) >= 8:
            # Already has a deadline — skip verification
            continue

        url = getattr(lead, 'source_url', '') or ''
        if not url:
            continue

        logger.info("deadline_verifier_checking", index=i, url=url[:80])

        try:
            vr = verify_deadline(url, timeout=timeout, use_llm=use_llm)
            results.append({"lead_index": i, "result": vr})

            if vr["deadline"]:
                # Update the lead's deadline field
                lead.submission_deadline = vr["deadline"]

        except Exception as exc:
            logger.debug("deadline_verifier_batch_error", index=i, error=str(exc))

    return results
