"""
Nexus AMS Bridge for Tender Agent
===================================
This script connects the tender-agent to the Nexus AMS platform.

It does 4 things:
1. Registers the agent and sends heartbeats (so AMS knows the agent is alive)
2. Polls /api/agents/tender-agent/search-jobs for filter-driven tender searches
   queued from the AMS UI, runs them, and pushes structured results back via
   /api/agents/tender-agent/discovered-tenders
3. Polls /api/agents/tender-agent/tasks for fill_form tasks (created from a
   TenderPursuit when the admin selects Agent fill mode), fills the attached
   form via the existing form_parser + form_filler + form_writer pipeline,
   and submits the completed file for human approval via /api/submissions/create
4. Sends Slack notifications on search completion (when any result has
   relevance > 0.70) and on form-fill task completion or failure

Usage:
    cd ~/Desktop/tender-agent
    source .venv/bin/activate
    python scripts/nexus_bridge.py

Press Ctrl+C to stop.
"""

import json
import os
import sys
import time
import tempfile
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from nexus_sdk import NexusClient, AgentConfig

# Import tender-agent's discovery modules
from src.discovery.sam_gov import SamGovScraper, score_relevance
from src.discovery.serp_search import SerpTenderSearcher, get_excluded_domains_for_region
from src.discovery.ted_europa import TedEuropaSearcher
from src.discovery.uk_tenders import UkTenderSearcher
from src.discovery.boamp_france import BoampSearcher
from src.discovery.world_bank import WorldBankSearcher
from src.discovery.prozorro import ProzorroSearcher
from src.discovery.canada_buys import CanadaBuysSearcher
from src.discovery.austender import AusTenderSearcher
from src.discovery.sa_etender import SaEtenderSearcher
from src.discovery.colombia_secop import ColombiaSecopSearcher
from src.discovery.brazil_compras import BrazilComprasSearcher
from src.discovery.germany_bkms import GermanyBkmsSearcher
from src.discovery.italy_anac import ItalyAnacSearcher
from src.discovery.dominican_dgcp import DominicanDgcpSearcher
from src.discovery.peru_oece import PeruOeceSearcher
from src.discovery.world_bank_v2 import WorldBankV2Searcher
from src.discovery.nigeria_nocopo import NigeriaNocopoSearcher
from src.discovery.kenya_ppra import KenyaPpraSearcher
from src.discovery.uganda_gpp import UgandaGppSearcher
from src.discovery.mexico_cdmx import MexicoCdmxSearcher

# Import form processing modules
from src.forms.form_parser import FormParser
from src.forms.form_filler import FormFiller, FillResult
from src.forms.form_writer import FormWriter

# Slack notifications
from slack_notifier import (
    notify_task_completed,
    notify_task_failed,
    notify_agent_online,
    notify_agent_offline,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEXUS_URL = os.getenv("NEXUS_AMS_URL", "http://localhost:3000")
OPENROUTER_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("QWEN_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = "anthropic/claude-sonnet-4"
DRY_RUN = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")

# Module-level reference to the NexusClient (set in main())
_client: "NexusClient | None" = None

# Current thread ID for audit context (set per-message)
_current_thread_id: str = ""


def _audit(
    action_type: str,
    description: str,
    **kwargs,
) -> None:
    """Write an audit log entry. Safe to call even if client isn't ready."""
    if not _client:
        return
    try:
        _client.audit(
            action_type,
            description,
            conversation_id=kwargs.pop("conversation_id", _current_thread_id or None),
            **kwargs,
        )
    except Exception as exc:
        print(f"  [audit] Failed to log {action_type}: {exc}")


# ---------------------------------------------------------------------------
# Document Context — fetches assigned docs from AMS for RAG
# ---------------------------------------------------------------------------

_cached_context: str = ""
_context_fetched_at: float = 0.0
CONTEXT_TTL_SECONDS = 300  # 5 minutes


def search_knowledge_base(query: str, top_k: int = 15) -> str:
    """Search the agent's knowledge base for chunks relevant to the query.

    Uses semantic vector search via the AMS /search endpoint.
    Returns a formatted context block with only the most relevant document chunks.
    Falls back to the legacy full-context approach if the KB has no indexed chunks.
    """
    try:
        import httpx
        resp = httpx.post(
            f"{NEXUS_URL}/api/agents/tender-agent/search",
            json={"query": query, "topK": top_k},
            timeout=15.0,
        )
        if resp.status_code != 200:
            print(f"KB search returned {resp.status_code}, falling back to legacy context")
            return fetch_agent_context()

        data = resp.json()
        mode = data.get("mode", "semantic")

        if mode == "legacy":
            # No chunks indexed yet — use the full fallback context
            fallback = data.get("fallbackContext", "")
            if fallback:
                _audit(
                    "kb_search",
                    f"Knowledge base not indexed, using legacy full-text context",
                    output_payload={"mode": "legacy", "query": query[:200]},
                )
                return f"<agent_context>\n{fallback}\n</agent_context>"
            return fetch_agent_context()

        results = data.get("results", [])
        total_tokens = data.get("totalTokens", 0)
        formatted = data.get("formattedContext", "")

        if not results:
            print(f"KB search returned 0 results for query: {query[:80]}")
            return fetch_agent_context()

        # Collect unique source filenames for audit
        filenames = list(set(r.get("filename", "?") for r in results))
        _audit(
            "kb_search",
            f"KB search: {len(results)} chunks from {len(filenames)} docs ({total_tokens} tokens)",
            output_payload={
                "mode": "semantic",
                "query": query[:200],
                "result_count": len(results),
                "total_tokens": total_tokens,
                "source_files": filenames,
            },
        )

        parts = [
            "<agent_context>",
            f"The following {len(results)} relevant excerpts were retrieved from your knowledge base.",
            f"Source documents: {', '.join(filenames)}",
            "",
            formatted,
            "</agent_context>",
        ]
        return "\n".join(parts)

    except Exception as exc:
        print(f"KB search failed: {exc}, falling back to legacy context")
        return fetch_agent_context()


def fetch_agent_context() -> str:
    """Fetch ALL documents assigned to tender-agent from the AMS (legacy fallback).

    Returns a formatted context block. Cached for 5 minutes.
    Used as fallback when the knowledge base hasn't been indexed yet.
    """
    global _cached_context, _context_fetched_at

    if _cached_context and (time.time() - _context_fetched_at) < CONTEXT_TTL_SECONDS:
        return _cached_context

    try:
        import httpx
        resp = httpx.get(f"{NEXUS_URL}/api/agents/tender-agent/context", timeout=10.0)
        if resp.status_code != 200:
            print(f"Context endpoint returned {resp.status_code}")
            return ""

        data = resp.json()
        docs = data.get("documents", [])
        if not docs:
            print("No documents assigned to tender-agent")
            return ""

        print(f"Loaded {len(docs)} context document(s) (legacy mode)")
        doc_names = [d.get("filename", "?") for d in docs]
        _audit(
            "document_read",
            f"Loaded {len(docs)} context document(s) (legacy): {', '.join(doc_names)}",
            output_payload={"document_count": len(docs), "filenames": doc_names},
        )

        parts = ["<agent_context>"]
        parts.append("The following company documents have been assigned to you. "
                     "Use this information to answer questions and fill forms.")
        parts.append("")

        for doc in docs:
            filename = doc.get("filename", "Unknown")
            text = doc.get("extractedText", "").strip()
            status = doc.get("extractionStatus", "unknown")
            minio_key = doc.get("minioKey", "")

            if text:
                parts.append(f"## Document: {filename}")
                parts.append(text)
                parts.append("")
            elif minio_key:
                print(f"  Document '{filename}' has no extracted text (status={status}), trying local extraction...")
                local_text = _extract_text_locally(minio_key, filename)
                if local_text:
                    parts.append(f"## Document: {filename}")
                    parts.append(local_text)
                    parts.append("")
                else:
                    parts.append(f"## Document: {filename}")
                    parts.append(f"(Document uploaded but text extraction pending — file available at {minio_key})")
                    parts.append("")

        parts.append("</agent_context>")

        _cached_context = "\n".join(parts)
        _context_fetched_at = time.time()
        return _cached_context

    except Exception as exc:
        print(f"Context fetch failed: {exc}")
        return ""


def _extract_text_locally(minio_key: str, filename: str) -> str:
    """Download a document from AMS and extract text locally as a fallback.

    Used when the AMS extractor Docker service hasn't processed the document.
    """
    try:
        import httpx

        # Download from AMS file endpoint
        url = f"{NEXUS_URL}/api/chat/files/{minio_key}"
        resp = httpx.get(url, timeout=30.0, follow_redirects=True)
        if resp.status_code != 200:
            print(f"    Download failed: HTTP {resp.status_code}")
            return ""

        content = resp.content
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        # PDF
        if ext == "pdf":
            try:
                import pypdf
                import io
                reader = pypdf.PdfReader(io.BytesIO(content))
                text = ""
                for page in reader.pages:
                    text += (page.extract_text() or "") + "\n"
                if text.strip():
                    print(f"    Locally extracted {len(text)} chars from PDF")
                    return text.strip()[:4000]
            except Exception as exc:
                print(f"    PDF extraction failed: {exc}")

        # DOCX
        elif ext in ("docx", "doc"):
            try:
                from docx import Document
                import io
                doc = Document(io.BytesIO(content))
                text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                if text.strip():
                    print(f"    Locally extracted {len(text)} chars from DOCX")
                    return text.strip()[:4000]
            except Exception as exc:
                print(f"    DOCX extraction failed: {exc}")

        # XLSX
        elif ext in ("xlsx", "xls"):
            try:
                from openpyxl import load_workbook
                import io
                wb = load_workbook(io.BytesIO(content), data_only=True)
                text = ""
                for ws in wb.worksheets:
                    for row in ws.iter_rows(values_only=True):
                        row_text = " | ".join(str(c) for c in row if c is not None)
                        if row_text.strip():
                            text += row_text + "\n"
                if text.strip():
                    print(f"    Locally extracted {len(text)} chars from XLSX")
                    return text.strip()[:4000]
            except Exception as exc:
                print(f"    XLSX extraction failed: {exc}")

        # Plain text / CSV / JSON
        elif ext in ("txt", "csv", "json", "md"):
            try:
                text = content.decode("utf-8", errors="replace")
                if text.strip():
                    print(f"    Read {len(text)} chars from text file")
                    return text.strip()[:4000]
            except Exception:
                pass

        return ""
    except Exception as exc:
        print(f"    Local extraction error: {exc}")
        return ""

# ---------------------------------------------------------------------------
# Prompt Studio — Fetch active system prompt from AMS
# ---------------------------------------------------------------------------

# Cache the active prompt for 60 seconds to avoid hitting AMS on every LLM call
_cached_prompt: dict | None = None
_prompt_cache_time: float = 0
_PROMPT_CACHE_TTL = 60  # seconds

DEFAULT_SYSTEM_PROMPT = (
    "You are the Tender Agent — an AI assistant for SDS Manager, "
    "specializing in Safety Data Sheet (SDS) and Environmental Health "
    "& Safety (EHS) software tenders.\n\n"
    "CAPABILITIES:\n"
    "- You CAN access and read company documents that have been "
    "uploaded and assigned to you in the Nexus AMS Documents section.\n"
    "- You CAN search for government tenders globally.\n"
    "- You CAN fill out tender forms using company context.\n"
    "- When asked about company details, reference the documents below.\n"
    "- Be concise, specific, and actionable."
)


def fetch_active_prompt() -> str:
    """Fetch the active system prompt from AMS Prompt Studio.

    Falls back to DEFAULT_SYSTEM_PROMPT if no active prompt is configured
    or if the AMS is unreachable.
    """
    global _cached_prompt, _prompt_cache_time
    import httpx

    # Return cached prompt if still fresh
    if _cached_prompt and (time.time() - _prompt_cache_time) < _PROMPT_CACHE_TTL:
        return _cached_prompt.get("systemPrompt", DEFAULT_SYSTEM_PROMPT)

    try:
        agent_name = "tender-agent"
        resp = httpx.get(
            f"{NEXUS_URL}/api/agents/{agent_name}/prompt",
            timeout=5.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("hasActivePrompt"):
                _cached_prompt = data
                _prompt_cache_time = time.time()
                print(f"  [Prompt Studio] Using active prompt v{data.get('version')} — {data.get('name', '')}")
                return data["systemPrompt"]
    except Exception as exc:
        print(f"  [Prompt Studio] Failed to fetch active prompt: {exc}")

    # Fallback to default
    return DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# LLM Helper — calls Claude Sonnet via OpenRouter
# ---------------------------------------------------------------------------

def call_llm(prompt: str, max_tokens: int = 1024) -> dict:
    """Call Claude Sonnet via OpenRouter and return response with usage stats.

    Returns:
        {
            "content": "the LLM response text",
            "tokens_input": 123,
            "tokens_output": 45,
            "cost_usd": 0.0001,
            "model": "anthropic/claude-sonnet-4",
            "duration_ms": 1234,
        }
    """
    import httpx

    # Fetch system prompt from Prompt Studio (with fallback)
    system_prompt = fetch_active_prompt()

    start_time = time.perf_counter()

    response = httpx.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://sdsmanager.com",
            "X-Title": "SDS Tender Agent",
        },
        json={
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        system_prompt + "\n\n"
                        + search_knowledge_base(prompt)
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()

    duration_ms = int((time.perf_counter() - start_time) * 1000)

    content = data["choices"][0]["message"].get("content", "")
    usage = data.get("usage", {})
    tokens_in = usage.get("prompt_tokens", 0)
    tokens_out = usage.get("completion_tokens", 0)

    # Claude Sonnet via OpenRouter: $3.00/M input, $15.00/M output
    cost_usd = (tokens_in * 3.00 / 1_000_000) + (tokens_out * 15.00 / 1_000_000)

    # Audit every LLM call
    _audit(
        "llm_call",
        f"LLM call: {prompt[:80]}{'...' if len(prompt) > 80 else ''}",
        model_used=LLM_MODEL,
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        cost_usd=round(cost_usd, 6),
        duration_ms=duration_ms,
        input_payload={"prompt": prompt[:500], "max_tokens": max_tokens},
        output_payload={"response": content[:500], "full_length": len(content)},
    )

    return {
        "content": content,
        "tokens_input": tokens_in,
        "tokens_output": tokens_out,
        "cost_usd": round(cost_usd, 6),
        "model": LLM_MODEL,
        "duration_ms": duration_ms,
    }


# ---------------------------------------------------------------------------
# Tender Discovery — runs the SAM.gov scraper and formats results
# ---------------------------------------------------------------------------

# URL patterns that indicate an actual tender/solicitation/bid page
TENDER_URL_PATTERNS = [
    "/opp/", "/notice/", "/solicitation", "/bid/", "/rfp/",
    "/tender/", "/procurement/", "/contract-opportunity/",
    "/open-bids/", "/view", "/detail", "/opportunity/",
    "/search-rfp/", "/searchrfp/", "/public/notice",
    "/contract/", "/award/", "/announcement/",
]

# URL patterns that indicate NOT a tender (info pages, wikis, articles, news)
NOT_TENDER_PATTERNS = [
    "/legislation/", "/directives/", "/guidelines/", "/themes/",
    "/wiki/", "/blog/", "/article/", "/articles/", "/news/",
    "/about/", "/glossary/", "/eguides/", "/films/", "/tools/",
    "/resources/", "/faq/", "/help/", "/contact/", "/policy/",
    "/regulation/", "/publications/", "/research/", "/reports/",
    "/events/", "/magazine/", "/webinar/", "/training/", "/podcast/",
    "/case-study/", "/case-studies/", "/white-paper/", "/whitepapers/",
    "/product/", "/products/", "/pricing/", "/demo/", "/solutions/",
    "/press-release/", "/press/", "/media/", "/insights/",
    "/category/", "/tag/", "/author/",
]

# Domains to always exclude — social, news, SaaS marketing, and industry
# magazines that will never contain actual tender listings
EXCLUDE_DOMAINS = {
    # Social media
    "linkedin.com", "youtube.com", "facebook.com", "twitter.com",
    "reddit.com", "quora.com", "instagram.com", "tiktok.com",
    # Content platforms
    "medium.com", "wikipedia.org", "slideshare.net",
    # Industry news & magazines (articles ABOUT safety, not tenders)
    "ohsonline.com", "ehstoday.com", "ishn.com", "safetyandhealthmagazine.com",
    "chemicalwatch.com", "chemweek.com", "chemicalprocessing.com",
    # SaaS / marketing sites
    "heyiris.ai", "hubspot.com", "salesforce.com", "g2.com",
    "capterra.com", "gartner.com", "softwareadvice.com",
    # OSHA info pages (not procurement)
    "oshwiki.osha.europa.eu", "eguides.osha.europa.eu",
}

# Domains where ONLY specific paths are tenders
STRICT_PATH_DOMAINS = {
    "osha.europa.eu": ["/procurement"],
    "gov.uk": ["/contracts-finder", "/tender"],
}


def is_valid_tender_link(url: str) -> bool:
    """Check if a URL is likely a tender/procurement page.

    Uses a blocklist approach: block known-bad domains and non-tender URL
    patterns, but allow everything else through.  Quality control is handled
    downstream by relevance scoring and deadline classification.
    """
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        path = parsed.path.lower()

        # Block 1: Known non-tender sites (social media, wikis, etc.)
        for excluded in EXCLUDE_DOMAINS:
            if excluded in domain:
                return False

        # Block 2: Info/wiki/legislation pages on otherwise-OK domains
        for pattern in NOT_TENDER_PATTERNS:
            if pattern in path:
                return False

        # Block 3: Strict-path domains — only specific paths are tenders
        for strict_domain, allowed_paths in STRICT_PATH_DOMAINS.items():
            if strict_domain in domain:
                return any(ap in path for ap in allowed_paths)

        # Everything else is allowed — relevance scoring + deadline
        # classification handle quality control downstream
        return True
    except Exception:
        return False


def _strict_deadline_filter(leads: list, source_label: str) -> list:
    """Strict deadline gate — blocks ANY lead without a verified future deadline.

    Used on BOTH API results and SERP results. No exceptions.
    If we can't prove the deadline is in the future, the lead is blocked.

    Args:
        leads: List of lead objects (API or SERP).
        source_label: "API" or "SERP" — for logging only.

    Returns:
        Filtered list where every lead has a verified, future deadline.
    """
    from dateutil import parser as dateparser

    today = datetime.now(timezone.utc).date()
    valid = []
    blocked_expired = 0
    blocked_no_deadline = 0

    for lead in leads:
        deadline = getattr(lead, 'submission_deadline', '') or ''

        if not deadline or len(deadline) < 8:
            blocked_no_deadline += 1
            continue

        try:
            parsed_date = dateparser.parse(deadline).date()
            if parsed_date >= today:
                valid.append(lead)
            else:
                blocked_expired += 1
        except Exception:
            blocked_no_deadline += 1

    print(f"    [{source_label} deadline gate] {len(valid)} active / "
          f"{blocked_expired} expired / {blocked_no_deadline} no-deadline "
          f"(from {len(leads)} total)")

    return valid


def _filter_serp_leads(leads: list, query: str) -> list:
    """Run the strict filtering pipeline on SERP results.

    Every SERP result gets its actual page fetched to extract and verify
    the deadline. No lead passes without a confirmed future deadline.

    Gates:
      1. Block known-bad URLs (blocklist)
      2. Block wrong region
      3. Require strong SDS/EHS keyword match
      4. Reject empty portal pages
      5. Require tender signal (proof it's procurement, not news)
      6. Minimum relevance score
      7. Deadline from snippet (expired → block immediately)
      8. PAGE-FETCH every remaining lead: extract deadline from actual page.
         If expired/closed/no-deadline-found → BLOCKED. No exceptions.
    """
    from dateutil import parser as dateparser
    from urllib.parse import urlparse
    from src.discovery.serp_search import extract_deadline_from_text
    from src.discovery.deadline_verifier import verify_deadline

    today = datetime.now(timezone.utc).date()
    excluded_domains = get_excluded_domains_for_region(query)

    MIN_RELEVANCE = 0.30

    def classify_deadline_from_snippet(lead: object) -> str:
        """Quick check from snippet text only. Returns valid/expired/missing."""
        deadline = getattr(lead, 'submission_deadline', '') or ''
        if not deadline or len(deadline) < 8:
            desc = getattr(lead, 'description', '') or ''
            title = getattr(lead, 'title', '') or ''
            deadline = extract_deadline_from_text(f"{title} {desc}")
            if deadline:
                lead.submission_deadline = deadline
        if deadline and len(deadline) >= 8:
            try:
                parsed_date = dateparser.parse(deadline).date()
                return 'valid' if parsed_date >= today else 'expired'
            except Exception:
                pass
        return 'missing'

    def is_correct_region(lead: object) -> bool:
        if not excluded_domains:
            return True
        try:
            domain = urlparse(lead.source_url).netloc.lower().replace("www.", "")
            return not any(exc in domain for exc in excluded_domains)
        except Exception:
            return True

    valid = []
    needs_page_verify: list = []
    counts = {
        "blocked_url": 0, "blocked_region": 0, "no_strong_kw": 0,
        "empty_page": 0, "no_tender_signal": 0,
        "low_relevance": 0, "expired_snippet": 0,
        "page_active": 0, "page_expired": 0, "page_no_deadline": 0,
    }

    # Gates 1-7: quick filters
    for lead in leads:
        raw = getattr(lead, 'raw_data', {}) or {}

        if not is_valid_tender_link(lead.source_url):
            counts["blocked_url"] += 1
            continue
        if not is_correct_region(lead):
            counts["blocked_region"] += 1
            continue
        if not raw.get("has_strong_match", False):
            counts["no_strong_kw"] += 1
            continue
        if raw.get("is_empty_page", False):
            counts["empty_page"] += 1
            continue
        if not raw.get("has_tender_signal", False):
            counts["no_tender_signal"] += 1
            continue
        if lead.relevance_score < MIN_RELEVANCE:
            counts["low_relevance"] += 1
            continue

        snippet_status = classify_deadline_from_snippet(lead)
        if snippet_status == 'expired':
            counts["expired_snippet"] += 1
            continue

        if snippet_status == 'valid':
            # Snippet had a valid future deadline — still page-verify to be sure
            needs_page_verify.append(lead)
        else:
            # No deadline in snippet — must page-verify
            needs_page_verify.append(lead)

    # ------------------------------------------------------------------
    # Gate 8: Page-fetch EVERY lead that passed gates 1-7.
    # No lead gets through without a verified future deadline.
    # ------------------------------------------------------------------
    if needs_page_verify:
        print(f"    [Gate 8] Fetching {len(needs_page_verify)} pages to verify deadlines...")

    for lead in needs_page_verify:
        url = getattr(lead, 'source_url', '') or ''
        existing_deadline = getattr(lead, 'submission_deadline', '') or ''

        # If we already have a snippet-extracted future deadline, trust it
        # but still double-check via page fetch for safety
        if existing_deadline and len(existing_deadline) >= 8:
            try:
                parsed = dateparser.parse(existing_deadline).date()
                if parsed >= today:
                    lead._effective_score = lead.relevance_score
                    counts["page_active"] += 1
                    valid.append(lead)
                    continue
            except Exception:
                pass

        # Must fetch the page to find deadline
        if not url:
            counts["page_no_deadline"] += 1
            print(f"      BLOCKED (no URL): {lead.title[:60]}...")
            continue

        try:
            vr = verify_deadline(url, timeout=12.0, use_llm=True)
            vr_status = vr.get("status", "unknown")
            vr_deadline = vr.get("deadline", "")
            vr_source = vr.get("source", "none")

            if vr_status == "active" and vr_deadline:
                lead.submission_deadline = vr_deadline
                lead._effective_score = lead.relevance_score
                counts["page_active"] += 1
                print(f"      ACTIVE: {lead.title[:50]}... "
                      f"[deadline={vr_deadline}, via={vr_source}]")
                valid.append(lead)

            elif vr_status in ("expired", "closed"):
                counts["page_expired"] += 1
                print(f"      BLOCKED ({vr_status}): {lead.title[:50]}... "
                      f"[deadline={vr_deadline}, via={vr_source}]")

            else:
                # Could not find any deadline on the page — BLOCKED
                counts["page_no_deadline"] += 1
                print(f"      BLOCKED (no deadline found): {lead.title[:50]}...")

        except Exception as exc:
            # Page fetch failed — BLOCKED (not graceful degradation anymore)
            counts["page_no_deadline"] += 1
            print(f"      BLOCKED (fetch failed): {lead.title[:50]}... [{exc}]")

    valid.sort(key=lambda l: getattr(l, '_effective_score', l.relevance_score), reverse=True)

    print(f"    Pipeline: {len(valid)} passed out of {len(leads)}")
    print(f"    quick-filters: url={counts['blocked_url']} region={counts['blocked_region']} "
          f"kw={counts['no_strong_kw']} empty={counts['empty_page']} "
          f"signal={counts['no_tender_signal']} rel={counts['low_relevance']} "
          f"expired-snippet={counts['expired_snippet']}")
    print(f"    page-verify: active={counts['page_active']} "
          f"expired={counts['page_expired']} "
          f"no-deadline={counts['page_no_deadline']}")

    return valid


def _is_deep_search(query: str) -> bool:
    """Detect if the user wants a deep/extended search (triggers SERP).

    Normal queries hit direct APIs only. Deep search adds Google SERP
    for broader coverage of smaller portals that don't have APIs.
    """
    deep_triggers = [
        "deep search", "deep-search", "extended search", "broad search",
        "search everywhere", "all sources", "include google",
        "serp", "google search", "wider search",
    ]
    q_lower = query.lower()
    return any(trigger in q_lower for trigger in deep_triggers)


def _strip_deep_trigger(query: str) -> str:
    """Remove the deep search trigger phrase from the query."""
    import re
    triggers = [
        r"deep[\s-]?search\s*", r"extended\s+search\s*", r"broad\s+search\s*",
        r"search\s+everywhere\s*", r"all\s+sources\s*", r"include\s+google\s*",
        r"serp\s*", r"google\s+search\s*", r"wider\s+search\s*",
    ]
    cleaned = query
    for trigger in triggers:
        cleaned = re.sub(trigger, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Tender Search — filter-driven, runs APIs first then SERP fallback
# ---------------------------------------------------------------------------

# Map source_portal (used by Lead objects) to the canonical source name we
# return in DiscoveredTender rows. Stable across both the bridge and the AMS.
_PORTAL_TO_SOURCE: dict[str, str] = {
    "ted.europa.eu":  "ted_europa",
    "uk_gov":         "uk_tenders",
    "sam.gov":        "sam_gov",
    "boamp":          "boamp_france",
    "world_bank":     "world_bank",
    "prozorro":       "prozorro",
    "canada_buys":    "canada_buys",
    "austender":      "austender",
    "sa_etender":     "sa_etender",
    "colombia_secop": "colombia_secop",
    "brazil_compras": "brazil_compras",
    "germany_bkms":   "germany_bkms",
    "italy_anac":     "italy_anac",
    "dominican_dgcp": "dominican_dgcp",
    "peru_oece":      "peru_oece",
    "world_bank_v2":  "world_bank_v2",
    "nigeria_nocopo": "nigeria_nocopo",
    "kenya_ppra":     "kenya_ppra",
    "uganda_gpp":     "uganda_gpp",
    "mexico_cdmx":    "mexico_cdmx",
    "google_serp":    "serp_fallback",
}

# Portal → canonical region (used to set DiscoveredTender.region when the
# Lead object doesn't carry it directly).
_PORTAL_TO_REGION: dict[str, str] = {
    "ted.europa.eu":  "europe",
    "uk_gov":         "uk",
    "sam.gov":        "usa",
    "boamp":          "europe",
    "world_bank":     "global",
    "prozorro":       "europe",
    "canada_buys":    "canada",
    "austender":      "australia",
    "sa_etender":     "africa",
    "colombia_secop": "south_america",
    "brazil_compras": "south_america",
    "germany_bkms":   "europe",
    "italy_anac":     "europe",
    "dominican_dgcp": "south_america",
    "peru_oece":      "south_america",
    "world_bank_v2":  "global",
    "nigeria_nocopo": "africa",
    "kenya_ppra":     "africa",
    "uganda_gpp":     "africa",
    "mexico_cdmx":    "south_america",
    "google_serp":    None,
}

# Absolute floor — wins even when the user-supplied minRelevance is lower.
ABSOLUTE_RELEVANCE_FLOOR = 0.30


def _parse_iso_date(value):
    """Parse a string like '2026-05-19' or '2026-05-19T00:00:00Z' to a date.

    Returns None on any failure or if value is falsy.
    """
    if not value:
        return None
    try:
        from dateutil import parser as dateparser
        return dateparser.parse(str(value)).date()
    except Exception:
        return None


def _iso_or_none(value):
    """Coerce a deadline/date field to ISO-8601 string with time, or None."""
    if not value:
        return None
    s = str(value).strip()
    if not s or len(s) < 8:
        return None
    try:
        from dateutil import parser as dateparser
        dt = dateparser.parse(s)
        # Anchor naive timestamps to UTC end-of-day so the AMS DateTime column
        # stays consistent across timezones.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return None


def _lead_to_tender(lead, search_query: str) -> dict:
    """Translate a Lead object into the DiscoveredTender payload shape.

    AMS expects a flat dict per the spec. fingerprint = sha256(source + ":" + sourceId)
    which deduplicates the same tender across overlapping queries / regions.
    """
    import hashlib

    portal = getattr(lead, "source_portal", "") or ""
    source = _PORTAL_TO_SOURCE.get(portal, portal or "unknown")
    source_id = (
        getattr(lead, "lead_id", None)
        or getattr(lead, "source_id", None)
        or getattr(lead, "ocid", None)
        or getattr(lead, "tender_id", None)
        or getattr(lead, "source_url", "")
    )
    fp = hashlib.sha256(f"{source}:{source_id}".encode("utf-8")).hexdigest()

    value_amount = getattr(lead, "value_amount", None) or 0
    currency = getattr(lead, "value_currency", None) or None
    value_raw = None
    if value_amount and currency:
        value_raw = f"{currency} {value_amount:,.0f}"
    elif value_amount:
        value_raw = f"{value_amount:,.0f}"

    raw_data = getattr(lead, "raw_data", None)
    if not isinstance(raw_data, dict):
        raw_data = {}
    cpv = raw_data.get("cpv_code") or getattr(lead, "cpv_code", None)
    country = raw_data.get("country") or getattr(lead, "country", None)

    submission_iso = _iso_or_none(getattr(lead, "submission_deadline", "") or "")
    # The strict + safety-net filters guarantee a deadline exists, but be defensive.
    if not submission_iso:
        return None

    return {
        "fingerprint": fp,
        "source": source,
        "sourceId": str(source_id)[:500],
        "title": (getattr(lead, "title", "") or "")[:1000],
        "description": (getattr(lead, "description", "") or None),
        "agency": (getattr(lead, "agency", "") or None),
        "country": (str(country)[:10] if country else None),
        "region": _PORTAL_TO_REGION.get(portal),
        "category": None,
        "submissionDeadline": submission_iso,
        "postedDate": _iso_or_none(getattr(lead, "posted_date", "") or ""),
        "contractValueUsd": (float(value_amount) if value_amount else None),
        "contractValueRaw": value_raw,
        "currency": currency,
        "url": (getattr(lead, "source_url", "") or "")[:2000],
        "cpvCode": (str(cpv)[:50] if cpv else None),
        "relevanceScore": float(getattr(lead, "relevance_score", 0) or 0),
        "rawPayload": {
            "search_query": search_query,
            "source_portal": portal,
            "raw_data": raw_data,
            "value_amount": value_amount,
            "value_currency": currency,
        },
    }


def run_tender_search(filters: dict) -> dict:
    """Run filter-driven tender discovery — APIs first, optional SERP fallback.

    Architecture preserved from Session 3:
      STEP 1: Run all 20 government APIs (region-routed via filters["regions"],
              source-gated via filters["sources"]).
      STEP 2: Strict deadline filter — block any API result without a verified
              future deadline. Zero exceptions.
      STEP 2b: Relevance floor — max(filters["minRelevance"], 0.30). Absolute
              30% floor always wins.
      STEP 3: If API results exist after filtering, return them. SERP skipped.
      STEP 4: Otherwise, if filters["includeSerpFallback"] is True, run SERP
              with page-fetch deadline verification on every lead.
      STEP 5: Final safety net at output time — re-verify every deadline.
      STEP 6: Apply post-filters from the form: deadline window, contract
              value range, posted-within-days.

    Returns a structured dict the AMS persists to DiscoveredTender:
        {
          "tenders": [ { fingerprint, source, sourceId, ... } ],
          "stats":   { totalFound, blockedExpired, blockedRelevance, deduplicated }
        }
    """
    from src.discovery.serp_search import detect_region

    start_time = time.perf_counter()

    search_query = (filters.get("keywords") or "").strip()
    regions_filter = set(filters.get("regions") or [])
    if not regions_filter:
        regions_filter = {"global"}
    # If user selects global, fire all region-gated APIs.
    if "global" in regions_filter:
        regions_filter = regions_filter | {
            "usa", "europe", "uk", "canada", "australia",
            "india", "africa", "south_america",
        }
    else:
        # Specific region(s) selected — still include the truly-global APIs
        # (World Bank v1 + v2) so the user gets cross-jurisdictional coverage,
        # not just one or two region-specific endpoints. Without this, picking
        # e.g. "USA" only fires SAM.gov + CanadaBuys; if SAM is rate-limited
        # or returns nothing, the user sees an empty inbox.
        regions_filter = regions_filter | {"global"}

    sources_filter = set(filters.get("sources") or [])
    all_sources_allowed = not sources_filter

    def region_match(*allowed: str) -> bool:
        return any(r in regions_filter for r in allowed)

    def src_on(name: str) -> bool:
        return all_sources_allowed or name in sources_filter

    include_serp = bool(filters.get("includeSerpFallback", True))
    user_min_relevance = float(filters.get("minRelevance") or ABSOLUTE_RELEVANCE_FLOOR)
    relevance_floor = max(user_min_relevance, ABSOLUTE_RELEVANCE_FLOOR)

    print(f"\n{'='*60}")
    print(f"  Tender search: '{search_query[:80]}'")
    print(f"  Regions: {sorted(regions_filter)}")
    print(f"  Sources: {'ALL' if all_sources_allowed else sorted(sources_filter)}")
    print(f"  Min relevance: {relevance_floor:.0%} | SERP fallback: {include_serp}")
    print(f"{'='*60}")

    api_leads: list = []

    # ------------------------------------------------------------------
    # Step 1: Direct API sources, fanned out in PARALLEL.
    # Each tuple = (source_name, region_gate, source_gate, callable).
    # We build the list first so we can submit only the ones that match
    # region/source filters, then call them concurrently via a thread pool.
    # Wall time goes from ~60s serial to ~10-15s parallel since these are
    # all I/O bound.
    # ------------------------------------------------------------------
    sam_api_key = os.getenv("SAM_GOV_API_KEY", "")

    def _ted():
        return TedEuropaSearcher().search(user_query=search_query, max_results=15)
    def _uk():
        return UkTenderSearcher().search(user_query=search_query, max_results=15, days_back=60)
    def _sam():
        return SamGovScraper(dry_run=False).fetch_opportunities(days_back=30)
    def _boamp():
        return BoampSearcher().search(user_query=search_query, max_results=10, days_back=60)
    def _wb():
        return WorldBankSearcher().search(user_query=search_query, max_results=10, days_back=90)
    def _prozorro():
        return ProzorroSearcher().search(user_query=search_query, max_results=10, days_back=60)
    def _canada():
        return CanadaBuysSearcher().search(user_query=search_query, max_results=10, days_back=60)
    def _austender():
        return AusTenderSearcher().search(user_query=search_query, max_results=10, days_back=60)
    def _sa():
        return SaEtenderSearcher().search(user_query=search_query, max_results=10, days_back=60)
    def _colombia():
        return ColombiaSecopSearcher().search(user_query=search_query, max_results=10, days_back=60)
    def _brazil():
        return BrazilComprasSearcher().search(user_query=search_query, max_results=10, days_back=60)
    def _germany():
        return GermanyBkmsSearcher().search(user_query=search_query, max_results=10, days_back=60)
    def _italy():
        return ItalyAnacSearcher().search(user_query=search_query, max_results=10, days_back=60)
    def _dr():
        return DominicanDgcpSearcher().search(user_query=search_query, max_results=10, days_back=60)
    def _peru():
        return PeruOeceSearcher().search(user_query=search_query, max_results=10, days_back=90)
    def _wb2():
        return WorldBankV2Searcher().search(user_query=search_query, max_results=10)
    def _nigeria():
        return NigeriaNocopoSearcher().search(user_query=search_query, max_results=10, days_back=60)
    def _kenya():
        return KenyaPpraSearcher().search(user_query=search_query, max_results=10, days_back=60)
    def _uganda():
        return UgandaGppSearcher().search(user_query=search_query, max_results=10, days_back=60)
    def _mexico():
        return MexicoCdmxSearcher().search(user_query=search_query, max_results=10, days_back=60)

    # Sources whose upstream JSON endpoints have been retired or changed
    # without backward compat (verified May 2026). Skipping them avoids
    # 20-30s of wasted retries per search. Re-enable any of these when
    # someone finds the new endpoint URL and patches the corresponding
    # discovery module.
    #
    # Investigation notes per source — last verified May 2026:
    #   germany_bkms:   oeffentlichevergabe.de /api/* paths all 404. The
    #                   BMI's published OCDS migration ("since Dec 16,
    #                   2025") doesn't yet have a discoverable public
    #                   endpoint. Recheck BMI dev portal in a few months.
    #   mexico_cdmx:    tianguisdigital.cdmx.gob.mx /api/* paths all 404
    #                   and the public web UI doesn't expose a JSON feed.
    #                   CDMX migrated to a new portal in 2025 with no
    #                   API yet.
    #   italy_anac:     dati.anticorruzione.it/opendata/* returns 269-byte
    #                   HTML stubs to bots (WAF-protected). The OCDS
    #                   dataset is downloadable as bulk JSON/CSV but no
    #                   query API; not viable for live polling.
    #   peru_oece:      contratacionesabiertas.oece.gob.pe returns 403 on
    #                   every path (looks IP-restricted). osce.gob.pe
    #                   returns HTML login pages.
    #   uganda_gpp:     gpp.ppda.go.ug returns HTML on every /api/* path;
    #                   the OCDS feed appears to have been retired.
    #   dominican_dgcp: api.dgcp.gob.do is documented but unreachable
    #                   (connection refused / DNS unstable as of probe).
    #
    # REVIVED (May 2026) and removed from this list:
    #   brazil_compras: migrated from dadosabertos.compras.gov.br to the
    #                   new PNCP portal at pncp.gov.br — module updated.
    #   sa_etender:     ocds-api.etenders.gov.za still works, the JSON
    #                   path is /api/OCDSReleases (Swagger-confirmed) —
    #                   module updated.
    KNOWN_BROKEN_APIS = {
        "germany_bkms":   "404 on every known endpoint path",
        "mexico_cdmx":    "tianguisdigital migrated; new feed unknown",
        "italy_anac":     "dati.anticorruzione.it WAF returns HTML stubs",
        "peru_oece":      "403 on every path (IP-restricted?)",
        "uganda_gpp":     "OCDS endpoint returns HTML",
        "dominican_dgcp": "api.dgcp.gob.do unreachable",
    }

    api_jobs = []
    def _add(name: str, fn):
        if name in KNOWN_BROKEN_APIS:
            print(f"  [{name}] skipped — {KNOWN_BROKEN_APIS[name]}")
            return
        api_jobs.append((name, fn))

    if region_match("europe", "uk", "global") and src_on("ted_europa"):
        _add("ted_europa", _ted)
    if region_match("uk", "europe", "global") and src_on("uk_tenders"):
        _add("uk_tenders", _uk)
    if region_match("usa", "global") and src_on("sam_gov") and sam_api_key:
        _add("sam_gov", _sam)
    elif region_match("usa", "global") and src_on("sam_gov") and not sam_api_key:
        print("  [SAM API] Skipped — SAM_GOV_API_KEY not set")
    if region_match("europe", "global") and src_on("boamp_france"):
        _add("boamp_france", _boamp)
    if region_match("global", "india", "australia") and src_on("world_bank"):
        _add("world_bank", _wb)
    if region_match("europe", "global") and src_on("prozorro"):
        _add("prozorro", _prozorro)
    if region_match("canada", "usa", "global") and src_on("canada_buys"):
        _add("canada_buys", _canada)
    if region_match("australia", "global") and src_on("austender"):
        _add("austender", _austender)
    if region_match("africa", "global") and src_on("sa_etender"):
        _add("sa_etender", _sa)
    if region_match("south_america", "global") and src_on("colombia_secop"):
        _add("colombia_secop", _colombia)
    if region_match("south_america", "global") and src_on("brazil_compras"):
        _add("brazil_compras", _brazil)
    if region_match("europe", "global") and src_on("germany_bkms"):
        _add("germany_bkms", _germany)
    if region_match("europe", "global") and src_on("italy_anac"):
        _add("italy_anac", _italy)
    if region_match("south_america", "global") and src_on("dominican_dgcp"):
        _add("dominican_dgcp", _dr)
    if region_match("south_america", "global") and src_on("peru_oece"):
        _add("peru_oece", _peru)
    if region_match("global", "india", "africa", "australia") and src_on("world_bank_v2"):
        _add("world_bank_v2", _wb2)
    if region_match("africa", "global") and src_on("nigeria_nocopo"):
        _add("nigeria_nocopo", _nigeria)
    if region_match("africa", "global") and src_on("kenya_ppra"):
        _add("kenya_ppra", _kenya)
    if region_match("africa", "global") and src_on("uganda_gpp"):
        _add("uganda_gpp", _uganda)
    if region_match("south_america", "global") and src_on("mexico_cdmx"):
        _add("mexico_cdmx", _mexico)

    print(f"  Fanning out to {len(api_jobs)} live APIs in parallel "
          f"({len(KNOWN_BROKEN_APIS)} known-broken skipped)…")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=10) as pool:
        future_to_name = {pool.submit(fn): name for name, fn in api_jobs}
        for future in as_completed(future_to_name, timeout=90):
            name = future_to_name[future]
            try:
                leads = future.result() or []
                api_leads.extend(leads)
                print(f"  [{name}] Got {len(leads)} results")
                _audit("tool_call", f"{name} returned {len(leads)} leads",
                       node_name="discover", status="success",
                       output_payload={"source": name, "count": len(leads)})
            except Exception as exc:
                print(f"  [{name}] Failed: {exc}")
                _audit("tool_call", f"{name} failed: {exc}",
                       node_name="discover", status="failure",
                       error_message=str(exc))


    api_raw_count = len(api_leads)
    print(f"  API total: {api_raw_count} raw leads from direct government APIs")

    # ------------------------------------------------------------------
    # Step 2: Strict deadline filter (Layer 1 of three).
    # ------------------------------------------------------------------
    api_verified = _strict_deadline_filter(api_leads, "API")
    blocked_expired = api_raw_count - len(api_verified)

    # Dedup by URL within the API set.
    seen_urls: set[str] = set()
    valid_leads: list = []
    duplicates_in_api = 0
    for lead in api_verified:
        url = getattr(lead, "source_url", "")
        if url and url in seen_urls:
            duplicates_in_api += 1
            continue
        if url:
            seen_urls.add(url)
        if not hasattr(lead, "_effective_score"):
            lead._effective_score = getattr(lead, "relevance_score", 0.80)
        valid_leads.append(lead)
    print(f"  API after deadline filter + dedup: {len(valid_leads)} verified leads")

    # ------------------------------------------------------------------
    # Step 2a (NEW): re-score every lead against the USER's actual keywords,
    # weighted by term specificity.
    #
    # The previous version treated every user term equally (+0.18 per match).
    # That meant "Student Management System" got the same boost as "Hazardous
    # Materials Management" — both have one term match ("management"). Bad.
    #
    # New scheme:
    #   STRONG (domain-specific):   +0.22 per match
    #   MEDIUM (regulation-ish):    +0.12 per match
    #   WEAK   (generic admin):     +0.03 per match
    #   default (unknown):          +0.10 per match
    #   bigram phrase bonus:        +0.20 once if any two consecutive user
    #                               terms appear together in the text
    #
    # Plus a guard: WEAK terms alone never push a lead over the 30% floor —
    # if only weak terms match, the boost caps below what's needed to clear.
    # ------------------------------------------------------------------
    STRONG_TERMS = {
        "chemical", "chemicals", "hazardous", "hazmat", "toxic", "carcinogen",
        "sds", "msds", "ghs", "ehs", "coshh", "reach", "clp",
        "biohazard", "radioactive", "asbestos", "lead-paint", "pesticide",
        "occupational",
    }
    MEDIUM_TERMS = {
        "safety", "compliance", "environmental", "materials", "substances",
        "substance", "regulation", "regulations", "classification",
        "labelling", "labeling", "waste", "spill", "disposal", "hazard",
        "hazards", "exposure", "remediation", "decontamination",
    }
    WEAK_TERMS = {
        "management", "services", "service", "system", "systems", "software",
        "platform", "contracts", "contract", "procurement", "tender",
        "tenders", "rfp", "training", "consulting", "support", "supply",
        "supplies", "agreement", "project", "program", "programme",
    }
    STOP_TERMS = {
        "with", "from", "into", "that", "this", "these", "those",
        "about", "what", "when", "where", "which", "while", "since",
        "global", "local", "regional", "national", "international",
        "year", "month", "week", "day",
    }

    _user_terms = [
        t.lower().strip(".,;:!?()[]")
        for t in (search_query or "").split()
        if len(t) >= 3 and t.lower().strip(".,;:!?()[]") not in STOP_TERMS
    ]
    _user_terms = [t for t in _user_terms if t]  # drop empties after strip

    def _term_weight(term: str) -> float:
        if term in STRONG_TERMS:
            return 0.22
        if term in MEDIUM_TERMS:
            return 0.12
        if term in WEAK_TERMS:
            return 0.03
        # Unknown words default to medium — assumed user-specific intent
        # (acronyms, product names, place names like "canada", "germany").
        return 0.10

    if _user_terms:
        rescored = 0
        for lead in valid_leads:
            text = (
                (getattr(lead, "title", "") or "") + " " +
                (getattr(lead, "description", "") or "")
            ).lower()

            # Per-term contributions
            boost = 0.0
            matched_terms: list[str] = []
            for t in _user_terms:
                if t in text:
                    boost += _term_weight(t)
                    matched_terms.append(t)

            # Bigram phrase bonus: any pair of consecutive user terms together
            # in the text indicates a much stronger match than disjoint terms.
            if len(_user_terms) >= 2:
                for i in range(len(_user_terms) - 1):
                    phrase = f"{_user_terms[i]} {_user_terms[i + 1]}"
                    if phrase in text:
                        boost += 0.20
                        break  # one bonus is enough — don't stack

            if boost > 0:
                boost = min(0.60, boost)
                current = float(getattr(lead, "relevance_score", 0) or 0)
                new_score = min(1.0, current + boost)
                lead.relevance_score = new_score
                if hasattr(lead, "_effective_score"):
                    lead._effective_score = new_score
                rescored += 1

        print(f"  [User-keyword rescore] Boosted {rescored}/{len(valid_leads)} "
              f"leads | weighted terms: strong={sorted(set(_user_terms) & STRONG_TERMS)} "
              f"medium={sorted(set(_user_terms) & MEDIUM_TERMS)} "
              f"weak={sorted(set(_user_terms) & WEAK_TERMS)}")

    # ------------------------------------------------------------------
    # Step 2b: Relevance floor — user's pick OR 30% absolute, whichever larger.
    # ------------------------------------------------------------------
    before_relevance = len(valid_leads)
    valid_leads = [
        l for l in valid_leads
        if getattr(l, "relevance_score", 0) >= relevance_floor
    ]
    blocked_relevance = before_relevance - len(valid_leads)
    if blocked_relevance > 0:
        print(f"  [Relevance gate] Blocked {blocked_relevance} leads below {relevance_floor:.0%} "
              f"(kept {len(valid_leads)} relevant)")
    else:
        print(f"  [Relevance gate] All {len(valid_leads)} leads above {relevance_floor:.0%} floor")

    # ------------------------------------------------------------------
    # Step 3: SERP automatic fallback — only when APIs returned 0 AND
    # the user opted in via filters["includeSerpFallback"].
    # ------------------------------------------------------------------
    serp_used = False
    if len(valid_leads) == 0 and include_serp:
        serp_used = True
        print(f"\n  [SERP FALLBACK] APIs returned 0 results — falling back to Google SERP...")
        _audit("tool_call",
               f"API sources returned 0 results, falling back to SERP for: {search_query[:100]}",
               node_name="discover",
               input_payload={"query": search_query, "tool": "brightdata_serp", "mode": "auto_fallback"})
        try:
            serp_leads = SerpTenderSearcher().search(user_query=search_query)
            print(f"  [SERP] Got {len(serp_leads)} raw leads — running strict pipeline...")
            serp_valid = _filter_serp_leads(serp_leads, search_query)
            print(f"  [SERP] {len(serp_valid)} passed (all have verified deadlines)")
            # Apply the same weighted rescore to SERP results.
            if _user_terms:
                for lead in serp_valid:
                    text = (
                        (getattr(lead, "title", "") or "") + " " +
                        (getattr(lead, "description", "") or "")
                    ).lower()
                    boost = 0.0
                    for t in _user_terms:
                        if t in text:
                            boost += _term_weight(t)
                    if len(_user_terms) >= 2:
                        for i in range(len(_user_terms) - 1):
                            phrase = f"{_user_terms[i]} {_user_terms[i + 1]}"
                            if phrase in text:
                                boost += 0.20
                                break
                    if boost > 0:
                        boost = min(0.60, boost)
                        current = float(getattr(lead, "relevance_score", 0) or 0)
                        new_score = min(1.0, current + boost)
                        lead.relevance_score = new_score
                        if hasattr(lead, "_effective_score"):
                            lead._effective_score = new_score
            for lead in serp_valid:
                if getattr(lead, "relevance_score", 0) < relevance_floor:
                    blocked_relevance += 1
                    continue
                url = getattr(lead, "source_url", "")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                valid_leads.append(lead)
        except Exception as exc:
            print(f"  [SERP] Failed: {exc}")
            _audit("tool_call", f"SERP fallback failed: {exc}",
                   node_name="discover", status="failure", error_message=str(exc))
    elif len(valid_leads) == 0:
        print("  [SERP] Skipped — APIs returned 0 results but includeSerpFallback=False")
    else:
        print("  [SERP] Skipped — APIs returned results, no fallback needed")

    valid_leads.sort(
        key=lambda l: getattr(l, "_effective_score", getattr(l, "relevance_score", 0)),
        reverse=True,
    )

    # ------------------------------------------------------------------
    # Step 4: FINAL SAFETY NET — re-verify every deadline right before we
    # serialize the output. If anything slipped past earlier filters, this
    # catches it. Zero expired tenders ever reach the AMS.
    # ------------------------------------------------------------------
    from dateutil import parser as dateparser
    today_final = datetime.now(timezone.utc).date()
    safety_net_leads: list = []
    safety_net_blocked = 0
    for lead in valid_leads:
        dl = getattr(lead, "submission_deadline", "") or ""
        if not dl or len(dl) < 8:
            safety_net_blocked += 1
            continue
        try:
            parsed = dateparser.parse(dl).date()
            if parsed < today_final:
                safety_net_blocked += 1
                continue
        except Exception:
            safety_net_blocked += 1
            continue
        safety_net_leads.append(lead)
    if safety_net_blocked > 0:
        print(f"  [SAFETY NET] Removed {safety_net_blocked} leads at final check")
        blocked_expired += safety_net_blocked
    valid_leads = safety_net_leads

    # ------------------------------------------------------------------
    # Step 5: Post-filters from the form — deadline window, value range,
    # postedWithinDays. Applied AFTER all gates.
    # ------------------------------------------------------------------
    deadline_from = _parse_iso_date(filters.get("deadlineFrom"))
    deadline_to = _parse_iso_date(filters.get("deadlineTo"))
    min_value = filters.get("minValueUsd")
    max_value = filters.get("maxValueUsd")
    posted_within_days = filters.get("postedWithinDays")
    today_d = datetime.now(timezone.utc).date()

    post_filter_blocked = 0
    after_post: list = []
    for lead in valid_leads:
        dl = _parse_iso_date(getattr(lead, "submission_deadline", ""))
        if deadline_from and dl and dl < deadline_from:
            post_filter_blocked += 1
            continue
        if deadline_to and dl and dl > deadline_to:
            post_filter_blocked += 1
            continue
        val = getattr(lead, "value_amount", None) or 0
        if (min_value is not None or max_value is not None) and not val:
            # Spec: skip leads without a contract value when a value filter is set.
            post_filter_blocked += 1
            continue
        if min_value is not None and val and float(val) < float(min_value):
            post_filter_blocked += 1
            continue
        if max_value is not None and val and float(val) > float(max_value):
            post_filter_blocked += 1
            continue
        if posted_within_days and str(posted_within_days).lower() != "all":
            try:
                window = int(posted_within_days)
                posted = _parse_iso_date(getattr(lead, "posted_date", ""))
                if posted:
                    delta_days = (today_d - posted).days
                    if delta_days > window:
                        post_filter_blocked += 1
                        continue
            except (ValueError, TypeError):
                pass
        after_post.append(lead)
    if post_filter_blocked > 0:
        print(f"  [POST FILTER] Dropped {post_filter_blocked} leads outside form bounds")
    valid_leads = after_post

    # ------------------------------------------------------------------
    # Step 6: Serialize to DiscoveredTender payloads. The HTTP push step
    # in main_loop() handles fingerprint-based dedup against the DB.
    # ------------------------------------------------------------------
    tenders: list[dict] = []
    seen_fingerprints: set[str] = set()
    in_batch_dupes = 0
    for lead in valid_leads:
        td = _lead_to_tender(lead, search_query)
        if td is None:
            continue
        fp = td["fingerprint"]
        if fp in seen_fingerprints:
            in_batch_dupes += 1
            continue
        seen_fingerprints.add(fp)
        tenders.append(td)

    duration_ms = int((time.perf_counter() - start_time) * 1000)
    print(f"  FINAL: {len(tenders)} tenders | "
          f"blockedExpired={blocked_expired} blockedRelevance={blocked_relevance} "
          f"deduped={duplicates_in_api + in_batch_dupes} | {duration_ms}ms")

    return {
        "tenders": tenders,
        "stats": {
            "totalFound": len(tenders),
            "blockedExpired": blocked_expired,
            "blockedRelevance": blocked_relevance,
            "deduplicated": duplicates_in_api + in_batch_dupes,
            "serpFallbackUsed": serp_used,
            "durationMs": duration_ms,
        },
    }


# ---------------------------------------------------------------------------
# Form Filling — triggered by fill_form AgentTask (no chat)
# ---------------------------------------------------------------------------

def handle_fill_form_task(client: NexusClient, task: dict) -> None:
    """Process a fill_form task created by a TenderPursuit Agent fill mode.

    The task carries (via metadata):
      - pursuitId: the TenderPursuit row to update on completion
      - formFileKey: MinIO key of the uploaded tender form
      - formFilename: original filename for the user-facing label
      - tenderTitle / tenderUrl: surface context for the Approval card

    Single-shot fill (no clarification round trips since chat is removed).
    Low-confidence fields are flagged in the submission metadata for human
    review; the operator approves or rejects in /approvals.
    """
    start_time = time.perf_counter()
    metadata = task.get("metadata") or {}
    pursuit_id = metadata.get("pursuitId")
    form_file_key = metadata.get("formFileKey")
    form_filename = metadata.get("formFilename") or "tender_form"
    tender_title = metadata.get("tenderTitle") or "Tender"
    task_id = task.get("id")

    if not form_file_key:
        msg = "fill_form task is missing metadata.formFileKey"
        print(f"  [fill_form] ERROR: {msg}")
        _audit("task_failed", msg, status="failure",
               input_payload={"task_id": task_id}, error_message=msg)
        complete_agent_task(client, task_id, status="failed",
                            result_summary=msg, duration_ms=0)
        return

    tmp_dir = tempfile.mkdtemp(prefix="tender_form_")
    local_path = os.path.join(tmp_dir, form_filename)

    try:
        print(f"  [fill_form] Downloading {form_file_key}")
        client.download_file(form_file_key, local_path)

        parse_result = FormParser().parse(local_path)
        print(f"  [fill_form] Parsed {len(parse_result.fields)} fields "
              f"({parse_result.file_format.value})")
        _audit("form_parsed",
               f"Parsed {form_filename}: {len(parse_result.fields)} fields",
               node_name="fill_form",
               input_payload={"filename": form_filename, "task_id": task_id},
               output_payload={
                   "fields_count": len(parse_result.fields),
                   "field_names": [f.name for f in parse_result.fields[:20]],
                   "parse_errors": parse_result.parse_errors,
               })

        if not parse_result.fields and not parse_result.raw_text:
            msg = f"Could not extract any fields from {form_filename}"
            complete_agent_task(client, task_id, status="failed",
                                result_summary=msg, duration_ms=0)
            mark_pursuit_status(client, pursuit_id, "backlog",
                                notes=f"Auto-revert: {msg}")
            return

        company_context = fetch_agent_context()
        if not company_context:
            msg = ("No company documents assigned to the tender-agent. "
                   "Upload context docs in Documents → assign to Tender Agent.")
            complete_agent_task(client, task_id, status="failed",
                                result_summary=msg, duration_ms=0)
            mark_pursuit_status(client, pursuit_id, "backlog",
                                notes=f"Auto-revert: {msg}")
            return

        filler = FormFiller(llm_call_fn=call_llm, confidence_threshold=0.7)
        fill_result = filler.fill(parse_result, company_context)
        print(f"  [fill_form] Filled {fill_result.filled_count}/{fill_result.total_fields}, "
              f"{fill_result.needs_clarification_count} low-confidence flagged")
        _audit("form_filled",
               f"Filled {fill_result.filled_count}/{fill_result.total_fields} fields, "
               f"{fill_result.needs_clarification_count} need review",
               node_name="fill_form",
               cost_usd=fill_result.llm_cost_usd,
               output_payload={
                   "filled_count": fill_result.filled_count,
                   "total_fields": fill_result.total_fields,
                   "low_confidence_count": fill_result.needs_clarification_count,
                   "filled_field_names": [f.name for f in fill_result.filled_fields],
               })

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        _submit_filled_form(
            client,
            task_id=task_id,
            pursuit_id=pursuit_id,
            tmp_dir=tmp_dir,
            local_path=local_path,
            original_filename=form_filename,
            tender_title=tender_title,
            fill_result=fill_result,
            duration_ms=duration_ms,
        )

        client.track("task_completion", 1.0, metadata={
            "node": "fill_form",
            "fields_filled": fill_result.filled_count,
            "fields_total": fill_result.total_fields,
            "pursuit_id": pursuit_id,
        })
        client.track("latency", float(duration_ms), metadata={"node": "fill_form"})
        if fill_result.llm_cost_usd > 0:
            client.track("cost", fill_result.llm_cost_usd,
                         metadata={"node": "fill_form", "model": LLM_MODEL})

    except Exception as exc:
        traceback.print_exc()
        msg = f"Error processing {form_filename}: {exc}"
        print(f"  [fill_form] ERROR: {msg}")
        _audit("task_failed", msg, node_name="fill_form",
               status="failure", error_message=str(exc),
               input_payload={"task_id": task_id, "filename": form_filename})
        complete_agent_task(client, task_id, status="failed",
                            result_summary=msg[:500], duration_ms=0)
        mark_pursuit_status(client, pursuit_id, "backlog",
                            notes=f"Auto-revert: {msg[:200]}")
        notify_task_failed(
            agent_name="tender-agent",
            task_title=f"Form fill failed: {form_filename}",
            error_message=str(exc)[:300],
        )
        client.track("error_rate", 1.0, metadata={
            "node": "fill_form", "error_type": type(exc).__name__,
        })

    try:
        client.flush()
    except Exception as exc:
        print(f"  [fill_form] Metric flush failed: {exc}")


def _submit_filled_form(
    client: NexusClient,
    task_id: str,
    pursuit_id,
    tmp_dir: str,
    local_path: str,
    original_filename: str,
    tender_title: str,
    fill_result: FillResult,
    duration_ms: int,
) -> None:
    """Write the filled form to disk, submit it for approval, advance the pursuit.

    No chat reply — the operator picks up the result in /approvals.
    """
    writer = FormWriter()
    stem = Path(original_filename).stem
    ext = Path(original_filename).suffix
    output_filename = f"{stem}_FILLED{ext}"
    output_path = os.path.join(tmp_dir, output_filename)
    writer.write(local_path, fill_result.filled_fields, output_path)

    filled_field_summary = {ff.name: ff.value for ff in fill_result.filled_fields[:30]}
    low_confidence_names = [
        ff.name for ff in fill_result.filled_fields if ff.confidence < 0.7
    ]
    output_summary = {
        "filled_count": fill_result.filled_count,
        "total_fields": fill_result.total_fields,
        "filled_fields": filled_field_summary,
        "low_confidence_field_names": low_confidence_names,
        "pursuit_id": pursuit_id,
    }

    try:
        print(f"  [fill_form] Submitting for approval: {output_filename}")
        result = client.submit_for_approval(
            thread_id=None,
            action_type="form_fill",
            title=f"Filled Form: {tender_title}",
            description=(
                f"Filled {fill_result.filled_count}/{fill_result.total_fields} fields on "
                f"{original_filename} for tender '{tender_title}'. "
                f"{len(low_confidence_names)} field(s) flagged low-confidence — "
                f"please review before approving."
            ),
            input_summary={
                "original_filename": original_filename,
                "total_fields": fill_result.total_fields,
                "pursuit_id": pursuit_id,
                "task_id": task_id,
            },
            output_summary=output_summary,
            file_path=output_path,
            filename=output_filename,
            metadata={
                "duration_ms": duration_ms,
                "cost_usd": fill_result.llm_cost_usd,
                "llm_model": LLM_MODEL,
                "pursuit_id": pursuit_id,
            },
        )
        submission_id = result.get("submissionId", "?")
        print(f"  [fill_form] Approval submission created: {submission_id}")

        notify_task_completed(
            agent_name="tender-agent",
            task_title=f"Form Filled: {tender_title}",
            summary=(
                f"Filled {fill_result.filled_count}/{fill_result.total_fields} fields "
                f"on '{original_filename}'. Submitted for human review."
            ),
            metrics={
                "duration_ms": duration_ms,
                "cost_usd": fill_result.llm_cost_usd,
                "fields_filled": fill_result.filled_count,
                "fields_total": fill_result.total_fields,
            },
            task_type="form_fill",
        )

        _audit("task_completed",
               f"Form submitted for approval: {output_filename} "
               f"({fill_result.filled_count}/{fill_result.total_fields})",
               node_name="fill_form",
               duration_ms=duration_ms,
               cost_usd=fill_result.llm_cost_usd,
               output_payload={"submission_id": submission_id, **output_summary})

        complete_agent_task(client, task_id, status="completed",
                            result_summary=(
                                f"Filled {fill_result.filled_count}/"
                                f"{fill_result.total_fields} fields; "
                                f"submission {submission_id}"
                            ),
                            duration_ms=duration_ms,
                            tokens_used=0,
                            cost_usd=fill_result.llm_cost_usd,
                            llm_calls=getattr(fill_result, "llm_calls", 0),
                            documents_used=1,
                            metadata={"submission_id": submission_id,
                                      "pursuit_id": pursuit_id})
        mark_pursuit_status(client, pursuit_id, "form_filling",
                            notes=f"Form submitted for approval (id={submission_id})")

    except Exception as exc:
        print(f"  [fill_form] Failed to submit for approval: {exc}")
        traceback.print_exc()
        _audit("task_failed", f"Submission creation failed: {exc}",
               node_name="fill_form", status="failure", error_message=str(exc),
               input_payload={"task_id": task_id, "pursuit_id": pursuit_id})
        complete_agent_task(client, task_id, status="failed",
                            result_summary=f"Submission API failed: {exc}",
                            duration_ms=duration_ms)


# ---------------------------------------------------------------------------
# HTTP helpers — search-jobs queue, agent-tasks queue, pursuit updates
# ---------------------------------------------------------------------------
# NOTE: There is no bearer auth on these endpoints in this codebase. The
# AMS-side routes look up the agent by URL slug and trust internal traffic.
# If the team later adds an agent API key, set it in the .env as
# NEXUS_AGENT_API_KEY and the AUTH_HEADERS dict below will be sent.

import httpx as _httpx  # noqa: E402  (lazy import to avoid top-of-file churn)

_AGENT_NAME = "tender-agent"
_AUTH_HEADERS: dict[str, str] = {}
_agent_api_key = os.getenv("NEXUS_AGENT_API_KEY", "").strip()
if _agent_api_key:
    _AUTH_HEADERS["Authorization"] = f"Bearer {_agent_api_key}"


def _ams_url(path: str) -> str:
    return f"{NEXUS_URL.rstrip('/')}{path}"


def poll_search_jobs() -> list[dict]:
    """GET /api/agents/tender-agent/search-jobs?status=queued

    Returns a list of job rows. Returns [] silently on any network error so
    the main loop doesn't crash when AMS is briefly down.
    """
    try:
        url = _ams_url(f"/api/agents/{_AGENT_NAME}/search-jobs?status=queued")
        resp = _httpx.get(url, headers=_AUTH_HEADERS, timeout=10.0)
        if resp.status_code != 200:
            return []
        data = resp.json()
        if isinstance(data, dict):
            return data.get("jobs") or data.get("searchJobs") or []
        if isinstance(data, list):
            return data
        return []
    except _httpx.ConnectError:
        return []
    except Exception as exc:
        print(f"  [poll_search_jobs] {exc}")
        return []


def claim_search_job(job_id: str) -> bool:
    """PATCH the job to status=running. Returns True on success."""
    try:
        url = _ams_url(f"/api/agents/{_AGENT_NAME}/search-jobs/{job_id}")
        resp = _httpx.patch(
            url,
            json={"status": "running", "startedAt": datetime.now(timezone.utc).isoformat()},
            headers=_AUTH_HEADERS,
            timeout=10.0,
        )
        return resp.status_code in (200, 204)
    except Exception as exc:
        print(f"  [claim_search_job {job_id}] {exc}")
        return False


def push_discovered_tenders(job_id: str, tenders: list[dict]) -> dict:
    """POST /api/agents/tender-agent/discovered-tenders — batch upsert.

    AMS upserts by fingerprint, so re-pushing the same tender across
    multiple searches is safe (idempotent).
    """
    print(f"  [push] About to POST {len(tenders)} tenders for job {job_id[:8]}")
    if not tenders:
        print(f"  [push] tender list is empty — nothing to send")
        return {"inserted": 0, "updated": 0}
    # Log the first tender's keys so we can spot a schema mismatch.
    print(f"  [push] first tender keys: {sorted(tenders[0].keys())}")
    print(f"  [push] first tender title: {(tenders[0].get('title') or '')[:80]}")
    print(f"  [push] first tender deadline: {tenders[0].get('submissionDeadline')}")
    url = _ams_url(f"/api/agents/{_AGENT_NAME}/discovered-tenders")
    print(f"  [push] POST {url}")
    try:
        resp = _httpx.post(
            url,
            json={"searchId": job_id, "tenders": tenders},
            headers=_AUTH_HEADERS,
            timeout=60.0,
        )
        print(f"  [push] HTTP {resp.status_code} | body[:300]: {resp.text[:300]}")
        resp.raise_for_status()
        return resp.json() if resp.content else {"inserted": len(tenders)}
    except Exception as exc:
        print(f"  [push_discovered_tenders {job_id}] {exc}")
        raise


def complete_search_job(job_id: str, stats: dict) -> None:
    """PATCH the job to status=completed with the run stats."""
    payload = {
        "status": "completed",
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "totalFound": int(stats.get("totalFound", 0)),
        "blockedExpired": int(stats.get("blockedExpired", 0)),
        "blockedRelevance": int(stats.get("blockedRelevance", 0)),
        "deduplicated": int(stats.get("deduplicated", 0)),
        "metadata": {
            "serpFallbackUsed": stats.get("serpFallbackUsed", False),
            "durationMs": stats.get("durationMs", 0),
            "brief": stats.get("brief"),
            "broadened": stats.get("broadened", False),
        },
    }
    try:
        url = _ams_url(f"/api/agents/{_AGENT_NAME}/search-jobs/{job_id}")
        _httpx.patch(url, json=payload, headers=_AUTH_HEADERS, timeout=10.0)
    except Exception as exc:
        print(f"  [complete_search_job {job_id}] {exc}")


def fail_search_job(job_id: str, error: str) -> None:
    """PATCH the job to status=failed with an error message."""
    payload = {
        "status": "failed",
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "errorMessage": str(error)[:1000],
    }
    try:
        url = _ams_url(f"/api/agents/{_AGENT_NAME}/search-jobs/{job_id}")
        _httpx.patch(url, json=payload, headers=_AUTH_HEADERS, timeout=10.0)
    except Exception as exc:
        print(f"  [fail_search_job {job_id}] {exc}")


def poll_agent_tasks() -> list[dict]:
    """GET /api/agents/tender-agent/tasks — queued/in_progress agent tasks."""
    try:
        url = _ams_url(f"/api/agents/{_AGENT_NAME}/tasks")
        resp = _httpx.get(url, headers=_AUTH_HEADERS, timeout=10.0)
        if resp.status_code != 200:
            return []
        data = resp.json()
        if isinstance(data, dict):
            return data.get("tasks") or []
        if isinstance(data, list):
            return data
        return []
    except _httpx.ConnectError:
        return []
    except Exception as exc:
        print(f"  [poll_agent_tasks] {exc}")
        return []


def mark_agent_task_running(client: NexusClient, task_id: str) -> None:
    """POST /api/agents/tender-agent/tasks — bump status to in_progress."""
    if not task_id:
        return
    try:
        url = _ams_url(f"/api/agents/{_AGENT_NAME}/tasks")
        _httpx.post(
            url,
            json={"taskId": task_id, "status": "in_progress"},
            headers=_AUTH_HEADERS,
            timeout=10.0,
        )
    except Exception as exc:
        print(f"  [mark_agent_task_running {task_id}] {exc}")


def complete_agent_task(
    client: NexusClient,
    task_id: str,
    status: str,
    result_summary: str,
    duration_ms: int,
    tokens_used: int = 0,
    cost_usd: float = 0.0,
    llm_calls: int = 0,
    documents_used: int = 0,
    metadata: dict | None = None,
) -> None:
    """POST /api/agents/tender-agent/tasks — final task status + metrics."""
    if not task_id:
        return
    payload = {
        "taskId": task_id,
        "status": status,
        "resultSummary": result_summary,
        "durationMs": duration_ms,
        "tokensUsed": tokens_used,
        "costUsd": cost_usd,
        "llmCalls": llm_calls,
        "documentsUsed": documents_used,
        "metadata": metadata or {},
    }
    try:
        url = _ams_url(f"/api/agents/{_AGENT_NAME}/tasks")
        _httpx.post(url, json=payload, headers=_AUTH_HEADERS, timeout=10.0)
    except Exception as exc:
        print(f"  [complete_agent_task {task_id}] {exc}")


def mark_pursuit_status(
    client: NexusClient,
    pursuit_id,
    new_status: str,
    notes: str | None = None,
) -> None:
    """PATCH /api/tender-pursuits/{id} — advance the linked pursuit lane.

    Quietly skipped if the pursuit_id is missing (manual tasks have none).
    """
    if not pursuit_id:
        return
    url = _ams_url(f"/api/tender-pursuits/{pursuit_id}")
    payload: dict = {"status": new_status}
    if notes:
        payload["notes"] = notes
    try:
        _httpx.patch(url, json=payload, headers=_AUTH_HEADERS, timeout=10.0)
    except Exception as exc:
        print(f"  [mark_pursuit_status {pursuit_id}] {exc}")


def _broaden_filters(filters: dict) -> dict:
    """Relax constraints when the strict pass returned too few results.

    Drops value range, extends deadline window by 60 days, opens posted
    window to "all", and re-enables every source. Keeps the 30% absolute
    relevance floor in place (non-negotiable per spec). Returns a fresh
    dict so the caller can compare before/after if needed.
    """
    relaxed = dict(filters or {})
    relaxed["minValueUsd"] = None
    relaxed["maxValueUsd"] = None
    relaxed["postedWithinDays"] = "all"
    relaxed["sources"] = []  # all 20 sources
    relaxed["includeSerpFallback"] = True
    # Extend the deadline window forward by 60 days so we catch tenders that
    # close later than the user's original window. We never push the window
    # earlier — expired tenders stay blocked by the deadline gates.
    try:
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        cur_to = relaxed.get("deadlineTo")
        if cur_to:
            base = _dt.fromisoformat(str(cur_to)).date()
            relaxed["deadlineTo"] = (base + _td(days=60)).isoformat()
        else:
            relaxed["deadlineTo"] = (
                _dt.now(_tz.utc).date() + _td(days=120)
            ).isoformat()
    except Exception:
        pass
    return relaxed


def _generate_brief(filters: dict, tenders: list[dict], stats: dict) -> str:
    """Write a 2–3 sentence executive brief over the result set.

    Uses the same OpenRouter LLM the bridge already uses. Falls back to a
    deterministic summary string if the LLM call fails so a brief is always
    written to TenderSearch.metadata.brief — the UI never shows a blank.
    """
    n = len(tenders)
    if n == 0:
        return (
            "No verified tenders matched these filters. The most common cause "
            "is too narrow a region+keyword combination. Try broadening keywords "
            "or selecting 'Global' to fan out across all 20 sources."
        )

    top = sorted(
        tenders,
        key=lambda t: float(t.get("relevanceScore") or 0),
        reverse=True,
    )[:5]
    bullet_lines = []
    for i, t in enumerate(top, 1):
        title = (t.get("title") or "Untitled")[:120]
        agency = (t.get("agency") or "Unknown agency")[:80]
        score = int(round(float(t.get("relevanceScore") or 0) * 100))
        deadline = (t.get("submissionDeadline") or "")[:10]
        bullet_lines.append(
            f"{i}. {title} — {agency} ({score}% relevance, closes {deadline})"
        )
    bullet_block = "\n".join(bullet_lines)

    deterministic = (
        f"Found {n} verified tenders. Top hit: {top[0].get('title', '?')[:100]} "
        f"({int(round(float(top[0].get('relevanceScore') or 0) * 100))}% relevance, "
        f"{top[0].get('agency', 'unknown agency')}). "
        f"{stats.get('blockedExpired', 0)} expired and "
        f"{stats.get('blockedRelevance', 0)} below relevance floor were filtered out."
    )

    if not OPENROUTER_API_KEY:
        return deterministic

    try:
        prompt = (
            f"You are summarising tender search results for a busy procurement lead. "
            f"Write a 2-3 sentence executive brief in plain English (no markdown, no bullet lists). "
            f"State the top opportunity, why it stands out, and one decision the user should make next.\n\n"
            f"Search keywords: {(filters.get('keywords') or '').strip() or '(none)'}\n"
            f"Region filter: {filters.get('regions') or ['global']}\n"
            f"Result count: {n}\n\n"
            f"Top results:\n{bullet_block}"
        )
        out = call_llm(prompt, max_tokens=200)
        text = (out.get("content") or "").strip()
        if len(text) > 40:
            return text
    except Exception as exc:
        print(f"  [brief] LLM brief failed, using deterministic: {exc}")

    return deterministic


def handle_search_job(client: NexusClient, job: dict) -> None:
    """End-to-end handler: claim → run → push results → complete (or fail).

    Includes one adaptive-broaden pass when the strict filters return fewer
    than 5 results, plus an LLM-generated executive brief that always lands
    in TenderSearch.metadata.brief.
    """
    job_id = job.get("id")
    filters = job.get("filters") or {}
    if not job_id:
        print("  [handle_search_job] dropped job with no id")
        return

    print(f"\n  >>> Claiming search job {job_id}")
    if not claim_search_job(job_id):
        print(f"  [handle_search_job {job_id}] could not claim (race or 404), skipping")
        return

    try:
        _audit("task_started",
               f"Search job {job_id} claimed",
               input_payload={"job_id": job_id, "filters": filters})

        # --- Pass 1: strict ------------------------------------------------
        result = run_tender_search(filters)
        tenders = result.get("tenders", [])
        stats = result.get("stats", {})
        broadened = False

        # --- Pass 2: auto-broaden when weak --------------------------------
        # Threshold: 5. Below that, retry with relaxed constraints and merge.
        # The user sees the union, with broadened=true flagged in metadata.
        # Auto-broaden only when pass 1 is genuinely sparse. 3+ results is
        # enough for a useful demo / triage; running pass 2 anyway just
        # doubles the user's wait time without adding meaningful coverage.
        if len(tenders) < 3:
            print(f"  [auto-broaden] Pass 1 returned {len(tenders)} — retrying with relaxed filters")
            broadened = True
            relaxed = _broaden_filters(filters)
            try:
                result2 = run_tender_search(relaxed)
                tenders2 = result2.get("tenders", [])
                stats2 = result2.get("stats", {})

                seen_fps = {t.get("fingerprint") for t in tenders if t.get("fingerprint")}
                new_only = [t for t in tenders2 if t.get("fingerprint") not in seen_fps]
                print(f"  [auto-broaden] Pass 2 added {len(new_only)} fresh tenders")
                tenders = tenders + new_only
                stats["totalFound"] = len(tenders)
                stats["blockedExpired"] = (
                    int(stats.get("blockedExpired", 0))
                    + int(stats2.get("blockedExpired", 0))
                )
                stats["blockedRelevance"] = (
                    int(stats.get("blockedRelevance", 0))
                    + int(stats2.get("blockedRelevance", 0))
                )
                stats["deduplicated"] = (
                    int(stats.get("deduplicated", 0))
                    + int(stats2.get("deduplicated", 0))
                )
            except Exception as exc:
                print(f"  [auto-broaden] Pass 2 failed: {exc}")

        # --- Brief ---------------------------------------------------------
        brief = _generate_brief(filters, tenders, stats)
        stats["brief"] = brief
        stats["broadened"] = broadened

        try:
            push_discovered_tenders(job_id, tenders)
        except Exception as push_exc:
            fail_search_job(job_id, f"push_discovered_tenders failed: {push_exc}")
            _audit("task_failed", f"Push failed for job {job_id}: {push_exc}",
                   status="failure", error_message=str(push_exc),
                   input_payload={"job_id": job_id})
            return

        complete_search_job(job_id, stats)
        _audit("task_completed",
               f"Search job {job_id} done: {stats.get('totalFound', 0)} tenders, "
               f"{stats.get('blockedExpired', 0)} expired blocked, "
               f"{stats.get('blockedRelevance', 0)} below relevance floor",
               node_name="discover",
               duration_ms=int(stats.get("durationMs", 0)),
               output_payload={"job_id": job_id, **stats})

        # Slack: notify the channel only when high-relevance hits land,
        # to keep the channel from getting noisy on speculative searches.
        any_high_relevance = any(
            (t.get("relevanceScore") or 0) > 0.70 for t in tenders
        )
        if any_high_relevance:
            notify_task_completed(
                agent_name="tender-agent",
                task_title=f"Tender search: {(filters.get('keywords') or '')[:60]}",
                summary=(
                    f"Found {len(tenders)} verified tenders; "
                    f"{sum(1 for t in tenders if (t.get('relevanceScore') or 0) > 0.70)} above 70% relevance."
                ),
                metrics={
                    "duration_ms": int(stats.get("durationMs", 0)),
                    "tenders_found": len(tenders),
                },
                task_type="tender_search",
            )
    except Exception as exc:
        traceback.print_exc()
        fail_search_job(job_id, str(exc))
        _audit("task_failed", f"Search job {job_id} failed: {exc}",
               status="failure", error_message=str(exc),
               input_payload={"job_id": job_id})
    finally:
        try:
            client.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main — Registration + Heartbeat + Job/Task Polling
# ---------------------------------------------------------------------------

def main() -> None:
    print("")
    print("=" * 60)
    print("  TENDER AGENT — Nexus AMS Bridge (filter-driven)")
    print("=" * 60)
    print(f"  AMS URL:     {NEXUS_URL}")
    print(f"  LLM Model:   {LLM_MODEL}")
    print(f"  Dry Run:     {DRY_RUN}")
    print(f"  OpenRouter:  {'configured' if OPENROUTER_API_KEY else 'NOT configured'}")
    print(f"  Agent token: {'set' if _agent_api_key else 'unset (open access)'}")
    print("=" * 60)
    print("")

    global _client
    client = NexusClient(base_url=NEXUS_URL)
    _client = client

    config = AgentConfig(
        name=_AGENT_NAME,
        display_name="Tender Agent",
        description=(
            "Discovers and evaluates government tenders related to SDS / EHS. "
            "Driven by filter forms in /tenders/discover; fills attached forms "
            "on demand from /tenders/pursuits when set to Agent fill mode."
        ),
        version="2.0.0",
        python_version="3.11",
        langgraph_version="0.4.0",
        llm_provider="openrouter",
        llm_models={
            LLM_MODEL: {
                "provider": "openrouter",
                "context_window": 200000,
            },
        },
        llm_pricing={
            LLM_MODEL: {
                "input_cost_per_million": 3.00,
                "output_cost_per_million": 15.00,
            },
        },
        embedding_model="voyage-3-large",
        embedding_dimensions=1024,
        state_fields_count=40,
        node_names=["discover", "fill_form", "submit"],
        tools=[
            "sam_gov_scraper", "ted_europa", "uk_tenders", "boamp_france",
            "world_bank", "prozorro", "canada_buys", "austender", "sa_etender",
            "colombia_secop", "brazil_compras", "germany_bkms", "italy_anac",
            "dominican_dgcp", "peru_oece", "world_bank_v2", "nigeria_nocopo",
            "kenya_ppra", "uganda_gpp", "mexico_cdmx",
            "brightdata_serp", "deadline_verifier",
            "form_parser", "form_filler", "form_writer",
            "voyage_embedder", "slack_client",
        ],
        health_endpoint="http://localhost:8100/health",
        slack_channels=["#agent-updates"],
        env_vars_count=23,
        dry_run=DRY_RUN,
        budget_monthly_usd=50.0,
        tags=["production", "government-tenders", "sds", "ehs", "filter-driven"],
        changelog="v2: filter-driven discovery, removed chat-driven inbox loop.",
    )

    print("Registering with Nexus AMS...")
    try:
        resp = client.register(config)
        print(f"Registered! Agent ID: {resp.agent_id}")
        notify_agent_online(_AGENT_NAME)
    except Exception as exc:
        print(f"Registration failed: {exc}")
        print("Make sure the AMS is running at", NEXUS_URL)
        sys.exit(1)

    print("Starting heartbeat (every 30 seconds)...")
    stop_event = threading.Event()

    def heartbeat_loop() -> None:
        while not stop_event.is_set():
            try:
                client.heartbeat(status="running")
            except Exception as exc:
                print(f"Heartbeat error (will retry): {exc}")
            stop_event.wait(30.0)

    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    hb_thread.start()

    print("")
    print("Polling search-jobs and agent tasks every 5s...")
    print("Press Ctrl+C to stop.")
    print("")

    try:
        while True:
            try:
                jobs = poll_search_jobs()
                for job in jobs:
                    handle_search_job(client, job)
            except Exception as exc:
                print(f"  [search-jobs loop] {exc}")
                traceback.print_exc()

            try:
                tasks = poll_agent_tasks()
                for task in tasks:
                    task_status = task.get("status")
                    if task_status not in ("queued",):
                        # Skip in_progress to avoid double-processing
                        continue
                    task_type = (
                        (task.get("metadata") or {}).get("type")
                        or task.get("type")
                        or "fill_form"
                    )
                    if task_type == "fill_form":
                        mark_agent_task_running(client, task.get("id"))
                        handle_fill_form_task(client, task)
                    else:
                        print(f"  [agent-tasks] Skipping unknown task type: {task_type}")
            except Exception as exc:
                print(f"  [agent-tasks loop] {exc}")
                traceback.print_exc()

            time.sleep(5.0)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        stop_event.set()
        notify_agent_offline(_AGENT_NAME)
        try:
            client.flush()
            client.close()
        except Exception:
            pass
        print("Tender Agent bridge stopped.")


if __name__ == "__main__":
    main()
