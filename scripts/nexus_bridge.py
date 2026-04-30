"""
Nexus AMS Bridge for Tender Agent
===================================
This script connects the tender-agent to the Nexus AMS platform.

It does 3 things:
1. Registers the agent and sends heartbeats (so AMS knows the agent is alive)
2. Polls for incoming chat messages from the AMS UI
3. Runs tender discovery searches and sends results back to the AMS chat

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
from src.discovery.serp_search import SerpTenderSearcher

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEXUS_URL = os.getenv("NEXUS_AMS_URL", "http://localhost:3000")
OPENROUTER_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-9b")
OPENROUTER_BASE_URL = os.getenv("QWEN_BASE_URL", "https://openrouter.ai/api/v1")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# LLM Helper — calls OpenRouter for tender evaluation summaries
# ---------------------------------------------------------------------------

def call_openrouter(prompt: str, max_tokens: int = 1024) -> dict:
    """Call OpenRouter API and return response with usage stats.

    Returns:
        {
            "content": "the LLM response text",
            "tokens_input": 123,
            "tokens_output": 45,
            "cost_usd": 0.0001,
            "model": "qwen/qwen3-9b",
            "duration_ms": 1234,
        }
    """
    import httpx

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
            "model": OPENROUTER_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a government procurement analyst specializing in "
                        "Safety Data Sheet (SDS) and Environmental Health & Safety (EHS) "
                        "software tenders. Be concise and actionable. "
                        "Respond directly without thinking out loud."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "thinking": {"type": "disabled"},
        },
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()

    duration_ms = int((time.perf_counter() - start_time) * 1000)

    # Handle models that return reasoning instead of content
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    if not content and msg.get("reasoning"):
        # Extract useful text from reasoning if content is empty
        content = msg["reasoning"].split("\n")[-1].strip()
    usage = data.get("usage", {})
    tokens_in = usage.get("prompt_tokens", 0)
    tokens_out = usage.get("completion_tokens", 0)

    # OpenRouter pricing for qwen/qwen3-9b: $0.20/M input, $0.60/M output
    cost_usd = (tokens_in * 0.20 / 1_000_000) + (tokens_out * 0.60 / 1_000_000)

    return {
        "content": content,
        "tokens_input": tokens_in,
        "tokens_output": tokens_out,
        "cost_usd": round(cost_usd, 6),
        "model": OPENROUTER_MODEL,
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

# URL patterns that indicate NOT a tender (info pages, wikis, articles)
NOT_TENDER_PATTERNS = [
    "/legislation/", "/directives/", "/guidelines/", "/themes/",
    "/wiki/", "/blog/", "/article/", "/news/", "/about/",
    "/glossary/", "/eguides/", "/films/", "/tools/", "/resources/",
    "/faq/", "/help/", "/contact/", "/policy/", "/regulation/",
    "/publications/", "/research/", "/reports/", "/events/",
]

# Domains to always exclude
EXCLUDE_DOMAINS = {
    "linkedin.com", "youtube.com", "facebook.com", "twitter.com",
    "reddit.com", "medium.com", "wikipedia.org", "quora.com",
    "heyiris.ai", "hubspot.com", "salesforce.com", "oshwiki.osha.europa.eu",
    "eguides.osha.europa.eu",
}

# Domains where ONLY specific paths are tenders
STRICT_PATH_DOMAINS = {
    "osha.europa.eu": ["/procurement"],
    "gov.uk": ["/contracts-finder", "/tender"],
}


def is_valid_tender_link(url: str) -> bool:
    """Check if a URL is an actual tender listing page."""
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        path = parsed.path.lower()

        # Always exclude known non-tender sites
        for excluded in EXCLUDE_DOMAINS:
            if excluded in domain:
                return False

        # Exclude info/wiki/legislation pages
        for pattern in NOT_TENDER_PATTERNS:
            if pattern in path:
                return False

        # Strict path domains — only allow specific paths
        for strict_domain, allowed_paths in STRICT_PATH_DOMAINS.items():
            if strict_domain in domain:
                return any(ap in path for ap in allowed_paths)

        # Known tender portals with any path are OK
        trusted_portals = {
            "sam.gov", "ted.europa.eu", "ungm.org", "merx.com",
            "bidnetdirect.com", "virginiabids.com", "highergov.com",
            "tendersontime.com", "buyandsell.gc.ca", "bidsandtenders.ca",
            "tenders.gov.au", "eprocure.gov.in", "etenders.gov.za",
            "globaltenders.com", "dgmarket.com", "devbusiness.com",
            "contracts.gov.sg", "etimaden.gov.tr",
        }
        for portal in trusted_portals:
            if portal in domain:
                return True

        # Check URL path for tender-like patterns
        for pattern in TENDER_URL_PATTERNS:
            if pattern in path:
                return True

        return False
    except Exception:
        return False


def run_tender_search(query: str) -> tuple[str, dict]:
    """Run a global tender discovery search via Bright Data SERP proxy.

    Returns only verified tender links with clean formatting.
    """
    start_time = time.perf_counter()

    # Step 1: Search globally via SERP proxy
    try:
        searcher = SerpTenderSearcher()
        leads = searcher.search(user_query=query)
    except Exception as exc:
        print(f"SERP search failed: {exc}, falling back to SAM.gov dry run")
        scraper = SamGovScraper(dry_run=True)
        raw = scraper.fetch_opportunities(days_back=7)
        leads = [
            type('Lead', (), {
                'title': l.title, 'description': l.description,
                'agency': l.agency, 'source_url': l.source_url,
                'relevance_score': l.relevance_score,
                'relevance_keywords': l.relevance_keywords,
                'submission_deadline': l.submission_deadline,
            })()
            for l in raw
        ]

    # Step 2: Filter — only real tender links
    valid_leads = [l for l in leads if is_valid_tender_link(l.source_url)]

    if not valid_leads:
        reply = "I searched globally but couldn't find verified tender listings matching your criteria. Try being more specific, e.g. *'SDS authoring software RFP United States'* or *'chemical safety compliance tender Europe'*."
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        return reply, {
            "node": "discover",
            "tenders_found": 0,
            "duration_ms": duration_ms,
            "cost_usd": 0,
        }

    # Step 3: Format each tender with title-as-link and description
    formatted_items = []
    for i, lead in enumerate(valid_leads[:10], 1):
        title = lead.title.strip()
        if len(title) > 80:
            title = title[:77] + "..."
        url = lead.source_url

        # Clean the description — remove URL breadcrumbs and duplicated text
        desc = getattr(lead, 'description', '') or ''
        # Remove breadcrumb patterns like "UNGM https://www.ungm.org › Public › Notice"
        import re
        desc = re.sub(r'https?://\S+', '', desc)
        desc = re.sub(r'[\w.-]+\s*›[^›\n]*(?:›[^›\n]*)*', '', desc)
        desc = re.sub(r'\s{2,}', ' ', desc).strip()
        # Remove if desc just repeats the title
        if desc.lower().strip('.') == title.lower().strip('.'):
            desc = ''
        if len(desc) > 150:
            desc = desc[:147] + "..."

        if desc:
            formatted_items.append(f"{i}. [{title}]({url})\n{desc}")
        else:
            formatted_items.append(f"{i}. [{title}]({url})")

    results_block = "\n\n".join(formatted_items)

    # Also build a plain text version for the LLM summary
    plain_list = "\n".join(
        f"{i}. {l.title} ({getattr(l, 'agency', 'Unknown')}) - {l.relevance_score:.0%}"
        for i, l in enumerate(valid_leads[:10], 1)
    )

    # Step 4: Get LLM summary
    llm_summary = ""
    llm_cost = 0.0
    llm_tokens_in = 0
    llm_tokens_out = 0
    if OPENROUTER_API_KEY:
        try:
            llm_result = call_openrouter(
                f"The user searched for: '{query}'\n\n"
                f"Found {len(valid_leads)} verified tender listings:\n{plain_list}\n\n"
                f"Write exactly 2 sentences: which opportunity is most promising and why. Be specific.",
                max_tokens=200,
            )
            llm_summary = llm_result["content"]
            llm_cost = llm_result["cost_usd"]
            llm_tokens_in = llm_result["tokens_input"]
            llm_tokens_out = llm_result["tokens_output"]
        except Exception as exc:
            print(f"LLM summary failed: {exc}")

    # Step 5: Assemble clean reply
    parts = []
    if llm_summary:
        parts.append(llm_summary)
        parts.append("")
        parts.append("---")
        parts.append("")
    parts.append(f"**{len(valid_leads)} verified tenders found:**")
    parts.append("")
    parts.append(results_block)

    reply = "\n".join(parts)

    duration_ms = int((time.perf_counter() - start_time) * 1000)

    metadata = {
        "node": "discover",
        "tenders_found": len(valid_leads),
        "top_relevance": valid_leads[0].relevance_score if valid_leads else 0,
        "tokens_input": llm_tokens_in,
        "tokens_output": llm_tokens_out,
        "cost_usd": llm_cost,
        "model": OPENROUTER_MODEL,
        "duration_ms": duration_ms,
        "search_source": "brightdata_serp",
    }

    return reply, metadata


# ---------------------------------------------------------------------------
# Message Handler — processes incoming chat messages from AMS
# ---------------------------------------------------------------------------

def classify_intent(user_message: str) -> dict:
    """Use the LLM to understand what the user wants.

    Returns:
        {
            "intent": "search_tenders" | "ask_question" | "greeting" | "refine_results",
            "search_query": "extracted search terms if intent is search",
            "response_hint": "brief note on how to respond",
        }
    """
    try:
        result = call_openrouter(
            f"""You are an intent classifier for a tender discovery agent.
Analyze this user message and respond with ONLY valid JSON (no markdown, no explanation):

User message: "{user_message}"

Respond with this exact JSON structure:
{{"intent": "search_tenders|ask_question|greeting|refine_results", "search_query": "extracted search keywords for tender search", "response_hint": "brief note"}}

Rules:
- "search_tenders": user wants to find tenders, RFPs, bids, procurement opportunities
- "ask_question": user asks about a specific tender, process, or general question
- "greeting": user says hi, hello, thanks, etc.
- "refine_results": user wants to narrow down or filter previous results
- "search_query": extract the core topic/keywords (e.g., "chemical safety tenders in Europe")
- Always extract a search_query even for greetings (use "SDS management tenders" as default)""",
            max_tokens=200,
        )
        # Parse JSON from response
        content = result["content"].strip()
        # Handle markdown code blocks
        if "```" in content:
            content = content.split("```")[1].strip()
            if content.startswith("json"):
                content = content[4:].strip()
        return json.loads(content)
    except Exception as exc:
        print(f"Intent classification failed: {exc}, defaulting to search")
        return {
            "intent": "search_tenders",
            "search_query": user_message,
            "response_hint": "default search",
        }


def generate_natural_response(user_message: str, intent: dict, results_text: str, num_results: int) -> str:
    """Use LLM to generate a natural, conversational response."""
    try:
        if intent["intent"] == "greeting":
            result = call_openrouter(
                f"""The user said: "{user_message}"
You are the Tender Agent — a friendly AI that helps find government and enterprise tenders related to Safety Data Sheets (SDS), chemical safety, and EHS compliance.
Respond naturally in 2-3 sentences. Introduce yourself briefly and ask how you can help with tender discovery.""",
                max_tokens=200,
            )
            return result["content"]

        if intent["intent"] == "ask_question":
            result = call_openrouter(
                f"""The user asked: "{user_message}"
You are the Tender Agent specializing in SDS/EHS/chemical safety tenders.
Answer their question concisely. If you don't know, say so and offer to search for relevant tenders instead.""",
                max_tokens=500,
            )
            return result["content"]

        # For search results, generate a summary
        if num_results > 0:
            result = call_openrouter(
                f"""The user asked: "{user_message}"
I searched globally and found {num_results} tender opportunities. Here are the results:

{results_text}

Write a natural 2-3 sentence executive summary. Mention the most promising opportunity, which region/agency it's from, and why it's relevant to SDS/EHS. Be specific, not generic.""",
                max_tokens=300,
            )
            return result["content"]
        else:
            return f"I searched globally for tenders matching your request but didn't find strong matches right now. Try rephrasing — for example: 'find chemical safety compliance RFPs in Europe' or 'SDS management tenders in the US'."

    except Exception as exc:
        print(f"Natural response generation failed: {exc}")
        return ""


def handle_message(client: NexusClient, message: dict) -> None:
    """Handle an incoming chat message from the AMS.

    Uses LLM to understand the user's intent and respond naturally.
    """
    thread_id = message.get("threadId", "")
    content = message.get("content", "")
    sender = message.get("senderId", "unknown")

    # Strip @tender-agent mention from content
    clean_content = content.replace("@tender-agent", "").strip()

    print(f"\n{'='*60}")
    print(f"  INCOMING MESSAGE")
    print(f"  Thread:  {thread_id}")
    print(f"  From:    {sender}")
    print(f"  Content: {clean_content[:100]}{'...' if len(clean_content) > 100 else ''}")
    print(f"{'='*60}\n")

    client.track(
        "conversation_volume",
        1.0,
        metadata={"thread_id": thread_id, "direction": "inbound"},
    )

    start_time = time.perf_counter()

    try:
        # Step 1: Understand what the user wants
        print("Classifying user intent...")
        intent = classify_intent(clean_content)
        print(f"  Intent: {intent.get('intent', 'unknown')}")
        print(f"  Search query: {intent.get('search_query', 'none')}")

        # Step 2: Handle based on intent
        if intent["intent"] in ("greeting", "ask_question") and intent["intent"] != "search_tenders":
            reply_text = generate_natural_response(clean_content, intent, "", 0)
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            client.reply(
                thread_id=thread_id,
                content=reply_text,
                metadata={"intent": intent["intent"], "duration_ms": duration_ms},
            )
            print(f"Conversational reply sent ({duration_ms}ms)")
            return

        # Step 3: Run tender search
        print("Running global tender search...")
        search_query = intent.get("search_query", clean_content)
        reply_text, metadata = run_tender_search(search_query)

        # Step 4: Generate natural summary and prepend it
        natural_intro = generate_natural_response(
            clean_content, intent,
            reply_text, metadata.get("tenders_found", 0)
        )

        if natural_intro:
            # Replace the generic header with the natural response
            final_reply = (
                f"{natural_intro}\n\n"
                f"---\n\n"
                f"{reply_text.split('---', 1)[-1] if '---' in reply_text else reply_text}"
            )
        else:
            final_reply = reply_text

        metadata["thread_id"] = thread_id
        metadata["intent"] = intent["intent"]
        client.reply(
            thread_id=thread_id,
            content=final_reply,
            metadata=metadata,
        )

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        print(f"Reply sent! Found {metadata['tenders_found']} tenders ({duration_ms}ms)")

        # Track metrics
        client.track("task_completion", 1.0, metadata={
            "node": "discover",
            "tenders_found": metadata["tenders_found"],
        })
        client.track("latency", float(duration_ms), metadata={
            "node": "discover",
        })
        if metadata.get("cost_usd", 0) > 0:
            client.track("cost", metadata["cost_usd"], metadata={
                "node": "discover",
                "model": OPENROUTER_MODEL,
                "cost_usd": metadata["cost_usd"],
            })

    except Exception as exc:
        error_msg = f"Error: {str(exc)}"
        print(f"ERROR: {error_msg}")
        traceback.print_exc()

        client.reply(
            thread_id=thread_id,
            content=f"Sorry, I ran into an issue while processing your request:\n\n`{str(exc)}`\n\nCould you try rephrasing? For example: 'find SDS management tenders in Europe'",
            metadata={"error": str(exc), "node": "discover"},
        )

        client.track("error_rate", 1.0, metadata={
            "node": "discover",
            "error_type": type(exc).__name__,
        })


# ---------------------------------------------------------------------------
# Main — Registration + Heartbeat + Inbox Polling
# ---------------------------------------------------------------------------

def main() -> None:
    print("")
    print("=" * 60)
    print("  TENDER AGENT — Nexus AMS Bridge")
    print("=" * 60)
    print(f"  AMS URL:     {NEXUS_URL}")
    print(f"  LLM Model:   {OPENROUTER_MODEL}")
    print(f"  Dry Run:     {DRY_RUN}")
    print(f"  OpenRouter:  {'configured' if OPENROUTER_API_KEY else 'NOT configured'}")
    print("=" * 60)
    print("")

    # Step 1: Create client and register
    client = NexusClient(base_url=NEXUS_URL)

    config = AgentConfig(
        name="tender-agent",
        display_name="Tender Agent",
        description=(
            "Discovers and evaluates government tenders related to SDS management. "
            "Searches SAM.gov and scores opportunities by relevance to EHS/chemical safety."
        ),
        version="1.0.0",
        python_version="3.11",
        langgraph_version="0.4.0",
        llm_provider="openrouter",
        llm_models={
            OPENROUTER_MODEL: {
                "provider": "openrouter",
                "context_window": 131072,
            },
        },
        llm_pricing={
            OPENROUTER_MODEL: {
                "input_cost_per_million": 0.20,
                "output_cost_per_million": 0.60,
            },
        },
        embedding_model="voyage-3-large",
        embedding_dimensions=1024,
        state_fields_count=40,
        node_names=["discover", "evaluate", "retrieve_draft", "gap_check",
                     "slack_escalate", "assemble", "submit"],
        tools=["sam_gov_scraper", "email_monitor", "voyage_embedder",
               "template_engine", "slack_client", "playwright_submitter"],
        health_endpoint="http://localhost:8100/health",
        slack_channels=["#tender-alerts"],
        env_vars_count=23,
        dry_run=DRY_RUN,
        budget_monthly_usd=50.0,
        tags=["production", "government-tenders", "sds", "ehs"],
        changelog="Bridge connection to Nexus AMS",
    )

    print("Registering with Nexus AMS...")
    try:
        resp = client.register(config)
        print(f"Registered! Agent ID: {resp.agent_id}")
    except Exception as exc:
        print(f"Registration failed: {exc}")
        print("Make sure the AMS is running at", NEXUS_URL)
        sys.exit(1)

    # Step 2: Start heartbeat in background
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

    # Step 3: Poll inbox for messages
    print("")
    print("Listening for messages from AMS chat...")
    print("Go to http://localhost:3000/chat and send a message mentioning @tender-agent")
    print("Example: '@tender-agent search for SDS management tenders'")
    print("")
    print("Press Ctrl+C to stop.")
    print("")

    try:
        while True:
            try:
                # Poll the inbox endpoint for new messages
                import httpx
                inbox_url = f"{NEXUS_URL}/api/agents/tender-agent/inbox"
                resp = httpx.get(inbox_url, timeout=10.0)

                if resp.status_code == 200:
                    data = resp.json()
                    # Handle both { "messages": [...] } and plain [...]
                    if isinstance(data, dict):
                        messages = data.get("messages", [])
                    elif isinstance(data, list):
                        messages = data
                    else:
                        messages = []
                    for msg in messages:
                        handle_message(client, msg)

            except httpx.ConnectError:
                pass  # AMS not reachable, will retry
            except Exception as exc:
                print(f"Inbox polling error: {exc}")

            time.sleep(2.0)  # Poll every 2 seconds

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        stop_event.set()
        client.flush()
        client.close()
        print("Tender Agent bridge stopped.")


if __name__ == "__main__":
    main()
