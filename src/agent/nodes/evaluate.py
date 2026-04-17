"""
Evaluate Node — Scores tender eligibility using Claude Haiku 4.5.

WHY HAIKU 4.5 (NOT SONNET OR OPUS):
Evaluation is a classification task — we're scoring a tender against a rubric,
not generating creative content. Haiku 4.5 handles this reliably at ~$0.001 per
call (~1000 input tokens + ~500 output tokens). Using Sonnet would cost 10x more
with negligible accuracy improvement for this task. We save the expensive models
for drafting (Sonnet) and compliance reasoning (Opus).

THE 8 SCORING DIMENSIONS:
Each dimension reflects a real procurement contersideration. The weights are tuned
for a B2B SaaS company in the chemical safety / SDS management space:

| Dimension      | Max Score | What It Measures                                      |
|---------------|-----------|-------------------------------------------------------|
| geography     | 15        | Can we serve this region? Local presence needed?      |
| budget        | 15        | Is the budget realistic for our pricing?              |
| scope         | 15        | Does the scope match our platform capabilities?       |
| compliance    | 10        | Can we meet all regulatory/compliance requirements?   |
| timeline      | 10        | Is the submission deadline achievable?                |
| certifications| 10        | Do we hold the required certs (ISO, SOC2, etc.)?     |
| domain_match  | 15        | Is this in our domain (EHS, SDS, chemical safety)?   |
| competition   | 10        | How competitive is this bid? Are we well-positioned? |
| TOTAL         | 100       | Score ≥ 60 = GO, Score < 60 = NO-GO                  |

DRY-RUN MODE:
When DRY_RUN=true (no API key), the node uses keyword-based heuristic scoring
instead of calling the LLM. This produces realistic, varied scores based on
the actual tender text — much better for testing than returning a fixed number.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

import structlog

from src.agent.state import TenderState, TenderStatus

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELIGIBILITY_THRESHOLD: int = 60

HAIKU_MODEL: str = "qwen3.5-flash"

# The scoring prompt sent to Haiku 4.5
EVALUATION_PROMPT: str = """You are an expert tender evaluation analyst for a B2B SaaS company that provides Safety Data Sheet (SDS) management software, GHS classification tools, and EHS compliance services.

Evaluate the following tender and score it on 8 dimensions. Our company:
- Provides cloud-based SDS management, chemical inventory tracking, GHS labeling
- Serves manufacturing, construction, oil & gas, pharma, education sectors
- Holds ISO 27001 and SOC 2 Type II certifications
- Supports OSHA HCS, WHMIS, CLP/REACH, and 140+ jurisdictions
- Has 500+ clients and 15 chemical safety specialists
- Pricing starts at $5,000/year for up to 100 users

SCORING RUBRIC (score each dimension as an integer):

1. geography (0-15): Can we serve this region? 15=our core market, 10=can serve but not primary, 5=possible with effort, 0=cannot serve
2. budget (0-15): Is the budget realistic? 15=perfect fit, 10=workable, 5=tight but possible, 0=way outside our range
3. scope (0-15): Does scope match our capabilities? 15=exact match, 10=strong match, 5=partial match, 0=outside our scope
4. compliance (0-10): Can we meet regulatory requirements? 10=fully compliant, 5=mostly, 0=major gaps
5. timeline (0-10): Is the deadline achievable? 10=comfortable, 5=tight, 0=impossible
6. certifications (0-10): Do we hold required certs? 10=all required, 5=most, 0=missing critical ones
7. domain_match (0-15): Is this in our domain (EHS/SDS/chemical safety)? 15=core domain, 10=adjacent, 5=tangential, 0=unrelated
8. competition (0-10): How well-positioned are we? 10=strong advantage, 5=competitive, 0=likely outmatched

TENDER TEXT:
{tender_text}

Respond with ONLY valid JSON (no markdown, no backticks, no explanation):
{{
    "scores": {{
        "geography": <int>,
        "budget": <int>,
        "scope": <int>,
        "compliance": <int>,
        "timeline": <int>,
        "certifications": <int>,
        "domain_match": <int>,
        "competition": <int>
    }},
    "reasoning": "<2-3 sentence explanation of the overall assessment>"
}}"""


# ---------------------------------------------------------------------------
# Dry-run keyword scoring
# ---------------------------------------------------------------------------

# Keywords that indicate relevance to each dimension.
# More matches = higher score for that dimension.
_KEYWORD_MAP: dict[str, dict[str, list[str]]] = {
    "geography": {
        "high": ["united states", "us ", "u.s.", "federal", "national", "state", "canada", "north america", "eu ", "europe"],
        "low": ["local only", "on-site only", "physical presence required"],
    },
    "budget": {
        "high": ["saas", "software", "platform", "cloud", "subscription", "annual"],
        "low": ["under $1,000", "volunteer", "pro bono", "unfunded"],
    },
    "scope": {
        "high": ["sds", "safety data sheet", "chemical", "inventory", "ghs", "label", "hazard", "msds", "ehs software", "compliance software"],
        "low": ["construction management", "accounting", "hr software", "erp"],
    },
    "compliance": {
        "high": ["osha", "hcs", "whmis", "clp", "reach", "regulatory", "compliance", "tier ii", "epcra"],
        "low": ["hipaa only", "pci only"],
    },
    "timeline": {
        "high": ["30 days", "60 days", "90 days", "flexible deadline"],
        "low": ["24 hours", "48 hours", "immediate", "urgent"],
    },
    "certifications": {
        "high": ["iso 27001", "soc 2", "soc2", "fedramp", "security certification"],
        "low": ["cmmi level 5", "iso 13485", "as9100"],
    },
    "domain_match": {
        "high": ["sds management", "safety data sheet", "chemical safety", "ehs", "environment health safety", "hazardous materials", "ghs classification", "chemical inventory"],
        "low": ["food safety", "cybersecurity", "financial audit"],
    },
    "competition": {
        "high": ["small business", "set-aside", "niche", "specialized", "sds", "chemical safety"],
        "low": ["large enterprise only", "incumbent", "sole source"],
    },
}

# Max scores per dimension
_MAX_SCORES: dict[str, int] = {
    "geography": 15,
    "budget": 15,
    "scope": 15,
    "compliance": 10,
    "timeline": 10,
    "certifications": 10,
    "domain_match": 15,
    "competition": 10,
}


def _dry_run_score(tender_text: str) -> tuple[dict[str, int], str]:
    """Score a tender using keyword heuristics (no LLM needed).

    Scans the tender text for relevant keywords in each dimension.
    More keyword matches = higher score. This gives realistic, varied
    scores based on actual tender content.

    Returns:
        Tuple of (score_breakdown, reasoning_string)
    """
    text_lower = tender_text.lower()
    breakdown: dict[str, int] = {}

    for dimension, max_score in _MAX_SCORES.items():
        keywords = _KEYWORD_MAP[dimension]
        high_matches = sum(1 for kw in keywords["high"] if kw in text_lower)
        low_matches = sum(1 for kw in keywords["low"] if kw in text_lower)

        # Base score: 40-60% of max (assume moderate relevance by default)
        base = int(max_score * 0.5)

        # Boost for high-relevance keywords (each match adds ~15% of max)
        boost = min(high_matches * int(max_score * 0.15), int(max_score * 0.5))

        # Penalty for low-relevance keywords
        penalty = min(low_matches * int(max_score * 0.2), int(max_score * 0.4))

        score = max(0, min(max_score, base + boost - penalty))
        breakdown[dimension] = score

    total = sum(breakdown.values())
    decision = "GO" if total >= ELIGIBILITY_THRESHOLD else "NO-GO"

    reasoning = (
        f"[DRY-RUN] Keyword-based evaluation. Total score: {total}/100. "
        f"Decision: {decision}. Strongest dimensions: "
        f"{', '.join(d for d, s in sorted(breakdown.items(), key=lambda x: -x[1])[:3])}."
    )

    return breakdown, reasoning


# ---------------------------------------------------------------------------
# Real LLM scoring
# ---------------------------------------------------------------------------

def _llm_score(tender_text: str) -> tuple[dict[str, int], str, int]:
    """Score a tender using Qwen3.5 Flash.

    Sends the tender text with a structured scoring prompt and parses
    the JSON response.

    Returns:
        Tuple of (score_breakdown, reasoning_string, tokens_used)

    Raises:
        RuntimeError: If the API call fails or response can't be parsed.
    """
    import os
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        ),
    )

    prompt = EVALUATION_PROMPT.format(tender_text=tender_text[:6000])
    # Truncate at 6000 chars (~1500 tokens) to keep costs low.

    response = client.chat.completions.create(
        model=HAIKU_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = (response.choices[0].message.content or "").strip()
    tokens_used = (
        (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)
        if response.usage else 0
    )

    # Parse JSON response — handle possible markdown code fences
    clean_text = raw_text.strip()
    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text)
        clean_text = re.sub(r"\s*```$", "", clean_text)

    try:
        parsed = json.loads(clean_text)
    except json.JSONDecodeError as exc:
        logger.error("eval_json_parse_failed", raw_response=raw_text[:200])
        raise RuntimeError(f"Failed to parse evaluation JSON: {exc}") from exc

    scores = parsed.get("scores", {})
    reasoning = parsed.get("reasoning", "No reasoning provided.")

    # Validate scores are within bounds
    validated_scores: dict[str, int] = {}
    for dim, max_val in _MAX_SCORES.items():
        raw_score = scores.get(dim, 0)
        validated_scores[dim] = max(0, min(max_val, int(raw_score)))

    return validated_scores, reasoning, tokens_used


# ---------------------------------------------------------------------------
# Main node function
# ---------------------------------------------------------------------------

def evaluate_node(state: TenderState) -> dict:
    """Node 2: EVALUATE — Score tender eligibility on 8 dimensions.

    Uses Claude Haiku 4.5 in production, keyword heuristics in dry-run mode.

    Input state fields:
        - tender_id, tender_title, tender_raw_text

    Output state fields:
        - eval_score, eval_breakdown, eval_decision, eval_reasoning
        - status, current_node, audit_log
    """
    tender_id = state.get("tender_id", "unknown")
    tender_text = state.get("tender_raw_text", "")

    logger.info("node_evaluate_start", tender_id=tender_id)

    dry_run = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")
    tokens_used = 0
    model_used = None

    if dry_run:
        breakdown, reasoning = _dry_run_score(tender_text)
        model_used = f"{HAIKU_MODEL}-dry-run"
    else:
        try:
            breakdown, reasoning, tokens_used = _llm_score(tender_text)
            model_used = HAIKU_MODEL
        except Exception as exc:
            # If LLM fails, fall back to dry-run scoring rather than crashing
            logger.error("eval_llm_failed", error=str(exc), fallback="dry_run")
            breakdown, reasoning = _dry_run_score(tender_text)
            reasoning = f"[LLM FALLBACK] {reasoning}"
            model_used = f"{HAIKU_MODEL}-fallback"

    total_score = sum(breakdown.values())
    decision = "go" if total_score >= ELIGIBILITY_THRESHOLD else "no_go"

    new_status = (
        TenderStatus.ELIGIBLE.value if decision == "go"
        else TenderStatus.REJECTED.value
    )

    logger.info(
        "node_evaluate_complete",
        tender_id=tender_id,
        score=total_score,
        decision=decision,
        dry_run=dry_run,
    )

    return {
        "eval_score": total_score,
        "eval_breakdown": breakdown,
        "eval_decision": decision,
        "eval_reasoning": reasoning,
        "status": new_status,
        "current_node": "evaluate",
        "audit_log": [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": "evaluate",
            "action": "tender_scored",
            "detail": (
                f"Score: {total_score}/100. Decision: {decision.upper()}. "
                f"Breakdown: {json.dumps(breakdown)}. {reasoning}"
            ),
            "model_used": model_used,
            "tokens_used": tokens_used,
        }],
    }