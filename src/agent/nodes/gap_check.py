"""
Gap Check Node — Verifies draft completeness against tender requirements.

WHY THIS NODE EXISTS:
The tender agent's #1 rule is: NEVER submit incomplete or inaccurate tenders.
The drafting node does its best with whatever's in the knowledge base, but it
can't know if it missed something — it doesn't know what it doesn't know.

This node catches those blind spots by checking:
1. LOW CONFIDENCE — Did the drafting LLM flag uncertainty? (confidence < 0.7)
2. PLACEHOLDER MARKERS — Did the draft contain "[INFORMATION NEEDED: ...]"?
3. SHORT SECTIONS — Is a mandatory section suspiciously brief? (<100 chars)
4. MISSING SECTIONS — Did a mandatory requirement get skipped entirely?
5. STALE DATA — Does the draft reference outdated years or expired certs?

For each gap found, the node generates a SPECIFIC, ACTIONABLE question that
gets sent to Slack. "What certifications do you have?" is a bad question.
"What is the expiry date of our ISO 27001 certification?" is a good one.

IN PRODUCTION (with real API keys):
The node sends the draft + requirements to Claude Sonnet 4.6 for deeper analysis.
The LLM can catch subtle gaps that heuristics miss — like a requirement asking for
"5 years of experience" when the draft only mentions "extensive experience" without
a specific number.

IN DRY-RUN MODE:
Uses the heuristic checks listed above. These catch the most common gap patterns
and produce realistic results for testing the full pipeline.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

import structlog

from src.agent.state import (
    DraftedSection,
    GapItem,
    TenderRequirement,
    TenderState,
    TenderStatus,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SONNET_MODEL: str = "claude-sonnet-4-6"

# Thresholds for heuristic gap detection
CONFIDENCE_THRESHOLD: float = 0.7      # Below this = flag as gap
MIN_SECTION_LENGTH: int = 100           # Chars — shorter = suspicious
CURRENT_YEAR: int = datetime.now().year

# Regex for detecting placeholder markers left by the drafting LLM
_INFO_NEEDED_RE = re.compile(
    r"\[INFORMATION NEEDED[:\s]*([^\]]*)\]",
    re.IGNORECASE,
)

# Regex for detecting stale year references
_YEAR_RE = re.compile(r"\b(20[0-2]\d)\b")

# The LLM prompt for deep gap analysis
GAP_ANALYSIS_PROMPT: str = """You are a tender quality assurance specialist. Compare each drafted section against its requirement and identify any gaps.

A "gap" is:
- Missing information that the requirement explicitly asks for
- Vague language where specifics are needed (e.g., "extensive experience" instead of "12 years")
- Claims not supported by the provided company data
- Mandatory requirements not fully addressed

TENDER REQUIREMENTS AND DRAFTS:
{sections_json}

For each gap found, provide:
- section_id: Which section has the gap
- description: What specific information is missing
- severity: "high" (mandatory info missing), "medium" (important but not critical), "low" (nice to have)
- suggested_question: A specific question to ask the team via Slack

Respond with ONLY valid JSON (no markdown, no backticks):
{{
    "gaps": [
        {{
            "section_id": "...",
            "description": "...",
            "severity": "high|medium|low",
            "suggested_question": "..."
        }}
    ],
    "summary": "Brief overall assessment"
}}

If there are NO gaps, return: {{"gaps": [], "summary": "All sections adequately address requirements."}}"""


# ---------------------------------------------------------------------------
# Heuristic gap detection (dry-run mode)
# ---------------------------------------------------------------------------

def _check_gaps_heuristic(
    requirements: list[TenderRequirement],
    sections: list[DraftedSection],
) -> list[GapItem]:
    """Detect gaps using rule-based heuristics (no LLM needed).

    Checks each section for common gap indicators:
    1. Low confidence score from the drafting model
    2. [INFORMATION NEEDED] placeholder markers
    3. Suspiciously short content for mandatory sections
    4. Missing sections (requirement exists but no corresponding draft)
    5. References to outdated years

    Returns:
        List of GapItem dicts describing each gap found.
    """
    gaps: list[GapItem] = []

    # Build a lookup: section_id → drafted section
    section_by_id: dict[str, DraftedSection] = {
        s.get("section_id", ""): s for s in sections
    }

    for req in requirements:
        section_id = req.get("section_id", "")
        section_title = req.get("section_title", "Untitled")
        is_mandatory = req.get("is_mandatory", True)
        draft = section_by_id.get(section_id)

        # --- Check 1: Missing section entirely ---
        if draft is None:
            if is_mandatory:
                gaps.append({
                    "section_id": section_id,
                    "description": f"Mandatory section '{section_title}' has no draft.",
                    "severity": "high",
                    "suggested_question": (
                        f"We need to draft section '{section_title}' for this tender. "
                        f"Can you provide the relevant information?"
                    ),
                })
            continue

        content = draft.get("content", "")
        confidence = draft.get("confidence", 1.0)

        # --- Check 2: Low confidence ---
        if confidence < CONFIDENCE_THRESHOLD:
            gaps.append({
                "section_id": section_id,
                "description": (
                    f"Section '{section_title}' has low confidence ({confidence:.0%}). "
                    f"The drafting model was uncertain about the content."
                ),
                "severity": "medium" if confidence >= 0.5 else "high",
                "suggested_question": (
                    f"The auto-drafted response for '{section_title}' has low confidence. "
                    f"Can you review and provide additional details for this section?"
                ),
            })

        # --- Check 3: Placeholder markers ---
        info_needed_matches = _INFO_NEEDED_RE.findall(content)
        for match in info_needed_matches:
            description = match.strip() if match.strip() else "unspecified information"
            gaps.append({
                "section_id": section_id,
                "description": (
                    f"Section '{section_title}' contains a placeholder: "
                    f"information needed for '{description}'."
                ),
                "severity": "high" if is_mandatory else "medium",
                "suggested_question": (
                    f"For the '{section_title}' section: {description}?"
                ),
            })

        # --- Check 4: Suspiciously short mandatory section ---
        if is_mandatory and len(content.strip()) < MIN_SECTION_LENGTH:
            gaps.append({
                "section_id": section_id,
                "description": (
                    f"Mandatory section '{section_title}' is very short "
                    f"({len(content)} chars). May be incomplete."
                ),
                "severity": "medium",
                "suggested_question": (
                    f"The '{section_title}' section seems incomplete. "
                    f"Can you provide more detail about: {req.get('requirement_text', section_title)}?"
                ),
            })

        # --- Check 5: Stale year references ---
        years_mentioned = [int(y) for y in _YEAR_RE.findall(content)]
        stale_years = [y for y in years_mentioned if y < CURRENT_YEAR - 2]
        if stale_years:
            gaps.append({
                "section_id": section_id,
                "description": (
                    f"Section '{section_title}' references potentially outdated "
                    f"year(s): {', '.join(str(y) for y in stale_years)}. "
                    f"Data may need refreshing."
                ),
                "severity": "low",
                "suggested_question": (
                    f"Section '{section_title}' mentions {', '.join(str(y) for y in stale_years)}. "
                    f"Are these dates still current, or do we have updated figures?"
                ),
            })

    return gaps


# ---------------------------------------------------------------------------
# LLM gap detection (production mode)
# ---------------------------------------------------------------------------

def _check_gaps_llm(
    requirements: list[TenderRequirement],
    sections: list[DraftedSection],
) -> tuple[list[GapItem], int]:
    """Detect gaps using Claude Sonnet 4.6 for deeper analysis.

    Returns:
        Tuple of (gaps_list, tokens_used)
    """
    import anthropic

    # Build a combined JSON of requirements + drafts for the LLM
    combined = []
    section_by_id = {s.get("section_id", ""): s for s in sections}

    for req in requirements:
        sid = req.get("section_id", "")
        draft = section_by_id.get(sid, {})
        combined.append({
            "section_id": sid,
            "section_title": req.get("section_title", ""),
            "requirement_text": req.get("requirement_text", ""),
            "is_mandatory": req.get("is_mandatory", True),
            "draft_content": draft.get("content", "[NO DRAFT]"),
            "draft_confidence": draft.get("confidence", 0),
        })

    prompt = GAP_ANALYSIS_PROMPT.format(
        sections_json=json.dumps(combined, indent=2)[:6000]
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=SONNET_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text.strip()
    tokens = response.usage.input_tokens + response.usage.output_tokens

    clean_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    clean_text = re.sub(r"\s*```$", "", clean_text)

    try:
        parsed = json.loads(clean_text)
        return parsed.get("gaps", []), tokens
    except json.JSONDecodeError:
        logger.error("gap_check_json_failed", raw=raw_text[:200])
        # Fall back to heuristic
        return _check_gaps_heuristic(requirements, sections), 0


# ---------------------------------------------------------------------------
# Main node function
# ---------------------------------------------------------------------------

def gap_check_node(state: TenderState) -> dict:
    """Node 4: GAP CHECK — Verify draft completeness against requirements.

    Input state fields:
        - tender_id, tender_requirements, drafted_sections

    Output state fields:
        - gaps, gap_check_passed
        - status, current_node, audit_log
    """
    tender_id = state.get("tender_id", "unknown")
    requirements = state.get("tender_requirements", [])
    sections = state.get("drafted_sections", [])

    logger.info(
        "node_gap_check_start",
        tender_id=tender_id,
        requirements_count=len(requirements),
        sections_count=len(sections),
    )

    dry_run = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")
    tokens_used = 0
    model_used = None

    if dry_run:
        gaps = _check_gaps_heuristic(requirements, sections)
        model_used = f"{SONNET_MODEL}-dry-run"
    else:
        try:
            gaps, tokens_used = _check_gaps_llm(requirements, sections)
            model_used = SONNET_MODEL
        except Exception as exc:
            logger.error("gap_check_llm_failed", error=str(exc))
            gaps = _check_gaps_heuristic(requirements, sections)
            model_used = f"{SONNET_MODEL}-fallback"

    passed = len(gaps) == 0

    # Categorize gaps by severity
    high_gaps = [g for g in gaps if g.get("severity") == "high"]
    medium_gaps = [g for g in gaps if g.get("severity") == "medium"]
    low_gaps = [g for g in gaps if g.get("severity") == "low"]

    logger.info(
        "node_gap_check_complete",
        tender_id=tender_id,
        passed=passed,
        total_gaps=len(gaps),
        high=len(high_gaps),
        medium=len(medium_gaps),
        low=len(low_gaps),
    )

    return {
        "gaps": gaps,
        "gap_check_passed": passed,
        "status": TenderStatus.GAP_CHECK.value,
        "current_node": "gap_check",
        "audit_log": [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": "gap_check",
            "action": "gap_analysis_complete",
            "detail": (
                f"{'No gaps found — ready for assembly.' if passed else ''}"
                f"{'Found ' + str(len(gaps)) + ' gaps: ' if not passed else ''}"
                f"{str(len(high_gaps)) + ' high, ' if high_gaps else ''}"
                f"{str(len(medium_gaps)) + ' medium, ' if medium_gaps else ''}"
                f"{str(len(low_gaps)) + ' low.' if low_gaps else ''}"
            ).strip(),
            "model_used": model_used,
            "tokens_used": tokens_used,
        }],
    }