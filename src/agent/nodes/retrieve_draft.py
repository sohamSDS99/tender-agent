"""
Retrieve & Draft Node — RAG-grounded section drafting with multi-model routing.

THIS IS THE CORE VALUE NODE:
Everything else in the pipeline (discovery, evaluation, assembly, submission) is
plumbing. This node is where the actual tender response gets written. Its quality
directly determines whether the company wins or loses bids.

HOW IT WORKS (3 PHASES):

Phase 1 — REQUIREMENT EXTRACTION:
    If `tender_requirements` is empty, the node first parses the raw tender text
    into structured requirements. In production, Sonnet 4.6 does this with a prompt
    that identifies each section/question the tender asks for. In dry-run mode,
    a regex-based heuristic splits the text into logical sections.

Phase 2 — RAG RETRIEVAL:
    For each requirement, the KnowledgeRetriever (Step 7) searches pgvector for the
    most relevant company data chunks. The retriever returns ranked results with
    similarity scores. These chunks become the "context" the drafting LLM uses.

    If the database has no embedded chunks (e.g., during early testing), the node
    generates mock context so the rest of the pipeline still works.

Phase 3 — SECTION DRAFTING:
    For each requirement + its retrieved context, an LLM drafts the response section.

    MULTI-MODEL ROUTING:
    - Standard sections → Claude Sonnet 4.6 (~$0.01/section)
      Used for: company overview, team bios, project approach, timeline
    - Compliance-critical sections → Claude Opus 4.6 (~$0.05/section)
      Used for: regulatory compliance, certifications, legal terms, data security
    
    WHY NOT USE OPUS FOR EVERYTHING:
    At ~5x the cost of Sonnet, using Opus for all 15-20 sections of a tender would
    cost $0.75-$1.00 per tender instead of $0.15-$0.20. Over 50 tenders/month,
    that's $50 vs $10. Opus is reserved for sections where mistakes have legal or
    compliance consequences.

RE-DRAFTING AFTER SLACK ESCALATION:
    If this node runs after a Slack escalation (escalation_count > 0), it incorporates
    the human's Slack responses into the context. Only the sections that had gaps get
    re-drafted — already-good sections are preserved to save time and money.
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
    TenderRequirement,
    TenderState,
    TenderStatus,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SONNET_MODEL: str = "claude-sonnet-4-6"
OPUS_MODEL: str = "claude-opus-4-6"

# Keywords that trigger Opus routing for compliance-critical sections
COMPLIANCE_KEYWORDS: set[str] = {
    "compliance", "regulatory", "osha", "hcs", "whmis", "clp", "reach",
    "certification", "iso 27001", "soc 2", "fedramp", "legal", "liability",
    "indemnification", "data security", "data privacy", "gdpr", "hipaa",
    "encryption", "audit", "tier ii", "epcra", "cercla", "prop 65",
}

# The prompt template for extracting requirements from tender text
REQUIREMENT_EXTRACTION_PROMPT: str = """You are an expert tender analyst. Extract all distinct requirements from this tender document.

For each requirement, identify:
- section_id: The section number (e.g., "3.1", "4.2") or generate sequential IDs
- section_title: A short descriptive title
- requirement_text: What the tender is asking for
- is_mandatory: Whether this is a must-have (true) or nice-to-have (false)

TENDER TEXT:
{tender_text}

Respond with ONLY valid JSON (no markdown, no backticks):
{{
    "requirements": [
        {{
            "section_id": "1.0",
            "section_title": "...",
            "requirement_text": "...",
            "is_mandatory": true
        }}
    ]
}}"""

# The prompt template for drafting a single section
SECTION_DRAFT_PROMPT: str = """You are an expert tender response writer for a B2B SaaS company that provides Safety Data Sheet (SDS) management software, GHS classification tools, and EHS compliance services.

Draft a professional, specific, and persuasive response for the following tender requirement. Use ONLY the company information provided in the context below. Do NOT make up facts, certifications, or capabilities that are not in the context.

If the context does not contain enough information to fully answer the requirement, write what you can and mark the gaps with [INFORMATION NEEDED: description of what's missing].

TENDER REQUIREMENT:
Section: {section_id} — {section_title}
{requirement_text}

COMPANY INFORMATION (from knowledge base):
{rag_context}

{slack_context}

INSTRUCTIONS:
- Be specific and factual — cite actual capabilities, numbers, and certifications from the context
- Use a professional but confident tone
- Keep the response focused and relevant to what was asked
- If a word limit is specified, respect it
- Structure the response with clear paragraphs (no bullet points unless the tender asks for them)

Write the response section now:"""


# ---------------------------------------------------------------------------
# Compliance routing helper
# ---------------------------------------------------------------------------

def _is_compliance_critical(requirement: TenderRequirement) -> bool:
    """Determine if a requirement needs Opus 4.6 (compliance-critical).

    Scans the requirement text for compliance-related keywords. If any
    match, the section is routed to Opus for more careful reasoning.

    This is deliberately conservative — it's better to send a borderline
    section to Opus (costs extra $0.04) than to let Sonnet make a
    compliance mistake that disqualifies the bid.
    """
    text_lower = (
        requirement.get("requirement_text", "").lower()
        + " "
        + requirement.get("section_title", "").lower()
    )
    return any(keyword in text_lower for keyword in COMPLIANCE_KEYWORDS)


# ---------------------------------------------------------------------------
# Dry-run helpers
# ---------------------------------------------------------------------------

def _extract_requirements_dry_run(tender_text: str) -> list[TenderRequirement]:
    """Extract requirements using regex heuristics (no LLM needed).

    Looks for common tender section patterns:
    - Numbered sections: "3.1 Technical Requirements"
    - Question patterns: "Describe your...", "Provide details of..."
    - Requirement keywords: "must", "shall", "required"

    Falls back to splitting the text into ~3 generic sections if no
    patterns are found.
    """
    requirements: list[TenderRequirement] = []

    # Try to find numbered sections
    section_pattern = re.compile(
        r"(?:^|\n)\s*(\d+\.?\d*)\s*[.:\-—]\s*(.+?)(?:\n|$)",
        re.MULTILINE,
    )
    matches = section_pattern.findall(tender_text)

    if matches:
        for section_id, title in matches:
            # Try to capture the text following this heading until the next heading
            title = title.strip()
            requirements.append({
                "section_id": section_id,
                "section_title": title,
                "requirement_text": title,
                "is_mandatory": True,
            })
    
    # If no numbered sections found, create generic ones from the text
    if not requirements:
        # Split into logical chunks and create requirements
        sentences = [s.strip() for s in tender_text.split(".") if len(s.strip()) > 30]
        
        generic_sections = [
            ("1.0", "Company Overview", "Provide an overview of your company, including experience and qualifications."),
            ("2.0", "Technical Capabilities", "Describe the technical capabilities of your proposed solution."),
            ("3.0", "Compliance & Certifications", "Detail your regulatory compliance and relevant certifications."),
            ("4.0", "Implementation Approach", "Describe your implementation methodology and timeline."),
            ("5.0", "Pricing", "Provide pricing details for the proposed solution."),
        ]

        for section_id, title, default_text in generic_sections:
            # Check if the tender text mentions this topic
            topic_keywords = title.lower().split()
            is_relevant = any(kw in tender_text.lower() for kw in topic_keywords)
            
            requirements.append({
                "section_id": section_id,
                "section_title": title,
                "requirement_text": default_text,
                "is_mandatory": is_relevant,
            })

    return requirements


def _draft_section_dry_run(
    requirement: TenderRequirement,
    context: str,
    is_compliance: bool,
) -> DraftedSection:
    """Generate a template-based mock draft (no LLM needed).

    Creates a realistic-looking section that references the requirement
    and indicates what model would be used in production.
    """
    section_id = requirement.get("section_id", "0.0")
    section_title = requirement.get("section_title", "Untitled")
    req_text = requirement.get("requirement_text", "")
    model = OPUS_MODEL if is_compliance else SONNET_MODEL

    content = (
        f"[DRY-RUN DRAFT — would use {model} in production]\n\n"
        f"Acme SDS Solutions is well-positioned to address this requirement. "
        f"In response to \"{section_title}\": our cloud-based SDS management "
        f"platform provides comprehensive capabilities aligned with the stated needs.\n\n"
        f"Our platform serves over 500 clients across manufacturing, construction, "
        f"oil & gas, and pharmaceutical industries, with demonstrated expertise in "
        f"the areas described in this section.\n\n"
        f"[In production, this section would be drafted using RAG-retrieved company "
        f"data relevant to: {req_text[:150]}]"
    )

    return {
        "section_id": section_id,
        "section_title": section_title,
        "content": content,
        "confidence": 0.75 if is_compliance else 0.85,
        "sources_used": ["[dry-run: no real RAG retrieval]"],
        "model_used": f"{model}-dry-run",
        "token_count": len(content) // 4,
    }


# ---------------------------------------------------------------------------
# Real LLM helpers
# ---------------------------------------------------------------------------

def _extract_requirements_llm(tender_text: str) -> list[TenderRequirement]:
    """Extract requirements using Claude Sonnet 4.6."""
    import anthropic

    client = anthropic.Anthropic()
    prompt = REQUIREMENT_EXTRACTION_PROMPT.format(
        tender_text=tender_text[:8000]
    )

    response = client.messages.create(
        model=SONNET_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text.strip()
    clean_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    clean_text = re.sub(r"\s*```$", "", clean_text)

    try:
        parsed = json.loads(clean_text)
        return parsed.get("requirements", [])
    except json.JSONDecodeError:
        logger.error("requirement_extraction_json_failed", raw=raw_text[:200])
        # Fall back to dry-run extraction
        return _extract_requirements_dry_run(tender_text)


def _draft_section_llm(
    requirement: TenderRequirement,
    context: str,
    slack_context: str,
    is_compliance: bool,
) -> tuple[DraftedSection, int]:
    """Draft a section using Claude Sonnet 4.6 or Opus 4.6.

    Returns:
        Tuple of (DraftedSection, tokens_used)
    """
    import anthropic

    client = anthropic.Anthropic()
    model = OPUS_MODEL if is_compliance else SONNET_MODEL

    prompt = SECTION_DRAFT_PROMPT.format(
        section_id=requirement.get("section_id", ""),
        section_title=requirement.get("section_title", ""),
        requirement_text=requirement.get("requirement_text", ""),
        rag_context=context or "[No relevant information found in the knowledge base.]",
        slack_context=slack_context,
    )

    response = client.messages.create(
        model=model,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.content[0].text.strip()
    tokens = response.usage.input_tokens + response.usage.output_tokens

    # Estimate confidence based on context quality
    has_info_gaps = "[INFORMATION NEEDED:" in content
    confidence = 0.5 if has_info_gaps else (0.8 if is_compliance else 0.9)

    return {
        "section_id": requirement.get("section_id", ""),
        "section_title": requirement.get("section_title", ""),
        "content": content,
        "confidence": confidence,
        "sources_used": ["[from RAG retrieval]"],
        "model_used": model,
        "token_count": tokens,
    }, tokens


# ---------------------------------------------------------------------------
# RAG retrieval helper
# ---------------------------------------------------------------------------

def _retrieve_context(requirement_text: str, session: Any = None) -> str:
    """Retrieve relevant context from the knowledge base.

    If a database session is available and has embedded chunks, uses
    the real KnowledgeRetriever. Otherwise returns a placeholder.
    """
    if session is None:
        return "[No database session — RAG retrieval skipped in dry-run mode.]"

    try:
        from src.knowledge.retriever import KnowledgeRetriever

        retriever = KnowledgeRetriever()
        results = retriever.search(
            query=requirement_text,
            session=session,
            top_k=5,
        )

        if not results:
            return "[No relevant chunks found in the knowledge base.]"

        return retriever.format_context(results)

    except Exception as exc:
        logger.warning("rag_retrieval_failed", error=str(exc))
        return f"[RAG retrieval failed: {exc}]"


# ---------------------------------------------------------------------------
# Main node function
# ---------------------------------------------------------------------------

def retrieve_draft_node(state: TenderState) -> dict:
    """Node 3: RETRIEVE & DRAFT — RAG retrieval + multi-model section drafting.

    Input state fields:
        - tender_id, tender_raw_text, tender_requirements (optional)
        - slack_responses (if re-drafting after escalation)
        - escalation_count

    Output state fields:
        - tender_requirements (if extracted)
        - drafted_sections
        - draft_rag_context
        - status, current_node, audit_log
    """
    tender_id = state.get("tender_id", "unknown")
    tender_text = state.get("tender_raw_text", "")
    escalation_count = state.get("escalation_count", 0)
    is_redraft = escalation_count > 0

    logger.info(
        "node_retrieve_draft_start",
        tender_id=tender_id,
        is_redraft=is_redraft,
        escalation_count=escalation_count,
    )

    dry_run = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")
    total_tokens = 0

    # ------------------------------------------------------------------
    # Phase 1: Extract requirements (if not already done)
    # ------------------------------------------------------------------
    requirements = state.get("tender_requirements") or []

    if not requirements:
        logger.info("extracting_requirements", method="dry_run" if dry_run else "llm")

        if dry_run:
            requirements = _extract_requirements_dry_run(tender_text)
        else:
            try:
                requirements = _extract_requirements_llm(tender_text)
            except Exception as exc:
                logger.error("requirement_extraction_failed", error=str(exc))
                requirements = _extract_requirements_dry_run(tender_text)

    logger.info("requirements_extracted", count=len(requirements))

    # ------------------------------------------------------------------
    # Phase 2 & 3: Retrieve context + Draft each section
    # ------------------------------------------------------------------

    # Build Slack context string (for re-drafts)
    slack_context = ""
    if is_redraft:
        slack_responses = state.get("slack_responses", [])
        if slack_responses:
            slack_context = (
                "ADDITIONAL INFORMATION FROM TEAM (via Slack):\n"
                + "\n".join(f"- {resp}" for resp in slack_responses)
            )

    # Get existing drafted sections (for re-draft: only redo the gaps)
    existing_sections = state.get("drafted_sections", [])
    existing_by_id = {s.get("section_id"): s for s in existing_sections}
    gap_section_ids = {g.get("section_id") for g in state.get("gaps", [])}

    drafted_sections: list[DraftedSection] = []
    all_rag_context_parts: list[str] = []

    for req in requirements:
        section_id = req.get("section_id", "")

        # If re-drafting, only redo sections that had gaps
        if is_redraft and section_id not in gap_section_ids and section_id in existing_by_id:
            drafted_sections.append(existing_by_id[section_id])
            logger.debug("section_preserved", section_id=section_id)
            continue

        is_compliance = _is_compliance_critical(req)
        req_text = req.get("requirement_text", "")

        # Phase 2: RAG retrieval
        rag_context = _retrieve_context(req_text)
        all_rag_context_parts.append(
            f"[Section {section_id}]\n{rag_context}"
        )

        # Phase 3: Draft the section
        if dry_run:
            section = _draft_section_dry_run(req, rag_context, is_compliance)
        else:
            try:
                section, tokens = _draft_section_llm(
                    req, rag_context, slack_context, is_compliance
                )
                total_tokens += tokens
            except Exception as exc:
                logger.error("section_draft_failed", section_id=section_id, error=str(exc))
                section = _draft_section_dry_run(req, rag_context, is_compliance)
                section["content"] = f"[LLM FAILED — fallback draft]\n{section['content']}"

        drafted_sections.append(section)

        model_label = "Opus" if is_compliance else "Sonnet"
        logger.info(
            "section_drafted",
            section_id=section_id,
            title=req.get("section_title", ""),
            model=model_label,
            compliance_critical=is_compliance,
            confidence=section.get("confidence", 0),
        )

    # ------------------------------------------------------------------
    # Compile results
    # ------------------------------------------------------------------
    combined_rag_context = "\n\n".join(all_rag_context_parts)
    action = "sections_redrafted" if is_redraft else "sections_drafted"

    logger.info(
        "node_retrieve_draft_complete",
        tender_id=tender_id,
        sections_drafted=len(drafted_sections),
        total_tokens=total_tokens,
        is_redraft=is_redraft,
    )

    return {
        "tender_requirements": requirements,
        "drafted_sections": drafted_sections,
        "draft_rag_context": combined_rag_context,
        "status": TenderStatus.DRAFTING.value,
        "current_node": "retrieve_draft",
        "audit_log": [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": "retrieve_draft",
            "action": action,
            "detail": (
                f"{'Re-drafted' if is_redraft else 'Drafted'} {len(drafted_sections)} sections. "
                f"Tokens used: {total_tokens}. "
                f"Compliance sections routed to Opus: "
                f"{sum(1 for s in drafted_sections if 'opus' in s.get('model_used', '').lower())}."
            ),
            "model_used": f"{SONNET_MODEL}+{OPUS_MODEL}",
            "tokens_used": total_tokens,
        }],
    }