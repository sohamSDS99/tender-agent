"""
Assemble Node — Generates the final submission document with quality checks.

WHAT THIS NODE DOES:
1. Takes all drafted sections from the state
2. Selects the appropriate template (standard/government/simple)
3. Calls TemplateEngine.assemble() to build the full Markdown document
4. Runs TemplateEngine.quality_check() to verify the document
5. Saves the document to disk
6. If quality fails and retry count < 3, increments retry and returns
   (the graph's route_after_assemble sends it back for another try)
7. If quality passes OR max retries reached, moves to submit

TEMPLATE SELECTION LOGIC:
- Source portal is "sam.gov" → "government" template
- Tender text mentions "simple" or "informal" → "simple" template
- Default → "standard" template

RETRY BEHAVIOUR:
The graph's conditional edge after assemble checks quality_check_passed:
- True → proceed to submit
- False + retry < 3 → loop back to assemble
- False + retry >= 3 → proceed to submit anyway (with warnings logged)

On retry, the node doesn't re-draft sections — it just re-runs assembly
with adjusted parameters (e.g., truncating long sections to fit page limits).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import structlog

from src.agent.state import TenderState, TenderStatus
from src.assembly.template_engine import TemplateEngine

logger = structlog.get_logger(__name__)


def _select_template(state: TenderState) -> str:
    """Choose a template based on tender source and content.

    Logic:
    - SAM.gov tenders → "government" (stricter formatting, page limits)
    - Tender text mentions informal/simple/brief → "simple"
    - Everything else → "standard"
    """
    source = state.get("source_portal", "").lower()
    raw_text = state.get("tender_raw_text", "").lower()

    if "sam.gov" in source:
        return "government"

    informal_keywords = ["simple proposal", "informal", "brief response", "short form"]
    if any(kw in raw_text for kw in informal_keywords):
        return "simple"

    return "standard"


def _truncate_long_sections(
    sections: list[dict],
    max_content_chars: int = 3000,
) -> list[dict]:
    """Truncate sections that are excessively long to help meet page limits.

    This is called on retry when the quality check failed due to page limits.
    Rather than re-drafting (expensive), we trim the longest sections.

    Args:
        sections: List of DraftedSection dicts.
        max_content_chars: Maximum chars per section content.

    Returns:
        New list with truncated sections (originals not modified).
    """
    truncated = []
    for section in sections:
        s = dict(section)  # Shallow copy
        content = s.get("content", "")
        if len(content) > max_content_chars:
            s["content"] = content[:max_content_chars].rsplit(" ", 1)[0] + "\n\n[Content truncated to meet page limits.]"
            logger.info(
                "section_truncated",
                section_id=s.get("section_id"),
                original_len=len(content),
                truncated_len=len(s["content"]),
            )
        truncated.append(s)
    return truncated


def assemble_node(state: TenderState) -> dict:
    """Node 6: ASSEMBLE — Generate final submission document with quality checks.

    Input state fields:
        - tender_id, tender_title, submission_deadline
        - source_portal (for template selection)
        - drafted_sections
        - assembly_retry_count

    Output state fields:
        - assembled_document_path
        - quality_check_passed, quality_issues
        - assembly_retry_count (incremented on failure)
        - status, current_node, audit_log
    """
    tender_id = state.get("tender_id", "unknown")
    tender_title = state.get("tender_title", "Untitled")
    deadline = state.get("submission_deadline", "")
    sections = state.get("drafted_sections", [])
    retry_count = state.get("assembly_retry_count", 0)

    logger.info(
        "node_assemble_start",
        tender_id=tender_id,
        sections=len(sections),
        retry=retry_count,
    )

    # Select template
    template_name = _select_template(state)

    # On retry, try to fix page limit issues by truncating
    if retry_count > 0:
        max_chars = 3000 - (retry_count * 500)  # Progressively shorter on each retry
        sections = _truncate_long_sections(sections, max_content_chars=max(1000, max_chars))
        logger.info("retry_truncation_applied", retry=retry_count, max_chars=max_chars)

    # Assemble the document
    engine = TemplateEngine()
    document = engine.assemble(
        sections=sections,
        tender_title=tender_title,
        tender_id=tender_id,
        template_name=template_name,
        submission_deadline=deadline,
    )

    # Run quality checks
    qc = engine.quality_check(
        document=document,
        sections=sections,
        template_name=template_name,
    )

    # Save the document regardless of quality (for debugging)
    safe_id = tender_id.replace("/", "_").replace(" ", "_")
    filename = f"tender_response_{safe_id}_v{retry_count + 1}.md"
    doc_path = engine.save(document, filename=str(engine.output_dir / filename))

    # Build quality issues list for state
    all_issues = qc.issues + [f"[WARNING] {w}" for w in qc.warnings]

    # Determine next action
    if qc.passed:
        status = TenderStatus.ASSEMBLING.value
        action_detail = (
            f"Document assembled and quality checks PASSED. "
            f"Template: {template_name}. "
            f"Stats: {qc.stats['word_count']} words, ~{qc.stats['page_estimate']} pages. "
            f"Saved to: {doc_path}"
        )
    else:
        new_retry = retry_count + 1
        status = TenderStatus.QUALITY_FAILED.value
        action_detail = (
            f"Quality check FAILED (attempt {new_retry}/3). "
            f"Issues: {'; '.join(qc.issues)}. "
            f"Template: {template_name}. "
            f"Saved draft to: {doc_path}"
        )

    logger.info(
        "node_assemble_complete",
        tender_id=tender_id,
        quality_passed=qc.passed,
        issues=len(qc.issues),
        warnings=len(qc.warnings),
        retry=retry_count,
        template=template_name,
        path=doc_path,
    )

    result = {
        "assembled_document_path": doc_path,
        "quality_check_passed": qc.passed,
        "quality_issues": all_issues,
        "status": status,
        "current_node": "assemble",
        "audit_log": [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": "assemble",
            "action": "document_assembled",
            "detail": action_detail,
            "model_used": None,
            "tokens_used": None,
        }],
    }

    # Only increment retry count on failure
    if not qc.passed:
        result["assembly_retry_count"] = retry_count + 1

    return result