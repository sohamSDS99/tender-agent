"""
Tender Agent Graph — The LangGraph state machine that orchestrates the 7-node pipeline.

WHY LANGGRAPH (NOT JUST FUNCTIONS):
You could chain 7 functions in a for loop. But that gives you:
- No checkpointing (crash = start over from scratch)
- No conditional routing (what if gap check fails?)
- No wait-and-resume (Slack escalation blocks until a human responds)
- No audit trail of which node ran when

LangGraph's StateGraph gives us all of these. Each node is a function that receives
the full state and returns updates. The graph defines the execution order and
branching logic. The PostgreSQL checkpointer saves state after every node, so if
the process crashes mid-pipeline, it resumes from the last completed node.

THE 7-NODE FLOW:
    DISCOVER → EVALUATE → (go/no_go?) → RETRIEVE_DRAFT → GAP_CHECK
                              ↓                              ↓
                          (archive)              (gaps found? / no gaps?)
                                                    ↓              ↓
                                            SLACK_ESCALATE     ASSEMBLE
                                                    ↓              ↓
                                          (human responds)    SUBMIT → END
                                                    ↓
                                          RETRIEVE_DRAFT (re-draft)

PLACEHOLDER NODES:
Each node in this file is a placeholder that logs what it would do and passes
through the state unchanged (except for status updates). Steps 9-22 will replace
these placeholders with real implementations, one at a time.

This approach lets us:
1. Verify the graph compiles and routes correctly before building any real logic
2. Run end-to-end dry runs to see the full flow
3. Build and test each node independently
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from langgraph.graph import END, START, StateGraph

from src.agent.state import TenderState, TenderStatus

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper: Create an audit entry
# ---------------------------------------------------------------------------

def _audit(node: str, action: str, detail: str) -> dict:
    """Create a standardised audit entry dict.

    Every node should call this to log what it did. The returned dict
    is merged into the state's audit_log list via the Annotated[list, operator.add]
    reducer — so you return it as: {"audit_log": [_audit(...)]}
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": node,
        "action": action,
        "detail": detail,
        "model_used": None,
        "tokens_used": None,
    }


# ---------------------------------------------------------------------------
# Placeholder node functions
# ---------------------------------------------------------------------------
# Each function receives the FULL state dict and returns a dict of fields
# to update. LangGraph merges the returned dict into the existing state.
# Fields not returned stay unchanged.
#
# IMPORTANT: For Annotated[list, operator.add] fields (error_messages, audit_log),
# return a LIST of new items — they get APPENDED to the existing list.
# For regular fields, the returned value REPLACES the old one.
# ---------------------------------------------------------------------------


def discover_node(state: TenderState) -> dict:
    """Node 1: DISCOVER — Find new tenders from portals, RSS, and email.

    PLACEHOLDER: Logs and passes through. Real implementation (Step 14) will:
    - Scrape SAM.gov RSS feeds for new tender postings
    - Check MERX and other procurement portals
    - Monitor IMAP email for tender notifications
    - Extract metadata (title, deadline, URL, category)
    - Parse attached tender documents (PDF/DOCX)
    - Store raw tender data in PostgreSQL
    """
    logger.info("node_discover", tender_id=state.get("tender_id", "unknown"))

    return {
        "status": TenderStatus.DISCOVERED.value,
        "current_node": "discover",
        "audit_log": [_audit("discover", "tender_discovered", (
            f"Tender '{state.get('tender_title', 'N/A')}' discovered "
            f"from {state.get('source_portal', 'unknown')}"
        ))],
    }


def evaluate_node(state: TenderState) -> dict:
    """Node 2: EVALUATE — Score tender eligibility on 8 dimensions.

    PLACEHOLDER: Returns a mock score of 75 (eligible). Real implementation
    (Step 9) will:
    - Send tender text to Claude Haiku 4.5 for fast classification
    - Score 8 dimensions: geography, budget, scope, compliance,
      timeline, certifications, domain match, competition level
    - Each dimension scored 0-15, total 0-100
    - Score ≥ 60 advances to drafting; below 60 is archived
    """
    logger.info("node_evaluate", tender_id=state.get("tender_id", "unknown"))

    mock_breakdown = {
        "geography": 10,
        "budget": 8,
        "scope": 9,
        "compliance": 10,
        "timeline": 8,
        "certifications": 8,
        "domain_match": 12,
        "competition": 10,
    }
    mock_score = sum(mock_breakdown.values())

    return {
        "eval_score": mock_score,
        "eval_breakdown": mock_breakdown,
        "eval_decision": "go" if mock_score >= 60 else "no_go",
        "eval_reasoning": f"Placeholder evaluation. Score: {mock_score}/100.",
        "status": TenderStatus.ELIGIBLE.value if mock_score >= 60 else TenderStatus.REJECTED.value,
        "current_node": "evaluate",
        "audit_log": [_audit("evaluate", "tender_scored", (
            f"Score: {mock_score}/100. Decision: {'GO' if mock_score >= 60 else 'NO-GO'}."
        ))],
    }


def retrieve_draft_node(state: TenderState) -> dict:
    """Node 3: RETRIEVE & DRAFT — RAG retrieval + LLM section drafting.

    PLACEHOLDER: Creates mock drafted sections. Real implementation (Step 10) will:
    - Decompose tender requirements into sub-queries
    - Search pgvector knowledge base using KnowledgeRetriever
    - Route to Sonnet 4.6 for standard sections
    - Route to Opus 4.6 Advisor for compliance-critical sections
    - Draft each section with RAG-grounded content
    - Track confidence scores and sources used
    """
    logger.info("node_retrieve_draft", tender_id=state.get("tender_id", "unknown"))

    # Check if we're re-drafting after Slack escalation
    escalation_count = state.get("escalation_count", 0)
    is_redraft = escalation_count > 0

    mock_sections = [
        {
            "section_id": "1.0",
            "section_title": "Company Overview",
            "content": "Acme SDS Solutions is a leading provider of SDS management software...",
            "confidence": 0.90,
            "sources_used": ["company_profile.pdf > Overview"],
            "model_used": "claude-sonnet-4-6",
            "token_count": 250,
        },
        {
            "section_id": "2.0",
            "section_title": "Technical Capabilities",
            "content": "Our platform provides GHS classification, label generation...",
            "confidence": 0.85,
            "sources_used": ["company_profile.pdf > Capabilities"],
            "model_used": "claude-sonnet-4-6",
            "token_count": 400,
        },
    ]

    detail = "Re-drafted sections with Slack responses" if is_redraft else "Drafted 2 sections"

    return {
        "drafted_sections": mock_sections,
        "draft_rag_context": "[Placeholder RAG context]",
        "status": TenderStatus.DRAFTING.value,
        "current_node": "retrieve_draft",
        "audit_log": [_audit("retrieve_draft", "sections_drafted", detail)],
    }


def gap_check_node(state: TenderState) -> dict:
    """Node 4: GAP CHECK — Verify draft completeness against requirements.

    PLACEHOLDER: Returns no gaps (pass). Real implementation (Step 11) will:
    - Compare each drafted section against its tender requirement
    - Check for missing data, outdated information, low confidence scores
    - Verify all mandatory sections have content
    - Check for fabricated/hallucinated claims not in the knowledge base
    - Generate specific questions for any gaps found
    """
    logger.info("node_gap_check", tender_id=state.get("tender_id", "unknown"))

    # Placeholder: no gaps found (clean pass)
    # When we build the real implementation, this will actually analyse the draft
    mock_gaps: list = []
    passed = len(mock_gaps) == 0

    return {
        "gaps": mock_gaps,
        "gap_check_passed": passed,
        "status": TenderStatus.GAP_CHECK.value,
        "current_node": "gap_check",
        "audit_log": [_audit("gap_check", "gap_analysis_complete", (
            f"{'No gaps found — ready for assembly.' if passed else f'{len(mock_gaps)} gaps identified.'}"
        ))],
    }


def slack_escalate_node(state: TenderState) -> dict:
    """Node 5: SLACK ESCALATE — Send questions to team, wait for response.

    PLACEHOLDER: Simulates immediate Slack response. Real implementation
    (Step 16) will:
    - Format gap items into clear Slack messages
    - Send to designated channel via Slack Bolt SDK
    - Enter checkpointed wait state (graph pauses)
    - Resume when human responds (via Slack event handler)
    - Handle 48h timeout with escalation to manager
    - Incorporate responses into state for re-drafting
    """
    logger.info("node_slack_escalate", tender_id=state.get("tender_id", "unknown"))

    gaps = state.get("gaps", [])
    questions = [g.get("suggested_question", "N/A") for g in gaps]
    current_count = state.get("escalation_count", 0)

    return {
        "slack_questions": questions,
        "slack_responses": ["[Mock response: information provided]"] * len(questions),
        "escalation_count": current_count + 1,
        "status": TenderStatus.AWAITING_HUMAN.value,
        "current_node": "slack_escalate",
        "audit_log": [_audit("slack_escalate", "questions_sent", (
            f"Sent {len(questions)} questions to Slack. Escalation #{current_count + 1}."
        ))],
    }


def assemble_node(state: TenderState) -> dict:
    """Node 6: ASSEMBLE — Generate final submission document.

    PLACEHOLDER: Returns mock success. Real implementation (Step 19) will:
    - Select the correct template (DOCX/PDF) based on tender requirements
    - Insert drafted sections into template with proper formatting
    - Generate table of contents, page numbers, headers/footers
    - Run quality checks: page limits, section ordering, attachments
    - Retry up to 3 times on quality failures
    - Save final document to filesystem and S3
    """
    logger.info("node_assemble", tender_id=state.get("tender_id", "unknown"))

    retry_count = state.get("assembly_retry_count", 0)

    return {
        "assembled_document_path": "/tmp/tender_response_mock.pdf",
        "quality_check_passed": True,
        "quality_issues": [],
        "assembly_retry_count": retry_count,
        "status": TenderStatus.ASSEMBLING.value,
        "current_node": "assemble",
        "audit_log": [_audit("assemble", "document_assembled", (
            "Document assembled and quality checks passed."
        ))],
    }


def submit_node(state: TenderState) -> dict:
    """Node 7: SUBMIT — Dispatch the tender to the procurement portal.

    PLACEHOLDER: Returns mock success. Real implementation (Step 22) will:
    - Determine submission method (portal upload, email, API)
    - Playwright automation for portal uploads (fill forms, upload files)
    - SMTP for email submissions
    - Capture confirmation screenshot and receipt
    - Handle submission failures with retry logic
    """
    logger.info("node_submit", tender_id=state.get("tender_id", "unknown"))

    return {
        "submission_method": "portal_upload",
        "submission_status": "success",
        "submission_confirmation": "MOCK-RECEIPT-001",
        "submission_screenshot_path": "/tmp/submission_screenshot_mock.png",
        "status": TenderStatus.SUBMITTED.value,
        "current_node": "submit",
        "audit_log": [_audit("submit", "tender_submitted", (
            f"Tender submitted via portal_upload. Receipt: MOCK-RECEIPT-001."
        ))],
    }


# ---------------------------------------------------------------------------
# Conditional routing functions
# ---------------------------------------------------------------------------

def route_after_evaluate(state: TenderState) -> str:
    """After evaluation, route based on the go/no-go decision.

    - Score ≥ 60 (go) → proceed to retrieve_draft
    - Score < 60 (no_go) → end the pipeline (tender archived)

    WHY NOT RAISE AN ERROR FOR NO-GO:
    A rejected tender isn't an error — it's a valid outcome. The audit log
    records why it was rejected, and the tender stays in the DB with status
    'rejected' for future reference.
    """
    decision = state.get("eval_decision", "no_go")
    if decision == "go":
        logger.info("route_evaluate", decision="go", next="retrieve_draft")
        return "retrieve_draft"
    else:
        logger.info("route_evaluate", decision="no_go", next="end")
        return "end"


def route_after_gap_check(state: TenderState) -> str:
    """After gap check, route based on whether gaps were found.

    - No gaps → proceed to assemble
    - Gaps found AND escalation count < 3 → escalate to Slack
    - Gaps found AND escalation count ≥ 3 → proceed to assemble anyway
      (we've asked humans 3 times; at some point we must move forward
      with what we have, flagging the gaps in the submission)

    WHY A MAX ESCALATION COUNT:
    Without it, a tender with a persistent gap could loop forever between
    gap_check and slack_escalate. Three rounds of human Q&A is generous —
    if the team can't provide the info after 3 asks, it probably doesn't exist.
    """
    gaps_passed = state.get("gap_check_passed", False)
    escalation_count = state.get("escalation_count", 0)

    if gaps_passed:
        logger.info("route_gap_check", result="no_gaps", next="assemble")
        return "assemble"
    elif escalation_count >= 3:
        logger.info("route_gap_check", result="max_escalations_reached", next="assemble")
        return "assemble"
    else:
        logger.info("route_gap_check", result="gaps_found", next="slack_escalate")
        return "slack_escalate"


def route_after_assemble(state: TenderState) -> str:
    """After assembly, check quality and decide whether to submit or retry.

    - Quality passed → submit
    - Quality failed AND retry < 3 → loop back to assemble
    - Quality failed AND retry ≥ 3 → submit anyway with quality warnings
    """
    quality_passed = state.get("quality_check_passed", False)
    retry_count = state.get("assembly_retry_count", 0)

    if quality_passed:
        logger.info("route_assemble", result="quality_passed", next="submit")
        return "submit"
    elif retry_count >= 3:
        logger.info("route_assemble", result="max_retries_reached", next="submit")
        return "submit"
    else:
        logger.info("route_assemble", result="quality_failed", next="assemble")
        return "assemble"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_tender_graph(checkpointer=None) -> StateGraph:
    """Construct and compile the 7-node tender agent graph.

    Args:
        checkpointer: Optional LangGraph checkpointer for state persistence.
            Pass a PostgresSaver for production, or None for testing.

    Returns:
        A compiled LangGraph StateGraph ready to invoke.

    GRAPH STRUCTURE:
        START → discover → evaluate → (go/no_go?)
                                         ↓ go          ↓ no_go
                                    retrieve_draft      END
                                         ↓
                                    gap_check → (gaps?)
                                      ↓ no gaps    ↓ gaps found
                                    assemble    slack_escalate
                                      ↓              ↓
                                    (quality?)   retrieve_draft (loop back)
                                    ↓ pass
                                    submit → END
    """
    graph = StateGraph(TenderState)

    # --- Add all 7 nodes ---
    graph.add_node("discover", discover_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("retrieve_draft", retrieve_draft_node)
    graph.add_node("gap_check", gap_check_node)
    graph.add_node("slack_escalate", slack_escalate_node)
    graph.add_node("assemble", assemble_node)
    graph.add_node("submit", submit_node)

    # --- Define edges ---

    # START → discover (entry point)
    graph.add_edge(START, "discover")

    # discover → evaluate (always)
    graph.add_edge("discover", "evaluate")

    # evaluate → conditional: retrieve_draft OR end
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "retrieve_draft": "retrieve_draft",
            "end": END,
        },
    )

    # retrieve_draft → gap_check (always)
    graph.add_edge("retrieve_draft", "gap_check")

    # gap_check → conditional: assemble OR slack_escalate
    graph.add_conditional_edges(
        "gap_check",
        route_after_gap_check,
        {
            "assemble": "assemble",
            "slack_escalate": "slack_escalate",
        },
    )

    # slack_escalate → retrieve_draft (loop back for re-drafting)
    graph.add_edge("slack_escalate", "retrieve_draft")

    # assemble → conditional: submit OR assemble (retry loop)
    graph.add_conditional_edges(
        "assemble",
        route_after_assemble,
        {
            "submit": "submit",
            "assemble": "assemble",
        },
    )

    # submit → END
    graph.add_edge("submit", END)

    # --- Compile ---
    compiled = graph.compile(checkpointer=checkpointer)

    logger.info("graph_compiled", nodes=7, checkpointer=checkpointer is not None)

    return compiled