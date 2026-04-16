"""
Slack Escalate Node — Sends gap questions to Slack and captures human responses.

WAIT-AND-RESUME PATTERN:
In production with LangGraph checkpointing:
1. This node sends questions to Slack → returns status=AWAITING_HUMAN
2. LangGraph checkpoints state to PostgreSQL → graph PAUSES
3. A Slack event listener watches for thread replies
4. Human responds → listener calls graph.update_state() with response
5. Graph resumes from checkpoint → continues to retrieve_draft for re-drafting

In dry-run mode:
Questions are "sent" and mock responses return immediately, so the graph
flows continuously without pausing. This tests the full pipeline end-to-end.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from src.agent.state import TenderState, TenderStatus
from src.slack_integration.slack_client import SlackClient

logger = structlog.get_logger(__name__)


def slack_escalate_node(state: TenderState) -> dict:
    """Node 5: SLACK ESCALATE — Send gap questions and capture responses.

    Input state fields:
        - tender_id, tender_title, submission_deadline
        - gaps (from gap check node)
        - escalation_count

    Output state fields:
        - slack_questions, slack_responses, slack_thread_ts
        - escalation_count (incremented)
        - status, current_node, audit_log
    """
    tender_id = state.get("tender_id", "unknown")
    tender_title = state.get("tender_title", "Untitled")
    deadline = state.get("submission_deadline", "Unknown")
    gaps = state.get("gaps", [])
    current_count = state.get("escalation_count", 0)

    logger.info(
        "node_slack_escalate_start",
        tender_id=tender_id,
        gaps_count=len(gaps),
        escalation_number=current_count + 1,
    )

    # Build specific questions from gaps
    questions = []
    for gap in gaps:
        question = gap.get("suggested_question", "")
        if question:
            severity = gap.get("severity", "medium").upper()
            questions.append(f"[{severity}] {question}")

    if not questions:
        questions = ["Please review the draft for completeness."]

    # Send to Slack
    client = SlackClient()
    msg = client.send_gap_questions(
        tender_title=tender_title,
        tender_deadline=deadline,
        questions=questions,
        tender_id=tender_id,
    )

    # In dry-run: get mock responses immediately
    # In production: this would be handled by checkpoint-and-resume
    responses = client.get_thread_responses(msg.channel, msg.ts)
    response_texts = [r.text for r in responses]

    logger.info(
        "node_slack_escalate_complete",
        tender_id=tender_id,
        questions_sent=len(questions),
        responses_received=len(response_texts),
    )

    return {
        "slack_questions": questions,
        "slack_responses": response_texts,
        "slack_thread_ts": msg.ts,
        "escalation_count": current_count + 1,
        "status": TenderStatus.AWAITING_HUMAN.value,
        "current_node": "slack_escalate",
        "audit_log": [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": "slack_escalate",
            "action": "questions_sent",
            "detail": (
                f"Sent {len(questions)} questions to Slack "
                f"(escalation #{current_count + 1}). "
                f"Received {len(response_texts)} response(s)."
            ),
            "model_used": None,
            "tokens_used": None,
        }],
    }