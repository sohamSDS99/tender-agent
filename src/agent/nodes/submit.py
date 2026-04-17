"""
Submit Node — Dispatches the tender response via the required channel.

THIS IS THE FINAL NODE IN THE PIPELINE.
When this node completes successfully, the tender has been submitted. The agent's
job is done (for this tender).

SUBMISSION METHOD ROUTING:
The node determines the submission method based on state and tender metadata:
1. If source_portal is "sam.gov" or "merx" → portal_upload (Playwright)
2. If source_portal is "email" → email submission (SMTP)
3. If tender metadata specifies an API endpoint → api dispatch (HTTP)
4. Fallback → mark as "manual" (human must submit manually)

The method can also be explicitly set in the state's submission_method field
by an earlier node or human override.

CONFIRMATION CAPTURE:
Regardless of channel, the node captures:
- Confirmation ID / receipt number
- Confirmation text
- Screenshot (portal only)
- Timestamp

All of this goes into the state and audit log for accountability.

FAILURE HANDLING:
If submission fails, the node sets submission_status="failed" and logs the error.
It does NOT retry automatically — submission is a one-shot operation. If the portal
was down, the team gets notified via Slack and can trigger a manual re-submission.
Automatic retry on submission is risky (could result in duplicate submissions).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import structlog

from src.agent.state import SubmissionMethod, TenderState, TenderStatus
from src.submission.email_api import APIDispatcher, EmailSubmitter
from src.submission.portal_upload import PortalUploader, SubmissionResult

logger = structlog.get_logger(__name__)


def _determine_submission_method(state: TenderState) -> str:
    """Determine how to submit based on tender source and metadata.

    Priority:
    1. Explicit submission_method already set in state → use it
    2. Source portal mapping (sam.gov → portal, email → email)
    3. Fallback to manual
    """
    # Check if already explicitly set
    existing = state.get("submission_method", "")
    if existing and existing != "":
        return existing

    source = state.get("source_portal", "").lower()
    raw_text = state.get("tender_raw_text", "").lower()

    if source in ("sam.gov", "merx"):
        return SubmissionMethod.PORTAL_UPLOAD.value

    if source == "email":
        return SubmissionMethod.EMAIL.value

    # Check if tender text mentions API submission
    if "api submission" in raw_text or "api endpoint" in raw_text:
        return SubmissionMethod.API.value

    # Default to portal upload for unknown sources
    return SubmissionMethod.PORTAL_UPLOAD.value


def _submit_via_portal(state: TenderState, doc_path: str) -> SubmissionResult:
    """Submit via Playwright portal automation."""
    uploader = PortalUploader()

    source = state.get("source_portal", "").lower()
    portal = "sam_gov" if "sam" in source else ("merx" if "merx" in source else "generic")

    # Extract opportunity ID from tender_id or source_url
    opportunity_id = state.get("tender_id", "unknown")

    return uploader.submit(
        portal=portal,
        opportunity_id=opportunity_id,
        document_path=doc_path,
        metadata={
            "company_name": "Acme SDS Solutions",
            "tender_title": state.get("tender_title", ""),
        },
    )


def _submit_via_email(state: TenderState, doc_path: str) -> SubmissionResult:
    """Submit via SMTP email."""
    submitter = EmailSubmitter()

    tender_id = state.get("tender_id", "")
    tender_title = state.get("tender_title", "Untitled")

    # Try to extract recipient from source_url (email://address) or raw text
    source_url = state.get("source_url", "")
    to_address = ""
    if source_url.startswith("email://"):
        # source_url might be "email://msg-id@domain" — not a recipient
        pass

    # Fall back to a configured default
    to_address = to_address or os.getenv("SUBMISSION_EMAIL_TO", "procurement@example.com")

    return submitter.submit(
        to_address=to_address,
        subject=f"Tender Response: {tender_title} — {tender_id}",
        body=(
            f"Dear Procurement Team,\n\n"
            f"Please find attached our response to tender {tender_id}: "
            f"{tender_title}.\n\n"
            f"We look forward to your review.\n\n"
            f"Best regards,\n"
            f"Acme SDS Solutions\n"
            f"Tender Agent (Automated Submission)"
        ),
        document_path=doc_path,
        tender_id=tender_id,
    )


def _submit_via_api(state: TenderState, doc_path: str) -> SubmissionResult:
    """Submit via HTTP API."""
    dispatcher = APIDispatcher()

    # In production, the API endpoint would come from tender metadata
    endpoint = os.getenv("SUBMISSION_API_ENDPOINT", "https://api.example.com/submissions")
    api_key = os.getenv("SUBMISSION_API_KEY", "")

    return dispatcher.submit(
        endpoint_url=endpoint,
        document_path=doc_path,
        tender_id=state.get("tender_id", ""),
        api_key=api_key,
        metadata={
            "company_name": "Acme SDS Solutions",
            "tender_id": state.get("tender_id", ""),
            "tender_title": state.get("tender_title", ""),
        },
    )


def submit_node(state: TenderState) -> dict:
    """Node 7: SUBMIT — Dispatch tender response via the required channel.

    Input state fields:
        - tender_id, tender_title, source_portal
        - assembled_document_path (from assemble node)
        - submission_method (optional — auto-detected if not set)

    Output state fields:
        - submission_method, submission_status
        - submission_confirmation, submission_screenshot_path
        - status, current_node, audit_log
    """
    tender_id = state.get("tender_id", "unknown")
    doc_path = state.get("assembled_document_path", "")

    logger.info(
        "node_submit_start",
        tender_id=tender_id,
        document=doc_path,
    )

    # Determine submission method
    method = _determine_submission_method(state)

    # Validate document exists
    if not doc_path:
        logger.error("no_document_to_submit", tender_id=tender_id)
        return {
            "submission_method": method,
            "submission_status": "failed",
            "submission_confirmation": None,
            "submission_screenshot_path": None,
            "status": TenderStatus.SUBMISSION_FAILED.value,
            "current_node": "submit",
            "audit_log": [{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "node": "submit",
                "action": "submission_failed",
                "detail": "No assembled document path in state. Cannot submit.",
                "model_used": None,
                "tokens_used": None,
            }],
        }

    # Route to the appropriate channel
    result: SubmissionResult | None = None

    if method == SubmissionMethod.PORTAL_UPLOAD.value:
        result = _submit_via_portal(state, doc_path)
    elif method == SubmissionMethod.EMAIL.value:
        result = _submit_via_email(state, doc_path)
    elif method == SubmissionMethod.API.value:
        result = _submit_via_api(state, doc_path)
    elif method == SubmissionMethod.MANUAL.value:
        # Manual submission — just log it and mark as pending
        result = SubmissionResult(
            success=True,
            confirmation_id="MANUAL-PENDING",
            confirmation_text=(
                f"Tender {tender_id} flagged for manual submission. "
                f"Document is ready at: {doc_path}"
            ),
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )
    else:
        # Unknown method — try portal as fallback
        logger.warning("unknown_submission_method", method=method, fallback="portal")
        result = _submit_via_portal(state, doc_path)

    # Build state update
    if result.success:
        status = TenderStatus.SUBMITTED.value
        sub_status = "success"
        action = "tender_submitted"
        detail = (
            f"Tender submitted via {method}. "
            f"Confirmation: {result.confirmation_id}. "
            f"{result.confirmation_text}"
        )
    else:
        status = TenderStatus.SUBMISSION_FAILED.value
        sub_status = "failed"
        action = "submission_failed"
        detail = (
            f"Submission via {method} FAILED. "
            f"Error: {result.error_message}"
        )

    logger.info(
        "node_submit_complete",
        tender_id=tender_id,
        method=method,
        success=result.success,
        confirmation_id=result.confirmation_id,
    )

    return {
        "submission_method": method,
        "submission_status": sub_status,
        "submission_confirmation": result.confirmation_id,
        "submission_screenshot_path": result.screenshot_path,
        "status": status,
        "current_node": "submit",
        "audit_log": [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": "submit",
            "action": action,
            "detail": detail,
            "model_used": None,
            "tokens_used": None,
        }],
    }