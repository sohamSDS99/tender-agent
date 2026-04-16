"""
Discover Node — Validates and enriches incoming tender state.

WHAT THIS NODE DOES IN THE GRAPH:
By the time the graph starts, the tender data has already been fetched by the
DiscoveryCoordinator (which runs outside the graph). The Discover node's job
is simpler but still important:

1. VALIDATE — Ensure required fields are present (tender_id, title, raw_text).
   If anything is missing, set an error status.

2. ENRICH — If we only have a URL but no raw text (common with email leads
   that just link to a portal), attempt to fetch the tender document.
   In dry-run mode, this generates placeholder text.

3. LOG — Record that this tender entered the pipeline with full metadata.

4. DEADLINE CHECK — If the submission deadline has already passed, reject
   immediately rather than wasting processing time.

WHY NOT JUST SKIP THIS NODE:
The coordinator could set status=DISCOVERED and feed directly into Evaluate.
But having a Discover node gives us:
- A clean audit trail entry marking when the tender entered the pipeline
- Validation that catches bad data before it reaches Evaluate
- A place to fetch full tender documents (not just the summary from scrapers)
- Deadline checking that prevents processing expired tenders
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import structlog

from src.agent.state import TenderState, TenderStatus

logger = structlog.get_logger(__name__)


def discover_node(state: TenderState) -> dict:
    """Node 1: DISCOVER — Validate, enrich, and log incoming tender.

    Input state fields:
        - tender_id, tender_title, tender_raw_text
        - source_portal, source_url
        - submission_deadline

    Output state fields:
        - status, current_node, audit_log
        - tender_raw_text (enriched if was empty)
        - error_messages (if validation fails)
    """
    tender_id = state.get("tender_id", "")
    tender_title = state.get("tender_title", "")
    tender_raw_text = state.get("tender_raw_text", "")
    source_portal = state.get("source_portal", "unknown")
    source_url = state.get("source_url", "")
    deadline = state.get("submission_deadline", "")

    logger.info(
        "node_discover_start",
        tender_id=tender_id,
        title=tender_title[:80],
        source=source_portal,
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    errors: list[str] = []

    if not tender_id:
        errors.append("Missing tender_id — cannot process without an identifier.")

    if not tender_title:
        errors.append("Missing tender_title.")

    if not tender_raw_text and not source_url:
        errors.append(
            "Missing both tender_raw_text and source_url — need at least one "
            "to proceed. Cannot evaluate a tender with no content."
        )

    # ------------------------------------------------------------------
    # Deadline check
    # ------------------------------------------------------------------
    deadline_passed = False
    if deadline:
        try:
            # Try ISO format first
            deadline_dt = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            if deadline_dt < datetime.now(timezone.utc):
                deadline_passed = True
                errors.append(
                    f"Submission deadline has already passed ({deadline}). "
                    f"Archiving tender."
                )
        except (ValueError, TypeError):
            # Non-ISO deadline string (e.g., "June 30, 2026") — can't parse,
            # skip the check. The Evaluate node can handle this later.
            pass

    # ------------------------------------------------------------------
    # Enrichment: fetch tender text if missing but URL is available
    # ------------------------------------------------------------------
    enriched_text = tender_raw_text

    if not tender_raw_text and source_url and not errors:
        dry_run = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")

        if dry_run:
            enriched_text = (
                f"[DRY-RUN: Tender document would be fetched from {source_url}]\n\n"
                f"Title: {tender_title}\n"
                f"Source: {source_portal}\n"
                f"This is placeholder text for the tender document. In production, "
                f"the full tender PDF/DOCX would be downloaded and parsed here."
            )
            logger.info("tender_text_enriched", method="dry-run")
        else:
            enriched_text = _fetch_tender_text(source_url)
            if enriched_text:
                logger.info("tender_text_enriched", method="web_fetch")
            else:
                errors.append(
                    f"Failed to fetch tender document from {source_url}. "
                    f"Will proceed with limited information."
                )

    # ------------------------------------------------------------------
    # Determine status
    # ------------------------------------------------------------------
    if errors and (not tender_id or deadline_passed or (not tender_raw_text and not enriched_text)):
        status = TenderStatus.ERROR.value if not deadline_passed else TenderStatus.REJECTED.value
    else:
        status = TenderStatus.DISCOVERED.value

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    result: dict = {
        "status": status,
        "current_node": "discover",
        "audit_log": [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": "discover",
            "action": "tender_discovered",
            "detail": (
                f"Tender '{tender_title}' from {source_portal}. "
                f"{'Errors: ' + '; '.join(errors) if errors else 'Validation passed.'}"
            ),
            "model_used": None,
            "tokens_used": None,
        }],
    }

    # Only update tender_raw_text if we enriched it
    if enriched_text and enriched_text != tender_raw_text:
        result["tender_raw_text"] = enriched_text

    # Only add errors if there are any
    if errors:
        result["error_messages"] = errors

    logger.info(
        "node_discover_complete",
        tender_id=tender_id,
        status=status,
        errors=len(errors),
    )

    return result


# ---------------------------------------------------------------------------
# Helper: fetch tender document from URL
# ---------------------------------------------------------------------------

def _fetch_tender_text(url: str) -> str:
    """Attempt to download and extract text from a tender URL.

    Tries to fetch the page content. In a full implementation, this would:
    - Download PDF/DOCX attachments from the page
    - Parse them using the document parser from Step 5
    - Return the combined extracted text

    For now, it fetches the HTML page and extracts visible text.
    """
    try:
        import httpx

        response = httpx.get(url, timeout=30.0, follow_redirects=True)
        response.raise_for_status()

        # Basic HTML text extraction (strip tags)
        import re
        text = re.sub(r"<[^>]+>", " ", response.text)
        text = re.sub(r"\s+", " ", text).strip()

        return text[:10000]  # Cap at 10k chars

    except Exception as exc:
        logger.error("tender_fetch_failed", url=url, error=str(exc))
        return ""