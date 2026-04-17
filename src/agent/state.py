"""
Tender Agent State Schema — The data that flows through the entire pipeline.

WHY A STATE SCHEMA MATTERS:
LangGraph's StateGraph passes a single state dictionary through every node in the
graph. Each node reads the fields it needs, does its work, and returns a dictionary
of fields to update. The state is the "shared memory" of the agent.

If we don't define this carefully upfront, we'll hit problems like:
- Node 3 (Draft) expects a field that Node 2 (Evaluate) forgot to set
- Two nodes write to the same field with different formats
- Debugging is impossible because nobody knows what the state looks like

THIS STATE REPRESENTS A SINGLE TENDER:
The graph processes one tender at a time. For each tender discovered, a new graph
invocation starts with fresh state. The Discover node finds tenders and kicks off
separate graph runs for each one.

HOW STATE UPDATES WORK IN LANGGRAPH:
When a node returns {"score": 85}, LangGraph merges that into the existing state.
It does NOT replace the entire state — only the fields you return get updated.
Fields you don't mention stay unchanged.

For list fields (like audit_log), we use Annotated[list, operator.add] which tells
LangGraph to APPEND to the list rather than replace it. This way every node can add
audit entries without overwriting previous ones.

FIELD NAMING CONVENTION:
- Fields set by a specific node are prefixed with the node's purpose:
  eval_* for Evaluate, draft_* for Draft, gap_* for Gap Check, etc.
- Shared fields (tender_id, current_node) have no prefix.
- This makes it obvious which node is responsible for which data.
"""

from __future__ import annotations

import operator
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, TypedDict


# ---------------------------------------------------------------------------
# Enums for type safety
# ---------------------------------------------------------------------------

class TenderStatus(str, Enum):
    """Tracks where a tender is in the pipeline.

    Using an enum instead of a raw string prevents typos like
    "evaluted" vs "evaluated" from causing silent bugs.
    """
    DISCOVERED = "discovered"
    EVALUATING = "evaluating"
    ELIGIBLE = "eligible"
    REJECTED = "rejected"
    DRAFTING = "drafting"
    GAP_CHECK = "gap_check"
    AWAITING_HUMAN = "awaiting_human"
    ASSEMBLING = "assembling"
    QUALITY_FAILED = "quality_failed"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    SUBMISSION_FAILED = "submission_failed"
    ERROR = "error"


class SubmissionMethod(str, Enum):
    """How the final tender gets submitted."""
    PORTAL_UPLOAD = "portal_upload"     # Playwright browser automation
    EMAIL = "email"                     # SMTP email dispatch
    API = "api"                         # Direct API submission
    MANUAL = "manual"                   # Flagged for human manual submission


# ---------------------------------------------------------------------------
# Nested data structures
# ---------------------------------------------------------------------------

class TenderRequirement(TypedDict, total=False):
    """A single requirement extracted from the tender document.

    Example:
        {
            "section_id": "3.2",
            "section_title": "Technical Capabilities",
            "requirement_text": "Describe your GHS classification capabilities...",
            "is_mandatory": True,
            "max_words": 500,
        }
    """
    section_id: str
    section_title: str
    requirement_text: str
    is_mandatory: bool
    max_words: int | None


class DraftedSection(TypedDict, total=False):
    """A single drafted section of the tender response.

    Example:
        {
            "section_id": "3.2",
            "section_title": "Technical Capabilities",
            "content": "Our platform provides comprehensive GHS classification...",
            "confidence": 0.85,
            "sources_used": ["company_profile.pdf > Certifications"],
            "model_used": "qwen3.5-plus",
            "token_count": 420,
        }
    """
    section_id: str
    section_title: str
    content: str
    confidence: float
    sources_used: list[str]
    model_used: str
    token_count: int


class GapItem(TypedDict, total=False):
    """A single identified gap (missing or outdated information).

    Example:
        {
            "section_id": "4.1",
            "description": "Pricing for 500+ user tier not found in knowledge base",
            "severity": "high",
            "suggested_question": "What is the per-user pricing for tiers above 500 users?",
        }
    """
    section_id: str
    description: str
    severity: str  # "high", "medium", "low"
    suggested_question: str


class AuditEntry(TypedDict, total=False):
    """One entry in the audit trail.

    Every action the agent takes gets logged here — what happened,
    which node did it, what the input/output was, and why.

    Example:
        {
            "timestamp": "2026-04-16T10:30:00Z",
            "node": "evaluate",
            "action": "scored_tender",
            "detail": "Score 78/100 — eligible. Geography: 10, Domain: 9, ...",
            "model_used": "qwen3.5-flash",
            "tokens_used": 1200,
        }
    """
    timestamp: str
    node: str
    action: str
    detail: str
    model_used: str | None
    tokens_used: int | None


# ---------------------------------------------------------------------------
# Main state schema
# ---------------------------------------------------------------------------

class TenderState(TypedDict, total=False):
    """The complete state of a tender as it flows through the 7-node pipeline.

    This TypedDict is the single source of truth. Every node reads from it
    and writes back to it. LangGraph handles the merge automatically.

    FIELD GROUPS:
    1. Identification — What tender is this?
    2. Evaluation — Should we bid on it?
    3. Drafting — What did we write?
    4. Gap Check — What's missing?
    5. Slack Escalation — What did we ask humans?
    6. Assembly — Is the document ready?
    7. Submission — Did it get sent?
    8. Tracking — Where are we in the pipeline?

    ANNOTATED LISTS:
    Fields marked with Annotated[list[X], operator.add] use LangGraph's
    "reducer" pattern. When a node returns one of these fields, the new
    items get APPENDED to the existing list instead of replacing it.
    This is essential for audit_log — every node adds entries, and we
    never want to lose previous entries.
    """

    # ------------------------------------------------------------------
    # 1. IDENTIFICATION — Set by Discover node
    # ------------------------------------------------------------------

    # Unique ID for this tender (from the source portal or generated UUID)
    tender_id: str

    # Human-readable title of the tender
    tender_title: str

    # Where this tender was found
    source_portal: str          # e.g., "sam.gov", "merx", "email"
    source_url: str             # Direct URL to the tender posting

    # Raw extracted text from the tender document
    tender_raw_text: str

    # Key dates
    submission_deadline: str    # ISO format: "2026-05-15T17:00:00Z"
    discovered_at: str          # ISO format: when we found it

    # Original document path (if downloaded)
    tender_document_path: str | None

    # ------------------------------------------------------------------
    # 2. EVALUATION — Set by Evaluate node
    # ------------------------------------------------------------------

    # Overall eligibility score (0-100). Threshold is 60.
    eval_score: int

    # Breakdown of the 8 scoring dimensions
    eval_breakdown: dict[str, int]
    # Example: {
    #     "geography": 10,    # max 15
    #     "budget": 8,        # max 15
    #     "scope": 9,         # max 15
    #     "compliance": 10,   # max 10
    #     "timeline": 7,      # max 10
    #     "certifications": 8,# max 10
    #     "domain_match": 12, # max 15
    #     "competition": 6,   # max 10
    # }

    # Go/No-Go decision
    eval_decision: str          # "go" or "no_go"
    eval_reasoning: str         # Plain-English explanation of the decision

    # ------------------------------------------------------------------
    # 3. DRAFTING — Set by Retrieve & Draft node
    # ------------------------------------------------------------------

    # Structured requirements extracted from the tender
    tender_requirements: list[TenderRequirement]

    # Drafted sections (one per requirement)
    drafted_sections: list[DraftedSection]

    # RAG context used for drafting (for audit/debugging)
    draft_rag_context: str

    # ------------------------------------------------------------------
    # 4. GAP CHECK — Set by Gap Check node
    # ------------------------------------------------------------------

    # List of identified gaps
    gaps: list[GapItem]

    # Did all sections pass the gap check?
    gap_check_passed: bool

    # ------------------------------------------------------------------
    # 5. SLACK ESCALATION — Set by Slack Escalate node
    # ------------------------------------------------------------------

    # Questions sent to the Slack channel
    slack_questions: list[str]

    # Responses received from the team
    slack_responses: list[str]

    # Slack message timestamp (for threading replies)
    slack_thread_ts: str | None

    # How many times we've escalated for this tender
    escalation_count: int

    # ------------------------------------------------------------------
    # 6. ASSEMBLY — Set by Assemble node
    # ------------------------------------------------------------------

    # Path to the assembled document (DOCX or PDF)
    assembled_document_path: str | None

    # Quality check results
    quality_check_passed: bool
    quality_issues: list[str]

    # How many times assembly has been retried (max 3)
    assembly_retry_count: int

    # ------------------------------------------------------------------
    # 7. SUBMISSION — Set by Submit node
    # ------------------------------------------------------------------

    # How the tender will be/was submitted
    submission_method: str      # SubmissionMethod value

    # Outcome
    submission_status: str      # "success", "failed", "pending"
    submission_confirmation: str | None  # Receipt ID or confirmation text
    submission_screenshot_path: str | None  # Screenshot of portal confirmation

    # ------------------------------------------------------------------
    # 8. TRACKING & AUDIT — Updated by every node
    # ------------------------------------------------------------------

    # Current pipeline status
    status: str                 # TenderStatus value

    # Which node is currently processing
    current_node: str

    # Accumulated errors (non-fatal)
    error_messages: Annotated[list[str], operator.add]

    # Full audit trail — every node appends entries here
    audit_log: Annotated[list[AuditEntry], operator.add]