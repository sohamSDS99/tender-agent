"""
Email Draft Writer — drafts a procurement-grade submission email for a
single TenderPursuit when the bridge's submission_classifier says the
tender wants an email-based submission.

Used from ``handle_draft_email_task`` (initial draft) and from
``handle_email_revision_task`` (HITL revision passes).  The output
shape mirrors ``FormFiller.FillResult`` so the FormFillEditor pattern
on the AMS side can render an EmailDraftEditor with the SAME three-
section accordion (high / medium / low confidence) — there are just
four "fields" (To, CC, Subject, Body) plus an attachments suggestion
list.

Inputs:
    - Tender info (title, agency, URL, deadline, description, instructions)
    - Extracted contact email + CC from the classifier
    - Detected language
    - Company KB context (already-retrieved chunks, joined into a single
      string the way FormFiller consumes ``company_context``)
    - Past email examples (also joined chunks — past sent submissions)
    - Optional ``user_answers`` map from a prior HITL revision pass:
        {"to": "...", "cc": "a@b, c@d", "subject": "...", "body": "..."}
      Any of these the operator has anchored overrides the LLM's output
      verbatim (source='user_input', confidence=1.0).

The writer never touches the database, MinIO, or HTTP.  Pure function
of its inputs given a callable ``llm_call_fn``.  Keep it that way so
the bridge handler is the only thing doing I/O.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types — analogous to form_filler.FillResult so the UI editor
# can be a 1:1 port of FormFillEditor with different field labels.
# ---------------------------------------------------------------------------

ConfidenceTier = str  # "high" | "medium" | "low"


def _tier_for(numeric_confidence: float, source: str) -> ConfidenceTier:
    """Mirror form_filler's tier function so the UI bucketing is
    consistent across form_fill and email_submission flows."""

    if source == "user_input":
        return "high"
    if source == "error":
        return "low"
    if "example" in source and numeric_confidence >= 0.65:
        return "high"
    if numeric_confidence >= 0.8:
        return "high"
    if numeric_confidence >= 0.55:
        return "medium"
    return "low"


@dataclass
class DraftedEmailField:
    """One row in the EmailDraftEditor's three-section accordion.

    ``name`` is one of: 'to', 'cc', 'subject', 'body'.  The UI uses
    the name as the field label and decides input control type
    (single-line for to/cc/subject, multi-line for body).
    """

    name: str
    value: str
    confidence: float
    confidence_tier: ConfidenceTier = "low"
    source: str = ""
    reasoning: str = ""
    example_doc_ids: list[str] = field(default_factory=list)


@dataclass
class SuggestedAttachment:
    """A document the LLM thinks should be attached to the email.

    The writer doesn't have access to the operator's actual Document
    library, so it returns *descriptions* (e.g. "Company capability
    statement").  The AMS UI lets the operator match each suggestion
    to a real Document by drag-and-drop.
    """

    filename_hint: str       # e.g. "company_profile.pdf"
    description: str         # what the LLM thinks this should be
    reason: str = ""         # why the LLM thinks the tender wants it


@dataclass
class EmailDraftResult:
    """Top-level output — what the bridge handler POSTs back into the
    Submission.outputSummary JSONB blob."""

    to: str | None
    cc: list[str]
    subject: str
    body: str
    suggested_attachments: list[SuggestedAttachment]
    fields: list[DraftedEmailField]
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0
    blank_count: int = 0
    examples_used: list[str] = field(default_factory=list)
    llm_cost_usd: float = 0.0
    llm_tokens: int = 0
    llm_model: str = ""
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def _valid_email(addr: str | None) -> bool:
    return bool(addr) and bool(_VALID_EMAIL_RE.match((addr or "").strip()))


def _normalise_cc(raw: str | list[str] | None) -> list[str]:
    """Accept either a list or a comma/semicolon-separated string."""

    if raw is None:
        return []
    if isinstance(raw, list):
        items = raw
    else:
        items = re.split(r"[,;]+", str(raw))
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        addr = (item or "").strip()
        if _valid_email(addr) and addr.lower() not in seen:
            seen.add(addr.lower())
            out.append(addr)
    return out[:5]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_LANG_NAME = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "de": "German",
    "it": "Italian",
}


def _build_prompt(
    *,
    tender_title: str,
    tender_agency: str | None,
    tender_url: str,
    tender_deadline: str | None,
    tender_description: str,
    contact_email: str | None,
    instructions: str | None,
    language: str | None,
    company_context: str,
    example_emails: str,
    user_anchors: dict[str, str] | None,
) -> str:
    """Compose the single prompt we send to Claude Sonnet.

    Returns the user-message text.  The system prompt is fetched by
    the LLM caller from Prompt Studio.
    """

    language_name = _LANG_NAME.get((language or "en").lower(), "English")

    anchor_block = ""
    if user_anchors:
        cleaned = {
            k: v.strip()
            for k, v in user_anchors.items()
            if isinstance(v, str) and v.strip()
        }
        if cleaned:
            anchor_block = (
                "\n\nOPERATOR ANCHORS (use these VERBATIM — they were "
                "edited by the human operator):\n"
                + "\n".join(f"  {k}: {_truncate(v, 1000)}" for k, v in cleaned.items())
                + "\n"
            )

    instructions_block = ""
    if instructions:
        instructions_block = (
            "\nSUBMISSION REQUIREMENTS (extracted from the tender page — "
            "the email body MUST address each of these):\n"
            f"{_truncate(instructions, 1500)}\n"
        )

    examples_block = ""
    if example_emails:
        examples_block = (
            "\nPAST SUBMISSION EMAILS (use these as style + voice "
            "examples; do not copy verbatim):\n"
            "----- BEGIN EXAMPLES -----\n"
            f"{_truncate(example_emails, 4000)}\n"
            "----- END EXAMPLES -----\n"
        )

    return (
        "You are drafting a procurement-grade submission email on behalf "
        "of SDS Manager (a chemical-safety / SDS / EHS specialist) in "
        f"response to a public tender.  Write the email in {language_name}.\n\n"
        f"TENDER: {_truncate(tender_title, 250)}\n"
        f"AGENCY: {tender_agency or 'unknown'}\n"
        f"DEADLINE: {tender_deadline or 'see tender page'}\n"
        f"SOURCE: {tender_url}\n"
        f"RECIPIENT: {contact_email or '(not extracted — leave To blank)'}\n\n"
        "TENDER DESCRIPTION:\n"
        f"{_truncate(tender_description, 2000)}\n"
        f"{instructions_block}"
        "\nCOMPANY KNOWLEDGE BASE (use only facts from here — do NOT "
        "invent capabilities, certs, or past projects):\n"
        f"{_truncate(company_context, 6000)}\n"
        f"{examples_block}"
        f"{anchor_block}"
        "\nReturn a STRICT JSON object with these exact keys:\n"
        "{\n"
        '  "subject":  string — a tight, deadline-respecting subject line\n'
        '              that names the tender (≤90 chars). If the tender\n'
        '              gave a reference number, include it.\n'
        '  "body":     string — the full email body, plain text (no\n'
        '              markdown). Greeting, brief intro of SDS Manager,\n'
        '              point-by-point response to each submission\n'
        '              requirement (if any), capability summary grounded\n'
        '              in the KB, closing + signature placeholder\n'
        '              ("Best regards,\\nSoham Sarker\\nSDS Manager").\n'
        '              350–700 words. Do NOT invent prices.\n'
        '  "suggested_attachments": array of objects, each with:\n'
        '              { "filename_hint": short suggested filename,\n'
        '                "description":  what the document should be,\n'
        '                "reason":       why this tender wants it }\n'
        '              Return 2–5 entries. Examples: company capability\n'
        '              statement, ISO 9001 cert, GHS labelling references,\n'
        '              past project portfolio. Use ONLY documents the KB\n'
        '              implies the company actually has.\n'
        '  "reasoning": one sentence about your overall drafting choices.\n'
        "}\n\n"
        "Output the JSON and NOTHING else — no preamble, no markdown "
        "code fence."
    )


# ---------------------------------------------------------------------------
# JSON parsing — tolerate code-fenced output even though we ask for plain
# ---------------------------------------------------------------------------

def _parse_llm_json(content: str) -> dict[str, Any]:
    """Strip optional ```json fences and parse.  Raises on bad shape."""

    s = (content or "").strip()
    if s.startswith("```"):
        # Drop the opening fence (```json or just ```) and the closing one
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        if s.endswith("```"):
            s = s[: -3].rstrip()
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError(f"LLM returned non-object JSON: {type(obj).__name__}")
    return obj


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class EmailDraftWriter:
    """Stateless writer — one instance per bridge process is fine.

    The LLM-call callable matches the signature used by FormFiller:

        llm_call_fn(prompt: str, max_tokens: int) -> {
            "content": str,
            "tokens_input": int,
            "tokens_output": int,
            "cost_usd": float,
            "model": str,
            "duration_ms": int,
        }
    """

    def __init__(
        self,
        *,
        llm_call_fn: Callable[..., dict[str, Any]],
    ):
        self.llm_call = llm_call_fn

    # -- main entry ---------------------------------------------------------

    def draft(
        self,
        *,
        tender_title: str,
        tender_description: str,
        tender_url: str,
        tender_agency: str | None = None,
        tender_deadline: str | None = None,
        contact_email: str | None = None,
        contact_cc: list[str] | None = None,
        instructions: str | None = None,
        language: str | None = None,
        company_context: str = "",
        example_emails: str = "",
        example_doc_ids: list[str] | None = None,
        user_answers: dict[str, str] | None = None,
    ) -> EmailDraftResult:
        """Draft a complete submission email.

        The user_answers map (from a prior HITL revision pass) is
        spliced in two places:
            - As anchors in the LLM prompt so the model knows what the
              operator already committed to.
            - As verbatim overrides in the returned field set
              (source='user_input', confidence=1.0).
        """

        example_doc_ids = example_doc_ids or []
        user_answers = {k: v for k, v in (user_answers or {}).items() if v}

        prompt = _build_prompt(
            tender_title=tender_title,
            tender_agency=tender_agency,
            tender_url=tender_url,
            tender_deadline=tender_deadline,
            tender_description=tender_description,
            contact_email=contact_email,
            instructions=instructions,
            language=language,
            company_context=company_context,
            example_emails=example_emails,
            user_anchors=user_answers,
        )

        # --------------------------------------------------------------
        # LLM call — graceful on any error.  We still return a draft
        # (with low-confidence fallbacks) so the operator can take over.
        # --------------------------------------------------------------
        llm_subject = ""
        llm_body = ""
        llm_suggested: list[SuggestedAttachment] = []
        llm_reasoning = ""
        llm_cost = 0.0
        llm_tokens = 0
        llm_model = ""
        llm_failed = False
        llm_error: str | None = None

        try:
            result = self.llm_call(prompt, max_tokens=3000)
            content = (result or {}).get("content", "") if isinstance(result, dict) else ""
            llm_cost = float((result or {}).get("cost_usd", 0.0)) if isinstance(result, dict) else 0.0
            llm_tokens = int(((result or {}).get("tokens_input", 0) or 0)) + int(
                ((result or {}).get("tokens_output", 0) or 0)
            ) if isinstance(result, dict) else 0
            llm_model = (result or {}).get("model", "") if isinstance(result, dict) else ""
            parsed = _parse_llm_json(content)
            llm_subject = (parsed.get("subject") or "").strip()
            llm_body = (parsed.get("body") or "").strip()
            raw_suggested = parsed.get("suggested_attachments") or []
            if isinstance(raw_suggested, list):
                for entry in raw_suggested[:8]:
                    if not isinstance(entry, dict):
                        continue
                    llm_suggested.append(
                        SuggestedAttachment(
                            filename_hint=str(entry.get("filename_hint") or "")[:200],
                            description=str(entry.get("description") or "")[:500],
                            reason=str(entry.get("reason") or "")[:500],
                        )
                    )
            llm_reasoning = str(parsed.get("reasoning") or "")[:1000]
        except Exception as exc:
            llm_failed = True
            llm_error = str(exc)
            logger.warning("[email-draft] LLM draft failed: %s", exc)

        # --------------------------------------------------------------
        # Resolve each field — operator anchor wins, else LLM, else
        # fallback.  We assemble the editor row list as we go.
        # --------------------------------------------------------------
        fields: list[DraftedEmailField] = []

        # --- To ----------------------------------------------------------
        to_anchor = user_answers.get("to")
        if to_anchor and _valid_email(to_anchor):
            to_value = to_anchor.strip()
            fields.append(
                DraftedEmailField(
                    name="to",
                    value=to_value,
                    confidence=1.0,
                    confidence_tier="high",
                    source="user_input",
                    reasoning="Operator anchored the recipient address.",
                )
            )
        elif _valid_email(contact_email):
            assert contact_email  # for type checker
            to_value = contact_email.strip()
            fields.append(
                DraftedEmailField(
                    name="to",
                    value=to_value,
                    confidence=0.95,
                    confidence_tier="high",
                    source="extracted",
                    reasoning="Extracted by classifier from tender submission instructions.",
                )
            )
        else:
            to_value = ""
            fields.append(
                DraftedEmailField(
                    name="to",
                    value="",
                    confidence=0.0,
                    confidence_tier="low",
                    source="error" if llm_failed else "missing",
                    reasoning="No recipient address could be extracted — operator must supply manually.",
                )
            )

        # --- CC ----------------------------------------------------------
        cc_anchor = user_answers.get("cc")
        if cc_anchor is not None:
            cc_list = _normalise_cc(cc_anchor)
            fields.append(
                DraftedEmailField(
                    name="cc",
                    value=", ".join(cc_list),
                    confidence=1.0 if cc_list else 0.0,
                    confidence_tier="high" if cc_list else "low",
                    source="user_input" if cc_list else "missing",
                    reasoning="Operator anchored the CC list." if cc_list else "Operator cleared CC.",
                )
            )
        else:
            cc_list = _normalise_cc(contact_cc)
            fields.append(
                DraftedEmailField(
                    name="cc",
                    value=", ".join(cc_list),
                    confidence=0.7 if cc_list else 0.0,
                    confidence_tier="medium" if cc_list else "low",
                    source="extracted" if cc_list else "missing",
                    reasoning=(
                        "Extracted by classifier from tender page."
                        if cc_list
                        else "No CC addresses extracted."
                    ),
                )
            )

        # --- Subject -----------------------------------------------------
        subject_anchor = user_answers.get("subject")
        if subject_anchor:
            subject_value = subject_anchor.strip()
            fields.append(
                DraftedEmailField(
                    name="subject",
                    value=subject_value,
                    confidence=1.0,
                    confidence_tier="high",
                    source="user_input",
                    reasoning="Operator anchored the subject.",
                )
            )
        elif llm_subject:
            subject_value = llm_subject
            fields.append(
                DraftedEmailField(
                    name="subject",
                    value=subject_value,
                    confidence=0.7,
                    confidence_tier="medium",
                    source="llm_inference",
                    reasoning="Drafted by LLM from tender title + reference.",
                )
            )
        else:
            # Fallback: a deterministic subject so the operator always
            # has something to send.
            fallback_subject = f"Proposal — {_truncate(tender_title, 80)}"
            subject_value = fallback_subject
            fields.append(
                DraftedEmailField(
                    name="subject",
                    value=subject_value,
                    confidence=0.3,
                    confidence_tier="low",
                    source="error" if llm_failed else "fallback",
                    reasoning=(
                        f"LLM unavailable ({llm_error}) — using a deterministic fallback."
                        if llm_failed
                        else "LLM returned no subject; using a deterministic fallback."
                    ),
                )
            )

        # --- Body --------------------------------------------------------
        body_anchor = user_answers.get("body")
        if body_anchor:
            body_value = body_anchor
            fields.append(
                DraftedEmailField(
                    name="body",
                    value=body_value,
                    confidence=1.0,
                    confidence_tier="high",
                    source="user_input",
                    reasoning="Operator anchored the body.",
                )
            )
        elif llm_body:
            body_value = llm_body
            # If we grounded in past examples, bump the confidence
            body_conf = 0.78 if example_doc_ids else 0.65
            body_source = "example_match" if example_doc_ids else "llm_inference"
            fields.append(
                DraftedEmailField(
                    name="body",
                    value=body_value,
                    confidence=body_conf,
                    confidence_tier=_tier_for(body_conf, body_source),
                    source=body_source,
                    reasoning=llm_reasoning or "Drafted by LLM from KB context.",
                    example_doc_ids=list(example_doc_ids),
                )
            )
        else:
            body_value = (
                "Dear Procurement Team,\n\n"
                f"We are writing in response to {tender_title}.\n\n"
                "Please find our proposal attached. We look forward to "
                "the opportunity to discuss next steps.\n\n"
                "Best regards,\nSoham Sarker\nSDS Manager"
            )
            fields.append(
                DraftedEmailField(
                    name="body",
                    value=body_value,
                    confidence=0.2,
                    confidence_tier="low",
                    source="error" if llm_failed else "fallback",
                    reasoning=(
                        f"LLM unavailable ({llm_error}) — using a minimal fallback body."
                        if llm_failed
                        else "LLM returned no body; using a minimal fallback."
                    ),
                )
            )

        # --------------------------------------------------------------
        # Tier counters for the editor's summary line
        # --------------------------------------------------------------
        high = sum(1 for f in fields if f.confidence_tier == "high")
        medium = sum(1 for f in fields if f.confidence_tier == "medium")
        low = sum(1 for f in fields if f.confidence_tier == "low")
        blank = sum(1 for f in fields if not f.value.strip())

        return EmailDraftResult(
            to=to_value or None,
            cc=cc_list,
            subject=subject_value,
            body=body_value,
            suggested_attachments=llm_suggested,
            fields=fields,
            high_confidence_count=high,
            medium_confidence_count=medium,
            low_confidence_count=low,
            blank_count=blank,
            examples_used=list(example_doc_ids),
            llm_cost_usd=round(llm_cost, 6),
            llm_tokens=llm_tokens,
            llm_model=llm_model,
            reasoning=llm_reasoning or ("LLM failed: " + (llm_error or "")) if llm_failed else llm_reasoning,
        )


__all__ = [
    "ConfidenceTier",
    "DraftedEmailField",
    "EmailDraftResult",
    "EmailDraftWriter",
    "SuggestedAttachment",
]
