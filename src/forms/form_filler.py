"""
Form Filler — LLM-powered form field filling using company context.

Takes extracted form fields + company documents and uses Claude to:
1. Map company information to the appropriate form fields
2. Fill fields with high confidence
3. Identify fields that need user clarification

Usage:
    filler = FormFiller(llm_call_fn=my_llm_function)
    result = filler.fill(parsed_form, company_context)
    print(result.filled_fields)
    print(result.questions)  # Fields that need user input
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .form_parser import FormField, ParseResult


# A confidence tier the Approvals UI renders as a coloured badge.
# Derived from the numeric `confidence` plus the source — set in the
# filler, never read inside the LLM.
ConfidenceTier = str  # "high" | "medium" | "low"


def _tier_for(numeric_confidence: float, source: str) -> ConfidenceTier:
    """Map a numeric confidence + source string to a coarse tier.

    `source` carries the provenance — e.g. "example_match", "kb",
    "user_input", "llm_inference", "error". We bump the tier up when
    we know the answer came from an example match, and down when the
    LLM hedged.
    """
    if source == "user_input":
        return "high"
    if source == "error":
        return "low"
    # Example-grounded answers get the higher tier even at slightly
    # lower numeric confidence — they're factually anchored to a real
    # past submission.
    if "example" in source and numeric_confidence >= 0.65:
        return "high"
    if numeric_confidence >= 0.8:
        return "high"
    if numeric_confidence >= 0.55:
        return "medium"
    return "low"


@dataclass
class FilledField:
    """A form field with its filled value and confidence score."""
    name: str
    value: str
    confidence: float            # 0.0 - 1.0 (numeric)
    source: str = ""             # Provenance ("example_match", "kb", "user_input", "llm_inference", "error")
    reasoning: str = ""          # Why this value was chosen
    confidence_tier: ConfidenceTier = "low"  # Coarse tier used by the Approvals UI
    example_doc_ids: list[str] = field(default_factory=list)  # Which past forms grounded this answer
    original_field: FormField | None = None


@dataclass
class ClarificationQuestion:
    """A question the agent needs to ask the user."""
    field_name: str
    question: str
    context: str = ""        # What the agent knows so far about this field
    suggestions: list[str] = field(default_factory=list)  # Possible answers


@dataclass
class FillResult:
    """Result of the form filling process."""
    filled_fields: list[FilledField]
    questions: list[ClarificationQuestion]  # Fields needing user input
    total_fields: int
    filled_count: int
    needs_clarification_count: int
    llm_cost_usd: float = 0.0
    llm_tokens: int = 0
    # How many fields landed in each confidence tier — the Approvals UI
    # surfaces this as a summary line ("3 high / 2 medium / 1 low").
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0
    # Distinct past-submission documents that grounded at least one
    # answer — drives the "Using N past submissions as reference" badge.
    examples_used: list[str] = field(default_factory=list)


class FormFiller:
    """Fills form fields using LLM and company context.

    Args:
        llm_call_fn: A function that takes (prompt: str, max_tokens: int) and
                     returns a dict with at least {"content": "...", "cost_usd": 0.0, ...}
        confidence_threshold: Fields below this confidence get flagged for user review.
    """

    def __init__(
        self,
        llm_call_fn: Callable[..., dict[str, Any]],
        confidence_threshold: float = 0.7,
        # Callable(question: str) -> list[{questionText, answerText,
        # documentId, similarity, metadata}]. Bridge wires this up to
        # search_similar_examples() from form_example_ingester.
        example_search_fn: Callable[[str], list[dict[str, Any]]] | None = None,
        examples_per_field: int = 3,
    ) -> None:
        self.llm_call = llm_call_fn
        self.confidence_threshold = confidence_threshold
        self.example_search_fn = example_search_fn
        self.examples_per_field = examples_per_field

    def fill(
        self,
        parsed_form: ParseResult,
        company_context: str,
        user_answers: dict[str, str] | None = None,
    ) -> FillResult:
        """Fill form fields using company context and optionally user answers.

        Args:
            parsed_form: The parsed form with extracted fields.
            company_context: Text from company documents assigned in AMS.
            user_answers: Optional dict of field_name → user-provided value
                          (from the clarification flow).

        Returns:
            FillResult with filled fields and any remaining questions.
        """
        fields = parsed_form.fields

        if not fields:
            return FillResult(
                filled_fields=[],
                questions=[],
                total_fields=0,
                filled_count=0,
                needs_clarification_count=0,
            )

        # Step 1: If user provided answers, apply them directly
        filled: list[FilledField] = []
        remaining_fields: list[FormField] = []

        if user_answers:
            for f in fields:
                if f.name in user_answers and user_answers[f.name]:
                    filled.append(FilledField(
                        name=f.name,
                        value=user_answers[f.name],
                        confidence=1.0,
                        source="user_input",
                        reasoning="Provided by user",
                        original_field=f,
                    ))
                else:
                    remaining_fields.append(f)
        else:
            remaining_fields = fields

        if not remaining_fields:
            return FillResult(
                filled_fields=filled,
                questions=[],
                total_fields=len(fields),
                filled_count=len(filled),
                needs_clarification_count=0,
            )

        # Step 2: For each remaining field, fetch top-N similar past Q&As
        # (if a search callable was wired up). The bridge passes this in
        # from form_example_ingester.search_similar_examples; tests can
        # leave it None and the filler degrades to KB-only mode.
        field_examples: dict[str, list[dict[str, Any]]] = {}
        if self.example_search_fn:
            for f in remaining_fields:
                try:
                    matches = self.example_search_fn(f.name) or []
                except Exception as exc:
                    # Search failures must not break form filling — log
                    # and fall through to KB-only mode for this field.
                    print(f"[FormFiller] example search failed for '{f.name[:60]}': {exc}")
                    matches = []
                if matches:
                    field_examples[f.name] = matches[: self.examples_per_field]

        # Step 3: Call LLM to fill remaining fields, now with example
        # context per field as few-shot grounding.
        llm_filled, llm_cost, llm_tokens = self._llm_fill_fields(
            remaining_fields,
            company_context,
            parsed_form.raw_text,
            field_examples,
        )

        # Step 4: Separate high-confidence fills from uncertain ones
        questions: list[ClarificationQuestion] = []
        examples_used: set[str] = set()

        for ff in llm_filled:
            # Stamp the coarse confidence tier the Approvals UI uses
            ff.confidence_tier = _tier_for(ff.confidence, ff.source)
            # Track which past submissions grounded this field
            if ff.example_doc_ids:
                examples_used.update(ff.example_doc_ids)

            if ff.confidence >= self.confidence_threshold and ff.value:
                filled.append(ff)
            else:
                question = self._build_question(ff, company_context)
                questions.append(question)

        # Re-tier user_input fields that came in via step 1
        for ff in filled:
            if not ff.confidence_tier:
                ff.confidence_tier = _tier_for(ff.confidence, ff.source)

        high = sum(1 for ff in filled if ff.confidence_tier == "high")
        medium = sum(1 for ff in filled if ff.confidence_tier == "medium")
        low = sum(1 for ff in filled if ff.confidence_tier == "low")

        return FillResult(
            filled_fields=filled,
            questions=questions,
            total_fields=len(fields),
            filled_count=len(filled),
            needs_clarification_count=len(questions),
            llm_cost_usd=llm_cost,
            llm_tokens=llm_tokens,
            high_confidence_count=high,
            medium_confidence_count=medium,
            low_confidence_count=low,
            examples_used=sorted(examples_used),
        )

    def _llm_fill_fields(
        self,
        fields: list[FormField],
        company_context: str,
        form_text: str,
        field_examples: dict[str, list[dict[str, Any]]] | None = None,
    ) -> tuple[list[FilledField], float, int]:
        """Use the LLM to fill form fields from context + past examples.

        `field_examples` is a per-field dict of relevant past Q&A pairs
        retrieved by vector similarity. When present, the prompt
        includes them inline so the LLM has concrete grounding for
        how this company has answered similar questions before.
        """

        field_examples = field_examples or {}

        # Build the field list for the prompt, with inline examples
        # where we have them. The LLM is told to PREFER the example
        # answer when the question is essentially the same.
        field_descriptions = []
        for i, f in enumerate(fields, 1):
            desc = f"  {i}. \"{f.name}\""
            if f.field_type != "text":
                desc += f" (type: {f.field_type})"
            if f.options:
                desc += f" (options: {', '.join(f.options)})"
            if f.current_value:
                desc += f" (current: {f.current_value})"
            if f.page_or_section:
                desc += f" [{f.page_or_section}]"

            examples = field_examples.get(f.name) or []
            if examples:
                desc += "\n     Past answers to similar questions:"
                for j, ex in enumerate(examples, 1):
                    q_prev = (ex.get("questionText") or "").strip().replace("\n", " ")
                    a_prev = (ex.get("answerText") or "").strip().replace("\n", " ")
                    sim = float(ex.get("similarity") or 0)
                    # Truncate aggressively — we just need shape, not bulk.
                    q_prev = q_prev[:180] + ("…" if len(q_prev) > 180 else "")
                    a_prev = a_prev[:280] + ("…" if len(a_prev) > 280 else "")
                    desc += f"\n       • [{sim:.2f}] Q: {q_prev}\n         A: {a_prev}"

            field_descriptions.append(desc)

        field_list = "\n".join(field_descriptions)

        # Truncate context to avoid token limits
        max_context_chars = 8000
        if len(company_context) > max_context_chars:
            company_context = company_context[:max_context_chars] + "\n... [truncated]"
        if len(form_text) > 4000:
            form_text = form_text[:4000] + "\n... [truncated]"

        prompt = f"""You are a government procurement form-filling specialist. You need to fill out a tender/RFP form using company information AND, where available, past answers this company has given to similar questions.

## Company Information (from documents):
{company_context}

## Form Context (raw text from the document):
{form_text[:2000]}

## Fields to Fill:
{field_list}

## Instructions:
For each field, provide:
1. The best value based on (a) past answers to similar questions if shown, then (b) company information.
2. A confidence score (0.0 to 1.0) — how certain you are the value is correct.
3. Where the information came from:
   - "example_match"   — copied/adapted from a high-similarity past answer
   - "kb"              — drawn from company-information documents
   - "llm_inference"   — composed from both / inferred
4. If you drew on a past answer, list the document IDs of the example(s) you used (these will appear in the "Past answers" list metadata; an empty list is fine if you didn't use any).
5. Brief reasoning.

Rules:
- When a past answer is shown at similarity ≥0.80 and the question is essentially the same, REUSE that answer verbatim unless company facts contradict it.
- Use EXACT company details from the documents (names, ABN, addresses, etc.). Never fabricate.
- For dates, use the format shown in the form (or DD/MM/YYYY if unclear).
- For fields with low example similarity AND no clear company-info match, set confidence below 0.5.
- For fields that are clearly specific to THIS tender (budget, timeline, pricing), set confidence to 0.0 — those are for the human to fill.

Respond with ONLY valid JSON (no markdown, no explanation). Use this structure:
{{
  "fields": [
    {{
      "name": "field name exactly as listed",
      "value": "filled value",
      "confidence": 0.85,
      "source": "example_match | kb | llm_inference",
      "exampleIds": [],
      "reasoning": "brief reason"
    }}
  ]
}}"""

        try:
            result = self.llm_call(prompt, max_tokens=2000)
            content = result.get("content", "").strip()
            cost = result.get("cost_usd", 0.0)
            tokens = result.get("tokens_input", 0) + result.get("tokens_output", 0)

            # Parse JSON from response
            if "```" in content:
                content = content.split("```")[1].strip()
                if content.startswith("json"):
                    content = content[4:].strip()

            data = json.loads(content)
            llm_fields = data.get("fields", [])

            # Map back to FilledField objects
            filled: list[FilledField] = []
            field_map = {f.name: f for f in fields}

            for lf in llm_fields:
                name = lf.get("name", "")
                original = field_map.get(name)
                if not original:
                    # Try fuzzy match
                    for fn in field_map:
                        if fn.lower().strip() == name.lower().strip():
                            original = field_map[fn]
                            name = fn
                            break

                example_ids_raw = lf.get("exampleIds") or []
                example_ids = [
                    str(x) for x in example_ids_raw if isinstance(x, (str, int))
                ] if isinstance(example_ids_raw, list) else []

                filled.append(FilledField(
                    name=name,
                    value=str(lf.get("value", "")),
                    confidence=float(lf.get("confidence", 0.5)),
                    source=lf.get("source", "llm_inference"),
                    reasoning=lf.get("reasoning", ""),
                    example_doc_ids=example_ids,
                    original_field=original,
                ))

            return filled, cost, tokens

        except json.JSONDecodeError as exc:
            print(f"LLM response was not valid JSON: {exc}")
            # Return all fields as low confidence
            return [
                FilledField(
                    name=f.name, value="", confidence=0.0,
                    source="error", reasoning=f"JSON parse error: {exc}",
                    original_field=f,
                )
                for f in fields
            ], 0.0, 0

        except Exception as exc:
            print(f"LLM form filling failed: {exc}")
            return [
                FilledField(
                    name=f.name, value="", confidence=0.0,
                    source="error", reasoning=f"LLM error: {exc}",
                    original_field=f,
                )
                for f in fields
            ], 0.0, 0

    def _build_question(self, ff: FilledField, context: str) -> ClarificationQuestion:
        """Build a clarification question for a low-confidence field."""
        suggestions = []
        if ff.value:
            suggestions.append(ff.value)

        question_text = f"What should I put for **\"{ff.name}\"**?"
        if ff.reasoning:
            question_text += f"\n_(Context: {ff.reasoning})_"

        return ClarificationQuestion(
            field_name=ff.name,
            question=question_text,
            context=ff.reasoning,
            suggestions=suggestions,
        )
