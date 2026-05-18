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


@dataclass
class FilledField:
    """A form field with its filled value and confidence score."""
    name: str
    value: str
    confidence: float        # 0.0 - 1.0
    source: str = ""         # Where the value came from ("context_doc", "llm_inference", "user_input")
    reasoning: str = ""      # Why this value was chosen
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
    ) -> None:
        self.llm_call = llm_call_fn
        self.confidence_threshold = confidence_threshold

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

        # Step 2: Call LLM to fill remaining fields
        llm_filled, llm_cost, llm_tokens = self._llm_fill_fields(
            remaining_fields, company_context, parsed_form.raw_text
        )

        # Step 3: Separate high-confidence fills from uncertain ones
        questions: list[ClarificationQuestion] = []

        for ff in llm_filled:
            if ff.confidence >= self.confidence_threshold and ff.value:
                filled.append(ff)
            else:
                # Generate a clarification question for this field
                question = self._build_question(ff, company_context)
                questions.append(question)

        return FillResult(
            filled_fields=filled,
            questions=questions,
            total_fields=len(fields),
            filled_count=len(filled),
            needs_clarification_count=len(questions),
            llm_cost_usd=llm_cost,
            llm_tokens=llm_tokens,
        )

    def _llm_fill_fields(
        self,
        fields: list[FormField],
        company_context: str,
        form_text: str,
    ) -> tuple[list[FilledField], float, int]:
        """Use the LLM to fill form fields from context."""

        # Build the field list for the prompt
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
            field_descriptions.append(desc)

        field_list = "\n".join(field_descriptions)

        # Truncate context to avoid token limits
        max_context_chars = 8000
        if len(company_context) > max_context_chars:
            company_context = company_context[:max_context_chars] + "\n... [truncated]"
        if len(form_text) > 4000:
            form_text = form_text[:4000] + "\n... [truncated]"

        prompt = f"""You are a government procurement form-filling specialist. You need to fill out a tender/RFP form using company information.

## Company Information (from documents):
{company_context}

## Form Context (raw text from the document):
{form_text[:2000]}

## Fields to Fill:
{field_list}

## Instructions:
For each field, provide:
1. The best value based on the company information
2. A confidence score (0.0 to 1.0) — how certain you are the value is correct
3. Where the information came from (specific document or inference)
4. Brief reasoning

Rules:
- Use EXACT company details from the documents (names, ABN, addresses, etc.)
- For dates, use the format shown in the form (or DD/MM/YYYY if unclear)
- For fields you cannot fill confidently, set confidence below 0.5
- For fields that are clearly specific to THIS tender (budget, timeline, pricing), set confidence to 0.0
- Never fabricate company information — if it's not in the documents, say so

Respond with ONLY valid JSON (no markdown, no explanation). Use this structure:
{{
  "fields": [
    {{
      "name": "field name exactly as listed",
      "value": "filled value",
      "confidence": 0.85,
      "source": "company context or inference",
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

                filled.append(FilledField(
                    name=name,
                    value=str(lf.get("value", "")),
                    confidence=float(lf.get("confidence", 0.5)),
                    source=lf.get("source", "llm_inference"),
                    reasoning=lf.get("reasoning", ""),
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
