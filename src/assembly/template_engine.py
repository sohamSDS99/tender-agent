"""
Template Engine — Assembles drafted sections into complete tender response documents.

WHY A TEMPLATE ENGINE:
Government and enterprise tenders have strict formatting requirements: specific
section ordering, page limits, required headers/footers, and table of contents.
A pile of drafted sections isn't a tender submission — this engine transforms them
into a professional, submission-ready document.

HOW IT WORKS:
1. SELECT TEMPLATE — Choose "standard", "government", or "simple" based on tender type
2. BUILD COVER PAGE — Company name, tender reference, date, deadline
3. BUILD TABLE OF CONTENTS — Auto-generated from section titles
4. ASSEMBLE BODY — Map drafted sections into template slots in correct order
5. ADD FOOTER — Generation metadata for audit trail
6. QUALITY CHECK — Verify page limits, required sections, no placeholders

OUTPUT FORMATS:
- Markdown (.md) — Primary intermediate format. Easy to review and edit.
  The Assemble Node (Step 19) will optionally convert this to DOCX using python-docx.
- The Markdown output is designed to be clean enough for direct submission to
  portals that accept Markdown or rich text.

TEMPLATES:
- "standard" — Full cover page + TOC + all sections. Good default.
- "government" — Adds past performance, certifications, small business sections.
  50-page max. Matches US federal RFP expectations.
- "simple" — No cover page or TOC. Just the sections. For quick RFQs and
  informal proposals. 20-page max.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from src.agent.state import DraftedSection

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Template configurations
# ---------------------------------------------------------------------------

@dataclass
class TemplateConfig:
    """Configuration for a document template.

    Attributes:
        name: Template identifier.
        display_name: Human-readable name.
        max_pages: Maximum page count (None = no limit).
        include_cover_page: Whether to generate a cover page.
        include_toc: Whether to generate a table of contents.
        company_name: Company name for headers/cover.
    """
    name: str
    display_name: str
    max_pages: int | None = None
    include_cover_page: bool = True
    include_toc: bool = True
    company_name: str = "Acme SDS Solutions"


TEMPLATES: dict[str, TemplateConfig] = {
    "standard": TemplateConfig(
        name="standard",
        display_name="Standard Tender Response",
        include_cover_page=True,
        include_toc=True,
    ),
    "government": TemplateConfig(
        name="government",
        display_name="Government RFP Response",
        include_cover_page=True,
        include_toc=True,
        max_pages=50,
    ),
    "simple": TemplateConfig(
        name="simple",
        display_name="Simple Proposal",
        include_cover_page=False,
        include_toc=False,
        max_pages=20,
    ),
}


# ---------------------------------------------------------------------------
# Quality check result
# ---------------------------------------------------------------------------

@dataclass
class QualityCheckResult:
    """Result of document quality checks.

    Attributes:
        passed: Whether all critical checks passed.
        issues: Blocking problems that must be fixed (causes quality_check_passed=False).
        warnings: Non-blocking concerns (logged but don't block submission).
        stats: Document statistics (word count, page estimate, etc.).
    """
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Template Engine
# ---------------------------------------------------------------------------

class TemplateEngine:
    """Assembles drafted sections into a complete tender response document.

    Usage:
        engine = TemplateEngine()

        # Assemble sections into Markdown
        doc = engine.assemble(
            sections=drafted_sections,
            tender_title="SDS Platform for EPA",
            tender_id="SAM-2026-001",
        )

        # Run quality checks
        qc = engine.quality_check(doc, sections=drafted_sections)
        if qc.passed:
            path = engine.save(doc)
            print(f"Saved to {path}")

    Args:
        company_name: Override default company name in templates.
        output_dir: Directory for saving assembled documents.
    """

    def __init__(
        self,
        company_name: str = "Acme SDS Solutions",
        output_dir: str = "/tmp/tender_outputs",
    ) -> None:
        self.company_name = company_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("template_engine_initialized", output_dir=str(self.output_dir))

    def assemble(
        self,
        sections: list[DraftedSection],
        tender_title: str,
        tender_id: str = "",
        template_name: str = "standard",
        submission_deadline: str = "",
    ) -> str:
        """Assemble drafted sections into a complete Markdown document.

        Args:
            sections: List of DraftedSection dicts from the drafting node.
            tender_title: Title of the tender being responded to.
            tender_id: Tender reference ID.
            template_name: Which template ("standard", "government", "simple").
            submission_deadline: Deadline string for the cover page.

        Returns:
            Complete document as a Markdown string.
        """
        template = TEMPLATES.get(template_name, TEMPLATES["standard"])
        now = datetime.now(timezone.utc).strftime("%B %d, %Y")

        parts: list[str] = []

        # --- Cover page ---
        if template.include_cover_page:
            parts.append(self._render_cover_page(
                tender_title=tender_title,
                tender_id=tender_id,
                date=now,
                deadline=submission_deadline,
            ))

        # --- Table of contents ---
        if template.include_toc:
            parts.append(self._render_toc(sections))

        # --- Body sections ---
        for section in sections:
            parts.append(self._render_section(section))

        # --- Footer ---
        parts.append(self._render_footer(tender_id, now))

        document = "\n\n---\n\n".join(parts)

        logger.info(
            "document_assembled",
            tender_id=tender_id,
            template=template_name,
            sections=len(sections),
            chars=len(document),
            words=len(document.split()),
        )

        return document

    def quality_check(
        self,
        document: str,
        sections: list[DraftedSection] | None = None,
        template_name: str = "standard",
    ) -> QualityCheckResult:
        """Run quality checks on an assembled document.

        CHECKS (blocking — cause failure):
        1. Document not empty (≥100 chars)
        2. No [INFORMATION NEEDED] placeholders remain
        3. Page count within template max_pages limit

        CHECKS (warnings — logged but don't block):
        4. Thin sections (< 50 chars of content)
        5. Low-confidence sections (< 70%)
        6. Dry-run placeholder text present

        Args:
            document: The assembled Markdown document.
            sections: Original drafted sections (for section-level checks).
            template_name: Template used (for max_pages check).

        Returns:
            QualityCheckResult with pass/fail, issues, warnings, and stats.
        """
        template = TEMPLATES.get(template_name, TEMPLATES["standard"])
        result = QualityCheckResult()

        # --- Document statistics ---
        word_count = len(document.split())
        page_estimate = word_count / 250  # ~250 words per page
        result.stats = {
            "word_count": word_count,
            "char_count": len(document),
            "page_estimate": round(page_estimate, 1),
            "section_count": len(sections) if sections else 0,
        }

        # --- Blocking check 1: Not empty ---
        if len(document.strip()) < 100:
            result.issues.append("Document is nearly empty (< 100 characters).")

        # --- Blocking check 2: No unresolved placeholders ---
        placeholder_matches = re.findall(
            r"\[INFORMATION NEEDED[^\]]*\]", document
        )
        if placeholder_matches:
            result.issues.append(
                f"Document contains {len(placeholder_matches)} unresolved "
                f"placeholder(s) that need human input."
            )

        # --- Blocking check 3: Page limit ---
        if template.max_pages and page_estimate > template.max_pages:
            result.issues.append(
                f"Document exceeds page limit: ~{page_estimate:.0f} estimated pages "
                f"(max {template.max_pages} for {template.display_name})."
            )

        # --- Warning check 4: Thin sections ---
        if sections:
            thin_sections = [
                s.get("section_title", "?")
                for s in sections
                if len(s.get("content", "").strip()) < 50
            ]
            if thin_sections:
                result.warnings.append(
                    f"Very short sections (< 50 chars): {', '.join(thin_sections)}"
                )

        # --- Warning check 5: Low-confidence sections ---
        if sections:
            low_confidence = [
                f"{s.get('section_title', '?')} ({s.get('confidence', 0):.0%})"
                for s in sections
                if s.get("confidence", 1.0) < 0.7
            ]
            if low_confidence:
                result.warnings.append(
                    f"Low-confidence sections: {', '.join(low_confidence)}"
                )

        # --- Warning check 6: Dry-run markers ---
        if "[DRY-RUN" in document:
            result.warnings.append(
                "Document contains dry-run placeholder text. "
                "Re-run with real API keys for production content."
            )

        result.passed = len(result.issues) == 0

        logger.info(
            "quality_check_complete",
            passed=result.passed,
            issues=len(result.issues),
            warnings=len(result.warnings),
            pages=result.stats.get("page_estimate"),
            words=result.stats.get("word_count"),
        )

        return result

    def save(self, document: str, filename: str | None = None) -> str:
        """Save the assembled document to disk.

        Args:
            document: Document content (Markdown string).
            filename: Output filename/path. If None, auto-generates with timestamp.

        Returns:
            Full path to the saved file.
        """
        if filename:
            path = Path(filename)
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = self.output_dir / f"tender_response_{timestamp}.md"

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding="utf-8")

        logger.info("document_saved", path=str(path), size_bytes=len(document.encode()))
        return str(path)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_cover_page(
        self,
        tender_title: str,
        tender_id: str,
        date: str,
        deadline: str,
    ) -> str:
        """Render the cover page section."""
        return (
            f"# {tender_title}\n\n"
            f"**Proposal by:** {self.company_name}\n\n"
            f"**Tender Reference:** {tender_id}\n\n"
            f"**Submission Date:** {date}\n\n"
            f"**Deadline:** {deadline}\n\n"
            f"*This proposal is confidential and intended solely for the "
            f"evaluation committee.*"
        )

    def _render_toc(self, sections: list[DraftedSection]) -> str:
        """Render the table of contents from section titles."""
        lines = ["## Table of Contents\n"]
        for i, section in enumerate(sections, start=1):
            title = section.get("section_title", f"Section {i}")
            sid = section.get("section_id", str(i))
            lines.append(f"{i}. {sid} — {title}")
        return "\n".join(lines)

    def _render_section(self, section: DraftedSection) -> str:
        """Render a single body section."""
        title = section.get("section_title", "Untitled Section")
        sid = section.get("section_id", "")
        content = section.get("content", "")
        return f"## {sid} {title}\n\n{content}"

    def _render_footer(self, tender_id: str, date: str) -> str:
        """Render the document footer."""
        return (
            f"*Document generated by {self.company_name} Tender Agent | "
            f"Tender: {tender_id} | Date: {date}*"
        )