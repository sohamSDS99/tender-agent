"""
Form Writer — Writes filled values back into the original file format.

Supports:
- PDF: Fill AcroForm fields, or create annotated overlay
- DOCX: Replace placeholders, fill table cells
- XLSX: Write values to identified cells

Usage:
    writer = FormWriter()
    output_path = writer.write(
        original_path="/path/to/form.pdf",
        filled_fields=[FilledField(name="Company", value="SDS Manager", ...)],
        output_path="/path/to/filled_form.pdf",
    )
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

from .form_parser import FileFormat, FormField
from .form_filler import FilledField


def _numeric_to_tier(confidence: float) -> str:
    """Fallback tier mapping when FilledField.confidence_tier is unset.

    Mirrors the thresholds in form_filler._tier_for so the canonical
    exports show the same colour-coding as the Approvals UI.
    """
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return "low"
    if c >= 0.8:
        return "high"
    if c >= 0.55:
        return "medium"
    return "low"


def _escape_pdf(text: str) -> str:
    """Escape characters that would otherwise break reportlab Paragraph parsing."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


class FormWriter:
    """Writes filled values back into form files."""

    def write(
        self,
        original_path: str,
        filled_fields: list[FilledField],
        output_path: str | None = None,
    ) -> str:
        """Write filled values into a copy of the original form.

        Args:
            original_path: Path to the original form file.
            filled_fields: List of fields with values to write.
            output_path: Where to save the filled form. Defaults to
                         original_name_FILLED.ext in the same directory.

        Returns:
            Path to the filled form file.
        """
        path = Path(original_path)
        if not path.exists():
            raise FileNotFoundError(f"Original file not found: {original_path}")

        # Determine output path
        if not output_path:
            stem = path.stem
            output_path = str(path.parent / f"{stem}_FILLED{path.suffix}")

        # Detect format and dispatch
        ext = path.suffix.lower()
        if ext == ".pdf":
            return self._write_pdf(path, filled_fields, output_path)
        elif ext in (".docx", ".doc"):
            return self._write_docx(path, filled_fields, output_path)
        elif ext in (".xlsx", ".xls"):
            return self._write_xlsx(path, filled_fields, output_path)
        else:
            return self._write_text_fallback(path, filled_fields, output_path)

    # -----------------------------------------------------------------------
    # PDF Writing
    # -----------------------------------------------------------------------

    def _write_pdf(
        self, path: Path, fields: list[FilledField], output_path: str
    ) -> str:
        """Fill a PDF form.

        Strategy 1: Use pypdf to fill AcroForm fields (fillable PDFs)
        Strategy 2: For non-fillable PDFs, create a new PDF with filled values
        """
        field_map = {f.name: f.value for f in fields if f.value}

        try:
            import pypdf

            reader = pypdf.PdfReader(str(path))
            writer = pypdf.PdfWriter()

            # Copy all pages
            for page in reader.pages:
                writer.add_page(page)

            # Try to fill AcroForm fields
            has_form = reader.get_form_text_fields() is not None
            if has_form:
                writer.update_page_form_field_values(
                    writer.pages[0], field_map
                )
                # Also try updating all pages
                for i, page in enumerate(writer.pages):
                    try:
                        writer.update_page_form_field_values(page, field_map)
                    except Exception:
                        pass

            with open(output_path, "wb") as f:
                writer.write(f)

            print(f"PDF form filled: {output_path}")
            return output_path

        except ImportError:
            # Fallback: copy the original and note that it couldn't be auto-filled
            shutil.copy2(str(path), output_path)
            print(f"pypdf not available — copied original to {output_path}")
            return output_path

        except Exception as exc:
            print(f"PDF write error: {exc}")
            # Copy original as fallback
            shutil.copy2(str(path), output_path)
            return output_path

    # -----------------------------------------------------------------------
    # DOCX Writing
    # -----------------------------------------------------------------------

    def _write_docx(
        self, path: Path, fields: list[FilledField], output_path: str
    ) -> str:
        """Fill a Word document form.

        Strategy 1: Replace values in table cells
        Strategy 2: Replace placeholder patterns in paragraphs
        """
        field_map = {f.name: f.value for f in fields if f.value}

        try:
            from docx import Document

            doc = Document(str(path))

            # Strategy 1: Fill table cells
            for table in doc.tables:
                for row in table.rows:
                    cells = row.cells
                    if len(cells) >= 2:
                        label = cells[0].text.strip().rstrip(":").rstrip("?").strip()
                        if label in field_map:
                            cells[1].text = field_map[label]

            # Strategy 2: Replace placeholder patterns
            for para in doc.paragraphs:
                original_text = para.text

                for name, value in field_map.items():
                    # Replace [Field Name] → value
                    pattern1 = f"[{name}]"
                    if pattern1 in original_text:
                        original_text = original_text.replace(pattern1, value)

                    # Replace {{field_name}} → value
                    snake_name = name.lower().replace(" ", "_")
                    pattern2 = f"{{{{{snake_name}}}}}"
                    if pattern2 in original_text:
                        original_text = original_text.replace(pattern2, value)

                    # Replace <FIELD NAME> → value
                    pattern3 = f"<{name.upper()}>"
                    if pattern3 in original_text:
                        original_text = original_text.replace(pattern3, value)

                # Replace _____ blanks near matching labels
                for name, value in field_map.items():
                    label_pattern = re.escape(name) + r'\s*:?\s*[_]{3,}'
                    replacement = f"{name}: {value}"
                    original_text = re.sub(label_pattern, replacement, original_text, flags=re.IGNORECASE)

                if original_text != para.text:
                    # Clear existing runs and set new text
                    # Preserve formatting of the first run
                    if para.runs:
                        first_run = para.runs[0]
                        for run in para.runs[1:]:
                            run.text = ""
                        first_run.text = original_text
                    else:
                        para.text = original_text

            doc.save(output_path)
            print(f"DOCX form filled: {output_path}")
            return output_path

        except ImportError:
            shutil.copy2(str(path), output_path)
            print(f"python-docx not available — copied original to {output_path}")
            return output_path

        except Exception as exc:
            print(f"DOCX write error: {exc}")
            shutil.copy2(str(path), output_path)
            return output_path

    # -----------------------------------------------------------------------
    # XLSX Writing
    # -----------------------------------------------------------------------

    def _write_xlsx(
        self, path: Path, fields: list[FilledField], output_path: str
    ) -> str:
        """Fill an Excel form by writing to identified cells."""
        field_map = {f.name: f for f in fields if f.value}

        try:
            from openpyxl import load_workbook

            wb = load_workbook(str(path))

            for ff in fields:
                if not ff.value or not ff.original_field:
                    continue

                meta = ff.original_field.metadata
                cell_ref = meta.get("cell_ref", "")
                sheet_name = meta.get("sheet", "")

                if cell_ref and sheet_name and sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                    ws[cell_ref] = ff.value
                else:
                    # Fallback: search for the label in all sheets and fill adjacent cell
                    for ws in wb.worksheets:
                        for row in ws.iter_rows():
                            for i, cell in enumerate(row):
                                label = str(cell.value or "").strip().rstrip(":").rstrip("?").strip()
                                if label == ff.name and i + 1 < len(row):
                                    row[i + 1].value = ff.value
                                    break

            wb.save(output_path)
            print(f"XLSX form filled: {output_path}")
            return output_path

        except ImportError:
            shutil.copy2(str(path), output_path)
            print(f"openpyxl not available — copied original to {output_path}")
            return output_path

        except Exception as exc:
            print(f"XLSX write error: {exc}")
            shutil.copy2(str(path), output_path)
            return output_path

    # -----------------------------------------------------------------------
    # Canonical multi-format export
    # -----------------------------------------------------------------------

    def write_canonical_set(
        self,
        filled_fields: list[FilledField],
        output_dir: str,
        base_name: str,
        *,
        title: str = "Filled Tender Form",
        include_low_confidence: bool = True,
    ) -> dict[str, str]:
        """Render the filled answers into three CLEAN editable formats.

        Unlike `write()` (which preserves the original document's
        template, logos, sections, etc.), this method generates
        canonical answer-only documents — flat Q&A tables ready for
        review, editing, or sharing.

        Always returns paths to all three formats; if a renderer
        fails the path is still returned but the file is best-effort.

        Args:
            filled_fields: The answers produced by FormFiller.
            output_dir: Directory to write the files into.
            base_name: Filename stem (no extension). The output files
                       will be `<base_name>_answers.{docx,xlsx,pdf}`.
            title: Human-readable title rendered at the top of each doc.
            include_low_confidence: When True, low-confidence fields are
                       rendered with a marker so the reviewer notices.

        Returns:
            Dict mapping format → absolute output path. Always has
            keys "docx", "xlsx", "pdf".
        """
        os.makedirs(output_dir, exist_ok=True)
        docx_path = os.path.join(output_dir, f"{base_name}_answers.docx")
        xlsx_path = os.path.join(output_dir, f"{base_name}_answers.xlsx")
        pdf_path = os.path.join(output_dir, f"{base_name}_answers.pdf")

        rows = self._canonical_rows(filled_fields, include_low_confidence)

        try:
            self._write_canonical_docx(rows, docx_path, title=title)
        except Exception as exc:
            print(f"[canonical docx] {type(exc).__name__}: {exc}")

        try:
            self._write_canonical_xlsx(rows, xlsx_path, title=title)
        except Exception as exc:
            print(f"[canonical xlsx] {type(exc).__name__}: {exc}")

        try:
            self._write_canonical_pdf(rows, pdf_path, title=title)
        except Exception as exc:
            print(f"[canonical pdf] {type(exc).__name__}: {exc}")

        return {"docx": docx_path, "xlsx": xlsx_path, "pdf": pdf_path}

    # ---- canonical row builder ----

    @staticmethod
    def _canonical_rows(
        fields: list[FilledField],
        include_low_confidence: bool,
    ) -> list[dict[str, str]]:
        """Flatten FilledField -> {question, answer, confidence_tier}."""
        rows: list[dict[str, str]] = []
        for ff in fields:
            value = (ff.value or "").strip()
            if not value:
                if not include_low_confidence:
                    continue
                value = "(needs human input)"
            tier = getattr(ff, "confidence_tier", "") or ""
            rows.append({
                "question": (ff.name or "").strip(),
                "answer": value,
                "confidence": tier or _numeric_to_tier(ff.confidence),
                "source": (ff.source or "").strip(),
            })
        return rows

    # ---- DOCX (editable) ----

    @staticmethod
    def _write_canonical_docx(
        rows: list[dict[str, str]],
        output_path: str,
        *,
        title: str,
    ) -> None:
        """Render the answers as a clean, editable DOCX.

        Layout: title at top, summary line, then a 3-column table
        (Question | Answer | Confidence). The DOCX is fully editable
        in Word, Google Docs, LibreOffice.
        """
        from docx import Document
        from docx.shared import Inches, Pt, RGBColor
        from docx.enum.table import WD_TABLE_ALIGNMENT

        doc = Document()

        # --- Title ---
        heading = doc.add_heading(title, level=1)
        for run in heading.runs:
            run.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)

        # --- Summary line ---
        high = sum(1 for r in rows if r["confidence"] == "high")
        medium = sum(1 for r in rows if r["confidence"] == "medium")
        low = sum(1 for r in rows if r["confidence"] == "low")
        summary = doc.add_paragraph()
        summary_run = summary.add_run(
            f"{len(rows)} answers · {high} high · {medium} medium · {low} low confidence"
        )
        summary_run.italic = True
        summary_run.font.size = Pt(10)
        summary_run.font.color.rgb = RGBColor(0x6B, 0x72, 0x80)

        # --- Q&A table ---
        if rows:
            table = doc.add_table(rows=1, cols=3)
            table.style = "Light Grid Accent 1"
            table.alignment = WD_TABLE_ALIGNMENT.LEFT
            table.autofit = False

            header_cells = table.rows[0].cells
            for idx, label in enumerate(("Question", "Answer", "Confidence")):
                cell = header_cells[idx]
                cell.text = label
                for run in cell.paragraphs[0].runs:
                    run.bold = True
                    run.font.size = Pt(10)

            # Column widths
            try:
                table.columns[0].width = Inches(2.4)
                table.columns[1].width = Inches(3.8)
                table.columns[2].width = Inches(1.0)
            except Exception:
                pass

            for r in rows:
                cells = table.add_row().cells
                cells[0].text = r["question"]
                cells[1].text = r["answer"]
                cells[2].text = (r["confidence"] or "—").upper()
                # subtle styling: dim low-confidence cells
                if r["confidence"] == "low":
                    for cell in cells:
                        for paragraph in cell.paragraphs:
                            for run in paragraph.runs:
                                run.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)

        doc.save(output_path)

    # ---- XLSX (editable) ----

    @staticmethod
    def _write_canonical_xlsx(
        rows: list[dict[str, str]],
        output_path: str,
        *,
        title: str,
    ) -> None:
        """Render the answers as a clean, editable XLSX spreadsheet."""
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter

        wb = Workbook()
        ws = wb.active
        ws.title = "Answers"

        # Title row
        ws["A1"] = title
        ws["A1"].font = Font(size=14, bold=True, color="1F2937")
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=3)

        # Summary row
        high = sum(1 for r in rows if r["confidence"] == "high")
        medium = sum(1 for r in rows if r["confidence"] == "medium")
        low = sum(1 for r in rows if r["confidence"] == "low")
        ws["A2"] = (
            f"{len(rows)} answers · {high} high · {medium} medium · {low} low confidence"
        )
        ws["A2"].font = Font(italic=True, color="6B7280", size=10)
        ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=3)

        # Header row
        headers = ["Question", "Answer", "Confidence"]
        for col_idx, label in enumerate(headers, start=1):
            cell = ws.cell(row=4, column=col_idx, value=label)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="4F46E5")
            cell.alignment = Alignment(horizontal="left", vertical="center")

        # Data rows
        tier_fill = {
            "high": PatternFill("solid", fgColor="ECFDF5"),
            "medium": PatternFill("solid", fgColor="FFFBEB"),
            "low": PatternFill("solid", fgColor="FEF2F2"),
        }
        for offset, r in enumerate(rows, start=5):
            ws.cell(row=offset, column=1, value=r["question"]).alignment = (
                Alignment(wrap_text=True, vertical="top")
            )
            ws.cell(row=offset, column=2, value=r["answer"]).alignment = (
                Alignment(wrap_text=True, vertical="top")
            )
            tier_cell = ws.cell(
                row=offset, column=3, value=(r["confidence"] or "—").upper()
            )
            tier_cell.alignment = Alignment(horizontal="center", vertical="top")
            fill = tier_fill.get(r["confidence"])
            if fill:
                for col_idx in (1, 2, 3):
                    ws.cell(row=offset, column=col_idx).fill = fill

        # Column widths
        ws.column_dimensions[get_column_letter(1)].width = 38
        ws.column_dimensions[get_column_letter(2)].width = 56
        ws.column_dimensions[get_column_letter(3)].width = 14

        # Freeze the header
        ws.freeze_panes = "A5"

        wb.save(output_path)

    # ---- PDF (canonical printable) ----

    @staticmethod
    def _write_canonical_pdf(
        rows: list[dict[str, str]],
        output_path: str,
        *,
        title: str,
    ) -> None:
        """Render the answers as a clean PDF report.

        Uses reportlab platypus so the layout flows across pages and
        long answers wrap correctly.
        """
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
        )

        styles = getSampleStyleSheet()
        body_style = ParagraphStyle(
            name="body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
        )
        title_style = ParagraphStyle(
            name="title",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=16,
            textColor=colors.HexColor("#1F2937"),
            spaceAfter=4,
        )
        summary_style = ParagraphStyle(
            name="summary",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            textColor=colors.HexColor("#6B7280"),
            spaceAfter=12,
        )

        doc = SimpleDocTemplate(
            output_path,
            pagesize=LETTER,
            leftMargin=0.6 * inch,
            rightMargin=0.6 * inch,
            topMargin=0.6 * inch,
            bottomMargin=0.6 * inch,
        )
        story: list = []

        story.append(Paragraph(title, title_style))
        high = sum(1 for r in rows if r["confidence"] == "high")
        medium = sum(1 for r in rows if r["confidence"] == "medium")
        low = sum(1 for r in rows if r["confidence"] == "low")
        story.append(
            Paragraph(
                f"{len(rows)} answers · {high} high · {medium} medium · {low} low confidence",
                summary_style,
            )
        )

        if rows:
            # Build the Q&A table. Wrap each cell in a Paragraph so
            # long answers flow correctly instead of overflowing.
            data = [
                [
                    Paragraph("<b>Question</b>", body_style),
                    Paragraph("<b>Answer</b>", body_style),
                    Paragraph("<b>Conf.</b>", body_style),
                ]
            ]
            tier_color = {
                "high": colors.HexColor("#ECFDF5"),
                "medium": colors.HexColor("#FFFBEB"),
                "low": colors.HexColor("#FEF2F2"),
            }
            row_styles = []
            for idx, r in enumerate(rows, start=1):
                data.append([
                    Paragraph(_escape_pdf(r["question"]), body_style),
                    Paragraph(_escape_pdf(r["answer"]), body_style),
                    Paragraph(
                        (r["confidence"] or "—").upper(),
                        body_style,
                    ),
                ])
                bg = tier_color.get(r["confidence"])
                if bg:
                    row_styles.append(("BACKGROUND", (0, idx), (-1, idx), bg))

            table = Table(
                data,
                colWidths=[2.4 * inch, 4.0 * inch, 0.6 * inch],
                repeatRows=1,
            )
            base_styles = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
            table.setStyle(TableStyle(base_styles + row_styles))
            story.append(table)
        else:
            story.append(Paragraph("(No answers produced.)", body_style))

        story.append(Spacer(1, 0.2 * inch))
        doc.build(story)

    # -----------------------------------------------------------------------
    # Text fallback
    # -----------------------------------------------------------------------

    def _write_text_fallback(
        self, path: Path, fields: list[FilledField], output_path: str
    ) -> str:
        """For unknown formats, create a companion text file with the answers."""
        lines = ["# Tender Form — Filled Fields\n"]
        lines.append(f"Original file: {path.name}\n")
        lines.append("=" * 60 + "\n")

        for ff in fields:
            if ff.value:
                lines.append(f"{ff.name}: {ff.value}")
                if ff.confidence < 0.7:
                    lines.append(f"  ⚠ Low confidence ({ff.confidence:.0%})")
                lines.append("")

        # Write as .txt alongside the original
        txt_output = str(Path(output_path).with_suffix(".txt"))
        with open(txt_output, "w") as f:
            f.write("\n".join(lines))

        # Also copy the original
        shutil.copy2(str(path), output_path)
        print(f"Text fallback: answers written to {txt_output}")
        return output_path
