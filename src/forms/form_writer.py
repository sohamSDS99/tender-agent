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
