#!/usr/bin/env python3
"""
Step 5 Verification — Document Ingestion Pipeline Tests

Runs 5 tests to verify that parsing, chunking, and the pipeline work correctly.
Creates temporary test files (no real company docs needed).

Usage:
    cd /Users/sohamsarker/tender-agent
    source .venv/bin/activate
    python scripts/test_ingestion.py

Expected output: All 5 tests pass.
"""

from __future__ import annotations

import sys
import tempfile
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_test_files(tmpdir: Path) -> dict[str, Path]:
    """Create temporary test documents for ingestion testing.

    Returns a dict mapping descriptive names to file paths.
    """
    files: dict[str, Path] = {}

    # --- Test TXT file ---
    txt_path = tmpdir / "company_overview.txt"
    txt_path.write_text(textwrap.dedent("""\
        Acme SDS Solutions is a leading provider of Safety Data Sheet management
        software. Founded in 2015, the company serves over 500 clients across
        manufacturing, construction, oil and gas, and pharmaceutical industries.

        Our platform provides cloud-based SDS management with features including
        GHS-compliant label generation, chemical inventory tracking, regulatory
        reporting automation, and mobile access via QR codes. We support OSHA HCS,
        WHMIS, CLP/REACH, and other global regulatory frameworks.

        The company holds ISO 27001 certification for information security and
        SOC 2 Type II compliance. Our data centres are located in AWS US-East
        and AWS EU-West regions, providing 99.95% uptime SLA.

        Key differentiators include our AI-powered SDS authoring engine, which
        reduces document creation time by 75%, and our regulatory change monitoring
        service that tracks updates across 140 jurisdictions in real time.
    """), encoding="utf-8")
    files["txt"] = txt_path

    # --- Test Markdown file ---
    md_path = tmpdir / "team_bios.md"
    md_path.write_text(textwrap.dedent("""\
        # Leadership Team

        ## Jane Chen — CEO & Co-Founder
        Jane has 15 years of experience in environmental health and safety.
        Before founding Acme SDS Solutions, she led the EHS division at
        ChemCorp International. She holds a PhD in Chemistry from MIT.

        ## Raj Patel — CTO & Co-Founder
        Raj brings 12 years of enterprise SaaS engineering experience.
        Previously a Staff Engineer at Google Cloud, he designed the
        distributed systems architecture that powers our platform.

        ## Maria Santos — VP of Regulatory Affairs
        Maria is a certified Dangerous Goods Safety Advisor (DGSA) with
        expertise in GHS classification, REACH registration, and
        OSHA compliance. She has authored over 2,000 Safety Data Sheets.
    """), encoding="utf-8")
    files["md"] = md_path

    # --- Test long file for chunking ---
    long_path = tmpdir / "certifications.txt"
    # Create a file long enough to require multiple chunks
    paragraphs = []
    for i in range(1, 21):
        paragraphs.append(
            f"Certification {i}: Acme SDS Solutions maintains compliance with "
            f"standard {i} across all operational regions. This certification was "
            f"first obtained in {2010 + i} and has been renewed annually since then. "
            f"The scope covers all aspects of our SDS management platform including "
            f"data storage, processing, transmission, and access controls. "
            f"Independent auditors from Ernst & Young have verified compliance for "
            f"the past {i} consecutive years without any major findings."
        )
    long_path.write_text("\n\n".join(paragraphs), encoding="utf-8")
    files["long"] = long_path

    return files


def test_1_parser_txt(files: dict[str, Path]) -> None:
    """Test 1: Parse a plain text file."""
    from src.ingestion.parser import DocumentParser

    parser = DocumentParser()
    pages = parser.parse(files["txt"])

    assert len(pages) == 1, f"Expected 1 page, got {len(pages)}"
    assert pages[0].source_file == "company_overview.txt"
    assert pages[0].page_number == 1
    assert pages[0].char_count > 100, f"Too few chars: {pages[0].char_count}"
    assert "ISO 27001" in pages[0].text, "Expected 'ISO 27001' in text"
    assert pages[0].content_hash, "content_hash should not be empty"

    print("  ✅ Test 1 passed: TXT parsing works correctly")


def test_2_parser_md(files: dict[str, Path]) -> None:
    """Test 2: Parse a Markdown file."""
    from src.ingestion.parser import DocumentParser

    parser = DocumentParser()
    pages = parser.parse(files["md"])

    assert len(pages) == 1, f"Expected 1 page, got {len(pages)}"
    assert "Jane Chen" in pages[0].text
    assert "Raj Patel" in pages[0].text
    assert pages[0].source_file == "team_bios.md"

    print("  ✅ Test 2 passed: Markdown parsing works correctly")


def test_3_parser_unsupported_extension() -> None:
    """Test 3: Parsing an unsupported file type raises ValueError."""
    from src.ingestion.parser import DocumentParser

    parser = DocumentParser()
    try:
        parser.parse("/tmp/fake_file.xlsx")
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "Unsupported file type" in str(exc)

    print("  ✅ Test 3 passed: Unsupported file types are rejected correctly")


def test_4_chunker(files: dict[str, Path]) -> None:
    """Test 4: Chunking a long document produces multiple overlapping chunks."""
    from src.ingestion.chunker import TextChunker
    from src.ingestion.parser import DocumentParser

    parser = DocumentParser()
    pages = parser.parse(files["long"])

    # Use a small chunk size to force multiple chunks.
    # chunk_overlap=300 ensures at least one ~450-char sentence fits in the
    # overlap window (each test paragraph is ~450 chars).
    chunker = TextChunker(chunk_size=1000, chunk_overlap=500, min_chunk_size=50)
    chunks = chunker.chunk_pages(pages)

    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"

    # Verify each chunk has correct metadata
    for chunk in chunks:
        assert chunk.source_file == "certifications.txt"
        assert chunk.page_number == 1
        assert chunk.char_count > 0
        assert chunk.content_hash, "content_hash should not be empty"

    # Verify overlap: consecutive chunks should share at least one sentence.
    # We check that some words from the end of chunk 0 appear in chunk 1.
    if len(chunks) >= 2:
        # Extract significant words from the last third of chunk 0
        words_end = [
            w for w in chunks[0].text.split()
            if len(w) > 5  # skip short common words
        ][-10:]  # last 10 significant words
        # At least 3 of those words should appear in chunk 1
        matches = sum(1 for w in words_end if w in chunks[1].text)
        assert matches >= 3, (
            f"Expected overlap between chunks, but only {matches}/10 words matched"
        )

    # Verify chunk indices are sequential
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks))), (
        f"Chunk indices should be sequential: {indices}"
    )

    print(f"  ✅ Test 4 passed: Chunking produced {len(chunks)} chunks with overlap")


def test_5_dry_run_pipeline(tmpdir: Path) -> None:
    """Test 5: Dry-run pipeline processes all files without database."""
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(chunk_size=1000, chunk_overlap=100)
    result = pipeline.ingest_directory_dry_run(tmpdir)

    assert result.files_processed >= 3, (
        f"Expected ≥3 files processed, got {result.files_processed}"
    )
    assert result.files_failed == 0, (
        f"Expected 0 failures, got {result.files_failed}: {result.errors}"
    )
    assert result.pages_extracted >= 3, (
        f"Expected ≥3 pages, got {result.pages_extracted}"
    )
    assert result.chunks_created >= 3, (
        f"Expected ≥3 chunks, got {result.chunks_created}"
    )
    # Dry run should NOT store anything
    assert result.chunks_stored == 0, (
        f"Dry run should store 0 chunks, got {result.chunks_stored}"
    )

    print(
        f"  ✅ Test 5 passed: Dry-run pipeline processed {result.files_processed} files, "
        f"created {result.chunks_created} chunks (0 stored — dry run)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  Step 5 Verification: Document Ingestion Pipeline")
    print("=" * 60 + "\n")

    # Create temporary test directory with sample files
    with tempfile.TemporaryDirectory(prefix="tender_test_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        files = create_test_files(tmpdir)

        print(f"  Test files created in: {tmpdir}\n")

        tests = [
            ("Test 1: Parse TXT file", lambda: test_1_parser_txt(files)),
            ("Test 2: Parse Markdown file", lambda: test_2_parser_md(files)),
            ("Test 3: Reject unsupported file type", test_3_parser_unsupported_extension),
            ("Test 4: Chunk long document with overlap", lambda: test_4_chunker(files)),
            ("Test 5: Dry-run pipeline (no DB)", lambda: test_5_dry_run_pipeline(tmpdir)),
        ]

        passed = 0
        failed = 0

        for name, test_fn in tests:
            try:
                test_fn()
                passed += 1
            except AssertionError as exc:
                print(f"  ❌ {name} FAILED: {exc}")
                failed += 1
            except Exception as exc:
                print(f"  ❌ {name} ERROR: {type(exc).__name__}: {exc}")
                failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}\n")

    if failed > 0:
        sys.exit(1)
    else:
        print("  🎉 All tests passed! Step 5 is complete.")
        print("  Next: git add -A && git commit -m 'Step 5: Document ingestion pipeline'")
        print()


if __name__ == "__main__":
    main()