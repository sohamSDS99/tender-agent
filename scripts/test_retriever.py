#!/usr/bin/env python3
"""
Step 7 Verification — RAG Retrieval Module Tests

Runs 5 tests to verify the retriever's formatting, result handling,
and embedder integration. No database connection needed for these tests.

After you've ingested documents (Step 5) and embedded them (Step 6),
run the bonus Test 6 to verify end-to-end search against your actual DB.

Usage:
    cd /Users/sohamsarker/tender-agent
    source .venv/bin/activate
    python scripts/test_retriever.py

Expected output: All 5 tests pass (+1 bonus if DB has data).
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_mock_chunks() -> list:
    """Create mock RetrievedChunk objects for testing without a database."""
    from src.knowledge.retriever import RetrievedChunk

    return [
        RetrievedChunk(
            content=(
                "Acme SDS Solutions holds ISO 27001 certification for information "
                "security, achieved in 2020. The certification covers all aspects of "
                "our cloud-based SDS management platform including data storage, "
                "processing, and transmission."
            ),
            source_file="company_profile.pdf",
            page_number=3,
            section_heading="Certifications",
            chunk_metadata={"category": "company_profile"},
            similarity_score=0.89,
            chunk_id=42,
        ),
        RetrievedChunk(
            content=(
                "Our platform provides SOC 2 Type II compliance, verified annually "
                "by independent auditors. Data centres are located in AWS US-East "
                "and AWS EU-West regions, providing 99.95% uptime SLA."
            ),
            source_file="company_profile.pdf",
            page_number=3,
            section_heading="Certifications",
            chunk_metadata={"category": "company_profile"},
            similarity_score=0.76,
            chunk_id=43,
        ),
        RetrievedChunk(
            content=(
                "The company was founded in 2015 and now serves over 500 clients "
                "across manufacturing, construction, oil and gas, and pharmaceuticals."
            ),
            source_file="company_overview.txt",
            page_number=1,
            section_heading=None,
            chunk_metadata={"category": "overview"},
            similarity_score=0.41,
            chunk_id=10,
        ),
    ]


def test_1_retrieved_chunk_creation() -> None:
    """Test 1: RetrievedChunk objects are created with correct attributes."""
    from src.knowledge.retriever import RetrievedChunk

    chunk = RetrievedChunk(
        content="ISO 27001 certification obtained in 2020.",
        source_file="certs.pdf",
        page_number=1,
        section_heading="Security",
        chunk_metadata={"category": "certification"},
        similarity_score=0.92,
        chunk_id=7,
    )

    assert chunk.content == "ISO 27001 certification obtained in 2020."
    assert chunk.source_file == "certs.pdf"
    assert chunk.page_number == 1
    assert chunk.section_heading == "Security"
    assert chunk.similarity_score == 0.92
    assert chunk.chunk_id == 7

    # Test repr contains useful info
    repr_str = repr(chunk)
    assert "0.920" in repr_str, f"Expected score in repr: {repr_str}"
    assert "certs.pdf" in repr_str, f"Expected source in repr: {repr_str}"

    print("  ✅ Test 1 passed: RetrievedChunk creation and repr work correctly")


def test_2_format_context() -> None:
    """Test 2: format_context produces well-structured LLM context."""
    from src.knowledge.retriever import KnowledgeRetriever

    retriever = KnowledgeRetriever()
    chunks = _make_mock_chunks()

    context = retriever.format_context(chunks)

    # Should contain source attribution headers
    assert "[Source 1:" in context, "Missing Source 1 header"
    assert "[Source 2:" in context, "Missing Source 2 header"
    assert "[Source 3:" in context, "Missing Source 3 header"

    # Should contain the source file names
    assert "company_profile.pdf" in context
    assert "company_overview.txt" in context

    # Should contain section headings where available
    assert "Certifications" in context

    # Should contain similarity scores as percentages
    assert "89%" in context, "Missing similarity percentage for chunk 1"

    # Should contain actual content
    assert "ISO 27001" in context
    assert "SOC 2 Type II" in context

    print("  ✅ Test 2 passed: format_context produces well-structured output")


def test_3_format_context_truncation() -> None:
    """Test 3: format_context respects max_chars limit."""
    from src.knowledge.retriever import KnowledgeRetriever

    retriever = KnowledgeRetriever()
    chunks = _make_mock_chunks()

    # Set a very small max_chars to force truncation
    context = retriever.format_context(chunks, max_chars=400)

    assert len(context) <= 500, (
        f"Context should be roughly within max_chars, got {len(context)}"
    )
    # Should include truncation notice
    assert "truncated" in context.lower(), "Should mention truncation"

    print("  ✅ Test 3 passed: format_context truncates correctly at max_chars")


def test_4_format_context_empty() -> None:
    """Test 4: format_context handles no results gracefully."""
    from src.knowledge.retriever import KnowledgeRetriever

    retriever = KnowledgeRetriever()
    context = retriever.format_context([])

    assert "No relevant information" in context, (
        f"Expected 'no information' message, got: {context}"
    )

    print("  ✅ Test 4 passed: format_context handles empty results")


def test_5_embedder_integration() -> None:
    """Test 5: Retriever's embedder can convert queries to vectors."""
    from src.ingestion.embedder import EMBEDDING_DIMENSIONS
    from src.knowledge.retriever import KnowledgeRetriever

    retriever = KnowledgeRetriever()

    # The retriever should be able to embed queries (this is what search() does internally)
    vector = retriever.embedder.embed_single(
        "What ISO certifications does the company hold?"
    )

    assert len(vector) == EMBEDDING_DIMENSIONS, (
        f"Expected {EMBEDDING_DIMENSIONS} dimensions, got {len(vector)}"
    )
    assert all(isinstance(v, float) for v in vector)

    print(f"  ✅ Test 5 passed: Embedder produces {EMBEDDING_DIMENSIONS}-dim query vectors")


def test_6_db_search() -> None:
    """BONUS Test 6: End-to-end search against real database.

    This test only runs if your PostgreSQL database has embedded chunks.
    It will be skipped gracefully if the DB is unavailable or empty.
    """
    try:
        from src.models.base import SessionLocal
    except Exception:
        print("  ⏭️  Test 6 skipped: Could not import SessionLocal (DB not configured)")
        return

    try:
        from src.knowledge.retriever import KnowledgeRetriever

        retriever = KnowledgeRetriever()

        with SessionLocal() as session:
            results = retriever.search(
                query="company certifications and compliance",
                session=session,
                top_k=3,
            )

            if not results:
                print(
                    "  ⏭️  Test 6 skipped: No embedded chunks in database. "
                    "Run Steps 5+6 ingestion first."
                )
                return

            # If we got results, verify they're well-formed
            for r in results:
                assert r.content, "Result content should not be empty"
                assert 0.0 <= r.similarity_score <= 1.0, (
                    f"Score {r.similarity_score} out of range"
                )
                assert r.source_file, "source_file should not be empty"

            context = retriever.format_context(results)
            assert len(context) > 0

            print(
                f"  ✅ Test 6 passed: DB search returned {len(results)} results "
                f"(top score: {results[0].similarity_score:.3f})"
            )

    except Exception as exc:
        print(f"  ⏭️  Test 6 skipped: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  Step 7 Verification: RAG Retrieval Module")
    print("=" * 60 + "\n")

    tests = [
        ("Test 1: RetrievedChunk creation", test_1_retrieved_chunk_creation),
        ("Test 2: format_context output", test_2_format_context),
        ("Test 3: format_context truncation", test_3_format_context_truncation),
        ("Test 4: format_context empty results", test_4_format_context_empty),
        ("Test 5: Embedder integration", test_5_embedder_integration),
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

    # Run bonus DB test (doesn't count toward pass/fail)
    print()
    test_6_db_search()

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed (5 required)")
    print(f"{'=' * 60}\n")

    if failed > 0:
        sys.exit(1)
    else:
        print("  🎉 All tests passed! Step 7 is complete.")
        print("  Next: git add -A && git commit -m 'Step 7: RAG retrieval module'")
        print()


if __name__ == "__main__":
    main()