#!/usr/bin/env python3
"""
Step 6 Verification — Embedding & Vector Storage Tests

Runs 5 tests to verify that the Voyage AI embedder (dry-run mode) and
embedding pipeline work correctly. No real API key or database needed.

Usage:
    cd /Users/sohamsarker/tender-agent
    source .venv/bin/activate
    python scripts/test_embeddings.py

Expected output: All 5 tests pass.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_1_dry_run_dimensions() -> None:
    """Test 1: Dry-run embedder produces vectors with correct dimensions."""
    from src.ingestion.embedder import EMBEDDING_DIMENSIONS, VoyageEmbedder

    embedder = VoyageEmbedder(dry_run=True)
    result = embedder.embed_texts(["Safety Data Sheet management software"])

    assert len(result.embeddings) == 1, (
        f"Expected 1 embedding, got {len(result.embeddings)}"
    )
    assert len(result.embeddings[0]) == EMBEDDING_DIMENSIONS, (
        f"Expected {EMBEDDING_DIMENSIONS} dimensions, got {len(result.embeddings[0])}"
    )
    assert result.is_dry_run is True
    assert result.model == "voyage-3-large-dry-run"

    print("  ✅ Test 1 passed: Dry-run produces 1024-dimensional vectors")


def test_2_deterministic_embeddings() -> None:
    """Test 2: Same text always produces the same embedding (determinism).

    This matters because in tests, you want reproducible results.
    If the same chunk text produces a different vector each run,
    similarity searches would give inconsistent results.
    """
    from src.ingestion.embedder import VoyageEmbedder

    embedder = VoyageEmbedder(dry_run=True)

    text = "ISO 27001 certification for information security"

    result_a = embedder.embed_texts([text])
    result_b = embedder.embed_texts([text])

    # Vectors should be identical
    for i, (a, b) in enumerate(zip(result_a.embeddings[0], result_b.embeddings[0])):
        assert a == b, f"Dimension {i} differs: {a} vs {b}"

    print("  ✅ Test 2 passed: Same text produces identical embeddings (deterministic)")


def test_3_different_texts_different_vectors() -> None:
    """Test 3: Different texts produce different vectors.

    If every text produced the same vector, the RAG pipeline couldn't
    distinguish between chunks — every query would match everything equally.
    """
    from src.ingestion.embedder import VoyageEmbedder

    embedder = VoyageEmbedder(dry_run=True)

    result = embedder.embed_texts([
        "ISO 27001 certification for information security",
        "Pricing starts at $5,000 per year for up to 100 users",
        "Our team includes 15 chemical safety specialists",
    ])

    assert len(result.embeddings) == 3, f"Expected 3 embeddings, got {len(result.embeddings)}"

    # Check that vectors are different using cosine similarity
    def cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    sim_01 = cosine_sim(result.embeddings[0], result.embeddings[1])
    sim_02 = cosine_sim(result.embeddings[0], result.embeddings[2])
    sim_12 = cosine_sim(result.embeddings[1], result.embeddings[2])

    # Random unit vectors in 1024d have expected cosine similarity ≈ 0
    # They should NOT be identical (similarity = 1.0)
    assert sim_01 < 0.99, f"Vectors 0 and 1 are too similar: {sim_01}"
    assert sim_02 < 0.99, f"Vectors 0 and 2 are too similar: {sim_02}"
    assert sim_12 < 0.99, f"Vectors 1 and 2 are too similar: {sim_12}"

    print(
        f"  ✅ Test 3 passed: Different texts produce different vectors "
        f"(similarities: {sim_01:.3f}, {sim_02:.3f}, {sim_12:.3f})"
    )


def test_4_batch_embedding() -> None:
    """Test 4: Batch embedding works correctly with many texts."""
    from src.ingestion.embedder import EMBEDDING_DIMENSIONS, VoyageEmbedder

    embedder = VoyageEmbedder(dry_run=True)

    # Create 10 different texts
    texts = [f"Certification number {i} covers regulatory compliance" for i in range(10)]
    result = embedder.embed_texts(texts)

    assert len(result.embeddings) == 10, f"Expected 10 embeddings, got {len(result.embeddings)}"
    for i, emb in enumerate(result.embeddings):
        assert len(emb) == EMBEDDING_DIMENSIONS, (
            f"Embedding {i} has {len(emb)} dims, expected {EMBEDDING_DIMENSIONS}"
        )

    assert result.token_count > 0, "Token count should be positive"

    print(f"  ✅ Test 4 passed: Batch of 10 texts embedded correctly ({result.token_count} tokens)")


def test_5_embed_query_and_cost() -> None:
    """Test 5: embed_single convenience method and cost tracking work."""
    from src.ingestion.embed_pipeline import EmbeddingPipeline
    from src.ingestion.embedder import EMBEDDING_DIMENSIONS, VoyageEmbedder

    embedder = VoyageEmbedder(dry_run=True)
    pipeline = EmbeddingPipeline(embedder=embedder)

    # Test embed_query (used by RAG pipeline in Step 7)
    vector = pipeline.embed_query("What ISO certifications does the company hold?")

    assert len(vector) == EMBEDDING_DIMENSIONS, (
        f"Expected {EMBEDDING_DIMENSIONS} dimensions, got {len(vector)}"
    )
    assert all(isinstance(v, float) for v in vector), "All values should be floats"

    # Test cost tracking
    cost = embedder.get_cost_estimate()
    assert cost >= 0, f"Cost should be non-negative, got {cost}"
    assert embedder.total_tokens_used > 0, "Should have tracked some tokens"

    print(
        f"  ✅ Test 5 passed: embed_query works, cost tracking reports "
        f"{embedder.total_tokens_used} tokens (${cost:.6f})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  Step 6 Verification: Embedding & Vector Storage")
    print("=" * 60 + "\n")

    tests = [
        ("Test 1: Dry-run vector dimensions", test_1_dry_run_dimensions),
        ("Test 2: Deterministic embeddings", test_2_deterministic_embeddings),
        ("Test 3: Different texts → different vectors", test_3_different_texts_different_vectors),
        ("Test 4: Batch embedding", test_4_batch_embedding),
        ("Test 5: embed_query + cost tracking", test_5_embed_query_and_cost),
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
        print("  🎉 All tests passed! Step 6 is complete.")
        print("  Next: git add -A && git commit -m 'Step 6: Embedding & vector storage'")
        print()


if __name__ == "__main__":
    main()