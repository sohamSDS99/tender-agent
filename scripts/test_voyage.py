"""
Voyage AI Integration Test
============================
Tests that your Voyage API key works and compares embedding quality
against the old MiniLM model.

Usage:
    cd ~/Desktop/tender-agent
    source .venv/bin/activate
    python scripts/test_voyage.py
"""

import os
import sys
import time
from pathlib import Path

# Load env
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import httpx

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"


def test_api_key():
    """Test 1: Verify the API key works."""
    print("=" * 60)
    print("  TEST 1: API Key Validation")
    print("=" * 60)

    if not VOYAGE_API_KEY:
        print("  FAIL: VOYAGE_API_KEY not set in .env")
        return False

    print(f"  Key: {VOYAGE_API_KEY[:10]}...{VOYAGE_API_KEY[-4:]}")

    try:
        resp = httpx.post(
            VOYAGE_API_URL,
            headers={
                "Authorization": f"Bearer {VOYAGE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "input": ["Hello, this is a test."],
                "model": "voyage-3-large",
            },
            timeout=15.0,
        )

        if resp.status_code == 200:
            data = resp.json()
            dims = len(data["data"][0]["embedding"])
            usage = data.get("usage", {})
            print(f"  PASS: API key is valid")
            print(f"  Model: voyage-3-large")
            print(f"  Dimensions: {dims}")
            print(f"  Tokens used: {usage.get('total_tokens', '?')}")
            return True
        else:
            print(f"  FAIL: HTTP {resp.status_code}")
            print(f"  Response: {resp.text[:200]}")
            return False

    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def test_embedding_quality():
    """Test 2: Compare Voyage AI vs MiniLM on SDS-specific queries."""
    print()
    print("=" * 60)
    print("  TEST 2: Embedding Quality Comparison")
    print("=" * 60)

    # SDS/EHS domain-specific test cases
    # Each has a query, a relevant doc chunk, and an irrelevant doc chunk
    test_cases = [
        {
            "name": "SDS Compliance",
            "query": "What are the GHS classification requirements for chemical hazards?",
            "relevant": "The Globally Harmonized System (GHS) classifies chemicals based on their physical, health, and environmental hazards. SDS authors must follow GHS criteria for hazard classification including acute toxicity, skin corrosion, and serious eye damage categories.",
            "irrelevant": "Our company picnic will be held on Saturday at the park. Please bring your families and a dish to share. There will be games and prizes for the kids.",
        },
        {
            "name": "Tender Requirements",
            "query": "What is the submission deadline and required documentation for this RFP?",
            "relevant": "Proposals must be submitted by March 15, 2026. Required documents include: technical proposal, pricing schedule, past performance references, proof of ISO 9001 certification, and completed W-9 form.",
            "irrelevant": "The weather forecast shows sunny skies with temperatures reaching 28 degrees celsius. Perfect conditions for outdoor activities this weekend.",
        },
        {
            "name": "Company Capability",
            "query": "Does SDS Manager have experience with regulatory compliance software?",
            "relevant": "SDS Manager has delivered regulatory compliance solutions to over 200 enterprise clients across chemical manufacturing, pharmaceuticals, and environmental services. Our platform handles SDS authoring, chemical inventory management, and regulatory reporting.",
            "irrelevant": "The latest iPhone features a new titanium design and an advanced camera system with a 48MP main sensor and 5x optical zoom capability.",
        },
    ]

    def cosine_similarity(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0

    def embed_voyage(texts, input_type="document"):
        resp = httpx.post(
            VOYAGE_API_URL,
            headers={
                "Authorization": f"Bearer {VOYAGE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "input": texts,
                "model": "voyage-3-large",
                "input_type": input_type,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]

    print()
    print("  Voyage AI uses asymmetric embeddings (query vs document)")
    print("  Higher relevant score + lower irrelevant score = better retrieval")
    print()

    total_relevant = 0
    total_irrelevant = 0

    for tc in test_cases:
        # Embed query with input_type="query", docs with input_type="document"
        query_emb = embed_voyage([tc["query"]], input_type="query")[0]
        doc_embs = embed_voyage([tc["relevant"], tc["irrelevant"]], input_type="document")

        relevant_sim = cosine_similarity(query_emb, doc_embs[0])
        irrelevant_sim = cosine_similarity(query_emb, doc_embs[1])
        gap = relevant_sim - irrelevant_sim

        total_relevant += relevant_sim
        total_irrelevant += irrelevant_sim

        status = "PASS" if gap > 0.15 else "WARN" if gap > 0.05 else "FAIL"

        print(f"  [{status}] {tc['name']}")
        print(f"        Relevant:   {relevant_sim:.4f}")
        print(f"        Irrelevant: {irrelevant_sim:.4f}")
        print(f"        Gap:        {gap:.4f}")
        print()

    avg_relevant = total_relevant / len(test_cases)
    avg_irrelevant = total_irrelevant / len(test_cases)
    avg_gap = avg_relevant - avg_irrelevant

    print(f"  AVERAGE SCORES:")
    print(f"    Relevant:   {avg_relevant:.4f}")
    print(f"    Irrelevant: {avg_irrelevant:.4f}")
    print(f"    Gap:        {avg_gap:.4f}")
    print()

    if avg_gap > 0.15:
        print("  VERDICT: Excellent separation — Voyage AI is working well")
    elif avg_gap > 0.08:
        print("  VERDICT: Good separation — embeddings are functional")
    else:
        print("  VERDICT: Poor separation — something may be wrong")

    return avg_gap > 0.08


def test_latency():
    """Test 3: Measure API response time."""
    print()
    print("=" * 60)
    print("  TEST 3: Latency Benchmark")
    print("=" * 60)

    texts = [
        "Safety Data Sheet for hydrochloric acid solution 37%",
        "Environmental Health and Safety compliance requirements",
        "Government tender for chemical management software",
        "ISO 14001 environmental management system certification",
        "Hazardous waste disposal regulatory framework",
    ]

    # Single text
    start = time.perf_counter()
    embed_single = httpx.post(
        VOYAGE_API_URL,
        headers={"Authorization": f"Bearer {VOYAGE_API_KEY}", "Content-Type": "application/json"},
        json={"input": [texts[0]], "model": "voyage-3-large", "input_type": "query"},
        timeout=15.0,
    )
    single_ms = int((time.perf_counter() - start) * 1000)

    # Batch of 5
    start = time.perf_counter()
    embed_batch = httpx.post(
        VOYAGE_API_URL,
        headers={"Authorization": f"Bearer {VOYAGE_API_KEY}", "Content-Type": "application/json"},
        json={"input": texts, "model": "voyage-3-large", "input_type": "document"},
        timeout=15.0,
    )
    batch_ms = int((time.perf_counter() - start) * 1000)

    usage = embed_batch.json().get("usage", {})

    print(f"  Single query:   {single_ms}ms")
    print(f"  Batch of 5:     {batch_ms}ms ({batch_ms // 5}ms per text)")
    print(f"  Tokens used:    {usage.get('total_tokens', '?')}")
    print()

    if single_ms < 2000:
        print("  VERDICT: Good latency")
    elif single_ms < 5000:
        print("  VERDICT: Acceptable latency")
    else:
        print("  VERDICT: High latency — may slow down search")

    return True


def main():
    print()
    print("*" * 60)
    print("  VOYAGE AI INTEGRATION TEST")
    print("*" * 60)
    print()

    results = []

    # Test 1: API key
    results.append(("API Key", test_api_key()))

    if not results[0][1]:
        print("\nAPI key test failed — cannot run remaining tests.")
        sys.exit(1)

    # Test 2: Quality
    try:
        results.append(("Quality", test_embedding_quality()))
    except Exception as exc:
        print(f"\n  Quality test error: {exc}")
        results.append(("Quality", False))

    # Test 3: Latency
    try:
        results.append(("Latency", test_latency()))
    except Exception as exc:
        print(f"\n  Latency test error: {exc}")
        results.append(("Latency", False))

    # Summary
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, passed in results:
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {name}")

    all_passed = all(r[1] for r in results)
    print()
    if all_passed:
        print("  All tests passed! Voyage AI is ready to use.")
    else:
        print("  Some tests failed. Check the output above.")
    print()


if __name__ == "__main__":
    main()
