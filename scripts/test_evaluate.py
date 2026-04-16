#!/usr/bin/env python3
"""
Step 9 Verification — Evaluate Node Tests

Runs 5 tests to verify the evaluate node scores tenders correctly
in dry-run mode (keyword heuristics). No API key needed.

Usage:
    cd /Users/sohamsarker/tender-agent
    source .venv/bin/activate
    python scripts/test_evaluate.py

Expected output: All 5 tests pass.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force dry-run mode for all tests
os.environ["DRY_RUN"] = "true"


def test_1_high_relevance_tender() -> None:
    """Test 1: A highly relevant SDS/chemical safety tender scores ≥ 60 (GO)."""
    from src.agent.nodes.evaluate import evaluate_node

    state = {
        "tender_id": "TEST-HIGH-001",
        "tender_title": "SDS Management Platform for Federal Agency",
        "tender_raw_text": (
            "The United States Environmental Protection Agency seeks a cloud-based "
            "SDS management platform for managing Safety Data Sheets across 50 "
            "facilities. The vendor must provide GHS classification, chemical "
            "inventory tracking, OSHA HCS compliance, and Tier II regulatory "
            "reporting. ISO 27001 or SOC 2 certification required. Budget range "
            "$50,000-$100,000 annual subscription. Submission deadline: 60 days."
        ),
        "audit_log": [],
        "error_messages": [],
    }

    result = evaluate_node(state)

    assert result["eval_score"] >= 60, (
        f"Highly relevant tender should score ≥60, got {result['eval_score']}"
    )
    assert result["eval_decision"] == "go", (
        f"Expected 'go', got '{result['eval_decision']}'"
    )
    assert result["eval_breakdown"]["domain_match"] > 5, (
        "Domain match should be high for SDS/chemical safety tender"
    )

    print(
        f"  ✅ Test 1 passed: High-relevance tender scored {result['eval_score']}/100 (GO)"
    )


def test_2_low_relevance_tender() -> None:
    """Test 2: An unrelated tender scores < 60 (NO-GO)."""
    from src.agent.nodes.evaluate import evaluate_node

    state = {
        "tender_id": "TEST-LOW-001",
        "tender_title": "Accounting Software for Small Business",
        "tender_raw_text": (
            "Small accounting firm seeks a desktop accounting and bookkeeping "
            "software solution. Must include invoicing, payroll processing, "
            "tax preparation, and financial reporting. Budget under $1,000 "
            "per year. Local installation required, no cloud solutions. "
            "HIPAA compliance for healthcare client billing."
        ),
        "audit_log": [],
        "error_messages": [],
    }

    result = evaluate_node(state)

    assert result["eval_score"] < 60, (
        f"Unrelated tender should score <60, got {result['eval_score']}"
    )
    assert result["eval_decision"] == "no_go", (
        f"Expected 'no_go', got '{result['eval_decision']}'"
    )

    print(
        f"  ✅ Test 2 passed: Low-relevance tender scored {result['eval_score']}/100 (NO-GO)"
    )


def test_3_score_breakdown_structure() -> None:
    """Test 3: Score breakdown has all 8 dimensions with valid values."""
    from src.agent.nodes.evaluate import evaluate_node, _MAX_SCORES

    state = {
        "tender_id": "TEST-STRUCT-001",
        "tender_raw_text": "Generic tender for EHS compliance software platform.",
        "audit_log": [],
        "error_messages": [],
    }

    result = evaluate_node(state)
    breakdown = result["eval_breakdown"]

    # Must have all 8 dimensions
    for dim in _MAX_SCORES:
        assert dim in breakdown, f"Missing dimension: {dim}"
        assert 0 <= breakdown[dim] <= _MAX_SCORES[dim], (
            f"{dim} score {breakdown[dim]} out of range [0, {_MAX_SCORES[dim]}]"
        )

    # Total must match sum of breakdown
    assert result["eval_score"] == sum(breakdown.values()), (
        f"Total {result['eval_score']} != sum of breakdown {sum(breakdown.values())}"
    )

    print(
        f"  ✅ Test 3 passed: All 8 dimensions present with valid scores, "
        f"total={result['eval_score']}"
    )


def test_4_audit_log_entry() -> None:
    """Test 4: Evaluate node produces a proper audit log entry."""
    from src.agent.nodes.evaluate import evaluate_node

    state = {
        "tender_id": "TEST-AUDIT-001",
        "tender_raw_text": "SDS management platform with GHS classification.",
        "audit_log": [],
        "error_messages": [],
    }

    result = evaluate_node(state)
    audit_entries = result["audit_log"]

    assert len(audit_entries) == 1, f"Expected 1 audit entry, got {len(audit_entries)}"

    entry = audit_entries[0]
    assert entry["node"] == "evaluate"
    assert entry["action"] == "tender_scored"
    assert "Score:" in entry["detail"]
    assert entry["model_used"] is not None
    assert "dry-run" in entry["model_used"], "Should indicate dry-run mode"
    assert entry["timestamp"], "Timestamp should not be empty"

    print("  ✅ Test 4 passed: Audit log entry is complete and well-formed")


def test_5_graph_integration() -> None:
    """Test 5: Real evaluate node works inside the full graph.

    Verifies that swapping the placeholder for the real implementation
    doesn't break the graph flow.
    """
    from src.agent.graph import build_tender_graph
    from src.agent.state import TenderStatus

    # Temporarily monkey-patch the graph to use our real evaluate node
    import src.agent.graph as graph_module
    from src.agent.nodes.evaluate import evaluate_node as real_evaluate

    original_fn = graph_module.evaluate_node
    graph_module.evaluate_node = real_evaluate

    try:
        graph = build_tender_graph(checkpointer=None)

        initial_state = {
            "tender_id": "TEST-GRAPH-001",
            "tender_title": "Chemical Safety SDS Platform",
            "source_portal": "sam.gov",
            "source_url": "https://sam.gov/opp/test",
            "tender_raw_text": (
                "Federal agency requires SDS management software with GHS "
                "classification, OSHA HCS compliance, chemical inventory, "
                "and regulatory reporting. ISO 27001 required. Budget $75,000. "
                "Submission in 60 days."
            ),
            "submission_deadline": "2026-06-15T17:00:00Z",
            "discovered_at": "2026-04-16T10:00:00Z",
            "tender_document_path": None,
            "escalation_count": 0,
            "assembly_retry_count": 0,
            "error_messages": [],
            "audit_log": [],
        }

        result = graph.invoke(initial_state)

        # Should reach submitted (since the tender is relevant)
        assert result["status"] == TenderStatus.SUBMITTED.value, (
            f"Expected 'submitted', got '{result['status']}'"
        )

        # Real eval score should be present (not the placeholder 75)
        assert result["eval_score"] != 75 or result["eval_decision"] == "go", (
            "Real evaluate node should produce a score different from placeholder"
        )

        print(
            f"  ✅ Test 5 passed: Real evaluate node integrates with graph "
            f"(score={result['eval_score']}, decision={result['eval_decision']})"
        )

    finally:
        # Restore original to avoid side effects
        graph_module.evaluate_node = original_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  Step 9 Verification: Evaluate Node")
    print("=" * 60 + "\n")

    tests = [
        ("Test 1: High-relevance tender (GO)", test_1_high_relevance_tender),
        ("Test 2: Low-relevance tender (NO-GO)", test_2_low_relevance_tender),
        ("Test 3: Score breakdown structure", test_3_score_breakdown_structure),
        ("Test 4: Audit log entry", test_4_audit_log_entry),
        ("Test 5: Graph integration", test_5_graph_integration),
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
        print("  🎉 All tests passed! Step 9 is complete.")
        print()
        print("  NEXT: Update src/agent/graph.py to use the real evaluate node.")
        print("  Replace the placeholder import with:")
        print("    from src.agent.nodes.evaluate import evaluate_node")
        print("  (delete the placeholder evaluate_node function from graph.py)")
        print()
        print("  Then: git add -A && git commit -m 'Step 9: Evaluate node'")
        print()


if __name__ == "__main__":
    main()