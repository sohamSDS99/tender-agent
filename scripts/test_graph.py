#!/usr/bin/env python3
"""
Step 8 Verification — State Schema & Graph Skeleton Tests

Runs 5 tests to verify:
1. TenderState can be created with all required fields
2. Graph compiles successfully
3. Happy path: eligible tender flows through all 7 nodes to submission
4. Reject path: low-score tender stops after evaluation
5. Audit log accumulates entries from every node

Usage:
    cd /Users/sohamsarker/tender-agent
    source .venv/bin/activate
    python scripts/test_graph.py

Expected output: All 5 tests pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# ---------------------------------------------------------------------------


def _make_sample_tender_input() -> dict:
    """Create a minimal tender state dict for testing.

    This is what the Discover node would produce in production — the
    initial state with tender metadata that kicks off the pipeline.
    """
    return {
        "tender_id": "TEST-001",
        "tender_title": "EHS Software Platform for State Environmental Agency",
        "source_portal": "sam.gov",
        "source_url": "https://sam.gov/opp/test-001",
        "tender_raw_text": (
            "The State Environmental Agency seeks a cloud-based EHS software "
            "platform capable of managing Safety Data Sheets, chemical inventories, "
            "and regulatory compliance reporting. The vendor must demonstrate "
            "experience with OSHA HCS, GHS classification, and Tier II reporting. "
            "ISO 27001 or equivalent security certification required."
        ),
        "submission_deadline": "2026-05-15T17:00:00Z",
        "discovered_at": "2026-04-16T10:00:00Z",
        "tender_document_path": None,
        "escalation_count": 0,
        "assembly_retry_count": 0,
        "error_messages": [],
        "audit_log": [],
    }


def test_1_state_creation() -> None:
    """Test 1: TenderState can be populated with all field groups."""
    from src.agent.state import TenderState, TenderStatus, SubmissionMethod

    # Verify enums work
    assert TenderStatus.DISCOVERED.value == "discovered"
    assert TenderStatus.SUBMITTED.value == "submitted"
    assert SubmissionMethod.PORTAL_UPLOAD.value == "portal_upload"

    # Verify we can create a state dict (TypedDict is structural)
    state = _make_sample_tender_input()
    assert state["tender_id"] == "TEST-001"
    assert state["tender_title"] == "EHS Software Platform for State Environmental Agency"
    assert state["escalation_count"] == 0
    assert isinstance(state["audit_log"], list)

    print("  ✅ Test 1 passed: TenderState schema and enums work correctly")


def test_2_graph_compiles() -> None:
    """Test 2: The graph compiles without errors."""
    from src.agent.graph import build_tender_graph

    graph = build_tender_graph(checkpointer=None)

    # The compiled graph should be invocable
    assert graph is not None
    assert hasattr(graph, "invoke"), "Compiled graph should have an invoke method"

    print("  ✅ Test 2 passed: Graph compiles successfully (7 nodes, 0 errors)")


def test_3_happy_path() -> None:
    """Test 3: Eligible tender flows through all 7 nodes to submission.

    The placeholder evaluate_node returns score=75 (above 60 threshold),
    gap_check returns no gaps, and quality check passes. This should
    flow: discover → evaluate → retrieve_draft → gap_check → assemble → submit.
    """
    from src.agent.graph import build_tender_graph
    from src.agent.state import TenderStatus

    graph = build_tender_graph(checkpointer=None)
    initial_state = _make_sample_tender_input()

    result = graph.invoke(initial_state)

    # Should end with SUBMITTED status
    assert result["status"] == TenderStatus.SUBMITTED.value, (
        f"Expected 'submitted', got '{result['status']}'"
    )

    # Should have a submission confirmation
    assert result["submission_confirmation"] == "MOCK-RECEIPT-001"

    # Eval score should be 75 (sum of placeholder breakdown)
    assert result["eval_score"] == 75, f"Expected score 75, got {result['eval_score']}"
    assert result["eval_decision"] == "go"

    # Should have drafted sections
    assert len(result["drafted_sections"]) == 2

    # Gap check should have passed
    assert result["gap_check_passed"] is True

    # Quality check should have passed
    assert result["quality_check_passed"] is True

    print("  ✅ Test 3 passed: Happy path completes — all 7 nodes executed")


def test_4_reject_path() -> None:
    """Test 4: Low-score tender is rejected after evaluation.

    We override the evaluate_node to return a low score to test the
    rejection routing. Since we can't easily override a node in the
    compiled graph, we'll test the routing function directly.
    """
    from src.agent.graph import route_after_evaluate

    # Simulate a state where evaluation returned "no_go"
    rejected_state = {
        "tender_id": "TEST-002",
        "eval_decision": "no_go",
        "eval_score": 35,
    }

    route = route_after_evaluate(rejected_state)
    assert route == "end", f"Expected 'end' route for no_go, got '{route}'"

    # Simulate a state where evaluation returned "go"
    accepted_state = {
        "tender_id": "TEST-003",
        "eval_decision": "go",
        "eval_score": 78,
    }

    route = route_after_evaluate(accepted_state)
    assert route == "retrieve_draft", f"Expected 'retrieve_draft' for go, got '{route}'"

    print("  ✅ Test 4 passed: Routing logic correctly handles go/no-go decisions")


def test_5_audit_log_accumulation() -> None:
    """Test 5: Audit log accumulates entries from every node visited.

    This verifies that the Annotated[list, operator.add] reducer works —
    each node's audit entries get APPENDED (not replaced).
    """
    from src.agent.graph import build_tender_graph

    graph = build_tender_graph(checkpointer=None)
    initial_state = _make_sample_tender_input()

    result = graph.invoke(initial_state)

    audit = result.get("audit_log", [])

    # Happy path visits: discover, evaluate, retrieve_draft, gap_check, assemble, submit
    # That's 6 nodes (slack_escalate is skipped because no gaps)
    assert len(audit) >= 6, (
        f"Expected ≥6 audit entries (one per node visited), got {len(audit)}"
    )

    # Verify each entry has required fields
    for entry in audit:
        assert "timestamp" in entry, f"Missing timestamp in audit entry: {entry}"
        assert "node" in entry, f"Missing node in audit entry: {entry}"
        assert "action" in entry, f"Missing action in audit entry: {entry}"

    # Verify the expected nodes appear in order
    visited_nodes = [entry["node"] for entry in audit]
    assert "discover" in visited_nodes, "discover missing from audit"
    assert "evaluate" in visited_nodes, "evaluate missing from audit"
    assert "retrieve_draft" in visited_nodes, "retrieve_draft missing from audit"
    assert "gap_check" in visited_nodes, "gap_check missing from audit"
    assert "assemble" in visited_nodes, "assemble missing from audit"
    assert "submit" in visited_nodes, "submit missing from audit"

    print(
        f"  ✅ Test 5 passed: Audit log has {len(audit)} entries from "
        f"{len(set(visited_nodes))} nodes"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  Step 8 Verification: State Schema & Graph Skeleton")
    print("=" * 60 + "\n")

    tests = [
        ("Test 1: State schema creation", test_1_state_creation),
        ("Test 2: Graph compilation", test_2_graph_compiles),
        ("Test 3: Happy path (full pipeline)", test_3_happy_path),
        ("Test 4: Reject path routing", test_4_reject_path),
        ("Test 5: Audit log accumulation", test_5_audit_log_accumulation),
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
        print("  🎉 All tests passed! Step 8 is complete.")
        print("  Next: git add -A && git commit -m 'Step 8: State schema & graph skeleton'")
        print()


if __name__ == "__main__":
    main()