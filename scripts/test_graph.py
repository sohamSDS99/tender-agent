#!/usr/bin/env python3
"""
Step 8 Verification — State Schema & Graph Skeleton Tests
(Updated for real node implementations)
"""
from __future__ import annotations
import os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["DRY_RUN"] = "true"

def _make_sample_input():
    return {
        "tender_id": "TEST-001",
        "tender_title": "EHS Software Platform for State Environmental Agency",
        "source_portal": "sam.gov",
        "source_url": "https://sam.gov/opp/test-001",
        "tender_raw_text": (
            "The State Environmental Agency seeks a cloud-based SDS management "
            "platform with GHS classification, chemical inventory tracking, "
            "OSHA HCS compliance, and Tier II regulatory reporting. "
            "ISO 27001 or equivalent security certification required. "
            "Budget range $50,000-$100,000 annual subscription."
        ),
        "submission_deadline": "2026-08-15T17:00:00Z",
        "discovered_at": "2026-04-16T10:00:00Z",
        "tender_document_path": None,
        "escalation_count": 0,
        "assembly_retry_count": 0,
        "error_messages": [],
        "audit_log": [],
    }

def test_1_state_creation():
    from src.agent.state import TenderState, TenderStatus, SubmissionMethod
    assert TenderStatus.DISCOVERED.value == "discovered"
    assert TenderStatus.SUBMITTED.value == "submitted"
    assert SubmissionMethod.PORTAL_UPLOAD.value == "portal_upload"
    state = _make_sample_input()
    assert state["tender_id"] == "TEST-001"
    assert isinstance(state["audit_log"], list)
    print("  ✅ Test 1 passed: TenderState schema and enums work correctly")

def test_2_graph_compiles():
    from src.agent.graph import build_tender_graph
    graph = build_tender_graph(checkpointer=None)
    assert graph is not None
    assert hasattr(graph, "invoke")
    print("  ✅ Test 2 passed: Graph compiles successfully")

def test_3_happy_path():
    from src.agent.graph import build_tender_graph
    graph = build_tender_graph(checkpointer=None)
    result = graph.invoke(_make_sample_input())
    assert result["status"] == "submitted", f"Expected submitted, got {result['status']}"
    assert result["submission_confirmation"], "Should have confirmation"
    assert result["eval_score"] >= 60, f"Should be eligible, got {result['eval_score']}"
    assert result["eval_decision"] == "go"
    assert len(result["drafted_sections"]) >= 1
    assert result["gap_check_passed"] is True
    assert result["quality_check_passed"] is True
    print(f"  ✅ Test 3 passed: Happy path (score={result['eval_score']}, sections={len(result['drafted_sections'])})")

def test_4_routing_logic():
    from src.agent.graph import route_after_evaluate, route_after_gap_check, route_after_assemble
    assert route_after_evaluate({"eval_decision": "no_go"}) == "end"
    assert route_after_evaluate({"eval_decision": "go"}) == "retrieve_draft"
    assert route_after_gap_check({"gap_check_passed": True}) == "assemble"
    assert route_after_gap_check({"gap_check_passed": False, "escalation_count": 0}) == "slack_escalate"
    assert route_after_gap_check({"gap_check_passed": False, "escalation_count": 3}) == "assemble"
    assert route_after_assemble({"quality_check_passed": True}) == "submit"
    assert route_after_assemble({"quality_check_passed": False, "assembly_retry_count": 0}) == "assemble"
    assert route_after_assemble({"quality_check_passed": False, "assembly_retry_count": 3}) == "submit"
    print("  ✅ Test 4 passed: All routing logic works correctly")

def test_5_audit_log_accumulation():
    from src.agent.graph import build_tender_graph
    graph = build_tender_graph(checkpointer=None)
    result = graph.invoke(_make_sample_input())
    audit = result.get("audit_log", [])
    assert len(audit) >= 6, f"Expected >=6 audit entries, got {len(audit)}"
    for entry in audit:
        assert "timestamp" in entry
        assert "node" in entry
        assert "action" in entry
    visited = {e["node"] for e in audit}
    for expected in ["discover", "evaluate", "retrieve_draft", "gap_check", "assemble", "submit"]:
        assert expected in visited, f"{expected} missing from audit"
    print(f"  ✅ Test 5 passed: Audit log has {len(audit)} entries from {len(visited)} nodes")

def main():
    print("\n" + "=" * 60)
    print("  Step 8 Verification: State Schema & Graph Skeleton")
    print("=" * 60 + "\n")
    tests = [
        ("Test 1: State schema creation", test_1_state_creation),
        ("Test 2: Graph compilation", test_2_graph_compiles),
        ("Test 3: Happy path (full pipeline)", test_3_happy_path),
        ("Test 4: Routing logic", test_4_routing_logic),
        ("Test 5: Audit log accumulation", test_5_audit_log_accumulation),
    ]
    passed = failed = 0
    for name, fn in tests:
        try: fn(); passed += 1
        except AssertionError as e: print(f"  ❌ {name} FAILED: {e}"); failed += 1
        except Exception as e: print(f"  ❌ {name} ERROR: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{'=' * 60}\n  Results: {passed} passed, {failed} failed\n{'=' * 60}\n")
    if failed: sys.exit(1)
    else: print("  🎉 All tests passed!\n")

if __name__ == "__main__":
    main()