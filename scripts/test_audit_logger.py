#!/usr/bin/env python3
"""Step 23 Verification — Structured Audit Logger Tests"""
from __future__ import annotations
import os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["DRY_RUN"] = "true"

def test_1_log_single_entry():
    """Test 1: Log a single audit entry and retrieve it."""
    from src.utils.audit_logger import AuditLogger
    audit = AuditLogger()
    entry = audit.log(
        tender_id="TEST-001", node="evaluate", action="tender_scored",
        detail="Score: 77/100. Decision: GO.", model_used="claude-haiku-4-5",
        tokens_used=1200,
    )
    assert entry.tender_id == "TEST-001"
    assert entry.node == "evaluate"
    assert entry.tokens_used == 1200
    assert entry.timestamp
    entries = audit.get_by_tender("TEST-001")
    assert len(entries) == 1
    assert entries[0].action == "tender_scored"
    print("  ✅ Test 1 passed: Single entry logged and retrieved")

def test_2_persist_from_state():
    """Test 2: Persist multiple entries from a graph state audit_log."""
    from src.utils.audit_logger import AuditLogger
    audit = AuditLogger()
    state_log = [
        {"timestamp": "2026-04-17T10:00:00Z", "node": "discover",
         "action": "tender_discovered", "detail": "Found on SAM.gov",
         "model_used": None, "tokens_used": None},
        {"timestamp": "2026-04-17T10:00:01Z", "node": "evaluate",
         "action": "tender_scored", "detail": "Score: 77/100",
         "model_used": "claude-haiku-4-5", "tokens_used": 1200},
        {"timestamp": "2026-04-17T10:00:05Z", "node": "retrieve_draft",
         "action": "sections_drafted", "detail": "Drafted 5 sections",
         "model_used": "claude-sonnet-4-6", "tokens_used": 4500},
    ]
    count = audit.persist_from_state("TEST-002", state_log)
    assert count == 3, f"Expected 3, got {count}"
    entries = audit.get_by_tender("TEST-002")
    assert len(entries) == 3
    print(f"  ✅ Test 2 passed: Persisted {count} entries from state audit_log")

def test_3_query_methods():
    """Test 3: Query by tender, node, and token totals."""
    from src.utils.audit_logger import AuditLogger
    audit = AuditLogger()
    audit.log("T-A", "evaluate", "scored", tokens_used=500)
    audit.log("T-A", "retrieve_draft", "drafted", tokens_used=3000)
    audit.log("T-B", "evaluate", "scored", tokens_used=400)
    audit.log("T-A", "submit", "submitted", tokens_used=0)
    # By tender
    assert len(audit.get_by_tender("T-A")) == 3
    assert len(audit.get_by_tender("T-B")) == 1
    # By node
    assert len(audit.get_by_node("evaluate")) == 2
    assert len(audit.get_by_node("submit")) == 1
    # Token totals
    assert audit.get_total_tokens("T-A") == 3500
    assert audit.get_total_tokens("T-B") == 400
    assert audit.get_total_tokens() == 3900
    print("  ✅ Test 3 passed: Query by tender, node, and token totals all work")

def test_4_summary():
    """Test 4: Summary provides a useful overview of a tender's processing."""
    from src.utils.audit_logger import AuditLogger
    audit = AuditLogger()
    audit.log("SUM-001", "discover", "discovered", model_used=None, tokens_used=0)
    audit.log("SUM-001", "evaluate", "scored", model_used="haiku", tokens_used=500)
    audit.log("SUM-001", "retrieve_draft", "drafted", model_used="sonnet", tokens_used=3000)
    audit.log("SUM-001", "gap_check", "checked", model_used="sonnet", tokens_used=800)
    audit.log("SUM-001", "assemble", "assembled", model_used=None, tokens_used=0)
    audit.log("SUM-001", "submit", "submitted", model_used=None, tokens_used=0)
    summary = audit.get_summary("SUM-001")
    assert summary["entries"] == 6
    assert summary["total_tokens"] == 4300
    assert len(summary["unique_nodes"]) == 6
    assert "haiku" in summary["models_used"]
    assert "sonnet" in summary["models_used"]
    print(f"  ✅ Test 4 passed: Summary shows {summary['entries']} entries, {summary['total_tokens']} tokens, {len(summary['models_used'])} models")

def test_5_format_timeline():
    """Test 5: Timeline produces readable output."""
    from src.utils.audit_logger import AuditLogger
    audit = AuditLogger()
    audit.log("TL-001", "discover", "discovered", detail="From SAM.gov")
    audit.log("TL-001", "evaluate", "scored", detail="77/100 GO", tokens_used=500)
    audit.log("TL-001", "submit", "submitted", detail="Portal upload CONF-ABC")
    timeline = audit.format_timeline("TL-001")
    assert "TL-001" in timeline
    assert "discover" in timeline
    assert "evaluate" in timeline
    assert "submit" in timeline
    assert "500 tokens" in timeline
    assert "3 events" in timeline
    # Empty tender
    empty = audit.format_timeline("NONEXISTENT")
    assert "No audit trail" in empty
    print("  ✅ Test 5 passed: Timeline is human-readable with correct data")

def test_6_end_to_end_with_graph():
    """Test 6: Audit logger captures entries from a full graph run."""
    import src.agent.graph as gm
    from src.agent.nodes.discover import discover_node
    from src.agent.nodes.evaluate import evaluate_node
    from src.agent.nodes.retrieve_draft import retrieve_draft_node
    from src.agent.nodes.gap_check import gap_check_node
    from src.agent.nodes.slack_escalate import slack_escalate_node
    from src.agent.nodes.assemble import assemble_node
    from src.agent.nodes.submit import submit_node
    from src.utils.audit_logger import AuditLogger
    orig = {}
    for name, fn in [("discover", discover_node), ("evaluate", evaluate_node),
                      ("retrieve_draft", retrieve_draft_node), ("gap_check", gap_check_node),
                      ("slack_escalate", slack_escalate_node), ("assemble", assemble_node),
                      ("submit", submit_node)]:
        orig[name] = getattr(gm, f"{name}_node")
        setattr(gm, f"{name}_node", fn)
    try:
        from src.agent.graph import build_tender_graph
        graph = build_tender_graph(checkpointer=None)
        result = graph.invoke({
            "tender_id": "AUDIT-E2E", "tender_title": "SDS Platform",
            "source_portal": "sam.gov", "source_url": "https://sam.gov/test",
            "tender_raw_text": "Safety Data Sheet SDS management platform with GHS classification OSHA HCS compliance chemical inventory tracking Tier II regulatory reporting ISO 27001 certification required annual subscription budget.",
            "submission_deadline": "2026-08-01T17:00:00Z",
            "discovered_at": "2026-04-17T00:00:00Z",
            "tender_document_path": None, "escalation_count": 0,
            "assembly_retry_count": 0, "error_messages": [], "audit_log": [],
        })
        # Persist the audit log
        audit = AuditLogger()
        count = audit.persist_from_state("AUDIT-E2E", result["audit_log"])
        assert count >= 6, f"Expected ≥6 entries, got {count}"
        summary = audit.get_summary("AUDIT-E2E")
        assert "discover" in summary["unique_nodes"]
        assert "submit" in summary["unique_nodes"]
        timeline = audit.format_timeline("AUDIT-E2E")
        assert "AUDIT-E2E" in timeline
        print(f"  ✅ Test 6 passed: Full graph audit captured ({count} entries, {summary['total_tokens']} tokens)")
    finally:
        for name, fn in orig.items():
            setattr(gm, f"{name}_node", fn)

def main():
    print("\n" + "=" * 60)
    print("  Step 23 Verification: Structured Audit Logger")
    print("=" * 60 + "\n")
    tests = [
        ("Test 1: Log single entry", test_1_log_single_entry),
        ("Test 2: Persist from state", test_2_persist_from_state),
        ("Test 3: Query methods", test_3_query_methods),
        ("Test 4: Summary", test_4_summary),
        ("Test 5: Format timeline", test_5_format_timeline),
        ("Test 6: End-to-end with graph", test_6_end_to_end_with_graph),
    ]
    passed = failed = 0
    for name, fn in tests:
        try: fn(); passed += 1
        except AssertionError as e: print(f"  ❌ {name} FAILED: {e}"); failed += 1
        except Exception as e: print(f"  ❌ {name} ERROR: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{'=' * 60}\n  Results: {passed} passed, {failed} failed\n{'=' * 60}\n")
    if failed: sys.exit(1)
    else:
        print("  🎉 All tests passed! Step 23 is complete.")
        print("  Next: git add -A && git commit -m 'Step 23: Structured audit logger'\n")

if __name__ == "__main__":
    main()