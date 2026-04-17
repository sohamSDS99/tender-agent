#!/usr/bin/env python3
"""Step 22 Verification — Submit Node Tests"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["DRY_RUN"] = "true"

def _make_doc(tmpdir: str) -> str:
    p = Path(tmpdir) / "tender_response.md"
    p.write_text("# Tender Response\n\nContent here.\n", encoding="utf-8")
    return str(p)

def test_1_portal_submission():
    """Test 1: SAM.gov tender routes to portal upload."""
    from src.agent.nodes.submit import submit_node
    with tempfile.TemporaryDirectory() as tmpdir:
        doc = _make_doc(tmpdir)
        state = {
            "tender_id": "SAM-2026-001", "tender_title": "SDS Platform",
            "source_portal": "sam.gov", "source_url": "https://sam.gov/opp/test",
            "tender_raw_text": "EHS software tender.",
            "assembled_document_path": doc,
            "audit_log": [], "error_messages": [],
        }
        result = submit_node(state)
        assert result["submission_status"] == "success"
        assert result["submission_method"] == "portal_upload"
        assert result["submission_confirmation"].startswith("CONF-")
        assert result["status"] == "submitted"
        print(f"  ✅ Test 1 passed: SAM.gov → portal upload (conf={result['submission_confirmation']})")

def test_2_email_submission():
    """Test 2: Email-sourced tender routes to email submission."""
    from src.agent.nodes.submit import submit_node
    with tempfile.TemporaryDirectory() as tmpdir:
        doc = _make_doc(tmpdir)
        state = {
            "tender_id": "EMAIL-001", "tender_title": "RFP: Chemical Safety",
            "source_portal": "email", "source_url": "email://msg@test.com",
            "tender_raw_text": "Chemical safety platform needed.",
            "assembled_document_path": doc,
            "audit_log": [], "error_messages": [],
        }
        result = submit_node(state)
        assert result["submission_status"] == "success"
        assert result["submission_method"] == "email"
        assert result["submission_confirmation"].startswith("EMAIL-")
        assert result["status"] == "submitted"
        print(f"  ✅ Test 2 passed: Email source → email submission (conf={result['submission_confirmation']})")

def test_3_manual_fallback():
    """Test 3: Explicit manual method flags for human submission."""
    from src.agent.nodes.submit import submit_node
    with tempfile.TemporaryDirectory() as tmpdir:
        doc = _make_doc(tmpdir)
        state = {
            "tender_id": "MANUAL-001", "tender_title": "Special Tender",
            "source_portal": "other", "source_url": "",
            "tender_raw_text": "Submit via registered mail.",
            "assembled_document_path": doc,
            "submission_method": "manual",
            "audit_log": [], "error_messages": [],
        }
        result = submit_node(state)
        assert result["submission_status"] == "success"
        assert result["submission_method"] == "manual"
        assert result["submission_confirmation"] == "MANUAL-PENDING"
        print("  ✅ Test 3 passed: Manual method flags for human submission")

def test_4_missing_document_fails():
    """Test 4: Missing document causes submission failure."""
    from src.agent.nodes.submit import submit_node
    state = {
        "tender_id": "FAIL-001", "tender_title": "Test",
        "source_portal": "sam.gov",
        "assembled_document_path": "",
        "audit_log": [], "error_messages": [],
    }
    result = submit_node(state)
    assert result["submission_status"] == "failed"
    assert result["status"] == "submission_failed"
    assert any("No assembled document" in e.get("detail", "") for e in result["audit_log"])
    print("  ✅ Test 4 passed: Missing document correctly fails submission")

def test_5_full_graph_end_to_end():
    """Test 5: Complete graph with ALL real nodes — discover through submit."""
    import src.agent.graph as gm
    from src.agent.nodes.discover import discover_node
    from src.agent.nodes.evaluate import evaluate_node
    from src.agent.nodes.retrieve_draft import retrieve_draft_node
    from src.agent.nodes.gap_check import gap_check_node
    from src.agent.nodes.slack_escalate import slack_escalate_node
    from src.agent.nodes.assemble import assemble_node
    from src.agent.nodes.submit import submit_node

    # Swap ALL placeholders with real implementations
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
            "tender_id": "E2E-FINAL-001",
            "tender_title": "SDS Management Platform for Federal Agency",
            "source_portal": "sam.gov",
            "source_url": "https://sam.gov/opp/final-test",
            "tender_raw_text": (
                "The federal agency seeks a cloud-based SDS management platform "
                "with GHS classification, OSHA HCS compliance, chemical inventory "
                "tracking, and Tier II regulatory reporting. ISO 27001 required. "
                "Budget: $75,000 annual subscription."
            ),
            "submission_deadline": "2026-08-01T17:00:00Z",
            "discovered_at": "2026-04-17T00:00:00Z",
            "tender_document_path": None,
            "escalation_count": 0,
            "assembly_retry_count": 0,
            "error_messages": [],
            "audit_log": [],
        })

        # Verify final state
        assert result["status"] == "submitted", f"Expected submitted, got {result['status']}"
        assert result["submission_status"] == "success"
        assert result["submission_confirmation"], "Should have confirmation ID"
        assert result["assembled_document_path"], "Should have document path"
        assert Path(result["assembled_document_path"]).exists(), "Document should exist"

        # Verify all 7 nodes ran (6 in happy path — no slack escalation)
        audit_nodes = [e["node"] for e in result.get("audit_log", [])]
        for expected in ["discover", "evaluate", "retrieve_draft", "gap_check", "assemble", "submit"]:
            assert expected in audit_nodes, f"Missing {expected} from audit trail"

        # Print summary
        print(f"  ✅ Test 5 passed: FULL END-TO-END PIPELINE COMPLETE")
        print(f"     Tender: {result['tender_title']}")
        print(f"     Score: {result.get('eval_score', '?')}/100 ({result.get('eval_decision', '?')})")
        print(f"     Sections: {len(result.get('drafted_sections', []))}")
        print(f"     Gaps: {len(result.get('gaps', []))}")
        print(f"     Document: {Path(result['assembled_document_path']).name}")
        print(f"     Method: {result['submission_method']}")
        print(f"     Confirmation: {result['submission_confirmation']}")
        print(f"     Audit trail: {len(audit_nodes)} entries")

    finally:
        for name, fn in orig.items():
            setattr(gm, f"{name}_node", fn)

def main():
    print("\n" + "=" * 60)
    print("  Step 22 Verification: Submit Node")
    print("=" * 60 + "\n")
    tests = [
        ("Test 1: Portal submission (SAM.gov)", test_1_portal_submission),
        ("Test 2: Email submission", test_2_email_submission),
        ("Test 3: Manual fallback", test_3_manual_fallback),
        ("Test 4: Missing document fails", test_4_missing_document_fails),
        ("Test 5: FULL END-TO-END PIPELINE", test_5_full_graph_end_to_end),
    ]
    passed = failed = 0
    for name, fn in tests:
        try: fn(); passed += 1
        except AssertionError as e: print(f"  ❌ {name} FAILED: {e}"); failed += 1
        except Exception as e: print(f"  ❌ {name} ERROR: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{'=' * 60}\n  Results: {passed} passed, {failed} failed\n{'=' * 60}\n")
    if failed: sys.exit(1)
    else:
        print("  🎉 All tests passed! Step 22 is complete.")
        print("  The entire 7-node pipeline now runs end-to-end with real implementations.")
        print("  NEXT: Update graph.py — import submit_node, delete placeholder")
        print("  Then: git add -A && git commit -m 'Step 22: Submit node'\n")

if __name__ == "__main__":
    main()