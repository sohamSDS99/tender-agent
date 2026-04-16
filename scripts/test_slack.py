#!/usr/bin/env python3
"""Steps 15-16 Verification — Slack Client & Escalate Node Tests"""
from __future__ import annotations
import os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["DRY_RUN"] = "true"

def test_1_slack_client_send():
    from src.slack_integration.slack_client import SlackClient
    client = SlackClient()
    msg = client.send_gap_questions(
        tender_title="SDS Platform for EPA", tender_deadline="2026-06-15",
        questions=["What is the ISO 27001 expiry date?", "Do we have FedRAMP?"],
        tender_id="TEST-001",
    )
    assert msg.channel == "C_MOCK_CHANNEL"
    assert msg.ts
    assert msg.thread_ts
    assert msg.blocks is not None
    print("  ✅ Test 1 passed: SlackClient sends messages with Block Kit blocks")

def test_2_slack_client_responses():
    from src.slack_integration.slack_client import SlackClient
    client = SlackClient()
    msg = client.send_gap_questions(
        tender_title="Test", tender_deadline="2026-07-01",
        questions=["Test?"], tender_id="TEST-002",
    )
    responses = client.get_thread_responses(msg.channel, msg.ts)
    assert len(responses) >= 1
    assert responses[0].is_response is True
    assert responses[0].user == "U_HUMAN_MOCK"
    print(f"  ✅ Test 2 passed: Got {len(responses)} mock response(s)")

def test_3_deadline_warning():
    from src.slack_integration.slack_client import SlackClient
    client = SlackClient()
    msg72 = client.send_deadline_warning("Test", "2026-06-15", 72, "T-003")
    msg24 = client.send_deadline_warning("Test", "2026-06-15", 24, "T-003")
    msg4 = client.send_deadline_warning("Test", "2026-06-15", 4, "T-003")
    assert "REMINDER" in msg72.text
    assert "WARNING" in msg24.text
    assert "URGENT" in msg4.text
    print("  ✅ Test 3 passed: Deadline warnings have correct urgency levels")

def test_4_escalate_node():
    from src.agent.nodes.slack_escalate import slack_escalate_node
    state = {
        "tender_id": "TEST-ESC-001", "tender_title": "EHS Platform",
        "submission_deadline": "2026-07-01T17:00:00Z",
        "gaps": [
            {"section_id": "3.0", "description": "Missing cert", "severity": "high",
             "suggested_question": "What is ISO 27001 expiry?"},
            {"section_id": "5.0", "description": "No pricing", "severity": "medium",
             "suggested_question": "Pricing above 500 users?"},
        ],
        "escalation_count": 0, "audit_log": [], "error_messages": [],
    }
    result = slack_escalate_node(state)
    assert len(result["slack_questions"]) == 2
    assert len(result["slack_responses"]) >= 1
    assert result["escalation_count"] == 1
    assert result["status"] == "awaiting_human"
    assert any("[HIGH]" in q for q in result["slack_questions"])
    print(f"  ✅ Test 4 passed: Escalate node sent 2 questions, got {len(result['slack_responses'])} response(s)")

def test_5_escalation_counter():
    from src.agent.nodes.slack_escalate import slack_escalate_node
    state = {
        "tender_id": "T-002", "tender_title": "Test", "submission_deadline": "2026-08-01",
        "gaps": [{"section_id": "1.0", "severity": "low",
                   "suggested_question": "Any updates?", "description": "test"}],
        "escalation_count": 2, "audit_log": [], "error_messages": [],
    }
    result = slack_escalate_node(state)
    assert result["escalation_count"] == 3
    print("  ✅ Test 5 passed: Escalation counter increments correctly (2 → 3)")

def test_6_graph_escalation_loop():
    import src.agent.graph as gm
    from src.agent.nodes.discover import discover_node as rd
    from src.agent.nodes.evaluate import evaluate_node as re
    from src.agent.nodes.retrieve_draft import retrieve_draft_node as rrd
    from src.agent.nodes.gap_check import gap_check_node as rgc
    from src.agent.nodes.slack_escalate import slack_escalate_node as rse
    orig = {n: getattr(gm, f"{n}_node") for n in ["discover","evaluate","retrieve_draft","gap_check","slack_escalate"]}
    gm.discover_node = rd; gm.evaluate_node = re; gm.retrieve_draft_node = rrd; gm.slack_escalate_node = rse
    call_count = {"n": 0}
    def gap_first_fail(state):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"gaps": [{"section_id": "3.0", "description": "Missing", "severity": "high",
                              "suggested_question": "ISO expiry?"}],
                    "gap_check_passed": False, "status": "gap_check", "current_node": "gap_check",
                    "audit_log": [{"timestamp": "2026-04-16T00:00:00Z", "node": "gap_check",
                                   "action": "gaps_found", "detail": "1 gap", "model_used": None, "tokens_used": None}]}
        return rgc(state)
    gm.gap_check_node = gap_first_fail
    try:
        from src.agent.graph import build_tender_graph
        graph = build_tender_graph(checkpointer=None)
        result = graph.invoke({
            "tender_id": "TEST-LOOP", "tender_title": "SDS Tender",
            "source_portal": "sam.gov", "source_url": "https://sam.gov/test",
            "tender_raw_text": "SDS management GHS OSHA chemical inventory ISO 27001 Budget $75K.",
            "submission_deadline": "2026-08-01T17:00:00Z", "discovered_at": "2026-04-16T00:00:00Z",
            "tender_document_path": None, "escalation_count": 0, "assembly_retry_count": 0,
            "error_messages": [], "audit_log": [],
        })
        assert result["status"] == "submitted"
        assert result["escalation_count"] >= 1
        nodes = [e["node"] for e in result.get("audit_log", [])]
        assert "slack_escalate" in nodes
        assert nodes.count("retrieve_draft") >= 2
        print(f"  ✅ Test 6 passed: Gap → Slack → Re-draft loop (escalations={result['escalation_count']}, drafts={nodes.count('retrieve_draft')}x)")
    finally:
        for n, fn in orig.items():
            setattr(gm, f"{n}_node", fn)

def main():
    print("\n" + "=" * 60)
    print("  Steps 15-16 Verification: Slack Client & Escalate Node")
    print("=" * 60 + "\n")
    tests = [
        ("Test 1: Slack client send", test_1_slack_client_send),
        ("Test 2: Thread responses", test_2_slack_client_responses),
        ("Test 3: Deadline warnings", test_3_deadline_warning),
        ("Test 4: Escalate node", test_4_escalate_node),
        ("Test 5: Escalation counter", test_5_escalation_counter),
        ("Test 6: Graph escalation loop", test_6_graph_escalation_loop),
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn(); passed += 1
        except AssertionError as e:
            print(f"  ❌ {name} FAILED: {e}"); failed += 1
        except Exception as e:
            print(f"  ❌ {name} ERROR: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{'=' * 60}\n  Results: {passed} passed, {failed} failed\n{'=' * 60}\n")
    if failed: sys.exit(1)
    else:
        print("  🎉 All tests passed! Steps 15-16 are complete.")
        print("  NEXT: Update graph.py — import slack_escalate_node, delete placeholder")
        print("  Then: git add -A && git commit -m 'Steps 15-16: Slack client & escalate node'\n")

if __name__ == "__main__":
    main()