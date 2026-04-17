#!/usr/bin/env python3
"""Step 19 Verification — Assemble Node Tests"""
from __future__ import annotations
import os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["DRY_RUN"] = "true"

def _make_good_state():
    return {
        "tender_id": "SAM-2026-001",
        "tender_title": "SDS Platform for EPA",
        "source_portal": "sam.gov",
        "source_url": "https://sam.gov/opp/test",
        "tender_raw_text": "SDS management platform with GHS and OSHA compliance.",
        "submission_deadline": "2026-06-15T17:00:00Z",
        "drafted_sections": [
            {"section_id": "1.0", "section_title": "Company Overview",
             "content": "Acme SDS Solutions is a leading provider of Safety Data Sheet management software serving over 500 clients across manufacturing, construction, oil and gas, and pharma. Our platform is trusted globally.",
             "confidence": 0.92, "sources_used": ["profile.pdf"], "model_used": "qwen3.5-plus", "token_count": 200},
            {"section_id": "2.0", "section_title": "Technical Capabilities",
             "content": "Our platform provides GHS classification, chemical inventory tracking, OSHA Tier II reporting, mobile access via QR codes, and support for 140+ regulatory jurisdictions worldwide.",
             "confidence": 0.88, "sources_used": ["caps.pdf"], "model_used": "qwen3.5-plus", "token_count": 250},
            {"section_id": "3.0", "section_title": "Compliance",
             "content": "Acme holds ISO 27001 certification and SOC 2 Type II compliance. AWS US-East and EU-West hosting with 99.95% uptime SLA and full disaster recovery capabilities.",
             "confidence": 0.85, "sources_used": ["certs.pdf"], "model_used": "qwen3-max", "token_count": 300},
        ],
        "assembly_retry_count": 0,
        "audit_log": [],
        "error_messages": [],
    }

def test_1_assemble_produces_document():
    """Test 1: Assemble node produces a document and passes quality check."""
    from src.agent.nodes.assemble import assemble_node
    result = assemble_node(_make_good_state())
    assert result["assembled_document_path"], "Should produce a document path"
    assert Path(result["assembled_document_path"]).exists(), "Document file should exist"
    assert result["quality_check_passed"] is True, f"Should pass, issues: {result.get('quality_issues')}"
    assert result["status"] == "assembling"
    content = Path(result["assembled_document_path"]).read_text()
    assert "SDS Platform for EPA" in content
    assert "Company Overview" in content
    print(f"  ✅ Test 1 passed: Document assembled at {Path(result['assembled_document_path']).name}")

def test_2_template_selection():
    """Test 2: SAM.gov tenders use government template."""
    from src.agent.nodes.assemble import _select_template
    assert _select_template({"source_portal": "sam.gov"}) == "government"
    assert _select_template({"source_portal": "email", "tender_raw_text": ""}) == "standard"
    assert _select_template({"source_portal": "email", "tender_raw_text": "simple proposal requested"}) == "simple"
    print("  ✅ Test 2 passed: Template selection works (sam.gov→government, default→standard, informal→simple)")

def test_3_quality_failure_increments_retry():
    """Test 3: Quality failure increments assembly_retry_count."""
    from src.agent.nodes.assemble import assemble_node
    state = _make_good_state()
    # Add unresolved placeholders to force quality failure
    state["drafted_sections"][0]["content"] = "Short. [INFORMATION NEEDED: details]"
    result = assemble_node(state)
    assert result["quality_check_passed"] is False, "Should fail (placeholder)"
    assert result["assembly_retry_count"] == 1, f"Expected retry=1, got {result.get('assembly_retry_count')}"
    assert result["status"] == "quality_failed"
    assert any("placeholder" in i.lower() for i in result["quality_issues"])
    print(f"  ✅ Test 3 passed: Quality failure sets retry=1, status=quality_failed")

def test_4_retry_truncation():
    """Test 4: Retry attempts truncate long sections to meet page limits."""
    from src.agent.nodes.assemble import assemble_node
    state = _make_good_state()
    state["source_portal"] = "email"  # Use simple template (20 page limit)
    state["tender_raw_text"] = "simple proposal"
    # Make sections very long to exceed 20 pages
    for s in state["drafted_sections"]:
        s["content"] = "This is detailed content for the tender response. " * 300
    state["assembly_retry_count"] = 1  # Simulate a retry
    result = assemble_node(state)
    # The document should have been truncated
    content = Path(result["assembled_document_path"]).read_text()
    assert "truncated" in content.lower(), "Retry should truncate long sections"
    print("  ✅ Test 4 passed: Retry truncation applied to long sections")

def test_5_graph_integration():
    """Test 5: Real assemble node works in the full graph."""
    import src.agent.graph as gm
    from src.agent.nodes.discover import discover_node
    from src.agent.nodes.evaluate import evaluate_node
    from src.agent.nodes.retrieve_draft import retrieve_draft_node
    from src.agent.nodes.gap_check import gap_check_node
    from src.agent.nodes.slack_escalate import slack_escalate_node
    from src.agent.nodes.assemble import assemble_node
    orig = {}
    for name, fn in [("discover", discover_node), ("evaluate", evaluate_node),
                      ("retrieve_draft", retrieve_draft_node), ("gap_check", gap_check_node),
                      ("slack_escalate", slack_escalate_node), ("assemble", assemble_node)]:
        orig[name] = getattr(gm, f"{name}_node")
        setattr(gm, f"{name}_node", fn)
    try:
        from src.agent.graph import build_tender_graph
        graph = build_tender_graph(checkpointer=None)
        result = graph.invoke({
            "tender_id": "TEST-ASM-001", "tender_title": "SDS Platform",
            "source_portal": "sam.gov", "source_url": "https://sam.gov/test",
            "tender_raw_text": "SDS management GHS OSHA chemical inventory ISO 27001 Budget $75K.",
            "submission_deadline": "2026-08-01T17:00:00Z", "discovered_at": "2026-04-16T00:00:00Z",
            "tender_document_path": None, "escalation_count": 0, "assembly_retry_count": 0,
            "error_messages": [], "audit_log": [],
        })
        assert result["status"] == "submitted"
        assert result["assembled_document_path"], "Should have document path"
        assert Path(result["assembled_document_path"]).exists(), "Document should exist on disk"
        audit_nodes = [e["node"] for e in result.get("audit_log", [])]
        assert "assemble" in audit_nodes
        print(f"  ✅ Test 5 passed: Full graph produces document at {Path(result['assembled_document_path']).name}")
    finally:
        for name, fn in orig.items():
            setattr(gm, f"{name}_node", fn)

def main():
    print("\n" + "=" * 60)
    print("  Step 19 Verification: Assemble Node")
    print("=" * 60 + "\n")
    tests = [
        ("Test 1: Assemble produces document", test_1_assemble_produces_document),
        ("Test 2: Template selection", test_2_template_selection),
        ("Test 3: Quality failure increments retry", test_3_quality_failure_increments_retry),
        ("Test 4: Retry truncation", test_4_retry_truncation),
        ("Test 5: Graph integration", test_5_graph_integration),
    ]
    passed = failed = 0
    for name, fn in tests:
        try: fn(); passed += 1
        except AssertionError as e: print(f"  ❌ {name} FAILED: {e}"); failed += 1
        except Exception as e: print(f"  ❌ {name} ERROR: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{'=' * 60}\n  Results: {passed} passed, {failed} failed\n{'=' * 60}\n")
    if failed: sys.exit(1)
    else:
        print("  🎉 All tests passed! Step 19 is complete.")
        print("  NEXT: Update graph.py — import assemble_node, delete placeholder")
        print("  Then: git add -A && git commit -m 'Step 19: Assemble node'\n")

if __name__ == "__main__":
    main()