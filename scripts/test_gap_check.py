#!/usr/bin/env python3
"""
Step 11 Verification — Gap Check Node Tests

Runs 5 tests to verify gap detection for various scenarios:
clean drafts, low confidence, placeholders, missing sections, and
full graph integration.

Usage:
    cd /Users/sohamsarker/tender-agent
    source .venv/bin/activate
    python scripts/test_gap_check.py

Expected output: All 5 tests pass.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["DRY_RUN"] = "true"


def test_1_clean_draft_passes() -> None:
    """Test 1: A complete, high-confidence draft passes with no gaps."""
    from src.agent.nodes.gap_check import gap_check_node

    state = {
        "tender_id": "TEST-CLEAN-001",
        "tender_requirements": [
            {"section_id": "1.0", "section_title": "Company Overview",
             "requirement_text": "Describe your company.", "is_mandatory": True},
            {"section_id": "2.0", "section_title": "Capabilities",
             "requirement_text": "Describe technical capabilities.", "is_mandatory": True},
        ],
        "drafted_sections": [
            {"section_id": "1.0", "section_title": "Company Overview",
             "content": (
                 "Acme SDS Solutions is a leading provider of Safety Data Sheet "
                 "management software. We serve over 500 clients across "
                 "manufacturing, construction, oil and gas, and pharmaceutical "
                 "industries with our cloud-based platform."
             ),
             "confidence": 0.90, "sources_used": ["profile.pdf"],
             "model_used": "claude-sonnet-4-6", "token_count": 200},
            {"section_id": "2.0", "section_title": "Capabilities",
             "content": (
                 "Our platform provides comprehensive SDS management including GHS "
                 "classification, chemical inventory tracking, regulatory compliance "
                 "reporting, and mobile access via QR codes. We support OSHA HCS, "
                 "WHMIS, CLP/REACH, and over 140 global jurisdictions."
             ),
             "confidence": 0.88, "sources_used": ["capabilities.pdf"],
             "model_used": "claude-sonnet-4-6", "token_count": 250},
        ],
        "audit_log": [],
        "error_messages": [],
    }

    result = gap_check_node(state)

    assert result["gap_check_passed"] is True, (
        f"Clean draft should pass, but got gaps: {result['gaps']}"
    )
    assert len(result["gaps"]) == 0

    print("  ✅ Test 1 passed: Clean, high-confidence draft passes with 0 gaps")


def test_2_low_confidence_detected() -> None:
    """Test 2: Low-confidence sections are flagged as gaps."""
    from src.agent.nodes.gap_check import gap_check_node

    state = {
        "tender_id": "TEST-LOWCONF-001",
        "tender_requirements": [
            {"section_id": "1.0", "section_title": "Pricing",
             "requirement_text": "Provide detailed pricing.", "is_mandatory": True},
        ],
        "drafted_sections": [
            {"section_id": "1.0", "section_title": "Pricing",
             "content": (
                 "Our pricing is competitive and varies based on the number of users "
                 "and modules selected. Contact us for a detailed quote tailored to "
                 "your specific needs and requirements."
             ),
             "confidence": 0.45, "sources_used": [],
             "model_used": "claude-sonnet-4-6", "token_count": 100},
        ],
        "audit_log": [],
        "error_messages": [],
    }

    result = gap_check_node(state)

    assert result["gap_check_passed"] is False, "Low confidence should trigger gap"
    assert len(result["gaps"]) >= 1

    conf_gap = next(
        (g for g in result["gaps"] if "confidence" in g.get("description", "").lower()),
        None,
    )
    assert conf_gap is not None, "Should have a confidence-related gap"
    assert conf_gap["severity"] in ("high", "medium")

    print(
        f"  ✅ Test 2 passed: Low-confidence section flagged "
        f"({len(result['gaps'])} gap(s), severity={conf_gap['severity']})"
    )


def test_3_placeholder_markers_detected() -> None:
    """Test 3: [INFORMATION NEEDED] placeholders are caught."""
    from src.agent.nodes.gap_check import gap_check_node

    state = {
        "tender_id": "TEST-PLACEHOLDER-001",
        "tender_requirements": [
            {"section_id": "3.0", "section_title": "Certifications",
             "requirement_text": "List all certifications.", "is_mandatory": True},
        ],
        "drafted_sections": [
            {"section_id": "3.0", "section_title": "Certifications",
             "content": (
                 "Acme SDS Solutions holds ISO 27001 certification for information "
                 "security. We also maintain SOC 2 Type II compliance. "
                 "[INFORMATION NEEDED: expiry date of ISO 27001 certification] "
                 "Additionally, [INFORMATION NEEDED: FedRAMP authorization status]."
             ),
             "confidence": 0.72, "sources_used": ["certs.pdf"],
             "model_used": "claude-sonnet-4-6", "token_count": 150},
        ],
        "audit_log": [],
        "error_messages": [],
    }

    result = gap_check_node(state)

    assert result["gap_check_passed"] is False
    placeholder_gaps = [
        g for g in result["gaps"] if "placeholder" in g.get("description", "").lower()
    ]
    assert len(placeholder_gaps) >= 2, (
        f"Expected ≥2 placeholder gaps, got {len(placeholder_gaps)}"
    )

    print(
        f"  ✅ Test 3 passed: {len(placeholder_gaps)} placeholder markers detected"
    )


def test_4_missing_mandatory_section() -> None:
    """Test 4: A mandatory requirement with no draft triggers a high-severity gap."""
    from src.agent.nodes.gap_check import gap_check_node

    state = {
        "tender_id": "TEST-MISSING-001",
        "tender_requirements": [
            {"section_id": "1.0", "section_title": "Company Overview",
             "requirement_text": "Describe your company.", "is_mandatory": True},
            {"section_id": "2.0", "section_title": "Security Plan",
             "requirement_text": "Provide a data security plan.", "is_mandatory": True},
            {"section_id": "3.0", "section_title": "References",
             "requirement_text": "List client references.", "is_mandatory": False},
        ],
        "drafted_sections": [
            # Only section 1.0 was drafted — 2.0 and 3.0 are missing
            {"section_id": "1.0", "section_title": "Company Overview",
             "content": (
                 "Acme SDS Solutions is a leading SDS management platform provider "
                 "serving over 500 clients across multiple industries worldwide."
             ),
             "confidence": 0.90, "sources_used": ["profile.pdf"],
             "model_used": "claude-sonnet-4-6", "token_count": 100},
        ],
        "audit_log": [],
        "error_messages": [],
    }

    result = gap_check_node(state)

    assert result["gap_check_passed"] is False
    
    # Section 2.0 (mandatory) should be a high-severity gap
    missing_mandatory = [
        g for g in result["gaps"]
        if g.get("section_id") == "2.0" and g.get("severity") == "high"
    ]
    assert len(missing_mandatory) >= 1, "Missing mandatory section should be high severity"

    print(
        f"  ✅ Test 4 passed: Missing mandatory section detected as high-severity gap "
        f"({len(result['gaps'])} total gaps)"
    )


def test_5_graph_integration() -> None:
    """Test 5: Gap check node works in the full graph with real nodes."""
    import src.agent.graph as graph_module
    from src.agent.nodes.evaluate import evaluate_node as real_evaluate
    from src.agent.nodes.gap_check import gap_check_node as real_gap_check
    from src.agent.nodes.retrieve_draft import retrieve_draft_node as real_draft

    orig_evaluate = graph_module.evaluate_node
    orig_draft = graph_module.retrieve_draft_node
    orig_gap = graph_module.gap_check_node
    graph_module.evaluate_node = real_evaluate
    graph_module.retrieve_draft_node = real_draft
    graph_module.gap_check_node = real_gap_check

    try:
        from src.agent.graph import build_tender_graph
        from src.agent.state import TenderStatus

        graph = build_tender_graph(checkpointer=None)

        initial_state = {
            "tender_id": "TEST-GRAPH-011",
            "tender_title": "SDS Platform for Agency",
            "source_portal": "sam.gov",
            "source_url": "https://sam.gov/opp/test",
            "tender_raw_text": (
                "The agency requires an SDS management platform with GHS "
                "classification, OSHA HCS compliance, and chemical inventory. "
                "ISO 27001 certification required. Budget $75,000 annual."
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

        # Verify gap check ran
        audit_nodes = [e["node"] for e in result.get("audit_log", [])]
        assert "gap_check" in audit_nodes, "Gap check should appear in audit log"

        # gap_check_passed should be a boolean
        assert isinstance(result.get("gap_check_passed"), bool)

        # gaps should be a list
        assert isinstance(result.get("gaps"), list)

        print(
            f"  ✅ Test 5 passed: Full graph with real evaluate + draft + gap_check "
            f"(gaps_found={len(result['gaps'])}, passed={result['gap_check_passed']}, "
            f"status={result['status']})"
        )

    finally:
        graph_module.evaluate_node = orig_evaluate
        graph_module.retrieve_draft_node = orig_draft
        graph_module.gap_check_node = orig_gap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  Step 11 Verification: Gap Check Node")
    print("=" * 60 + "\n")

    tests = [
        ("Test 1: Clean draft passes", test_1_clean_draft_passes),
        ("Test 2: Low confidence detected", test_2_low_confidence_detected),
        ("Test 3: Placeholder markers detected", test_3_placeholder_markers_detected),
        ("Test 4: Missing mandatory section", test_4_missing_mandatory_section),
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
        print("  🎉 All tests passed! Step 11 is complete.")
        print()
        print("  NEXT: Update src/agent/graph.py —")
        print("    Add: from src.agent.nodes.gap_check import gap_check_node")
        print("    Delete the placeholder gap_check_node function")
        print()
        print("  Then: git add -A && git commit -m 'Step 11: Gap check node'")
        print()


if __name__ == "__main__":
    main()