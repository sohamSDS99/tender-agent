#!/usr/bin/env python3
"""
Step 10 Verification — Retrieve & Draft Node Tests

Runs 5 tests to verify requirement extraction, section drafting,
multi-model routing, and graph integration. All in dry-run mode.

Usage:
    cd /Users/sohamsarker/tender-agent
    source .venv/bin/activate
    python scripts/test_retrieve_draft.py

Expected output: All 5 tests pass.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["DRY_RUN"] = "true"


SAMPLE_TENDER_TEXT = (
    "The State Environmental Agency seeks a cloud-based SDS management platform. "
    "Requirements include: "
    "1. Company Overview - Provide a detailed overview of your company including "
    "years of experience and number of clients served. "
    "2. Technical Capabilities - Describe your platform's SDS management, GHS "
    "classification, and chemical inventory tracking capabilities. "
    "3. Regulatory Compliance - Detail your compliance with OSHA HCS, WHMIS, "
    "CLP/REACH regulations and any relevant ISO certifications. "
    "4. Implementation Plan - Provide a timeline for implementation including "
    "data migration, training, and go-live milestones. "
    "5. Pricing - Provide annual subscription pricing for 200 users."
)


def test_1_requirement_extraction() -> None:
    """Test 1: Dry-run extracts structured requirements from tender text."""
    from src.agent.nodes.retrieve_draft import _extract_requirements_dry_run

    requirements = _extract_requirements_dry_run(SAMPLE_TENDER_TEXT)

    assert len(requirements) >= 3, (
        f"Expected ≥3 requirements, got {len(requirements)}"
    )

    # Each requirement should have required fields
    for req in requirements:
        assert "section_id" in req, f"Missing section_id: {req}"
        assert "section_title" in req, f"Missing section_title: {req}"
        assert "requirement_text" in req, f"Missing requirement_text: {req}"

    print(
        f"  ✅ Test 1 passed: Extracted {len(requirements)} requirements "
        f"from tender text"
    )


def test_2_section_drafting() -> None:
    """Test 2: Dry-run drafts sections with correct structure."""
    from src.agent.nodes.retrieve_draft import retrieve_draft_node

    state = {
        "tender_id": "TEST-DRAFT-001",
        "tender_raw_text": SAMPLE_TENDER_TEXT,
        "escalation_count": 0,
        "audit_log": [],
        "error_messages": [],
    }

    result = retrieve_draft_node(state)

    sections = result["drafted_sections"]
    assert len(sections) >= 3, f"Expected ≥3 drafted sections, got {len(sections)}"

    for section in sections:
        assert section.get("section_id"), f"Missing section_id: {section}"
        assert section.get("section_title"), f"Missing section_title: {section}"
        assert section.get("content"), f"Missing content: {section}"
        assert section.get("confidence", 0) > 0, f"Confidence should be > 0"
        assert section.get("model_used"), f"Missing model_used: {section}"
        assert "dry-run" in section["model_used"], "Should indicate dry-run"

    print(
        f"  ✅ Test 2 passed: Drafted {len(sections)} sections with complete metadata"
    )


def test_3_compliance_routing() -> None:
    """Test 3: Compliance-critical sections are routed to Opus model."""
    from src.agent.nodes.retrieve_draft import (
        OPUS_MODEL,
        SONNET_MODEL,
        _is_compliance_critical,
    )

    # Compliance section
    compliance_req = {
        "section_id": "3.0",
        "section_title": "Regulatory Compliance",
        "requirement_text": "Detail OSHA HCS compliance and ISO 27001 certification.",
    }
    assert _is_compliance_critical(compliance_req) is True, (
        "OSHA/ISO section should be compliance-critical"
    )

    # Non-compliance section
    overview_req = {
        "section_id": "1.0",
        "section_title": "Company Overview",
        "requirement_text": "Describe your company history and experience.",
    }
    assert _is_compliance_critical(overview_req) is False, (
        "Company overview should NOT be compliance-critical"
    )

    # Now verify the full node routes correctly
    from src.agent.nodes.retrieve_draft import retrieve_draft_node

    state = {
        "tender_id": "TEST-ROUTE-001",
        "tender_raw_text": SAMPLE_TENDER_TEXT,
        "escalation_count": 0,
        "audit_log": [],
        "error_messages": [],
    }

    result = retrieve_draft_node(state)
    sections = result["drafted_sections"]

    # At least one section should use Opus (the compliance one)
    opus_sections = [s for s in sections if "opus" in s.get("model_used", "").lower()]
    sonnet_sections = [s for s in sections if "sonnet" in s.get("model_used", "").lower()]

    assert len(opus_sections) >= 1, (
        f"Expected ≥1 Opus-routed section, got {len(opus_sections)}"
    )
    assert len(sonnet_sections) >= 1, (
        f"Expected ≥1 Sonnet-routed section, got {len(sonnet_sections)}"
    )

    print(
        f"  ✅ Test 3 passed: Multi-model routing works — "
        f"{len(opus_sections)} Opus, {len(sonnet_sections)} Sonnet"
    )


def test_4_redraft_preserves_good_sections() -> None:
    """Test 4: Re-drafting after Slack only re-does sections with gaps."""
    from src.agent.nodes.retrieve_draft import retrieve_draft_node

    # Simulate a state AFTER gap check identified a gap in section 3.0
    state = {
        "tender_id": "TEST-REDRAFT-001",
        "tender_raw_text": SAMPLE_TENDER_TEXT,
        "tender_requirements": [
            {"section_id": "1.0", "section_title": "Company Overview",
             "requirement_text": "Describe your company.", "is_mandatory": True},
            {"section_id": "3.0", "section_title": "Regulatory Compliance",
             "requirement_text": "Detail OSHA compliance.", "is_mandatory": True},
        ],
        "drafted_sections": [
            {"section_id": "1.0", "section_title": "Company Overview",
             "content": "Original good draft for overview.", "confidence": 0.9,
             "sources_used": ["profile.pdf"], "model_used": "claude-sonnet-4-6",
             "token_count": 200},
            {"section_id": "3.0", "section_title": "Regulatory Compliance",
             "content": "Incomplete draft.", "confidence": 0.4,
             "sources_used": [], "model_used": "claude-opus-4-6",
             "token_count": 100},
        ],
        "gaps": [
            {"section_id": "3.0", "description": "Missing OSHA details",
             "severity": "high", "suggested_question": "What OSHA certs?"},
        ],
        "slack_responses": ["We have OSHA 300 log compliance and VPP Star status."],
        "escalation_count": 1,
        "audit_log": [],
        "error_messages": [],
    }

    result = retrieve_draft_node(state)
    sections = result["drafted_sections"]

    # Section 1.0 should be preserved (no gap)
    section_1 = next((s for s in sections if s["section_id"] == "1.0"), None)
    assert section_1 is not None, "Section 1.0 should be in results"
    assert section_1["content"] == "Original good draft for overview.", (
        "Section 1.0 should be preserved unchanged"
    )

    # Section 3.0 should be re-drafted (had a gap)
    section_3 = next((s for s in sections if s["section_id"] == "3.0"), None)
    assert section_3 is not None, "Section 3.0 should be in results"
    assert section_3["content"] != "Incomplete draft.", (
        "Section 3.0 should have been re-drafted"
    )

    # Audit log should say "re-drafted"
    audit = result["audit_log"]
    assert any("Re-drafted" in e.get("detail", "") or "redrafted" in e.get("action", "")
               for e in audit), "Audit should mention re-drafting"

    print("  ✅ Test 4 passed: Re-draft preserves good sections, re-does only gaps")


def test_5_graph_integration() -> None:
    """Test 5: Real retrieve_draft node works in the full graph."""
    import src.agent.graph as graph_module
    from src.agent.nodes.evaluate import evaluate_node as real_evaluate
    from src.agent.nodes.retrieve_draft import retrieve_draft_node as real_draft

    # Swap both placeholders
    orig_evaluate = graph_module.evaluate_node
    orig_draft = graph_module.retrieve_draft_node
    graph_module.evaluate_node = real_evaluate
    graph_module.retrieve_draft_node = real_draft

    try:
        from src.agent.graph import build_tender_graph
        from src.agent.state import TenderStatus

        graph = build_tender_graph(checkpointer=None)

        initial_state = {
            "tender_id": "TEST-GRAPH-010",
            "tender_title": "SDS Platform for State Agency",
            "source_portal": "sam.gov",
            "source_url": "https://sam.gov/opp/test",
            "tender_raw_text": SAMPLE_TENDER_TEXT,
            "submission_deadline": "2026-06-15T17:00:00Z",
            "discovered_at": "2026-04-16T10:00:00Z",
            "tender_document_path": None,
            "escalation_count": 0,
            "assembly_retry_count": 0,
            "error_messages": [],
            "audit_log": [],
        }

        result = graph.invoke(initial_state)

        assert result["status"] == TenderStatus.SUBMITTED.value
        assert len(result["drafted_sections"]) >= 3, (
            f"Expected ≥3 sections, got {len(result['drafted_sections'])}"
        )
        assert len(result["tender_requirements"]) >= 3, (
            f"Expected ≥3 requirements, got {len(result['tender_requirements'])}"
        )

        print(
            f"  ✅ Test 5 passed: Full graph with real evaluate + draft nodes "
            f"({len(result['drafted_sections'])} sections drafted, "
            f"status={result['status']})"
        )

    finally:
        graph_module.evaluate_node = orig_evaluate
        graph_module.retrieve_draft_node = orig_draft


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  Step 10 Verification: Retrieve & Draft Node")
    print("=" * 60 + "\n")

    tests = [
        ("Test 1: Requirement extraction", test_1_requirement_extraction),
        ("Test 2: Section drafting", test_2_section_drafting),
        ("Test 3: Compliance routing (Sonnet vs Opus)", test_3_compliance_routing),
        ("Test 4: Re-draft preserves good sections", test_4_redraft_preserves_good_sections),
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
        print("  🎉 All tests passed! Step 10 is complete.")
        print()
        print("  NEXT: Update src/agent/graph.py —")
        print("    Add: from src.agent.nodes.retrieve_draft import retrieve_draft_node")
        print("    Delete the placeholder retrieve_draft_node function")
        print()
        print("  Then: git add -A && git commit -m 'Step 10: Retrieve & draft node'")
        print()


if __name__ == "__main__":
    main()