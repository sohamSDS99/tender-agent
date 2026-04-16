#!/usr/bin/env python3
"""
Step 14 Verification — Discover Node Integration Tests

Runs 5 tests to verify the discovery coordinator, tender state conversion,
validation, deadline checking, and full graph integration.

Usage:
    cd /Users/sohamsarker/tender-agent
    source .venv/bin/activate
    python scripts/test_discover.py

Expected output: All 5 tests pass.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["DRY_RUN"] = "true"


def test_1_coordinator_discovers_tenders() -> None:
    """Test 1: Coordinator runs both scrapers and returns combined results."""
    from src.discovery.coordinator import DiscoveryCoordinator

    coordinator = DiscoveryCoordinator()
    tender_states = coordinator.discover_new_tenders(known_ids=set())

    assert len(tender_states) >= 3, (
        f"Expected ≥3 tenders from combined sources, got {len(tender_states)}"
    )

    # Each should be a valid initial TenderState dict
    for ts in tender_states:
        assert ts.get("tender_id"), f"Missing tender_id: {ts}"
        assert ts.get("tender_title"), f"Missing tender_title: {ts}"
        assert ts.get("tender_raw_text"), f"Missing tender_raw_text: {ts}"
        assert ts.get("source_portal") in ("sam.gov", "email"), (
            f"Unexpected source: {ts.get('source_portal')}"
        )
        assert isinstance(ts.get("audit_log"), list)
        assert isinstance(ts.get("error_messages"), list)
        assert ts.get("escalation_count") == 0

    # Should have tenders from both sources
    sources = {ts["source_portal"] for ts in tender_states}
    assert "sam.gov" in sources, "Should have SAM.gov tenders"
    assert "email" in sources, "Should have email tenders"

    print(
        f"  ✅ Test 1 passed: Coordinator discovered {len(tender_states)} tenders "
        f"from {len(sources)} sources"
    )


def test_2_deduplication_across_sources() -> None:
    """Test 2: Known IDs are excluded from both sources."""
    from src.discovery.coordinator import DiscoveryCoordinator

    coordinator = DiscoveryCoordinator(min_relevance=0.0)

    # First run — get all
    all_tenders = coordinator.discover_new_tenders(known_ids=set())
    all_ids = {t["tender_id"] for t in all_tenders}

    # Mark some as known
    known = set(list(all_ids)[:2])

    # Second run — should exclude known
    new_tenders = coordinator.discover_new_tenders(known_ids=known)
    new_ids = {t["tender_id"] for t in new_tenders}

    assert known.isdisjoint(new_ids), "Known IDs should not appear in results"
    assert len(new_tenders) == len(all_tenders) - len(known), (
        f"Expected {len(all_tenders) - len(known)} new, got {len(new_tenders)}"
    )

    print(
        f"  ✅ Test 2 passed: Deduplication excluded {len(known)} known IDs, "
        f"{len(new_tenders)} new tenders returned"
    )


def test_3_discover_node_validation() -> None:
    """Test 3: Discover node validates required fields and catches errors."""
    from src.agent.nodes.discover import discover_node

    # Valid state
    valid_state = {
        "tender_id": "TEST-001",
        "tender_title": "SDS Management Platform",
        "tender_raw_text": "Full tender document text here...",
        "source_portal": "sam.gov",
        "source_url": "https://sam.gov/opp/test",
        "submission_deadline": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
        "audit_log": [],
        "error_messages": [],
    }

    result = discover_node(valid_state)
    assert result["status"] == "discovered", f"Valid state should be 'discovered', got {result['status']}"

    # Invalid state — missing everything
    empty_state = {
        "tender_id": "",
        "tender_title": "",
        "tender_raw_text": "",
        "source_url": "",
        "audit_log": [],
        "error_messages": [],
    }

    result = discover_node(empty_state)
    assert result["status"] == "error", f"Empty state should be 'error', got {result['status']}"
    assert len(result.get("error_messages", [])) >= 1, "Should have error messages"

    print("  ✅ Test 3 passed: Validation catches missing fields correctly")


def test_4_deadline_check() -> None:
    """Test 4: Expired deadlines are caught and tender is rejected."""
    from src.agent.nodes.discover import discover_node

    # Expired deadline
    expired_state = {
        "tender_id": "TEST-EXPIRED",
        "tender_title": "Old Tender",
        "tender_raw_text": "Some tender content",
        "source_portal": "sam.gov",
        "source_url": "",
        "submission_deadline": "2024-01-01T00:00:00Z",  # Past date
        "audit_log": [],
        "error_messages": [],
    }

    result = discover_node(expired_state)
    assert result["status"] == "rejected", (
        f"Expired tender should be 'rejected', got {result['status']}"
    )
    assert any("deadline" in e.lower() for e in result.get("error_messages", [])), (
        "Should mention deadline in error messages"
    )

    # Future deadline — should pass
    future_state = {
        "tender_id": "TEST-FUTURE",
        "tender_title": "Active Tender",
        "tender_raw_text": "Some tender content",
        "source_portal": "sam.gov",
        "source_url": "",
        "submission_deadline": (datetime.now(timezone.utc) + timedelta(days=60)).isoformat(),
        "audit_log": [],
        "error_messages": [],
    }

    result = discover_node(future_state)
    assert result["status"] == "discovered", f"Future deadline should pass, got {result['status']}"

    print("  ✅ Test 4 passed: Expired deadlines are rejected, future deadlines pass")


def test_5_graph_integration() -> None:
    """Test 5: Real discover node works in the full graph."""
    import src.agent.graph as graph_module
    from src.agent.nodes.discover import discover_node as real_discover
    from src.agent.nodes.evaluate import evaluate_node as real_evaluate
    from src.agent.nodes.retrieve_draft import retrieve_draft_node as real_draft
    from src.agent.nodes.gap_check import gap_check_node as real_gap

    # Swap all implemented placeholders
    originals = {
        "discover": graph_module.discover_node,
        "evaluate": graph_module.evaluate_node,
        "retrieve_draft": graph_module.retrieve_draft_node,
        "gap_check": graph_module.gap_check_node,
    }
    graph_module.discover_node = real_discover
    graph_module.evaluate_node = real_evaluate
    graph_module.retrieve_draft_node = real_draft
    graph_module.gap_check_node = real_gap

    try:
        from src.agent.graph import build_tender_graph
        from src.discovery.coordinator import DiscoveryCoordinator

        graph = build_tender_graph(checkpointer=None)
        coordinator = DiscoveryCoordinator()

        # Get a tender from the coordinator
        tenders = coordinator.discover_new_tenders()
        assert len(tenders) >= 1, "Coordinator should return at least 1 tender"

        # Run the first tender through the graph
        tender_state = tenders[0]
        result = graph.invoke(tender_state)

        # Should complete the full pipeline
        assert result["status"] == "submitted", (
            f"Expected 'submitted', got {result['status']}"
        )

        # Audit should show discover as first node
        audit_nodes = [e["node"] for e in result.get("audit_log", [])]
        assert audit_nodes[0] == "discover", (
            f"First audit entry should be 'discover', got {audit_nodes[0]}"
        )

        print(
            f"  ✅ Test 5 passed: Coordinator → Graph end-to-end "
            f"(tender='{tender_state['tender_title'][:50]}...', "
            f"status={result['status']})"
        )

    finally:
        for name, fn in originals.items():
            setattr(graph_module, f"{name}_node", fn)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  Step 14 Verification: Discover Node Integration")
    print("=" * 60 + "\n")

    tests = [
        ("Test 1: Coordinator discovers tenders", test_1_coordinator_discovers_tenders),
        ("Test 2: Deduplication across sources", test_2_deduplication_across_sources),
        ("Test 3: Discover node validation", test_3_discover_node_validation),
        ("Test 4: Deadline check", test_4_deadline_check),
        ("Test 5: Graph integration (end-to-end)", test_5_graph_integration),
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
        print("  🎉 All tests passed! Step 14 is complete.")
        print()
        print("  NEXT: Update src/agent/graph.py —")
        print("    Add: from src.agent.nodes.discover import discover_node")
        print("    Delete the placeholder discover_node function")
        print()
        print("  Then: git add -A && git commit -m 'Step 14: Discover node integration'")
        print()


if __name__ == "__main__":
    main()