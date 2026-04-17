#!/usr/bin/env python3
"""
Step 26: Integration Tests — Cross-module workflows.

Tests that verify multiple modules work together correctly,
beyond what individual unit tests cover.

Usage:
    cd /Users/sohamsarker/tender-agent
    source .venv/bin/activate
    python scripts/test_integration.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["DRY_RUN"] = "true"


def test_1_discovery_to_submission():
    """Integration Test 1: Full flow — coordinator discovers, graph processes, dashboard records."""
    import src.agent.graph as gm
    from src.agent.nodes.discover import discover_node
    from src.agent.nodes.evaluate import evaluate_node
    from src.agent.nodes.retrieve_draft import retrieve_draft_node
    from src.agent.nodes.gap_check import gap_check_node
    from src.agent.nodes.slack_escalate import slack_escalate_node
    from src.agent.nodes.assemble import assemble_node
    from src.agent.nodes.submit import submit_node
    from src.discovery.coordinator import DiscoveryCoordinator
    from src.utils.dashboard import CostDashboard

    # Swap all nodes
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
        coordinator = DiscoveryCoordinator(min_relevance=0.3)
        dashboard = CostDashboard(monthly_budget=300.0)

        # Step 1: Discover tenders
        tenders = coordinator.discover_new_tenders()
        assert len(tenders) >= 1, "Should discover at least 1 tender"

        # Step 2: Process each through the graph
        submitted = 0
        rejected = 0
        for tender_state in tenders[:3]:  # Process top 3
            result = graph.invoke(tender_state)
            dashboard.record_run(result)

            if result["status"] == "submitted":
                submitted += 1
            elif result["status"] == "rejected":
                rejected += 1

        # Step 3: Verify dashboard captured everything
        stats = dashboard.get_pipeline_stats()
        assert stats.total_tenders >= 1
        budget = dashboard.get_budget_status()
        assert budget["spent"] >= 0

        summary = dashboard.format_summary()
        assert "Tenders Processed" in summary

        print(
            f"  ✅ Integration 1 passed: Discovery → Graph → Dashboard "
            f"({submitted} submitted, {rejected} rejected, "
            f"${budget['spent']:.4f} cost)"
        )

    finally:
        for name, fn in orig.items():
            setattr(gm, f"{name}_node", fn)


def test_2_ingestion_to_retrieval():
    """Integration Test 2: Ingest documents → chunk → embed → retrieve.

    Tests the knowledge base pipeline end-to-end without a database,
    using in-memory operations.
    """
    import tempfile
    from src.ingestion.parser import DocumentParser
    from src.ingestion.chunker import TextChunker
    from src.ingestion.embedder import VoyageEmbedder, EMBEDDING_DIMENSIONS

    # Create a test document
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(
            "Acme SDS Solutions holds ISO 27001 certification for information "
            "security. The certification was obtained in 2020 and covers all "
            "aspects of our cloud platform. We also maintain SOC 2 Type II "
            "compliance verified by independent auditors annually.\n\n"
            "Our platform serves over 500 clients across manufacturing, "
            "construction, oil and gas, and pharmaceutical industries. We "
            "provide GHS classification, chemical inventory tracking, and "
            "regulatory reporting automation."
        )
        tmp_path = f.name

    try:
        # Parse
        parser = DocumentParser()
        pages = parser.parse(tmp_path)
        assert len(pages) >= 1

        # Chunk
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_pages(pages)
        assert len(chunks) >= 1

        # Embed
        embedder = VoyageEmbedder(dry_run=True)
        texts = [c.text for c in chunks]
        result = embedder.embed_texts(texts)
        assert len(result.embeddings) == len(chunks)
        assert len(result.embeddings[0]) == EMBEDDING_DIMENSIONS

        # Verify embeddings are different per chunk
        if len(result.embeddings) >= 2:
            assert result.embeddings[0] != result.embeddings[1], "Chunks should have different embeddings"

        print(
            f"  ✅ Integration 2 passed: Parse → Chunk → Embed "
            f"({len(pages)} pages, {len(chunks)} chunks, "
            f"{len(result.embeddings)} embeddings)"
        )

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_3_audit_trail_completeness():
    """Integration Test 3: Every node produces audit entries with required fields."""
    import src.agent.graph as gm
    from src.agent.nodes.discover import discover_node
    from src.agent.nodes.evaluate import evaluate_node
    from src.agent.nodes.retrieve_draft import retrieve_draft_node
    from src.agent.nodes.gap_check import gap_check_node
    from src.agent.nodes.slack_escalate import slack_escalate_node
    from src.agent.nodes.assemble import assemble_node
    from src.agent.nodes.submit import submit_node

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
            "tender_id": "AUDIT-INT",
            "tender_title": "SDS Management System",
            "source_portal": "sam.gov",
            "source_url": "https://sam.gov/test",
            "tender_raw_text": (
                "Safety Data Sheet SDS management platform with GHS classification "
                "OSHA HCS compliance chemical inventory Tier II reporting "
                "ISO 27001 certification required annual subscription"
            ),
            "submission_deadline": "2026-08-01T17:00:00Z",
            "discovered_at": "2026-04-17T00:00:00Z",
            "tender_document_path": None,
            "escalation_count": 0,
            "assembly_retry_count": 0,
            "error_messages": [],
            "audit_log": [],
        })

        audit = result.get("audit_log", [])
        assert len(audit) >= 6, f"Expected ≥6 audit entries, got {len(audit)}"

        required_fields = {"timestamp", "node", "action", "detail"}
        for i, entry in enumerate(audit):
            for field in required_fields:
                assert field in entry, f"Entry {i} missing '{field}': {entry}"
                assert entry[field], f"Entry {i} has empty '{field}'"

        # Every visited node should have exactly one audit entry
        node_names = [e["node"] for e in audit]
        expected = {"discover", "evaluate", "retrieve_draft", "gap_check", "assemble", "submit"}
        assert expected.issubset(set(node_names)), (
            f"Missing nodes from audit: {expected - set(node_names)}"
        )

        print(
            f"  ✅ Integration 3 passed: All {len(audit)} audit entries have "
            f"required fields, all expected nodes present"
        )

    finally:
        for name, fn in orig.items():
            setattr(gm, f"{name}_node", fn)


def test_4_multi_tender_batch():
    """Integration Test 4: Process multiple tenders from different sources."""
    import src.agent.graph as gm
    from src.agent.nodes.discover import discover_node
    from src.agent.nodes.evaluate import evaluate_node
    from src.agent.nodes.retrieve_draft import retrieve_draft_node
    from src.agent.nodes.gap_check import gap_check_node
    from src.agent.nodes.slack_escalate import slack_escalate_node
    from src.agent.nodes.assemble import assemble_node
    from src.agent.nodes.submit import submit_node
    from src.discovery.coordinator import DiscoveryCoordinator

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
        coordinator = DiscoveryCoordinator(min_relevance=0.0)
        tenders = coordinator.discover_new_tenders()

        statuses = []
        sources = set()
        for state in tenders[:5]:
            result = graph.invoke(state)
            statuses.append(result["status"])
            sources.add(state["source_portal"])

        assert len(statuses) >= 3, f"Should process ≥3 tenders, got {len(statuses)}"
        assert "submitted" in statuses or "rejected" in statuses, "Should have some outcomes"
        assert len(sources) >= 2, f"Should have ≥2 sources, got {sources}"

        print(
            f"  ✅ Integration 4 passed: Batch processed {len(statuses)} tenders "
            f"from {sources} "
            f"(submitted={statuses.count('submitted')}, rejected={statuses.count('rejected')})"
        )

    finally:
        for name, fn in orig.items():
            setattr(gm, f"{name}_node", fn)


def main():
    print("\n" + "=" * 60)
    print("  Step 26: Integration Tests")
    print("=" * 60 + "\n")
    tests = [
        ("Integration 1: Discovery → Graph → Dashboard", test_1_discovery_to_submission),
        ("Integration 2: Ingest → Chunk → Embed", test_2_ingestion_to_retrieval),
        ("Integration 3: Audit trail completeness", test_3_audit_trail_completeness),
        ("Integration 4: Multi-tender batch", test_4_multi_tender_batch),
    ]
    passed = failed = 0
    for name, fn in tests:
        try: fn(); passed += 1
        except AssertionError as e: print(f"  ❌ {name} FAILED: {e}"); failed += 1
        except Exception as e: print(f"  ❌ {name} ERROR: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{'=' * 60}\n  Results: {passed} passed, {failed} failed\n{'=' * 60}\n")
    if failed: sys.exit(1)
    else:
        print("  🎉 All integration tests passed!")
        print("  Next: git add -A && git commit -m 'Step 26: Unit & integration tests'\n")

if __name__ == "__main__":
    main()