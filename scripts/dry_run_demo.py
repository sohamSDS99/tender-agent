#!/usr/bin/env python3
"""
Step 27: End-to-End Dry Run Demo

Runs the complete AI Tender Agent pipeline with narrated output.
Discovers tenders, evaluates them, drafts responses, assembles documents,
and submits — all in dry-run mode with mock data.

This is the script you show to stakeholders to demonstrate the agent works.

Usage:
    cd /Users/sohamsarker/tender-agent
    source .venv/bin/activate
    python scripts/dry_run_demo.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["DRY_RUN"] = "true"

# Suppress structlog noise for clean demo output
import logging
logging.disable(logging.DEBUG)


def banner(text: str) -> None:
    width = 66
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}\n")


def section(text: str) -> None:
    print(f"\n  --- {text} ---\n")


def main() -> None:
    banner("AI TENDER AGENT — END-TO-END DRY RUN DEMO")
    print("  This demo runs the complete pipeline with mock data.")
    print("  No real API keys, Slack workspace, or portals are needed.\n")
    start_time = time.time()

    # =====================================================================
    # PHASE 1: DISCOVERY
    # =====================================================================
    section("PHASE 1: TENDER DISCOVERY")

    from src.discovery.coordinator import DiscoveryCoordinator
    coordinator = DiscoveryCoordinator(min_relevance=0.15)
    tenders = coordinator.discover_new_tenders()

    print(f"  Discovered {len(tenders)} tenders from SAM.gov + Email:\n")
    for i, t in enumerate(tenders, 1):
        source = t["source_portal"]
        title = t["tender_title"][:55]
        print(f"    {i}. [{source:<7}] {title}")

    # =====================================================================
    # PHASE 2: GRAPH PROCESSING
    # =====================================================================
    section("PHASE 2: GRAPH PROCESSING")

    # Swap all placeholder nodes with real implementations
    import src.agent.graph as gm
    from src.agent.nodes.discover import discover_node
    from src.agent.nodes.evaluate import evaluate_node
    from src.agent.nodes.retrieve_draft import retrieve_draft_node
    from src.agent.nodes.gap_check import gap_check_node
    from src.agent.nodes.slack_escalate import slack_escalate_node
    from src.agent.nodes.assemble import assemble_node
    from src.agent.nodes.submit import submit_node

    originals = {}
    for name, fn in [("discover", discover_node), ("evaluate", evaluate_node),
                      ("retrieve_draft", retrieve_draft_node), ("gap_check", gap_check_node),
                      ("slack_escalate", slack_escalate_node), ("assemble", assemble_node),
                      ("submit", submit_node)]:
        originals[name] = getattr(gm, f"{name}_node")
        setattr(gm, f"{name}_node", fn)

    from src.agent.graph import build_tender_graph
    from src.utils.dashboard import CostDashboard

    graph = build_tender_graph(checkpointer=None)
    dashboard = CostDashboard(monthly_budget=300.0)

    results = []
    max_to_process = min(len(tenders), 4)  # Process top 4 for demo

    for i, tender_state in enumerate(tenders[:max_to_process], 1):
        tid = tender_state["tender_id"]
        title = tender_state["tender_title"][:50]
        source = tender_state["source_portal"]

        print(f"  Processing tender {i}/{max_to_process}: {title}...")
        t0 = time.time()

        result = graph.invoke(tender_state)
        duration = time.time() - t0

        dashboard.record_run(result)
        results.append(result)

        status = result["status"]
        score = result.get("eval_score", "?")
        decision = result.get("eval_decision", "?")
        sections = len(result.get("drafted_sections", []))
        gaps = len(result.get("gaps", []))
        conf = result.get("submission_confirmation", "N/A")
        method = result.get("submission_method", "N/A")

        status_icon = "✅" if status == "submitted" else ("⛔" if status == "rejected" else "❌")

        print(f"    {status_icon} Status: {status}")
        print(f"       Score: {score}/100 ({decision})")
        if status == "submitted":
            print(f"       Sections: {sections} | Gaps: {gaps}")
            print(f"       Method: {method} | Confirmation: {conf}")
        print(f"       Duration: {duration:.1f}s")
        print()

    # Restore originals
    for name, fn in originals.items():
        setattr(gm, f"{name}_node", fn)

    # =====================================================================
    # PHASE 3: RESULTS SUMMARY
    # =====================================================================
    section("PHASE 3: RESULTS SUMMARY")

    submitted = sum(1 for r in results if r["status"] == "submitted")
    rejected = sum(1 for r in results if r["status"] == "rejected")
    failed = len(results) - submitted - rejected

    print(f"  Tenders processed: {len(results)}")
    print(f"  Submitted:         {submitted} ✅")
    print(f"  Rejected:          {rejected} ⛔")
    if failed:
        print(f"  Failed:            {failed} ❌")

    # =====================================================================
    # PHASE 4: SAMPLE DOCUMENT
    # =====================================================================
    section("PHASE 4: SAMPLE ASSEMBLED DOCUMENT")

    # Show the first submitted tender's document
    submitted_results = [r for r in results if r["status"] == "submitted"]
    if submitted_results:
        doc_path = submitted_results[0].get("assembled_document_path", "")
        if doc_path and Path(doc_path).exists():
            content = Path(doc_path).read_text(encoding="utf-8")
            # Show first 1500 chars
            preview = content[:1500]
            if len(content) > 1500:
                preview += "\n\n  [...document continues...]"
            print(f"  File: {Path(doc_path).name}")
            print(f"  Size: {len(content)} chars ({len(content.split())} words)\n")
            for line in preview.split("\n"):
                print(f"    {line}")
        else:
            print("  (No document file found)")
    else:
        print("  (No tenders were submitted)")

    # =====================================================================
    # PHASE 5: AUDIT TRAIL
    # =====================================================================
    section("PHASE 5: AUDIT TRAIL (first tender)")

    if results:
        audit = results[0].get("audit_log", [])
        tid = results[0].get("tender_id", "?")
        print(f"  Tender: {tid}")
        print(f"  Entries: {len(audit)}\n")
        for j, entry in enumerate(audit, 1):
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            node = entry.get("node", "?")
            action = entry.get("action", "?")
            tokens = entry.get("tokens_used", 0)
            tokens_str = f" [{tokens} tokens]" if tokens else ""
            print(f"    {j}. [{ts}] {node} → {action}{tokens_str}")

    # =====================================================================
    # PHASE 6: COST & DASHBOARD
    # =====================================================================
    section("PHASE 6: COST DASHBOARD")
    print(dashboard.format_summary())

    # =====================================================================
    # FINAL
    # =====================================================================
    total_time = time.time() - start_time
    banner(f"DEMO COMPLETE — {len(results)} tenders in {total_time:.1f}s")

    print("  The AI Tender Agent successfully:")
    print(f"    ✅ Discovered {len(tenders)} tenders from 2 sources")
    print(f"    ✅ Evaluated {len(results)} tenders on 8 dimensions")
    print(f"    ✅ Drafted {sum(len(r.get('drafted_sections', [])) for r in results)} sections")
    print(f"    ✅ Assembled {submitted} submission documents")
    print(f"    ✅ Submitted via portal upload and email")
    print(f"    ✅ Logged {sum(len(r.get('audit_log', [])) for r in results)} audit entries")
    print()
    print("  NEXT STEPS:")
    print("    1. Get real API keys (Anthropic, Voyage AI)")
    print("    2. Set DRY_RUN=false in .env")
    print("    3. Ingest company documents into the knowledge base")
    print("    4. Run: python scripts/dry_run_demo.py (with real LLMs)")
    print("    5. Deploy to AWS (Step 28)")
    print()


if __name__ == "__main__":
    main()