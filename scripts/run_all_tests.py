#!/usr/bin/env python3
"""
Step 26: Master Test Runner — Runs ALL test suites and reports results.

Usage:
    cd /Users/sohamsarker/tender-agent
    source .venv/bin/activate
    python scripts/run_all_tests.py

This runs every test script from Steps 5-25 in order, captures results,
and prints a unified summary. Exit code 0 = all passed, 1 = failures.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
os.environ["DRY_RUN"] = "true"

# All test scripts in dependency order
TEST_SCRIPTS: list[tuple[str, str]] = [
    ("Step 5:  Document Ingestion",       "scripts/test_ingestion.py"),
    ("Step 6:  Embeddings",               "scripts/test_embeddings.py"),
    ("Step 7:  RAG Retriever",            "scripts/test_retriever.py"),
    ("Step 8:  State & Graph Skeleton",   "scripts/test_graph.py"),
    ("Step 9:  Evaluate Node",            "scripts/test_evaluate.py"),
    ("Step 10: Retrieve & Draft Node",    "scripts/test_retrieve_draft.py"),
    ("Step 11: Gap Check Node",           "scripts/test_gap_check.py"),
    ("Step 12: SAM.gov Scraper",          "scripts/test_sam_gov.py"),
    ("Step 13: Email Monitor",            "scripts/test_email_monitor.py"),
    ("Step 14: Discover Node",            "scripts/test_discover.py"),
    ("Steps 15-16: Slack & Escalate",     "scripts/test_slack.py"),
    ("Step 17: Timeout Handler",          "scripts/test_timeout.py"),
    ("Step 18: Template Engine",          "scripts/test_template_engine.py"),
    ("Step 19: Assemble Node",            "scripts/test_assemble.py"),
    ("Step 20: Portal Upload",            "scripts/test_portal_upload.py"),
    ("Step 21: Email & API Submission",   "scripts/test_email_api.py"),
    ("Step 22: Submit Node",              "scripts/test_submit.py"),
    ("Step 23: Audit Logger",             "scripts/test_audit_logger.py"),
    ("Step 24: LangSmith & Costs",        "scripts/test_langsmith.py"),
    ("Step 25: Dashboard",                "scripts/test_dashboard.py"),
]


def run_test(name: str, script: str) -> tuple[bool, float, str]:
    """Run a single test script and capture results.

    Returns:
        Tuple of (passed: bool, duration: float, output: str)
    """
    script_path = PROJECT_ROOT / script
    if not script_path.exists():
        return False, 0.0, f"Script not found: {script}"

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "DRY_RUN": "true"},
        timeout=60,
    )
    duration = time.time() - start

    passed = result.returncode == 0
    output = result.stdout + result.stderr

    return passed, duration, output


def main() -> None:
    print("\n" + "=" * 70)
    print("  AI TENDER AGENT — MASTER TEST SUITE")
    print("  Running all test scripts from Steps 5-25")
    print("=" * 70 + "\n")

    results: list[tuple[str, bool, float]] = []
    total_start = time.time()

    for name, script in TEST_SCRIPTS:
        sys.stdout.write(f"  Running {name:<35} ... ")
        sys.stdout.flush()

        try:
            passed, duration, output = run_test(name, script)
        except subprocess.TimeoutExpired:
            passed, duration, output = False, 60.0, "TIMEOUT after 60s"
        except Exception as exc:
            passed, duration, output = False, 0.0, str(exc)

        results.append((name, passed, duration))

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  ({duration:.1f}s)")

        if not passed:
            # Show failure details (last 10 lines)
            lines = output.strip().split("\n")
            for line in lines[-10:]:
                print(f"    | {line}")
            print()

    # Summary
    total_duration = time.time() - total_start
    passed_count = sum(1 for _, p, _ in results if p)
    failed_count = sum(1 for _, p, _ in results if not p)
    skipped_count = len(TEST_SCRIPTS) - len(results)

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total:    {len(results)} test suites")
    print(f"  Passed:   {passed_count} ✅")
    print(f"  Failed:   {failed_count} ❌")
    if skipped_count:
        print(f"  Skipped:  {skipped_count} ⏭️")
    print(f"  Duration: {total_duration:.1f}s")
    print("=" * 70)

    if failed_count == 0:
        print("\n  🎉 ALL TESTS PASSED! The agent is ready for deployment.\n")
    else:
        failed_names = [n for n, p, _ in results if not p]
        print(f"\n  ⚠️  {failed_count} suite(s) failed: {', '.join(failed_names)}")
        print("  Fix the failures above before proceeding.\n")

    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()