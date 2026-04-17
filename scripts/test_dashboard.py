#!/usr/bin/env python3
"""Step 25 Verification — Cost Tracking Dashboard Tests"""
from __future__ import annotations
import os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["DRY_RUN"] = "true"

def _make_completed_state(tender_id, title, score, decision, status, sections=3, gaps=0, escalations=0):
    """Build a mock completed graph state."""
    return {
        "tender_id": tender_id,
        "tender_title": title,
        "status": status,
        "eval_score": score,
        "eval_decision": decision,
        "drafted_sections": [{"section_id": f"{i}.0", "section_title": f"Sec {i}", "content": "x"*100,
                               "confidence": 0.85, "sources_used": [], "model_used": "qwen3.5-plus", "token_count": 200}
                              for i in range(1, sections+1)],
        "gaps": [{"section_id": "1.0", "description": "gap", "severity": "high", "suggested_question": "?"}] * gaps,
        "escalation_count": escalations,
        "submission_method": "portal_upload" if status == "submitted" else "",
        "submission_confirmation": "CONF-123" if status == "submitted" else "",
        "audit_log": [
            {"timestamp": "2026-04-17T10:00:00Z", "node": "discover", "action": "discovered",
             "detail": "Found", "model_used": None, "tokens_used": 0},
            {"timestamp": "2026-04-17T10:00:01Z", "node": "evaluate", "action": "scored",
             "detail": f"Score: {score}", "model_used": "qwen3.5-flash", "tokens_used": 1200},
            {"timestamp": "2026-04-17T10:00:05Z", "node": "retrieve_draft", "action": "drafted",
             "detail": f"Drafted {sections}", "model_used": "qwen3.5-plus", "tokens_used": 4500},
            {"timestamp": "2026-04-17T10:00:10Z", "node": "gap_check", "action": "checked",
             "detail": "Checked", "model_used": "qwen3.5-plus", "tokens_used": 800},
            {"timestamp": "2026-04-17T10:00:15Z", "node": "assemble", "action": "assembled",
             "detail": "Done", "model_used": None, "tokens_used": 0},
            {"timestamp": "2026-04-17T10:00:20Z", "node": "submit", "action": "submitted",
             "detail": "OK", "model_used": None, "tokens_used": 0},
        ],
    }

def test_1_record_and_summary():
    """Test 1: Dashboard records runs and produces a summary."""
    from src.utils.dashboard import CostDashboard
    dash = CostDashboard(monthly_budget=300.0)
    dash.record_run(_make_completed_state("T-001", "SDS Platform", 77, "go", "submitted"))
    dash.record_run(_make_completed_state("T-002", "ERP System", 35, "no_go", "rejected", sections=0))
    summary = dash.format_summary()
    assert "Tenders Processed:  2" in summary
    assert "Submitted:          1" in summary
    assert "Rejected:           1" in summary
    assert "$" in summary  # Should show costs
    print("  ✅ Test 1 passed: Summary shows 2 tenders (1 submitted, 1 rejected)")

def test_2_pipeline_report():
    """Test 2: Pipeline report shows per-tender breakdown."""
    from src.utils.dashboard import CostDashboard
    dash = CostDashboard()
    dash.record_run(_make_completed_state("SAM-001", "SDS Platform", 82, "go", "submitted", sections=5))
    dash.record_run(_make_completed_state("SAM-002", "Chemical Inv", 71, "go", "submitted", sections=4, gaps=1, escalations=1))
    dash.record_run(_make_completed_state("SAM-003", "Accounting", 42, "no_go", "rejected"))
    report = dash.format_pipeline_report()
    assert "SAM-001" in report
    assert "SAM-002" in report
    assert "SAM-003" in report
    assert "3 tenders" in report
    print("  ✅ Test 2 passed: Pipeline report shows 3 tenders with scores and costs")

def test_3_cost_aggregation():
    """Test 3: Costs aggregate correctly across multiple tenders."""
    from src.utils.dashboard import CostDashboard
    dash = CostDashboard()
    dash.record_run(_make_completed_state("C-001", "Tender A", 80, "go", "submitted"))
    dash.record_run(_make_completed_state("C-002", "Tender B", 75, "go", "submitted"))
    cost_report = dash.format_cost_report()
    assert "qwen3.5-flash" in cost_report
    assert "qwen3.5-plus" in cost_report
    assert "C-001" in cost_report
    assert "C-002" in cost_report
    budget = dash.get_budget_status()
    assert budget["spent"] > 0, "Should have some spend"
    assert not budget["exceeded"], "Should not exceed $300 with 2 tenders"
    print(f"  ✅ Test 3 passed: Cost aggregation works (${budget['spent']:.4f} across 2 tenders)")

def test_4_budget_warning():
    """Test 4: Budget warning triggers at 80% usage."""
    from src.utils.dashboard import CostDashboard
    dash = CostDashboard(monthly_budget=0.001)  # Tiny budget — Qwen is much cheaper than Claude
    dash.record_run(_make_completed_state("B-001", "Test", 80, "go", "submitted"))
    status = dash.get_budget_status()
    assert status["exceeded"], "Should exceed tiny budget"
    summary = dash.format_summary()
    assert "EXCEEDED" in summary or "WARNING" in summary
    print(f"  ✅ Test 4 passed: Budget exceeded detected (${status['spent']:.4f} > ${status['budget']})")

def test_5_pipeline_stats():
    """Test 5: Pipeline stats calculate averages and totals correctly."""
    from src.utils.dashboard import CostDashboard
    dash = CostDashboard()
    dash.record_run(_make_completed_state("S-001", "T1", 80, "go", "submitted", sections=5, gaps=0))
    dash.record_run(_make_completed_state("S-002", "T2", 60, "go", "submitted", sections=4, gaps=2, escalations=1))
    dash.record_run(_make_completed_state("S-003", "T3", 40, "no_go", "rejected", sections=0, gaps=0))
    stats = dash.get_pipeline_stats()
    assert stats.total_tenders == 3
    assert stats.tenders_by_status.get("submitted") == 2
    assert stats.tenders_by_status.get("rejected") == 1
    assert stats.total_sections_drafted == 5 + 4 + 0
    assert stats.total_gaps_found == 2
    assert stats.total_escalations == 1
    assert 55 < stats.avg_eval_score < 65, f"Avg should be ~60, got {stats.avg_eval_score}"
    print(f"  ✅ Test 5 passed: Stats correct (avg_score={stats.avg_eval_score:.0f}, sections={stats.total_sections_drafted}, gaps={stats.total_gaps_found})")

def test_6_recent_activity():
    """Test 6: Recent activity shows audit entries."""
    from src.utils.dashboard import CostDashboard
    dash = CostDashboard()
    dash.record_run(_make_completed_state("A-001", "Test", 75, "go", "submitted"))
    activity = dash.format_recent_activity()
    assert "A-001" in activity
    assert "discover" in activity
    assert "evaluate" in activity
    print("  ✅ Test 6 passed: Recent activity shows audit entries with timestamps")

def main():
    print("\n" + "=" * 60)
    print("  Step 25 Verification: Cost Tracking Dashboard")
    print("=" * 60 + "\n")
    tests = [
        ("Test 1: Record runs and summary", test_1_record_and_summary),
        ("Test 2: Pipeline report", test_2_pipeline_report),
        ("Test 3: Cost aggregation", test_3_cost_aggregation),
        ("Test 4: Budget warning", test_4_budget_warning),
        ("Test 5: Pipeline stats", test_5_pipeline_stats),
        ("Test 6: Recent activity", test_6_recent_activity),
    ]
    passed = failed = 0
    for name, fn in tests:
        try: fn(); passed += 1
        except AssertionError as e: print(f"  ❌ {name} FAILED: {e}"); failed += 1
        except Exception as e: print(f"  ❌ {name} ERROR: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{'=' * 60}\n  Results: {passed} passed, {failed} failed\n{'=' * 60}\n")
    if failed: sys.exit(1)
    else:
        print("  🎉 All tests passed! Step 25 is complete.")
        print("  Next: git add -A && git commit -m 'Step 25: Cost tracking dashboard'\n")

if __name__ == "__main__":
    main()