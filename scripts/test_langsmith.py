#!/usr/bin/env python3
"""Step 24 Verification — LangSmith Integration & Cost Tracking Tests"""
from __future__ import annotations
import os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["DRY_RUN"] = "true"

def test_1_configure_langsmith():
    """Test 1: LangSmith config sets correct environment variables."""
    from src.utils.langsmith_config import configure_langsmith
    # In dry-run, tracing should be disabled by default
    config = configure_langsmith()
    assert config["LANGCHAIN_TRACING_V2"] == "false", "Should be disabled in dry-run"
    assert config["LANGCHAIN_PROJECT"] == "tender-agent"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    # Explicit enable without key should fall back to disabled
    config2 = configure_langsmith(enabled=True)
    assert config2["LANGCHAIN_TRACING_V2"] == "false", "No API key → disabled"
    # Explicit enable with key
    config3 = configure_langsmith(api_key="ls-test-key", project="my-project", enabled=True)
    assert config3["LANGCHAIN_TRACING_V2"] == "true"
    assert config3["LANGCHAIN_PROJECT"] == "my-project"
    assert os.environ["LANGCHAIN_PROJECT"] == "my-project"
    print("  ✅ Test 1 passed: LangSmith config handles dry-run, missing key, and explicit enable")

def test_2_cost_recording():
    """Test 2: CostTracker records usage and estimates costs correctly."""
    from src.utils.langsmith_config import CostTracker
    tracker = CostTracker()
    # Haiku call: 1000 input tokens, 500 output
    # Price: $0.07/1M input + $0.26/1M output
    cost = tracker.record("T-001", "qwen3.5-flash", 1000, 500)
    expected = (1000 / 1e6) * 0.07 + (500 / 1e6) * 0.26
    assert abs(cost - expected) < 0.0001, f"Expected {expected}, got {cost}"
    # Sonnet call
    cost2 = tracker.record("T-001", "qwen3.5-plus", 2000, 1000)
    expected2 = (2000 / 1e6) * 0.26 + (1000 / 1e6) * 1.56
    assert abs(cost2 - expected2) < 0.0001
    # Tender total
    tender_cost = tracker.get_tender_cost("T-001")
    assert abs(tender_cost - (cost + cost2)) < 0.0001
    print(f"  ✅ Test 2 passed: Cost tracking works (Haiku=${cost:.6f}, Sonnet=${cost2:.6f}, total=${tender_cost:.6f})")

def test_3_record_from_audit():
    """Test 3: CostTracker extracts usage from audit log entries."""
    from src.utils.langsmith_config import CostTracker
    tracker = CostTracker()
    audit_log = [
        {"node": "evaluate", "action": "scored", "model_used": "qwen3.5-flash",
         "tokens_used": 1500, "timestamp": "2026-04-17T10:00:00Z"},
        {"node": "retrieve_draft", "action": "drafted", "model_used": "qwen3.5-plus",
         "tokens_used": 5000, "timestamp": "2026-04-17T10:00:05Z"},
        {"node": "gap_check", "action": "checked", "model_used": "qwen3.5-plus",
         "tokens_used": 2000, "timestamp": "2026-04-17T10:00:10Z"},
        {"node": "assemble", "action": "assembled", "model_used": None,
         "tokens_used": 0, "timestamp": "2026-04-17T10:00:15Z"},
    ]
    total = tracker.record_from_audit("T-002", audit_log)
    assert total > 0, "Should have recorded some cost"
    assert abs(tracker.get_tender_cost("T-002") - total) < 0.0001
    report = tracker.get_report()
    assert report.total_calls == 3, f"Expected 3 calls (skips 0-token), got {report.total_calls}"
    print(f"  ✅ Test 3 passed: Extracted cost from audit log (${total:.6f} from 3 LLM calls)")

def test_4_cost_report():
    """Test 4: Cost report aggregates by model and tender."""
    from src.utils.langsmith_config import CostTracker
    tracker = CostTracker()
    tracker.record("T-A", "qwen3.5-flash", 1000, 500)
    tracker.record("T-A", "qwen3.5-plus", 3000, 1500)
    tracker.record("T-B", "qwen3.5-flash", 800, 400)
    tracker.record("T-B", "qwen3-max", 500, 200)
    report = tracker.get_report()
    assert report.total_calls == 4
    assert report.total_tokens == 1000+500+3000+1500+800+400+500+200
    assert len(report.by_model) == 3, f"Expected 3 models, got {len(report.by_model)}"
    assert len(report.by_tender) == 2
    assert report.by_tender["T-A"] > 0
    assert report.by_tender["T-B"] > 0
    formatted = tracker.format_report()
    assert "Cost Tracking Report" in formatted
    assert "qwen3.5-flash" in formatted
    assert "qwen3.5-plus" in formatted
    assert "qwen3-max" in formatted
    assert "T-A" in formatted
    print(f"  ✅ Test 4 passed: Report has {report.total_calls} calls, {len(report.by_model)} models, ${report.total_cost:.4f}")

def test_5_budget_check():
    """Test 5: Budget monitoring detects overspend."""
    from src.utils.langsmith_config import CostTracker
    tracker = CostTracker()
    # Record minimal usage — well under budget
    tracker.record("T-001", "qwen3.5-flash", 1000, 500)
    status = tracker.check_budget(monthly_budget=300.0)
    assert not status["warning"], "Should not warn at minimal usage"
    assert not status["exceeded"]
    assert status["remaining"] > 299
    # Record huge usage to trigger warning
    tracker2 = CostTracker()
    for i in range(100):
        tracker2.record(f"T-{i}", "qwen3-max", 100_000, 50_000)
    # 100 calls × (100k/1M × $0.78 + 50k/1M × $3.90) ≈ $27.30 — use $25 budget to trigger warning
    status2 = tracker2.check_budget(monthly_budget=25.0)
    assert status2["warning"] or status2["exceeded"], "Should warn at heavy qwen3-max usage"
    print(f"  ✅ Test 5 passed: Budget check works (low=${status['spent']:.4f}, high=${status2['spent']:.2f})")

def main():
    print("\n" + "=" * 60)
    print("  Step 24 Verification: LangSmith Integration & Cost Tracking")
    print("=" * 60 + "\n")
    tests = [
        ("Test 1: LangSmith configuration", test_1_configure_langsmith),
        ("Test 2: Cost recording", test_2_cost_recording),
        ("Test 3: Record from audit log", test_3_record_from_audit),
        ("Test 4: Cost report", test_4_cost_report),
        ("Test 5: Budget check", test_5_budget_check),
    ]
    passed = failed = 0
    for name, fn in tests:
        try: fn(); passed += 1
        except AssertionError as e: print(f"  ❌ {name} FAILED: {e}"); failed += 1
        except Exception as e: print(f"  ❌ {name} ERROR: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{'=' * 60}\n  Results: {passed} passed, {failed} failed\n{'=' * 60}\n")
    if failed: sys.exit(1)
    else:
        print("  🎉 All tests passed! Step 24 is complete.")
        print("  Next: git add -A && git commit -m 'Step 24: LangSmith integration & cost tracking'\n")

if __name__ == "__main__":
    main()