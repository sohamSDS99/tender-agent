#!/usr/bin/env python3
"""
Step 17 Verification — Timeout & Deadline Escalation Tests

Runs 5 tests covering timeout detection, deadline alerts,
auto-proceed logic, and manager escalation.

Usage:
    cd /Users/sohamsarker/tender-agent
    source .venv/bin/activate
    python scripts/test_timeout.py

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


def test_1_no_timeout_normal_waiting() -> None:
    """Test 1: Recent escalation with distant deadline = normal waiting."""
    from src.slack_integration.timeout_handler import TimeoutHandler

    handler = TimeoutHandler(timeout_hours=48)

    # Escalated 2 hours ago, deadline in 30 days
    escalated_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    deadline = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()

    status = handler.check_escalation_status(
        tender_id="TEST-001",
        escalated_at=escalated_at,
        deadline=deadline,
    )

    assert not status.is_timed_out, "2h wait should not be timed out"
    assert not status.is_deadline_critical, "30-day deadline should not be critical"
    assert not status.should_auto_proceed, "Should not auto-proceed"
    assert status.action_taken == "waiting"
    assert 1.5 < status.hours_waiting < 3.0, f"Expected ~2h waiting, got {status.hours_waiting}"

    print(
        f"  ✅ Test 1 passed: Normal waiting state "
        f"(waiting={status.hours_waiting:.1f}h, action={status.action_taken})"
    )


def test_2_timeout_reached() -> None:
    """Test 2: 50h wait with distant deadline = timeout, escalate to manager."""
    from src.slack_integration.timeout_handler import TimeoutHandler

    handler = TimeoutHandler(timeout_hours=48)

    escalated_at = (datetime.now(timezone.utc) - timedelta(hours=50)).isoformat()
    deadline = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()

    status = handler.check_escalation_status(
        tender_id="TEST-002",
        escalated_at=escalated_at,
        deadline=deadline,
    )

    assert status.is_timed_out, "50h wait should be timed out"
    assert not status.is_deadline_critical, "10-day deadline should not be critical"
    assert not status.should_auto_proceed, "Should escalate, not auto-proceed"
    assert status.action_taken == "timeout_escalate_to_manager"

    print(
        f"  ✅ Test 2 passed: Timeout reached after {status.hours_waiting:.1f}h "
        f"(action={status.action_taken})"
    )


def test_3_deadline_critical_no_timeout() -> None:
    """Test 3: Recent escalation but deadline within 24h = deadline warning."""
    from src.slack_integration.timeout_handler import TimeoutHandler

    handler = TimeoutHandler(timeout_hours=48)

    escalated_at = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    deadline = (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat()

    status = handler.check_escalation_status(
        tender_id="TEST-003",
        escalated_at=escalated_at,
        deadline=deadline,
    )

    assert not status.is_timed_out, "5h wait should not be timed out"
    assert status.is_deadline_critical, "12h deadline should be critical"
    assert not status.should_auto_proceed, "No timeout yet, don't auto-proceed"
    assert status.action_taken == "deadline_warning"

    print(
        f"  ✅ Test 3 passed: Deadline critical, no timeout "
        f"(hours_to_deadline={status.hours_to_deadline:.1f}h)"
    )


def test_4_timeout_and_deadline_auto_proceed() -> None:
    """Test 4: Timeout + deadline critical = auto-proceed."""
    from src.slack_integration.timeout_handler import TimeoutHandler

    handler = TimeoutHandler(timeout_hours=48)

    escalated_at = (datetime.now(timezone.utc) - timedelta(hours=50)).isoformat()
    deadline = (datetime.now(timezone.utc) + timedelta(hours=8)).isoformat()

    status = handler.check_escalation_status(
        tender_id="TEST-004",
        escalated_at=escalated_at,
        deadline=deadline,
    )

    assert status.is_timed_out, "50h should be timed out"
    assert status.is_deadline_critical, "8h deadline should be critical"
    assert status.should_auto_proceed, "Timeout + critical deadline = auto-proceed"
    assert status.action_taken == "auto_proceed_timeout_and_deadline"

    print(
        f"  ✅ Test 4 passed: Auto-proceed triggered "
        f"(waiting={status.hours_waiting:.1f}h, deadline={status.hours_to_deadline:.1f}h)"
    )


def test_5_deadline_alerts() -> None:
    """Test 5: Deadline alerts fire at correct thresholds."""
    from src.slack_integration.timeout_handler import TimeoutHandler

    handler = TimeoutHandler()

    # Deadline in 3 hours — should trigger 72h, 24h, and 4h alerts
    deadline = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()

    # No alerts sent yet
    new_alerts = handler.check_deadline_alerts(
        tender_id="TEST-005",
        tender_title="Urgent Tender",
        deadline=deadline,
        alerts_already_sent=set(),
    )

    assert 72 in new_alerts, "Should send 72h alert (we're within 72h)"
    assert 24 in new_alerts, "Should send 24h alert (we're within 24h)"
    assert 4 in new_alerts, "Should send 4h alert (we're within 4h)"

    # Now simulate running again with 72h already sent
    new_alerts_2 = handler.check_deadline_alerts(
        tender_id="TEST-005",
        tender_title="Urgent Tender",
        deadline=deadline,
        alerts_already_sent={72, 24, 4},  # All already sent
    )

    assert len(new_alerts_2) == 0, (
        f"Should not re-send alerts, got {new_alerts_2}"
    )

    # Deadline far away — no alerts
    far_deadline = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    far_alerts = handler.check_deadline_alerts(
        tender_id="TEST-006",
        tender_title="Distant Tender",
        deadline=far_deadline,
        alerts_already_sent=set(),
    )

    assert len(far_alerts) == 0, "30-day deadline should trigger no alerts"

    print(
        f"  ✅ Test 5 passed: Alerts fired correctly "
        f"(3h deadline: {new_alerts}, re-run: {new_alerts_2}, 30d: {far_alerts})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  Step 17 Verification: Timeout & Deadline Escalation")
    print("=" * 60 + "\n")

    tests = [
        ("Test 1: Normal waiting", test_1_no_timeout_normal_waiting),
        ("Test 2: Timeout → manager escalation", test_2_timeout_reached),
        ("Test 3: Deadline critical, no timeout", test_3_deadline_critical_no_timeout),
        ("Test 4: Timeout + deadline → auto-proceed", test_4_timeout_and_deadline_auto_proceed),
        ("Test 5: Deadline alert thresholds", test_5_deadline_alerts),
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
        print("  🎉 All tests passed! Step 17 is complete.")
        print("  Next: git add -A && git commit -m 'Step 17: Timeout & deadline escalation'")
        print()


if __name__ == "__main__":
    main()