#!/usr/bin/env python3
"""
Step 13 Verification — Email IMAP Monitor Tests

Runs 5 tests to verify email parsing, relevance scoring,
deadline extraction, filtering, and deduplication.

Usage:
    cd /Users/sohamsarker/tender-agent
    source .venv/bin/activate
    python scripts/test_email_monitor.py

Expected output: All 5 tests pass.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["DRY_RUN"] = "true"


def test_1_fetch_mock_emails() -> None:
    """Test 1: Dry-run fetches mock email leads."""
    from src.discovery.email_monitor import EmailMonitor

    monitor = EmailMonitor()
    leads = monitor.check_inbox()

    assert len(leads) >= 1, f"Expected ≥1 leads, got {len(leads)}"

    for lead in leads:
        assert lead.lead_id.startswith("EMAIL-"), f"Email leads should have EMAIL- prefix: {lead.lead_id}"
        assert lead.title, "title should not be empty"
        assert lead.source_portal == "email"
        assert lead.relevance_score >= 0.0

    print(f"  ✅ Test 1 passed: Fetched {len(leads)} leads from mock emails")


def test_2_deadline_extraction() -> None:
    """Test 2: Deadlines are extracted from email body text."""
    from src.discovery.email_monitor import _extract_deadline

    # Standard format
    assert _extract_deadline("Deadline: June 30, 2026") == "June 30, 2026"

    # ISO format
    assert _extract_deadline("Closing Date: 2026-07-15") == "2026-07-15"

    # Slash format
    assert _extract_deadline("Due: 06/15/2026") == "06/15/2026"

    # Submit by
    assert _extract_deadline("Submit by: July 20, 2026") == "July 20, 2026"

    # No deadline
    assert _extract_deadline("No deadline mentioned here") == ""

    print("  ✅ Test 2 passed: Deadline extraction handles all common formats")


def test_3_relevance_filtering() -> None:
    """Test 3: Irrelevant emails (office supplies) are scored low."""
    from src.discovery.email_monitor import EmailMonitor

    monitor = EmailMonitor(min_relevance=0.0)  # Keep everything
    all_leads = monitor.check_inbox()

    # Find the office supply email
    office_leads = [l for l in all_leads if "office supply" in l.title.lower()]
    sds_leads = [l for l in all_leads if "sds" in l.title.lower() or "safety data" in l.title.lower()]

    assert len(office_leads) >= 1, "Should have the office supply mock email"
    assert len(sds_leads) >= 1, "Should have SDS-related mock emails"

    # Office supply should score lower than SDS
    office_score = office_leads[0].relevance_score
    sds_score = sds_leads[0].relevance_score

    assert sds_score > office_score, (
        f"SDS lead ({sds_score}) should score higher than office supplies ({office_score})"
    )

    # With a reasonable threshold, office supplies should be filtered out
    monitor_strict = EmailMonitor(min_relevance=0.15)
    filtered_leads = monitor_strict.check_inbox()
    filtered_titles = [l.title.lower() for l in filtered_leads]
    assert not any("office supply" in t for t in filtered_titles), (
        "Office supply email should be filtered out at 0.15 threshold"
    )

    print(
        f"  ✅ Test 3 passed: SDS email ({sds_score:.2f}) outscores "
        f"office supplies ({office_score:.2f}), filtering works"
    )


def test_4_deduplication() -> None:
    """Test 4: Already-known email leads are excluded."""
    from src.discovery.email_monitor import EmailMonitor

    monitor = EmailMonitor(min_relevance=0.0)
    all_leads = monitor.check_inbox()

    # Mark first lead as known
    known_ids = {all_leads[0].lead_id}
    new_leads = monitor.check_and_deduplicate(known_ids=known_ids)

    assert len(new_leads) == len(all_leads) - 1, (
        f"Expected {len(all_leads) - 1} new leads, got {len(new_leads)}"
    )
    assert all_leads[0].lead_id not in {l.lead_id for l in new_leads}

    print(
        f"  ✅ Test 4 passed: Deduplication removed 1 known lead, "
        f"{len(new_leads)} new remain"
    )


def test_5_parsed_email_structure() -> None:
    """Test 5: ParsedEmail objects have all expected fields."""
    from src.discovery.email_monitor import EmailMonitor, _email_to_tender_lead, ParsedEmail

    monitor = EmailMonitor()

    # Create a ParsedEmail and convert to TenderLead
    parsed = ParsedEmail(
        message_id="test-123@example.com",
        subject="RFP: Chemical Inventory Management System",
        sender="buyer@agency.gov",
        date="2026-04-16T10:00:00Z",
        body_text=(
            "Seeking a chemical inventory and SDS management system "
            "with GHS classification capabilities. OSHA compliance required. "
            "Deadline: August 1, 2026"
        ),
        extracted_deadline="August 1, 2026",
        attachment_names=["RFP_Document.pdf"],
    )

    lead = _email_to_tender_lead(parsed)

    assert lead.lead_id.startswith("EMAIL-")
    assert lead.title == parsed.subject
    assert lead.source_portal == "email"
    assert lead.submission_deadline == "August 1, 2026"
    assert lead.relevance_score > 0, "SDS/GHS/OSHA email should have positive relevance"

    # Verify to_dict works
    d = lead.to_dict()
    assert "lead_id" in d
    assert "relevance_score" in d

    print(
        f"  ✅ Test 5 passed: ParsedEmail → TenderLead conversion works "
        f"(score={lead.relevance_score:.2f}, deadline={lead.submission_deadline})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  Step 13 Verification: Email IMAP Monitor")
    print("=" * 60 + "\n")

    tests = [
        ("Test 1: Fetch mock email leads", test_1_fetch_mock_emails),
        ("Test 2: Deadline extraction", test_2_deadline_extraction),
        ("Test 3: Relevance filtering", test_3_relevance_filtering),
        ("Test 4: Deduplication", test_4_deduplication),
        ("Test 5: ParsedEmail structure", test_5_parsed_email_structure),
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
        print("  🎉 All tests passed! Step 13 is complete.")
        print("  Next: git add -A && git commit -m 'Step 13: Email IMAP monitor'")
        print()


if __name__ == "__main__":
    main()