#!/usr/bin/env python3
"""
Step 12 Verification — SAM.gov RSS & Portal Scraper Tests

Runs 5 tests to verify opportunity fetching, relevance scoring,
keyword filtering, and deduplication.

Usage:
    cd /Users/sohamsarker/tender-agent
    source .venv/bin/activate
    python scripts/test_sam_gov.py

Expected output: All 5 tests pass.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["DRY_RUN"] = "true"


def test_1_fetch_mock_opportunities() -> None:
    """Test 1: Dry-run fetches mock tender leads."""
    from src.discovery.sam_gov import SamGovScraper

    scraper = SamGovScraper()
    leads = scraper.fetch_opportunities()

    assert len(leads) >= 1, f"Expected ≥1 leads, got {len(leads)}"

    for lead in leads:
        assert lead.lead_id, "lead_id should not be empty"
        assert lead.title, "title should not be empty"
        assert lead.description, "description should not be empty"
        assert lead.source_portal == "sam.gov"
        assert lead.relevance_score >= 0.1, (
            f"Lead '{lead.title}' has score {lead.relevance_score} below min_relevance"
        )

    print(f"  ✅ Test 1 passed: Fetched {len(leads)} relevant mock leads")


def test_2_relevance_scoring() -> None:
    """Test 2: Relevance scoring ranks SDS-specific tenders highest."""
    from src.discovery.sam_gov import score_relevance

    # Highly relevant — SDS + chemical safety + OSHA
    score_high, kw_high = score_relevance(
        "SDS Management Platform",
        "Safety data sheet management with GHS classification and OSHA compliance",
    )

    # Somewhat relevant — environmental but not SDS-specific
    score_mid, kw_mid = score_relevance(
        "Environmental Monitoring Platform",
        "Environmental compliance and regulatory reporting software",
    )

    # Irrelevant — ERP system
    score_low, kw_low = score_relevance(
        "ERP System Modernization",
        "Financial management and human resources modules",
    )

    assert score_high > score_mid > score_low, (
        f"Expected high({score_high}) > mid({score_mid}) > low({score_low})"
    )
    assert score_high >= 0.3, f"SDS tender should score ≥0.3, got {score_high}"
    assert len(kw_high) >= 2, f"SDS tender should match ≥2 keywords, got {kw_high}"

    print(
        f"  ✅ Test 2 passed: Relevance scoring works — "
        f"SDS={score_high}, ENV={score_mid}, ERP={score_low}"
    )


def test_3_filtering_removes_irrelevant() -> None:
    """Test 3: Irrelevant opportunities are filtered out."""
    from src.discovery.sam_gov import SamGovScraper

    # Use a high min_relevance to aggressively filter
    scraper = SamGovScraper(min_relevance=0.3)
    leads = scraper.fetch_opportunities()

    # The ERP tender (SAM-2026-IT-004) should be filtered out
    erp_leads = [l for l in leads if "ERP" in l.title]
    assert len(erp_leads) == 0, (
        f"ERP tender should be filtered out at min_relevance=0.3, "
        f"but found: {[l.title for l in erp_leads]}"
    )

    # SDS-related tenders should remain
    assert len(leads) >= 1, "Should have at least 1 relevant lead after filtering"

    print(
        f"  ✅ Test 3 passed: Filtering at 0.3 threshold keeps {len(leads)} "
        f"relevant leads, removes irrelevant ones"
    )


def test_4_deduplication() -> None:
    """Test 4: Already-known tenders are excluded."""
    from src.discovery.sam_gov import SamGovScraper

    scraper = SamGovScraper(min_relevance=0.0)  # Keep everything for this test

    # First fetch — get all leads
    all_leads = scraper.fetch_opportunities()
    all_ids = {l.lead_id for l in all_leads}

    # Mark some as known
    known_ids = {all_leads[0].lead_id} if all_leads else set()

    # Fetch with deduplication
    new_leads = scraper.fetch_and_deduplicate(known_ids=known_ids)

    assert len(new_leads) == len(all_leads) - len(known_ids), (
        f"Expected {len(all_leads) - len(known_ids)} new leads, got {len(new_leads)}"
    )

    # Verify known IDs are excluded
    new_ids = {l.lead_id for l in new_leads}
    assert known_ids.isdisjoint(new_ids), "Known IDs should not appear in new leads"

    print(
        f"  ✅ Test 4 passed: Deduplication removed {len(known_ids)} known lead(s), "
        f"returning {len(new_leads)} new"
    )


def test_5_lead_to_dict() -> None:
    """Test 5: TenderLead.to_dict() produces a clean serializable dict."""
    from src.discovery.sam_gov import SamGovScraper

    scraper = SamGovScraper()
    leads = scraper.fetch_opportunities()

    assert len(leads) >= 1
    lead = leads[0]
    d = lead.to_dict()

    required_keys = {
        "lead_id", "title", "description", "agency", "source_portal",
        "source_url", "naics_code", "submission_deadline", "posted_date",
        "relevance_score", "relevance_keywords",
    }

    missing = required_keys - set(d.keys())
    assert not missing, f"Missing keys in to_dict(): {missing}"

    # Verify it's JSON-serializable
    import json
    try:
        json.dumps(d)
    except (TypeError, ValueError) as exc:
        assert False, f"to_dict() should be JSON-serializable: {exc}"

    print(
        f"  ✅ Test 5 passed: TenderLead.to_dict() has all {len(required_keys)} fields, "
        f"JSON-serializable"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  Step 12 Verification: SAM.gov RSS & Portal Scraper")
    print("=" * 60 + "\n")

    tests = [
        ("Test 1: Fetch mock opportunities", test_1_fetch_mock_opportunities),
        ("Test 2: Relevance scoring", test_2_relevance_scoring),
        ("Test 3: Filtering removes irrelevant", test_3_filtering_removes_irrelevant),
        ("Test 4: Deduplication", test_4_deduplication),
        ("Test 5: TenderLead serialization", test_5_lead_to_dict),
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
        print("  🎉 All tests passed! Step 12 is complete.")
        print("  Next: git add -A && git commit -m 'Step 12: SAM.gov portal scraper'")
        print()


if __name__ == "__main__":
    main()