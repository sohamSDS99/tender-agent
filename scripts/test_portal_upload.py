#!/usr/bin/env python3
"""Step 20 Verification — Playwright Portal Automation Tests"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["DRY_RUN"] = "true"

def _make_mock_document(tmpdir: str) -> str:
    """Create a mock tender document for upload testing."""
    doc_path = Path(tmpdir) / "tender_response_TEST.md"
    doc_path.write_text(
        "# Test Tender Response\n\nThis is a mock document for testing.\n"
        "## 1.0 Company Overview\nAcme SDS Solutions is a leading provider.\n",
        encoding="utf-8",
    )
    return str(doc_path)

def test_1_dry_run_submit():
    """Test 1: Dry-run submission returns success with confirmation."""
    from src.submission.portal_upload import PortalUploader
    with tempfile.TemporaryDirectory() as tmpdir:
        doc = _make_mock_document(tmpdir)
        uploader = PortalUploader(screenshots_dir=tmpdir)
        result = uploader.submit(
            portal="sam_gov",
            opportunity_id="OPP-2026-001",
            document_path=doc,
            metadata={"company_name": "Acme SDS Solutions"},
        )
        assert result.success is True, f"Should succeed, got error: {result.error_message}"
        assert result.confirmation_id, "Should have confirmation ID"
        assert result.confirmation_id.startswith("CONF-")
        assert result.screenshot_path, "Should have screenshot path"
        assert Path(result.screenshot_path).exists(), "Screenshot file should exist"
        assert result.submitted_at, "Should have timestamp"
        assert "sam.gov" in result.portal_url
        print(f"  ✅ Test 1 passed: Dry-run submission succeeded (conf={result.confirmation_id})")

def test_2_missing_document():
    """Test 2: Submission with missing document returns failure."""
    from src.submission.portal_upload import PortalUploader
    uploader = PortalUploader()
    result = uploader.submit(
        portal="sam_gov",
        opportunity_id="OPP-FAIL",
        document_path="/nonexistent/file.pdf",
    )
    assert result.success is False, "Should fail with missing document"
    assert "not found" in result.error_message.lower()
    print("  ✅ Test 2 passed: Missing document correctly returns failure")

def test_3_portal_configs():
    """Test 3: Pre-configured portals have valid configurations."""
    from src.submission.portal_upload import PORTAL_CONFIGS
    assert "sam_gov" in PORTAL_CONFIGS
    assert "merx" in PORTAL_CONFIGS
    assert "generic" in PORTAL_CONFIGS
    sam = PORTAL_CONFIGS["sam_gov"]
    assert sam.login_url, "SAM.gov should have login URL"
    assert "{opportunity_id}" in sam.submission_url_template
    assert sam.file_upload_field, "Should have file upload selector"
    assert sam.submit_button, "Should have submit button selector"
    print(f"  ✅ Test 3 passed: {len(PORTAL_CONFIGS)} portal configs valid")

def test_4_screenshot_capture():
    """Test 4: Dry-run creates a screenshot file with submission details."""
    from src.submission.portal_upload import PortalUploader
    with tempfile.TemporaryDirectory() as tmpdir:
        doc = _make_mock_document(tmpdir)
        uploader = PortalUploader(screenshots_dir=tmpdir)
        result = uploader.submit(
            portal="merx", opportunity_id="MERX-2026-100",
            document_path=doc,
        )
        assert result.success
        screenshot_content = Path(result.screenshot_path).read_text()
        assert "MERX-2026-100" in screenshot_content
        assert "SUBMITTED SUCCESSFULLY" in screenshot_content
        assert result.confirmation_id in screenshot_content
        print(f"  ✅ Test 4 passed: Screenshot contains submission details ({Path(result.screenshot_path).name})")

def test_5_submission_result_structure():
    """Test 5: SubmissionResult has all expected fields."""
    from src.submission.portal_upload import PortalUploader
    with tempfile.TemporaryDirectory() as tmpdir:
        doc = _make_mock_document(tmpdir)
        uploader = PortalUploader(screenshots_dir=tmpdir)
        result = uploader.submit(
            portal="sam_gov", opportunity_id="OPP-STRUCT",
            document_path=doc,
            metadata={"company_name": "Acme", "uei": "ABC123"},
        )
        # Verify all fields are populated
        assert isinstance(result.success, bool)
        assert isinstance(result.confirmation_id, str) and len(result.confirmation_id) > 0
        assert isinstance(result.confirmation_text, str) and len(result.confirmation_text) > 0
        assert isinstance(result.screenshot_path, str) and len(result.screenshot_path) > 0
        assert isinstance(result.portal_url, str) and len(result.portal_url) > 0
        assert isinstance(result.submitted_at, str) and len(result.submitted_at) > 0
        assert isinstance(result.metadata, dict)
        assert result.metadata.get("company_name") == "Acme"
        assert result.metadata.get("dry_run") is True
        print("  ✅ Test 5 passed: SubmissionResult has all fields populated")

def main():
    print("\n" + "=" * 60)
    print("  Step 20 Verification: Playwright Portal Automation")
    print("=" * 60 + "\n")
    tests = [
        ("Test 1: Dry-run submit", test_1_dry_run_submit),
        ("Test 2: Missing document", test_2_missing_document),
        ("Test 3: Portal configs", test_3_portal_configs),
        ("Test 4: Screenshot capture", test_4_screenshot_capture),
        ("Test 5: Result structure", test_5_submission_result_structure),
    ]
    passed = failed = 0
    for name, fn in tests:
        try: fn(); passed += 1
        except AssertionError as e: print(f"  ❌ {name} FAILED: {e}"); failed += 1
        except Exception as e: print(f"  ❌ {name} ERROR: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{'=' * 60}\n  Results: {passed} passed, {failed} failed\n{'=' * 60}\n")
    if failed: sys.exit(1)
    else:
        print("  🎉 All tests passed! Step 20 is complete.")
        print("  Next: git add -A && git commit -m 'Step 20: Playwright portal automation'\n")

if __name__ == "__main__":
    main()