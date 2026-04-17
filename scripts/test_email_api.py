#!/usr/bin/env python3
"""Step 21 Verification — Email Submission & API Dispatch Tests"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["DRY_RUN"] = "true"

def _make_doc(tmpdir: str) -> str:
    p = Path(tmpdir) / "tender_response.md"
    p.write_text("# Test Response\n\nCompany overview content here.\n", encoding="utf-8")
    return str(p)

def test_1_email_dry_run():
    """Test 1: Email submitter sends dry-run email with confirmation."""
    from src.submission.email_api import EmailSubmitter
    with tempfile.TemporaryDirectory() as tmpdir:
        doc = _make_doc(tmpdir)
        sub = EmailSubmitter()
        result = sub.submit(
            to_address="procurement@agency.gov",
            subject="Proposal: SDS Platform — SAM-2026-001",
            body="Please find our proposal attached.",
            document_path=doc,
            tender_id="SAM-2026-001",
        )
        assert result.success is True, f"Should succeed: {result.error_message}"
        assert result.confirmation_id.startswith("EMAIL-")
        assert "procurement@agency.gov" in result.confirmation_text
        assert result.metadata.get("method") == "email"
        assert result.metadata.get("dry_run") is True
        print(f"  ✅ Test 1 passed: Email dry-run sent (conf={result.confirmation_id})")

def test_2_email_missing_document():
    """Test 2: Email submission with missing document fails gracefully."""
    from src.submission.email_api import EmailSubmitter
    sub = EmailSubmitter()
    result = sub.submit(
        to_address="test@test.com", subject="Test", body="Test",
        document_path="/nonexistent/file.pdf",
    )
    assert result.success is False
    assert "not found" in result.error_message.lower()
    print("  ✅ Test 2 passed: Missing document returns failure")

def test_3_api_dry_run():
    """Test 3: API dispatcher sends dry-run submission."""
    from src.submission.email_api import APIDispatcher
    with tempfile.TemporaryDirectory() as tmpdir:
        doc = _make_doc(tmpdir)
        disp = APIDispatcher()
        result = disp.submit(
            endpoint_url="https://portal.example.com/api/v1/submissions",
            document_path=doc,
            tender_id="API-001",
            metadata={"company_name": "Acme SDS", "opportunity_id": "OPP-100"},
        )
        assert result.success is True, f"Should succeed: {result.error_message}"
        assert result.confirmation_id.startswith("API-")
        assert result.metadata.get("method") == "api"
        assert result.metadata.get("endpoint") == "https://portal.example.com/api/v1/submissions"
        print(f"  ✅ Test 3 passed: API dry-run submitted (conf={result.confirmation_id})")

def test_4_api_missing_document():
    """Test 4: API submission with missing document fails gracefully."""
    from src.submission.email_api import APIDispatcher
    disp = APIDispatcher()
    result = disp.submit(
        endpoint_url="https://example.com/api", document_path="/no/file.pdf",
    )
    assert result.success is False
    assert "not found" in result.error_message.lower()
    print("  ✅ Test 4 passed: API missing document returns failure")

def test_5_all_methods_return_same_structure():
    """Test 5: All three submission methods return consistent SubmissionResult."""
    from src.submission.portal_upload import PortalUploader
    from src.submission.email_api import EmailSubmitter, APIDispatcher
    with tempfile.TemporaryDirectory() as tmpdir:
        doc = _make_doc(tmpdir)
        results = []

        # Portal
        portal = PortalUploader(screenshots_dir=tmpdir)
        results.append(("portal", portal.submit("sam_gov", "OPP-1", doc)))

        # Email
        email = EmailSubmitter()
        results.append(("email", email.submit("test@gov.com", "Subj", "Body", doc)))

        # API
        api = APIDispatcher()
        results.append(("api", api.submit("https://api.example.com/submit", doc)))

        for method, result in results:
            assert isinstance(result.success, bool), f"{method}: success should be bool"
            assert isinstance(result.confirmation_id, str), f"{method}: conf_id should be str"
            assert isinstance(result.submitted_at, str), f"{method}: submitted_at should be str"
            assert isinstance(result.metadata, dict), f"{method}: metadata should be dict"
            assert result.success is True, f"{method}: should succeed in dry-run"

        print(f"  ✅ Test 5 passed: All 3 methods (portal, email, API) return consistent SubmissionResult")

def main():
    print("\n" + "=" * 60)
    print("  Step 21 Verification: Email Submission & API Dispatch")
    print("=" * 60 + "\n")
    tests = [
        ("Test 1: Email dry-run", test_1_email_dry_run),
        ("Test 2: Email missing document", test_2_email_missing_document),
        ("Test 3: API dry-run", test_3_api_dry_run),
        ("Test 4: API missing document", test_4_api_missing_document),
        ("Test 5: Consistent result structure", test_5_all_methods_return_same_structure),
    ]
    passed = failed = 0
    for name, fn in tests:
        try: fn(); passed += 1
        except AssertionError as e: print(f"  ❌ {name} FAILED: {e}"); failed += 1
        except Exception as e: print(f"  ❌ {name} ERROR: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{'=' * 60}\n  Results: {passed} passed, {failed} failed\n{'=' * 60}\n")
    if failed: sys.exit(1)
    else:
        print("  🎉 All tests passed! Step 21 is complete.")
        print("  Next: git add -A && git commit -m 'Step 21: Email submission & API dispatch'\n")

if __name__ == "__main__":
    main()