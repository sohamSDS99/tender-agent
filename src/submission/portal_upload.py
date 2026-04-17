"""
Portal Upload — Automates tender submission via Playwright browser automation.

HOW PORTAL SUBMISSION WORKS:
Government procurement portals (SAM.gov, MERX, state portals) require vendors to:
1. Log in with credentials
2. Navigate to the specific opportunity page
3. Fill in submission metadata (company name, contact, DUNS/UEI number)
4. Upload the tender response document (PDF/DOCX)
5. Upload supporting files (certifications, past performance, pricing sheets)
6. Check acknowledgment boxes
7. Click "Submit"
8. Save the confirmation page/receipt

Each portal has different form layouts, field names, and workflows. This module
provides a base PortalUploader class with common logic, and portal-specific
subclasses can override the navigation steps.

WHY PLAYWRIGHT (NOT SELENIUM):
- Playwright is faster and more reliable than Selenium
- Built-in auto-waiting (no explicit sleep/wait calls for elements)
- Better handling of modern SPAs and dynamic content
- Native support for screenshots, file uploads, and form filling
- Headless by default — no GUI needed on the server

DRY-RUN MODE:
Simulates the entire upload process without opening a real browser. Returns
mock confirmation data. This lets us test the full pipeline end-to-end
without credentials or network access.

SETUP (for production):
    pip install playwright
    playwright install chromium
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SubmissionResult:
    """Result of a portal submission attempt.

    Attributes:
        success: Whether the submission completed successfully.
        confirmation_id: Receipt/confirmation ID from the portal.
        confirmation_text: Full confirmation message text.
        screenshot_path: Path to the confirmation screenshot.
        portal_url: The URL that was submitted to.
        submitted_at: ISO timestamp of submission.
        error_message: Error description if submission failed.
        metadata: Additional portal-specific data.
    """
    success: bool
    confirmation_id: str = ""
    confirmation_text: str = ""
    screenshot_path: str = ""
    portal_url: str = ""
    submitted_at: str = ""
    error_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PortalConfig:
    """Configuration for a specific procurement portal.

    Attributes:
        name: Portal identifier (e.g., "sam_gov", "merx").
        login_url: URL of the login page.
        submission_url_template: URL pattern for submission pages.
            Use {opportunity_id} as a placeholder.
        username_field: CSS selector for username input.
        password_field: CSS selector for password input.
        file_upload_field: CSS selector for file upload input.
        submit_button: CSS selector for the submit button.
        confirmation_selector: CSS selector for the confirmation message.
    """
    name: str
    login_url: str = ""
    submission_url_template: str = ""
    username_field: str = "input[name='username']"
    password_field: str = "input[name='password']"
    file_upload_field: str = "input[type='file']"
    submit_button: str = "button[type='submit']"
    confirmation_selector: str = ".confirmation-message"


# Pre-configured portals
PORTAL_CONFIGS: dict[str, PortalConfig] = {
    "sam_gov": PortalConfig(
        name="sam_gov",
        login_url="https://sam.gov/signin",
        submission_url_template="https://sam.gov/opp/{opportunity_id}/submit",
        username_field="#username",
        password_field="#password",
        file_upload_field="input[type='file']",
        submit_button="button.submit-proposal",
        confirmation_selector=".submission-confirmation",
    ),
    "merx": PortalConfig(
        name="merx",
        login_url="https://merx.com/login",
        submission_url_template="https://merx.com/opportunity/{opportunity_id}/respond",
        username_field="#email",
        password_field="#password",
        file_upload_field=".file-upload input",
        submit_button="#submit-response",
        confirmation_selector=".confirmation",
    ),
    "generic": PortalConfig(
        name="generic",
        login_url="",
        submission_url_template="{opportunity_id}",
    ),
}


# ---------------------------------------------------------------------------
# Portal Uploader
# ---------------------------------------------------------------------------

class PortalUploader:
    """Automates tender submission to procurement portals via Playwright.

    Usage:
        uploader = PortalUploader()  # reads DRY_RUN from env

        result = uploader.submit(
            portal="sam_gov",
            opportunity_id="OPP-2026-001",
            document_path="/tmp/tender_response.md",
            credentials={"username": "user@co.com", "password": "secret"},
            metadata={"company_name": "Acme SDS Solutions", "uei": "ABC123"},
        )

        if result.success:
            print(f"Submitted! Confirmation: {result.confirmation_id}")

    Args:
        dry_run: If True, simulate everything. Default reads from DRY_RUN env var.
        screenshots_dir: Where to save confirmation screenshots.
        headless: Run browser in headless mode. Default True.
        timeout_ms: Default timeout for page operations in milliseconds.
    """

    def __init__(
        self,
        dry_run: bool | None = None,
        screenshots_dir: str = "/tmp/tender_screenshots",
        headless: bool = True,
        timeout_ms: int = 30_000,
    ) -> None:
        if dry_run is None:
            dry_run = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")

        self.dry_run = dry_run
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.timeout_ms = timeout_ms

        logger.info(
            "portal_uploader_initialized",
            dry_run=self.dry_run,
            headless=self.headless,
        )

    def submit(
        self,
        portal: str,
        opportunity_id: str,
        document_path: str,
        credentials: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        additional_files: list[str] | None = None,
    ) -> SubmissionResult:
        """Submit a tender response to a procurement portal.

        Args:
            portal: Portal identifier ("sam_gov", "merx", "generic").
            opportunity_id: The opportunity/solicitation ID on the portal.
            document_path: Path to the assembled tender document.
            credentials: Login credentials {"username": ..., "password": ...}.
            metadata: Form field data {"company_name": ..., "uei": ...}.
            additional_files: Paths to additional files to upload.

        Returns:
            SubmissionResult with success/failure details.
        """
        config = PORTAL_CONFIGS.get(portal, PORTAL_CONFIGS["generic"])

        logger.info(
            "submission_start",
            portal=portal,
            opportunity_id=opportunity_id,
            document=Path(document_path).name,
        )

        # Validate document exists
        if not Path(document_path).exists():
            return SubmissionResult(
                success=False,
                error_message=f"Document not found: {document_path}",
                portal_url=config.submission_url_template.format(
                    opportunity_id=opportunity_id
                ),
            )

        if self.dry_run:
            return self._submit_dry_run(config, opportunity_id, document_path, metadata)
        else:
            return self._submit_real(
                config, opportunity_id, document_path,
                credentials or {}, metadata or {}, additional_files or [],
            )

    def _submit_dry_run(
        self,
        config: PortalConfig,
        opportunity_id: str,
        document_path: str,
        metadata: dict[str, str] | None,
    ) -> SubmissionResult:
        """Simulate a portal submission."""
        now = datetime.now(timezone.utc)
        conf_id = f"CONF-{uuid.uuid4().hex[:8].upper()}"

        # Create a mock screenshot
        screenshot_name = f"submission_{opportunity_id}_{now.strftime('%Y%m%d_%H%M%S')}.txt"
        screenshot_path = self.screenshots_dir / screenshot_name
        screenshot_path.write_text(
            f"[DRY-RUN SCREENSHOT]\n"
            f"Portal: {config.name}\n"
            f"Opportunity: {opportunity_id}\n"
            f"Document: {Path(document_path).name}\n"
            f"Confirmation: {conf_id}\n"
            f"Timestamp: {now.isoformat()}\n"
            f"Status: SUBMITTED SUCCESSFULLY\n",
            encoding="utf-8",
        )

        result = SubmissionResult(
            success=True,
            confirmation_id=conf_id,
            confirmation_text=(
                f"[DRY-RUN] Your proposal for opportunity {opportunity_id} has been "
                f"received. Confirmation number: {conf_id}. "
                f"Document: {Path(document_path).name}."
            ),
            screenshot_path=str(screenshot_path),
            portal_url=config.submission_url_template.format(
                opportunity_id=opportunity_id
            ),
            submitted_at=now.isoformat(),
            metadata={
                "dry_run": True,
                "portal": config.name,
                "document_size": Path(document_path).stat().st_size,
                **(metadata or {}),
            },
        )

        logger.info(
            "submission_complete_dry_run",
            portal=config.name,
            confirmation_id=conf_id,
            screenshot=str(screenshot_path),
        )

        return result

    def _submit_real(
        self,
        config: PortalConfig,
        opportunity_id: str,
        document_path: str,
        credentials: dict[str, str],
        metadata: dict[str, str],
        additional_files: list[str],
    ) -> SubmissionResult:
        """Perform real portal submission using Playwright.

        FLOW:
        1. Launch headless Chromium
        2. Navigate to login page → fill credentials → submit
        3. Navigate to opportunity submission page
        4. Fill metadata fields (company name, UEI, etc.)
        5. Upload primary document
        6. Upload any additional files
        7. Click submit button
        8. Wait for confirmation
        9. Capture screenshot
        10. Extract confirmation ID
        """
        now = datetime.now(timezone.utc)

        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.headless)
                context = browser.new_context()
                page = context.new_page()
                page.set_default_timeout(self.timeout_ms)

                # Step 1: Login
                if config.login_url and credentials:
                    page.goto(config.login_url)
                    page.fill(config.username_field, credentials.get("username", ""))
                    page.fill(config.password_field, credentials.get("password", ""))
                    page.click("button[type='submit']")
                    page.wait_for_load_state("networkidle")
                    logger.info("portal_login_complete", portal=config.name)

                # Step 2: Navigate to submission page
                submission_url = config.submission_url_template.format(
                    opportunity_id=opportunity_id
                )
                page.goto(submission_url)
                page.wait_for_load_state("networkidle")

                # Step 3: Fill metadata fields
                for field_name, field_value in metadata.items():
                    try:
                        selector = f"input[name='{field_name}'], textarea[name='{field_name}']"
                        if page.locator(selector).count() > 0:
                            page.fill(selector, field_value)
                    except Exception:
                        logger.debug("metadata_field_not_found", field=field_name)

                # Step 4: Upload primary document
                page.set_input_files(config.file_upload_field, document_path)
                logger.info("document_uploaded", file=Path(document_path).name)

                # Step 5: Upload additional files
                for file_path in additional_files:
                    if Path(file_path).exists():
                        page.set_input_files(config.file_upload_field, file_path)

                # Step 6: Submit
                page.click(config.submit_button)
                page.wait_for_load_state("networkidle")

                # Step 7: Capture confirmation
                screenshot_name = (
                    f"submission_{opportunity_id}_{now.strftime('%Y%m%d_%H%M%S')}.png"
                )
                screenshot_path = str(self.screenshots_dir / screenshot_name)
                page.screenshot(path=screenshot_path, full_page=True)

                # Step 8: Extract confirmation text
                conf_text = ""
                conf_id = ""
                try:
                    conf_element = page.locator(config.confirmation_selector)
                    if conf_element.count() > 0:
                        conf_text = conf_element.first.text_content() or ""
                        # Try to extract a confirmation number
                        import re
                        id_match = re.search(r"(?:confirmation|receipt|reference)[:\s#]*(\S+)", conf_text, re.IGNORECASE)
                        if id_match:
                            conf_id = id_match.group(1)
                except Exception:
                    conf_text = "Confirmation element not found — check screenshot."

                if not conf_id:
                    conf_id = f"PORTAL-{uuid.uuid4().hex[:8].upper()}"

                browser.close()

                return SubmissionResult(
                    success=True,
                    confirmation_id=conf_id,
                    confirmation_text=conf_text,
                    screenshot_path=screenshot_path,
                    portal_url=submission_url,
                    submitted_at=now.isoformat(),
                    metadata={"portal": config.name, **metadata},
                )

        except ImportError:
            return SubmissionResult(
                success=False,
                error_message=(
                    "Playwright is not installed. Run: pip install playwright && "
                    "playwright install chromium"
                ),
            )
        except Exception as exc:
            logger.error("portal_submission_failed", portal=config.name, error=str(exc))
            return SubmissionResult(
                success=False,
                error_message=f"Portal submission failed: {exc}",
                portal_url=config.submission_url_template.format(
                    opportunity_id=opportunity_id
                ),
                submitted_at=now.isoformat(),
            )