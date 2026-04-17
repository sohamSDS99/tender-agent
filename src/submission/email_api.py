"""
Email Submission & API Dispatch — Alternative submission channels.

WHY MULTIPLE CHANNELS:
Different tenders require different submission methods:
- Portal upload (Step 20) — Most government tenders (SAM.gov, MERX)
- Email — Many state/local government and private enterprise tenders
- API — Newer procurement platforms that offer programmatic submission

The Submit node (Step 22) checks the tender's required submission method
and routes to the appropriate channel.

EMAIL SUBMISSION:
Uses Python's built-in smtplib (no extra dependencies). Composes a professional
email with the tender response attached as a file. Supports multiple attachments
(main document + certifications + pricing sheets).

For Gmail: Use an App Password, not your regular password.
Set SMTP_HOST=smtp.gmail.com, SMTP_PORT=587, SMTP_USE_TLS=true.

API DISPATCH:
A generic HTTP POST/PUT sender for portals with submission APIs. Sends the
document as multipart form data with metadata fields. Each portal's API
format is slightly different, so the dispatcher accepts a flexible config.

DRY-RUN MODE:
Both channels simulate sending and return mock confirmations.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import structlog

from src.submission.portal_upload import SubmissionResult

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Email Submission
# ---------------------------------------------------------------------------

class EmailSubmitter:
    """Submits tender responses via SMTP email.

    Usage:
        submitter = EmailSubmitter()  # reads DRY_RUN from env
        result = submitter.submit(
            to_address="procurement@agency.gov",
            subject="Proposal: SDS Management Platform — SAM-2026-001",
            body="Please find our proposal attached.",
            document_path="/tmp/tender_response.md",
            tender_id="SAM-2026-001",
        )

    Args:
        dry_run: If True, simulate sending. Default reads from DRY_RUN env var.
        smtp_host: SMTP server. Reads from SMTP_HOST env var.
        smtp_port: SMTP port. Reads from SMTP_PORT env var. Default 587.
        smtp_user: SMTP username. Reads from SMTP_USER env var.
        smtp_password: SMTP password. Reads from SMTP_PASSWORD env var.
        smtp_use_tls: Whether to use STARTTLS. Reads from SMTP_USE_TLS env var.
        from_address: Sender email. Reads from SMTP_FROM env var.
    """

    def __init__(
        self,
        dry_run: bool | None = None,
        smtp_host: str | None = None,
        smtp_port: int | None = None,
        smtp_user: str | None = None,
        smtp_password: str | None = None,
        smtp_use_tls: bool | None = None,
        from_address: str | None = None,
    ) -> None:
        if dry_run is None:
            dry_run = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")

        self.dry_run = dry_run
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = smtp_user or os.getenv("SMTP_USER", "")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD", "")
        self.from_address = from_address or os.getenv("SMTP_FROM", self.smtp_user)

        if smtp_use_tls is not None:
            self.use_tls = smtp_use_tls
        else:
            self.use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes")

        if not self.dry_run and not all([self.smtp_host, self.smtp_user, self.smtp_password]):
            raise ValueError(
                "SMTP_HOST, SMTP_USER, and SMTP_PASSWORD are required when "
                "DRY_RUN is disabled. Set them in .env."
            )

        logger.info(
            "email_submitter_initialized",
            dry_run=self.dry_run,
            host=self.smtp_host if not self.dry_run else "(dry-run)",
        )

    def submit(
        self,
        to_address: str,
        subject: str,
        body: str,
        document_path: str,
        tender_id: str = "",
        additional_files: list[str] | None = None,
        cc: list[str] | None = None,
    ) -> SubmissionResult:
        """Send tender response via email with attachments.

        Args:
            to_address: Recipient email (procurement officer).
            subject: Email subject line.
            body: Email body text (plain text).
            document_path: Path to the main tender document.
            tender_id: Tender reference ID (for logging).
            additional_files: Paths to extra attachments.
            cc: CC addresses.

        Returns:
            SubmissionResult with success/failure details.
        """
        logger.info(
            "email_submission_start",
            to=to_address,
            tender_id=tender_id,
            document=Path(document_path).name,
        )

        if not Path(document_path).exists():
            return SubmissionResult(
                success=False,
                error_message=f"Document not found: {document_path}",
            )

        if self.dry_run:
            return self._submit_dry_run(to_address, subject, body, document_path, tender_id)
        else:
            return self._submit_real(
                to_address, subject, body, document_path,
                tender_id, additional_files or [], cc or [],
            )

    def _submit_dry_run(
        self, to: str, subject: str, body: str, doc_path: str, tender_id: str,
    ) -> SubmissionResult:
        now = datetime.now(timezone.utc)
        conf_id = f"EMAIL-{uuid.uuid4().hex[:8].upper()}"

        result = SubmissionResult(
            success=True,
            confirmation_id=conf_id,
            confirmation_text=(
                f"[DRY-RUN] Email sent to {to}. "
                f"Subject: {subject}. "
                f"Attachment: {Path(doc_path).name}. "
                f"Message ID: {conf_id}."
            ),
            screenshot_path="",
            portal_url=f"mailto:{to}",
            submitted_at=now.isoformat(),
            metadata={
                "dry_run": True,
                "method": "email",
                "to": to,
                "subject": subject,
                "attachment": Path(doc_path).name,
            },
        )

        logger.info("email_submission_dry_run", to=to, conf_id=conf_id)
        return result

    def _submit_real(
        self, to: str, subject: str, body: str, doc_path: str,
        tender_id: str, additional_files: list[str], cc: list[str],
    ) -> SubmissionResult:
        """Send the actual email via SMTP."""
        import smtplib

        now = datetime.now(timezone.utc)

        try:
            # Build the email
            msg = MIMEMultipart()
            msg["From"] = self.from_address
            msg["To"] = to
            msg["Subject"] = subject
            if cc:
                msg["Cc"] = ", ".join(cc)

            msg.attach(MIMEText(body, "plain"))

            # Attach main document
            all_files = [doc_path] + additional_files
            for file_path in all_files:
                path = Path(file_path)
                if not path.exists():
                    continue

                part = MIMEBase("application", "octet-stream")
                part.set_payload(path.read_bytes())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename={path.name}",
                )
                msg.attach(part)

            # Send
            recipients = [to] + cc
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_address, recipients, msg.as_string())

            conf_id = f"EMAIL-{uuid.uuid4().hex[:8].upper()}"

            logger.info("email_sent", to=to, subject=subject, tender_id=tender_id)

            return SubmissionResult(
                success=True,
                confirmation_id=conf_id,
                confirmation_text=f"Email sent to {to}. Subject: {subject}.",
                portal_url=f"mailto:{to}",
                submitted_at=now.isoformat(),
                metadata={"method": "email", "to": to, "subject": subject},
            )

        except Exception as exc:
            logger.error("email_send_failed", to=to, error=str(exc))
            return SubmissionResult(
                success=False,
                error_message=f"Email send failed: {exc}",
                submitted_at=now.isoformat(),
            )


# ---------------------------------------------------------------------------
# API Dispatch
# ---------------------------------------------------------------------------

class APIDispatcher:
    """Submits tender responses via HTTP API.

    For procurement platforms that offer programmatic submission endpoints.
    Sends the document as multipart form data with metadata.

    Usage:
        dispatcher = APIDispatcher()
        result = dispatcher.submit(
            endpoint_url="https://portal.example.com/api/v1/submissions",
            document_path="/tmp/tender_response.md",
            tender_id="T-001",
            api_key="sk-...",
            metadata={"company_name": "Acme", "opportunity_id": "OPP-001"},
        )

    Args:
        dry_run: If True, simulate. Default reads from DRY_RUN env var.
    """

    def __init__(self, dry_run: bool | None = None) -> None:
        if dry_run is None:
            dry_run = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")
        self.dry_run = dry_run
        logger.info("api_dispatcher_initialized", dry_run=self.dry_run)

    def submit(
        self,
        endpoint_url: str,
        document_path: str,
        tender_id: str = "",
        api_key: str = "",
        metadata: dict[str, str] | None = None,
        additional_files: list[str] | None = None,
    ) -> SubmissionResult:
        """Submit a tender response via HTTP POST.

        Args:
            endpoint_url: The API endpoint to POST to.
            document_path: Path to the tender document.
            tender_id: Tender reference ID.
            api_key: API authentication key.
            metadata: Key-value pairs sent as form fields.
            additional_files: Extra files to upload.

        Returns:
            SubmissionResult with success/failure details.
        """
        logger.info(
            "api_submission_start",
            endpoint=endpoint_url,
            tender_id=tender_id,
        )

        if not Path(document_path).exists():
            return SubmissionResult(
                success=False,
                error_message=f"Document not found: {document_path}",
            )

        if self.dry_run:
            return self._submit_dry_run(endpoint_url, document_path, tender_id, metadata)
        else:
            return self._submit_real(
                endpoint_url, document_path, tender_id,
                api_key, metadata or {}, additional_files or [],
            )

    def _submit_dry_run(
        self, url: str, doc_path: str, tender_id: str, metadata: dict | None,
    ) -> SubmissionResult:
        now = datetime.now(timezone.utc)
        conf_id = f"API-{uuid.uuid4().hex[:8].upper()}"

        return SubmissionResult(
            success=True,
            confirmation_id=conf_id,
            confirmation_text=(
                f"[DRY-RUN] API submission to {url}. "
                f"Document: {Path(doc_path).name}. "
                f"Response ID: {conf_id}."
            ),
            portal_url=url,
            submitted_at=now.isoformat(),
            metadata={
                "dry_run": True,
                "method": "api",
                "endpoint": url,
                **(metadata or {}),
            },
        )

    def _submit_real(
        self, url: str, doc_path: str, tender_id: str,
        api_key: str, metadata: dict[str, str], additional_files: list[str],
    ) -> SubmissionResult:
        """Send the actual API request."""
        import httpx

        now = datetime.now(timezone.utc)

        try:
            headers: dict[str, str] = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            # Build multipart form data
            files_list: list[tuple[str, tuple[str, bytes, str]]] = []

            # Main document
            doc = Path(doc_path)
            files_list.append(
                ("document", (doc.name, doc.read_bytes(), "application/octet-stream"))
            )

            # Additional files
            for fp in additional_files:
                p = Path(fp)
                if p.exists():
                    files_list.append(
                        ("attachments", (p.name, p.read_bytes(), "application/octet-stream"))
                    )

            response = httpx.post(
                url,
                headers=headers,
                data=metadata,
                files=files_list,
                timeout=60.0,
            )

            if response.status_code in (200, 201, 202):
                resp_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                conf_id = resp_data.get("confirmation_id", resp_data.get("id", f"API-{uuid.uuid4().hex[:8].upper()}"))

                return SubmissionResult(
                    success=True,
                    confirmation_id=str(conf_id),
                    confirmation_text=f"API submission accepted. Status: {response.status_code}.",
                    portal_url=url,
                    submitted_at=now.isoformat(),
                    metadata={"method": "api", "status_code": response.status_code, **metadata},
                )
            else:
                return SubmissionResult(
                    success=False,
                    error_message=f"API returned {response.status_code}: {response.text[:500]}",
                    portal_url=url,
                    submitted_at=now.isoformat(),
                )

        except Exception as exc:
            logger.error("api_submission_failed", url=url, error=str(exc))
            return SubmissionResult(
                success=False,
                error_message=f"API submission failed: {exc}",
                portal_url=url,
                submitted_at=now.isoformat(),
            )