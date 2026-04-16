"""
Email IMAP Monitor — Discovers tenders from email notifications and direct RFPs.

WHY EMAIL MONITORING:
Not all tenders come from SAM.gov. The company receives tender notifications from:
- Procurement alert services (MERX, BidSync, GovWin, FindRFP)
- Direct RFP emails from enterprise clients
- State/local government procurement offices
- Industry association tender digests

These emails follow recognizable patterns: subject lines contain "RFP", "RFQ",
"tender", "bid", "solicitation", or "proposal request". The body usually includes
a deadline, a description, and sometimes an attached tender document (PDF/DOCX).

HOW IT WORKS:
1. Connect to an IMAP inbox (Gmail, Outlook, or any IMAP server)
2. Search for unread emails matching tender-related keywords
3. Parse subject, body, and sender for tender metadata
4. Extract attachments (PDF/DOCX) for later document parsing
5. Convert each email into a TenderLead (same format as SAM.gov scraper)
6. Mark emails as processed (move to a "Processed" folder or flag them)

IMAP CONFIGURATION:
Set these in your .env file:
    IMAP_HOST=imap.gmail.com
    IMAP_PORT=993
    IMAP_USER=tenders@yourcompany.com
    IMAP_PASSWORD=your-app-password
    IMAP_FOLDER=INBOX

For Gmail: Use an App Password (not your regular password).
Go to Google Account → Security → 2-Step Verification → App passwords.

DRY-RUN MODE:
Returns mock email data that simulates typical tender notifications.
No IMAP connection is made.
"""

from __future__ import annotations

import email
import email.policy
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any

import structlog

from src.discovery.sam_gov import TenderLead, score_relevance

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IMAP search criteria — emails with these words in subject get checked
SUBJECT_KEYWORDS: list[str] = [
    "RFP", "RFQ", "RFI", "tender", "bid", "solicitation",
    "proposal request", "request for proposal", "request for quote",
    "procurement", "contract opportunity",
]

# Regex to extract deadlines from email body
_DEADLINE_PATTERNS: list[re.Pattern] = [
    re.compile(r"deadline[:\s]+(\w+ \d{1,2},?\s*\d{4})", re.IGNORECASE),
    re.compile(r"due[:\s]+(\w+ \d{1,2},?\s*\d{4})", re.IGNORECASE),
    re.compile(r"closes?[:\s]+(\w+ \d{1,2},?\s*\d{4})", re.IGNORECASE),
    re.compile(r"submit by[:\s]+(\w+ \d{1,2},?\s*\d{4})", re.IGNORECASE),
    re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4})", re.IGNORECASE),
    re.compile(r"(\d{4}-\d{2}-\d{2})", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Parsed email data
# ---------------------------------------------------------------------------

@dataclass
class ParsedEmail:
    """An email parsed for tender-relevant information.

    Attributes:
        message_id: Email Message-ID header (for deduplication).
        subject: Email subject line.
        sender: From address.
        date: When the email was sent.
        body_text: Plain-text body content.
        extracted_deadline: Deadline found in the body (if any).
        attachment_names: List of attachment filenames.
        attachment_paths: List of saved attachment file paths (after download).
    """
    message_id: str
    subject: str
    sender: str
    date: str
    body_text: str
    extracted_deadline: str = ""
    attachment_names: list[str] = field(default_factory=list)
    attachment_paths: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Email parser helpers
# ---------------------------------------------------------------------------

def _extract_deadline(body: str) -> str:
    """Try to find a deadline date in the email body.

    Searches for common patterns like "Deadline: June 15, 2026",
    "Due: 06/15/2026", "Closes: 2026-06-15".

    Returns the first match as a string, or empty string if none found.
    """
    for pattern in _DEADLINE_PATTERNS:
        match = pattern.search(body)
        if match:
            return match.group(1).strip()
    return ""


def _is_tender_email(subject: str) -> bool:
    """Check if an email subject suggests it's tender-related."""
    subject_lower = subject.lower()
    return any(kw.lower() in subject_lower for kw in SUBJECT_KEYWORDS)


def _email_to_tender_lead(parsed: ParsedEmail) -> TenderLead:
    """Convert a ParsedEmail into a TenderLead for pipeline processing."""
    # Generate a stable ID from the message-ID
    lead_id = f"EMAIL-{parsed.message_id[:20]}" if parsed.message_id else f"EMAIL-{uuid.uuid4().hex[:12]}"

    # Score relevance using the same function as SAM.gov scraper
    score, keywords = score_relevance(parsed.subject, parsed.body_text[:2000])

    return TenderLead(
        lead_id=lead_id,
        title=parsed.subject,
        description=parsed.body_text[:2000],
        agency=parsed.sender,
        source_portal="email",
        source_url=f"email://{parsed.message_id}",
        naics_code="",
        submission_deadline=parsed.extracted_deadline,
        posted_date=parsed.date,
        relevance_score=score,
        relevance_keywords=keywords,
    )


# ---------------------------------------------------------------------------
# Email IMAP Monitor
# ---------------------------------------------------------------------------

class EmailMonitor:
    """Monitors an IMAP inbox for tender notifications and RFPs.

    Usage:
        monitor = EmailMonitor()  # uses DRY_RUN from env
        leads = monitor.check_inbox()
        for lead in leads:
            print(f"[{lead.relevance_score:.0%}] {lead.title}")

    Args:
        dry_run: If True, return mock data. Default reads from DRY_RUN env var.
        imap_host: IMAP server hostname. Reads from IMAP_HOST env var.
        imap_port: IMAP port (usually 993 for SSL). Reads from IMAP_PORT env var.
        imap_user: IMAP username/email. Reads from IMAP_USER env var.
        imap_password: IMAP password. Reads from IMAP_PASSWORD env var.
        imap_folder: Which folder to check. Default "INBOX".
        min_relevance: Minimum relevance score. Default 0.05.
    """

    def __init__(
        self,
        dry_run: bool | None = None,
        imap_host: str | None = None,
        imap_port: int | None = None,
        imap_user: str | None = None,
        imap_password: str | None = None,
        imap_folder: str = "INBOX",
        min_relevance: float = 0.05,
    ) -> None:
        if dry_run is None:
            dry_run = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")

        self.dry_run = dry_run
        self.min_relevance = min_relevance
        self.imap_host = imap_host or os.getenv("IMAP_HOST", "")
        self.imap_port = imap_port or int(os.getenv("IMAP_PORT", "993"))
        self.imap_user = imap_user or os.getenv("IMAP_USER", "")
        self.imap_password = imap_password or os.getenv("IMAP_PASSWORD", "")
        self.imap_folder = imap_folder

        if not self.dry_run and not all([self.imap_host, self.imap_user, self.imap_password]):
            raise ValueError(
                "IMAP_HOST, IMAP_USER, and IMAP_PASSWORD are required when "
                "DRY_RUN is disabled. Set them in your .env file."
            )

        logger.info(
            "email_monitor_initialized",
            dry_run=self.dry_run,
            host=self.imap_host or "(dry-run)",
            folder=self.imap_folder,
        )

    def check_inbox(
        self,
        max_emails: int = 50,
    ) -> list[TenderLead]:
        """Check the inbox for tender-related emails and return leads.

        Args:
            max_emails: Maximum emails to process per check. Default 50.

        Returns:
            List of TenderLead objects from tender-related emails,
            scored and filtered for relevance.
        """
        logger.info("checking_inbox", max_emails=max_emails, dry_run=self.dry_run)

        if self.dry_run:
            parsed_emails = self._check_dry_run()
        else:
            parsed_emails = self._check_real(max_emails)

        # Convert to TenderLeads and filter
        leads: list[TenderLead] = []
        for parsed in parsed_emails:
            lead = _email_to_tender_lead(parsed)
            if lead.relevance_score >= self.min_relevance:
                leads.append(lead)

        # Sort by relevance
        leads.sort(key=lambda l: l.relevance_score, reverse=True)

        logger.info(
            "inbox_checked",
            emails_found=len(parsed_emails),
            leads_after_filter=len(leads),
        )

        return leads

    def check_and_deduplicate(
        self,
        known_ids: set[str],
        max_emails: int = 50,
    ) -> list[TenderLead]:
        """Check inbox and exclude already-known tender IDs.

        Args:
            known_ids: Set of lead IDs already in the system.
            max_emails: Maximum emails to process.

        Returns:
            Only NEW leads not in known_ids.
        """
        all_leads = self.check_inbox(max_emails=max_emails)
        new_leads = [l for l in all_leads if l.lead_id not in known_ids]

        logger.info(
            "email_dedup_complete",
            total=len(all_leads),
            new=len(new_leads),
        )

        return new_leads

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _check_real(self, max_emails: int) -> list[ParsedEmail]:
        """Connect to IMAP and fetch unread tender-related emails.

        Uses Python's built-in imaplib (no extra dependencies).
        """
        import imaplib

        parsed: list[ParsedEmail] = []

        try:
            # Connect with SSL
            conn = imaplib.IMAP4_SSL(self.imap_host, self.imap_port)
            conn.login(self.imap_user, self.imap_password)
            conn.select(self.imap_folder)

            # Search for unread emails
            status, message_ids = conn.search(None, "UNSEEN")
            if status != "OK" or not message_ids[0]:
                conn.logout()
                return []

            ids = message_ids[0].split()[-max_emails:]  # Take most recent

            for msg_id in ids:
                status, msg_data = conn.fetch(msg_id, "(RFC822)")
                if status != "OK" or not msg_data[0]:
                    continue

                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(
                    raw_email, policy=email.policy.default
                )

                subject = str(msg.get("Subject", ""))

                # Only process emails with tender-related subjects
                if not _is_tender_email(subject):
                    continue

                # Extract body text
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            payload = part.get_payload(decode=True)
                            if payload:
                                body = payload.decode("utf-8", errors="replace")
                            break
                else:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        body = payload.decode("utf-8", errors="replace")

                # Extract attachment names
                attachments = []
                if msg.is_multipart():
                    for part in msg.walk():
                        filename = part.get_filename()
                        if filename:
                            attachments.append(filename)

                parsed.append(ParsedEmail(
                    message_id=str(msg.get("Message-ID", uuid.uuid4().hex)),
                    subject=subject,
                    sender=str(msg.get("From", "")),
                    date=str(msg.get("Date", "")),
                    body_text=body[:5000],
                    extracted_deadline=_extract_deadline(body),
                    attachment_names=attachments,
                ))

            conn.logout()

        except Exception as exc:
            logger.error("imap_connection_failed", error=str(exc))

        return parsed

    def _check_dry_run(self) -> list[ParsedEmail]:
        """Return mock email data for testing."""
        now = datetime.now(timezone.utc).isoformat()

        return [
            ParsedEmail(
                message_id="mock-001@procurement.gov",
                subject="RFP: Safety Data Sheet Management System for State DOT",
                sender="procurement@dot.state.gov",
                date=now,
                body_text=(
                    "The State Department of Transportation is seeking proposals for "
                    "a Safety Data Sheet (SDS) management system. The system must "
                    "provide GHS-compliant chemical inventory tracking, automated "
                    "regulatory reporting, and mobile access for field workers.\n\n"
                    "Key requirements:\n"
                    "- Cloud-based SaaS platform\n"
                    "- OSHA HCS compliance\n"
                    "- Integration with existing ERP systems\n"
                    "- Training for 200+ users\n\n"
                    "Deadline: June 30, 2026\n"
                    "Contact: procurement@dot.state.gov\n"
                    "Budget: $60,000-$80,000 annually"
                ),
                extracted_deadline="June 30, 2026",
                attachment_names=["RFP_SDS_Management_2026.pdf"],
            ),
            ParsedEmail(
                message_id="mock-002@merx.com",
                subject="Tender Alert: Chemical Safety Compliance Software (MERX #TN-2026-4521)",
                sender="alerts@merx.com",
                date=now,
                body_text=(
                    "New opportunity matching your profile on MERX:\n\n"
                    "Title: Chemical Safety Compliance Software\n"
                    "Organization: Canadian Department of National Defence\n"
                    "Category: Software - Environmental & Safety\n\n"
                    "Description: Enterprise chemical safety and SDS management "
                    "platform for military installations across Canada. Must support "
                    "WHMIS 2015 compliance, bilingual (English/French), and PROTECTED B "
                    "security classification.\n\n"
                    "Closing Date: 2026-07-15\n"
                    "Estimated Value: CAD $120,000"
                ),
                extracted_deadline="2026-07-15",
                attachment_names=[],
            ),
            ParsedEmail(
                message_id="mock-003@company.com",
                subject="RFQ: Office Supply Delivery Services Q3 2026",
                sender="purchasing@randomcorp.com",
                date=now,
                body_text=(
                    "We are requesting quotes for office supply delivery services "
                    "for our corporate headquarters. Items include paper, toner, "
                    "pens, and general office supplies. Monthly delivery schedule. "
                    "Budget: $5,000/month. Deadline: May 1, 2026."
                ),
                extracted_deadline="May 1, 2026",
                attachment_names=["Office_Supply_RFQ.xlsx"],
            ),
            ParsedEmail(
                message_id="mock-004@bidnotify.com",
                subject="Bid Opportunity: EHS Software Platform for Pharma Manufacturer",
                sender="notifications@bidnotify.com",
                date=now,
                body_text=(
                    "A major pharmaceutical manufacturer is seeking an Environment, "
                    "Health & Safety (EHS) software platform to manage chemical "
                    "inventories, safety data sheets, and regulatory compliance "
                    "across 8 manufacturing facilities.\n\n"
                    "Requirements include:\n"
                    "- SDS authoring and distribution\n"
                    "- GHS classification engine\n"
                    "- OSHA 300 log management\n"
                    "- ISO 45001 compliance tracking\n"
                    "- Mobile app for floor workers\n\n"
                    "Submit by: July 20, 2026\n"
                    "Estimated contract value: $150,000/year"
                ),
                extracted_deadline="July 20, 2026",
                attachment_names=["EHS_RFP_Pharma_2026.pdf", "Technical_Requirements.docx"],
            ),
        ]