"""
Timeout & Deadline Escalation — Manages Slack response timeouts and deadline alerts.

WHY TIMEOUT HANDLING MATTERS:
If the agent sends questions to Slack and nobody responds, the tender sits in limbo
forever. This module prevents that by enforcing three rules:

1. RESPONSE TIMEOUT (48h) — If no Slack response after 48 hours, escalate to a
   manager channel with the unanswered questions. This creates social pressure
   and ensures visibility.

2. DEADLINE ALERTS (72h / 24h / 4h) — Send increasingly urgent warnings as the
   tender submission deadline approaches. These fire regardless of whether there
   are pending Slack questions — they're a safety net for the whole pipeline.

3. AUTO-PROCEED — If response timeout is reached AND the deadline is within 24h,
   the agent should stop waiting and proceed with the best available draft. A
   slightly incomplete submission is better than no submission.

HOW IT INTEGRATES:
In production, a background scheduler (APScheduler or cron) runs every hour and calls:
- check_escalation_status() for each tender in AWAITING_HUMAN state
- check_deadline_alerts() for all active tenders

These run OUTSIDE the graph — they monitor PostgreSQL state and trigger actions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import structlog

from src.slack_integration.slack_client import SlackClient

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESPONSE_TIMEOUT_HOURS: int = 48
DEADLINE_ALERT_THRESHOLDS: list[int] = [72, 24, 4]  # Hours before deadline


@dataclass
class EscalationStatus:
    """Status of a pending Slack escalation.

    Attributes:
        tender_id: Which tender.
        hours_waiting: How long since questions were sent.
        is_timed_out: Whether RESPONSE_TIMEOUT_HOURS has passed.
        hours_to_deadline: Hours until submission deadline (None if unparseable).
        is_deadline_critical: Whether deadline is within 24h.
        should_auto_proceed: Whether to proceed without human response.
        action_taken: What action was taken (for logging).
    """
    tender_id: str
    hours_waiting: float
    is_timed_out: bool
    hours_to_deadline: float | None
    is_deadline_critical: bool
    should_auto_proceed: bool
    action_taken: str


class TimeoutHandler:
    """Manages Slack response timeouts and deadline-aware escalation.

    Usage:
        handler = TimeoutHandler()

        # Check a specific tender's escalation status
        status = handler.check_escalation_status(
            tender_id="TEST-001",
            escalated_at="2026-04-14T10:00:00Z",
            deadline="2026-04-17T17:00:00Z",
        )

        if status.is_timed_out:
            handler.send_timeout_escalation(...)

        if status.should_auto_proceed:
            # Resume graph without waiting for human input

        # Check deadline alerts for any tender
        new_alerts = handler.check_deadline_alerts(
            tender_id="TEST-001",
            tender_title="SDS Platform",
            deadline="2026-04-18T17:00:00Z",
            alerts_already_sent={72},  # Already sent 72h warning
        )

    Args:
        slack_client: SlackClient instance. Created automatically if None.
        timeout_hours: Hours to wait before timeout. Default 48.
        manager_channel: Channel for timeout escalations. Reads from env.
    """

    def __init__(
        self,
        slack_client: SlackClient | None = None,
        timeout_hours: int = RESPONSE_TIMEOUT_HOURS,
        manager_channel: str | None = None,
    ) -> None:
        self.slack_client = slack_client or SlackClient()
        self.timeout_hours = timeout_hours
        self.manager_channel = manager_channel or os.getenv(
            "SLACK_MANAGER_CHANNEL_ID", "C_MANAGER_MOCK"
        )

        logger.info(
            "timeout_handler_initialized",
            timeout_hours=self.timeout_hours,
            manager_channel=self.manager_channel,
        )

    def check_escalation_status(
        self,
        tender_id: str,
        escalated_at: str,
        deadline: str | None = None,
    ) -> EscalationStatus:
        """Check the status of a pending Slack escalation.

        Determines whether the response has timed out, whether the deadline
        is approaching, and whether the agent should auto-proceed.

        AUTO-PROCEED LOGIC:
        - Timed out + deadline critical (≤24h) → auto-proceed
        - Timed out + deadline not critical → escalate to manager, keep waiting
        - Not timed out + deadline critical → send deadline warning, keep waiting
        - Not timed out + deadline not critical → keep waiting (normal)

        Args:
            tender_id: Tender being checked.
            escalated_at: ISO timestamp of when questions were sent.
            deadline: ISO timestamp of submission deadline (optional).

        Returns:
            EscalationStatus with analysis and recommended action.
        """
        now = datetime.now(timezone.utc)

        # Calculate wait time
        try:
            esc_dt = datetime.fromisoformat(escalated_at.replace("Z", "+00:00"))
            hours_waiting = (now - esc_dt).total_seconds() / 3600
        except (ValueError, TypeError):
            hours_waiting = 0.0

        is_timed_out = hours_waiting >= self.timeout_hours

        # Calculate time to deadline
        hours_to_deadline: float | None = None
        is_deadline_critical = False
        if deadline:
            try:
                deadline_dt = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
                hours_to_deadline = (deadline_dt - now).total_seconds() / 3600
                is_deadline_critical = hours_to_deadline is not None and hours_to_deadline <= 24
            except (ValueError, TypeError):
                pass

        # Determine action
        should_auto_proceed = False
        if is_timed_out and is_deadline_critical:
            action = "auto_proceed_timeout_and_deadline"
            should_auto_proceed = True
        elif is_timed_out:
            action = "timeout_escalate_to_manager"
        elif is_deadline_critical:
            action = "deadline_warning"
        else:
            action = "waiting"

        status = EscalationStatus(
            tender_id=tender_id,
            hours_waiting=round(hours_waiting, 1),
            is_timed_out=is_timed_out,
            hours_to_deadline=round(hours_to_deadline, 1) if hours_to_deadline is not None else None,
            is_deadline_critical=is_deadline_critical,
            should_auto_proceed=should_auto_proceed,
            action_taken=action,
        )

        logger.info(
            "escalation_status_checked",
            tender_id=tender_id,
            hours_waiting=status.hours_waiting,
            timed_out=is_timed_out,
            hours_to_deadline=status.hours_to_deadline,
            action=action,
        )

        return status

    def send_timeout_escalation(
        self,
        tender_id: str,
        tender_title: str,
        original_questions: list[str],
        hours_waiting: float,
    ) -> None:
        """Escalate to manager channel when response timeout is reached.

        Sends a message to the manager channel with the unanswered questions
        and how long the team has been unresponsive.

        Args:
            tender_id: Tender ID.
            tender_title: Tender title.
            original_questions: The unanswered questions.
            hours_waiting: How long since questions were sent.
        """
        self.slack_client.send_gap_questions(
            tender_title=f"[TIMEOUT {hours_waiting:.0f}h] {tender_title}",
            tender_deadline="OVERDUE — needs immediate response",
            questions=original_questions,
            tender_id=tender_id,
            channel=self.manager_channel,
        )

        logger.warning(
            "timeout_escalation_sent",
            tender_id=tender_id,
            hours_waiting=round(hours_waiting, 1),
            channel=self.manager_channel,
            questions=len(original_questions),
        )

    def check_deadline_alerts(
        self,
        tender_id: str,
        tender_title: str,
        deadline: str,
        alerts_already_sent: set[int] | None = None,
    ) -> list[int]:
        """Check if deadline alerts need to be sent and send them.

        Compares current time against deadline and sends warnings at
        each threshold (72h, 24h, 4h) that hasn't been sent yet.

        Args:
            tender_id: Tender ID.
            tender_title: Tender title.
            deadline: Submission deadline ISO string.
            alerts_already_sent: Set of thresholds already alerted (e.g., {72}).

        Returns:
            List of new thresholds that were alerted (e.g., [24, 4]).
        """
        sent = alerts_already_sent or set()
        new_alerts: list[int] = []

        try:
            deadline_dt = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            hours_remaining = (deadline_dt - datetime.now(timezone.utc)).total_seconds() / 3600
        except (ValueError, TypeError):
            return []

        for threshold in DEADLINE_ALERT_THRESHOLDS:
            if threshold not in sent and hours_remaining <= threshold:
                self.slack_client.send_deadline_warning(
                    tender_title=tender_title,
                    tender_deadline=deadline,
                    hours_remaining=threshold,
                    tender_id=tender_id,
                )
                new_alerts.append(threshold)
                logger.info(
                    "deadline_alert_sent",
                    tender_id=tender_id,
                    threshold_hours=threshold,
                    hours_remaining=round(hours_remaining, 1),
                )

        return new_alerts