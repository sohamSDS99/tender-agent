"""
Slack Client — Sends gap questions and receives human responses via Slack.

HOW THE AGENT USES SLACK:
When the Gap Check node finds missing information, the Slack Escalate node sends
specific questions to a designated channel (e.g., #tender-agent-questions). The
message uses Block Kit for clear formatting, includes tender title and deadline,
and tags relevant team members.

DRY-RUN MODE:
Returns mock Slack data without connecting to any workspace. Mock responses
simulate immediate human replies so the pipeline can be tested end-to-end.

SETUP (for production):
1. Create a Slack App at https://api.slack.com/apps
2. Add Bot Token Scopes: chat:write, channels:read, channels:history
3. Install to workspace, get SLACK_BOT_TOKEN
4. Set SLACK_CHANNEL_ID to the target channel
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SlackMessage:
    """A message sent to or received from Slack."""
    channel: str
    text: str
    thread_ts: str | None = None
    blocks: list[dict] | None = None
    ts: str = ""
    user: str = ""
    is_response: bool = False


class SlackClient:
    """Client for sending and receiving Slack messages.

    Usage:
        client = SlackClient()  # reads DRY_RUN from env
        msg = client.send_gap_questions(
            tender_title="SDS Platform for EPA",
            tender_deadline="2026-06-15",
            questions=["What is our ISO 27001 expiry date?"],
        )
        responses = client.get_thread_responses(msg.channel, msg.ts)

    Args:
        dry_run: If True, mock all Slack interactions.
        bot_token: Slack Bot Token. Reads from SLACK_BOT_TOKEN env var.
        channel_id: Default channel. Reads from SLACK_CHANNEL_ID env var.
    """

    def __init__(
        self,
        dry_run: bool | None = None,
        bot_token: str | None = None,
        channel_id: str | None = None,
    ) -> None:
        if dry_run is None:
            dry_run = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")

        self.dry_run = dry_run
        self._bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN", "")
        self.default_channel = channel_id or os.getenv("SLACK_CHANNEL_ID", "C_MOCK_CHANNEL")

        if not self.dry_run and not self._bot_token:
            raise ValueError(
                "SLACK_BOT_TOKEN is required when DRY_RUN is disabled. "
                "Create a Slack App and set the token in .env."
            )

        self._client = None
        if not self.dry_run:
            from slack_sdk import WebClient
            self._client = WebClient(token=self._bot_token)

        logger.info("slack_client_initialized", dry_run=self.dry_run)

    def send_gap_questions(
        self,
        tender_title: str,
        tender_deadline: str,
        questions: list[str],
        tender_id: str = "",
        channel: str | None = None,
    ) -> SlackMessage:
        """Send gap escalation questions to Slack with Block Kit formatting.

        Returns:
            SlackMessage with sent message details (including thread_ts).
        """
        target_channel = channel or self.default_channel

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Tender Agent Needs Input"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Tender:*\n{tender_title}"},
                    {"type": "mrkdwn", "text": f"*Deadline:*\n{tender_deadline}"},
                ],
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Questions:*\n" + "\n".join(
                        f"{i+1}. {q}" for i, q in enumerate(questions)
                    ),
                },
            },
            {"type": "divider"},
            {
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": (
                        f"Reply in this thread to answer. "
                        f"Tender ID: `{tender_id}` | "
                        f"The agent will resume automatically after your response."
                    ),
                }],
            },
        ]

        plain_text = (
            f"Tender Agent needs input for: {tender_title}\n"
            + "\n".join(f"  {i+1}. {q}" for i, q in enumerate(questions))
        )

        if self.dry_run:
            return self._send_dry_run(target_channel, plain_text, blocks)
        else:
            return self._send_real(target_channel, plain_text, blocks)

    def send_deadline_warning(
        self,
        tender_title: str,
        tender_deadline: str,
        hours_remaining: int,
        tender_id: str = "",
        channel: str | None = None,
    ) -> SlackMessage:
        """Send a deadline warning at 72h, 24h, or 4h before expiry."""
        target_channel = channel or self.default_channel
        urgency = "URGENT" if hours_remaining <= 4 else (
            "WARNING" if hours_remaining <= 24 else "REMINDER"
        )
        text = (
            f"[{urgency}] Tender '{tender_title}' ({tender_id}) has "
            f"{hours_remaining}h remaining before deadline ({tender_deadline})."
        )
        if self.dry_run:
            return self._send_dry_run(target_channel, text, None)
        else:
            return self._send_real(target_channel, text, None)

    def get_thread_responses(
        self,
        channel: str,
        thread_ts: str,
    ) -> list[SlackMessage]:
        """Fetch human responses from a Slack thread."""
        if self.dry_run:
            return [
                SlackMessage(
                    channel=channel,
                    text=(
                        "Our ISO 27001 certification was renewed in March 2026, "
                        "valid until March 2029. FedRAMP authorization is in "
                        "progress, expected Q3 2026."
                    ),
                    thread_ts=thread_ts,
                    ts=f"{thread_ts}.001",
                    user="U_HUMAN_MOCK",
                    is_response=True,
                ),
            ]
        else:
            return self._get_responses_real(channel, thread_ts)

    # --- Private methods ---

    def _send_dry_run(self, channel: str, text: str, blocks: list[dict] | None) -> SlackMessage:
        ts = datetime.now(timezone.utc).strftime("%s.%f")
        logger.info("slack_message_sent_dry_run", channel=channel, preview=text[:80])
        return SlackMessage(channel=channel, text=text, blocks=blocks, ts=ts, thread_ts=ts)

    def _send_real(self, channel: str, text: str, blocks: list[dict] | None) -> SlackMessage:
        kwargs: dict[str, Any] = {"channel": channel, "text": text}
        if blocks:
            kwargs["blocks"] = blocks
        response = self._client.chat_postMessage(**kwargs)
        return SlackMessage(
            channel=response["channel"], text=text, blocks=blocks,
            ts=response["ts"], thread_ts=response["ts"],
        )

    def _get_responses_real(self, channel: str, thread_ts: str) -> list[SlackMessage]:
        response = self._client.conversations_replies(channel=channel, ts=thread_ts)
        messages = []
        for msg in response.get("messages", [])[1:]:
            if not msg.get("bot_id"):
                messages.append(SlackMessage(
                    channel=channel, text=msg.get("text", ""),
                    thread_ts=thread_ts, ts=msg.get("ts", ""),
                    user=msg.get("user", ""), is_response=True,
                ))
        return messages