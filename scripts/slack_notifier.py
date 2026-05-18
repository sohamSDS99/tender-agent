"""
Slack Notifier for Nexus Agent Bridge
======================================
Sends formatted task completion notifications to the #agent-updates channel.

Each agent posts with its own display name and emoji icon, so the channel
shows a clear feed of which agent completed what.

Usage:
    from slack_notifier import notify_task_completed, notify_task_failed

    notify_task_completed(
        agent_name="Tender Agent",
        task_title="Search for SDS tenders in Europe",
        summary="Found 8 verified tenders across 3 countries...",
        metrics={"duration_ms": 4500, "cost_usd": 0.003, "tokens_used": 1200},
    )
"""

import os
import json
from datetime import datetime, timezone

import httpx


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID", "")

# Agent identity map — each agent gets its own display name and emoji
AGENT_IDENTITIES = {
    "tender-agent": {
        "display_name": "Tender Agent",
        "icon_emoji": ":mag:",  # 🔍
    },
    "compliance-agent": {
        "display_name": "Compliance Agent",
        "icon_emoji": ":shield:",  # 🛡️
    },
    "sds-author-agent": {
        "display_name": "SDS Author Agent",
        "icon_emoji": ":memo:",  # 📝
    },
    # Add more agents here as you build them
    "_default": {
        "display_name": "Nexus Agent",
        "icon_emoji": ":robot_face:",  # 🤖
    },
}


def _get_identity(agent_name: str) -> dict:
    """Get the Slack display identity for an agent."""
    return AGENT_IDENTITIES.get(agent_name, AGENT_IDENTITIES["_default"])


def _post_message(blocks: list, text: str, agent_name: str = "tender-agent") -> bool:
    """Post a message to the #agent-updates Slack channel.

    Returns True if successful, False otherwise.
    """
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID:
        print("[Slack] Missing SLACK_BOT_TOKEN or SLACK_CHANNEL_ID — skipping notification")
        return False

    identity = _get_identity(agent_name)

    try:
        resp = httpx.post(
            "https://slack.com/api/chat.postMessage",
            headers={
                "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "channel": SLACK_CHANNEL_ID,
                "username": identity["display_name"],
                "icon_emoji": identity["icon_emoji"],
                "text": text,  # Fallback for notifications
                "blocks": blocks,
            },
            timeout=10.0,
        )

        data = resp.json()
        if data.get("ok"):
            print(f"[Slack] Notification sent to #{SLACK_CHANNEL_ID}")
            return True
        else:
            error = data.get("error", "unknown")
            print(f"[Slack] API error: {error}")
            return False

    except Exception as exc:
        print(f"[Slack] Failed to send notification: {exc}")
        return False


# ---------------------------------------------------------------------------
# Public API — call these from the bridge
# ---------------------------------------------------------------------------

def notify_task_completed(
    agent_name: str,
    task_title: str,
    summary: str = "",
    metrics: dict | None = None,
    task_type: str = "task",
    thread_id: str = "",
) -> bool:
    """Send a task completion notification to Slack.

    Args:
        agent_name: Internal agent name (e.g., "tender-agent")
        task_title: What the task was about
        summary: Brief description of the result
        metrics: Optional dict with duration_ms, cost_usd, tokens_used, tenders_found, etc.
        task_type: Type of task (e.g., "tender_search", "form_fill", "question")
        thread_id: Chat thread ID for reference

    Returns:
        True if the message was sent successfully
    """
    identity = _get_identity(agent_name)
    now = datetime.now(timezone.utc).strftime("%b %d, %Y at %H:%M UTC")
    metrics = metrics or {}

    # Build the metrics line
    metric_parts = []
    if metrics.get("duration_ms"):
        duration = metrics["duration_ms"]
        if duration >= 60000:
            metric_parts.append(f":stopwatch: {duration / 1000:.1f}s")
        else:
            metric_parts.append(f":stopwatch: {duration}ms")
    if metrics.get("cost_usd"):
        metric_parts.append(f":moneybag: ${metrics['cost_usd']:.4f}")
    if metrics.get("tokens_used"):
        metric_parts.append(f":abacus: {metrics['tokens_used']:,} tokens")
    if metrics.get("tenders_found"):
        metric_parts.append(f":page_facing_up: {metrics['tenders_found']} tenders found")
    if metrics.get("fields_filled"):
        total = metrics.get("fields_total", "?")
        metric_parts.append(f":pencil2: {metrics['fields_filled']}/{total} fields filled")
    if metrics.get("llm_calls"):
        metric_parts.append(f":brain: {metrics['llm_calls']} LLM calls")

    metrics_line = "  |  ".join(metric_parts) if metric_parts else "No metrics recorded"

    # Task type emoji
    type_emojis = {
        "tender_search": ":mag:",
        "form_fill": ":clipboard:",
        "question": ":speech_balloon:",
        "greeting": ":wave:",
        "task": ":white_check_mark:",
    }
    type_emoji = type_emojis.get(task_type, ":white_check_mark:")

    # Build Slack Block Kit message
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{identity['display_name']} completed a task",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{type_emoji}  *{task_title}*",
            },
        },
    ]

    # Add summary if provided
    if summary:
        # Truncate long summaries
        display_summary = summary[:500] + "..." if len(summary) > 500 else summary
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": display_summary,
            },
        })

    # Add metrics
    blocks.append({"type": "divider"})
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": metrics_line,
            },
        ],
    })

    # Add timestamp
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f":clock1: {now}",
            },
        ],
    })

    fallback_text = f"{identity['display_name']} completed: {task_title}"
    return _post_message(blocks, fallback_text, agent_name)


def notify_task_failed(
    agent_name: str,
    task_title: str,
    error_message: str = "",
    metrics: dict | None = None,
) -> bool:
    """Send a task failure notification to Slack.

    Args:
        agent_name: Internal agent name
        task_title: What the task was about
        error_message: What went wrong
        metrics: Optional performance metrics

    Returns:
        True if the message was sent successfully
    """
    identity = _get_identity(agent_name)
    now = datetime.now(timezone.utc).strftime("%b %d, %Y at %H:%M UTC")
    metrics = metrics or {}

    # Build metrics line
    metric_parts = []
    if metrics.get("duration_ms"):
        metric_parts.append(f":stopwatch: {metrics['duration_ms']}ms")
    metrics_line = "  |  ".join(metric_parts) if metric_parts else ""

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{identity['display_name']} — Task Failed",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":x:  *{task_title}*",
            },
        },
    ]

    if error_message:
        display_error = error_message[:400] + "..." if len(error_message) > 400 else error_message
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"```{display_error}```",
            },
        })

    blocks.append({"type": "divider"})

    context_parts = [f":clock1: {now}"]
    if metrics_line:
        context_parts.append(metrics_line)

    blocks.append({
        "type": "context",
        "elements": [
            {"type": "mrkdwn", "text": "  |  ".join(context_parts)},
        ],
    })

    fallback_text = f"{identity['display_name']} FAILED: {task_title}"
    return _post_message(blocks, fallback_text, agent_name)


def notify_agent_online(agent_name: str) -> bool:
    """Send a notification that an agent has come online."""
    identity = _get_identity(agent_name)
    now = datetime.now(timezone.utc).strftime("%b %d, %Y at %H:%M UTC")

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":green_circle:  *{identity['display_name']}* is now online and ready for tasks.",
            },
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f":clock1: {now}"},
            ],
        },
    ]

    fallback_text = f"{identity['display_name']} is now online"
    return _post_message(blocks, fallback_text, agent_name)


def notify_agent_offline(agent_name: str) -> bool:
    """Send a notification that an agent is going offline."""
    identity = _get_identity(agent_name)
    now = datetime.now(timezone.utc).strftime("%b %d, %Y at %H:%M UTC")

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":red_circle:  *{identity['display_name']}* has gone offline.",
            },
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f":clock1: {now}"},
            ],
        },
    ]

    fallback_text = f"{identity['display_name']} is now offline"
    return _post_message(blocks, fallback_text, agent_name)
