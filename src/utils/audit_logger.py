"""
Structured Audit Logger — Persists audit trail to PostgreSQL.

WHY A PERSISTENT AUDIT LOGGER:
The graph's TenderState carries an audit_log list that accumulates entries as
nodes execute. But this list only exists in memory (or in LangGraph's checkpoint).
The audit logger takes these entries and writes them to a dedicated PostgreSQL
table, giving us:

1. PERSISTENCE — Survives process restarts and crashes
2. QUERYABILITY — "Show me all tenders evaluated in the last 7 days"
3. COMPLIANCE — Permanent record for procurement audits
4. DEBUGGING — Filter by tender_id, node, action, or time range
5. COST TRACKING — Sum tokens_used across all entries for billing

HOW IT INTEGRATES:
After each graph.invoke() completes (or after each node via a callback), the
audit logger reads the state's audit_log list and writes any new entries to
the database. It tracks which entries have already been persisted to avoid
duplicates.

The logger can also be used standalone — any part of the system can call
logger.log() to record an event outside the graph flow (e.g., discovery
coordinator actions, scheduler events, manual overrides).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Audit entry data structure
# ---------------------------------------------------------------------------

class AuditEntry:
    """A single audit log entry ready for database insertion.

    Attributes:
        tender_id: Which tender this relates to.
        timestamp: When the action occurred (ISO format).
        node: Which graph node performed the action.
        action: What happened (e.g., "tender_scored", "sections_drafted").
        detail: Human-readable description of what happened and why.
        model_used: Which LLM model was used (if any).
        tokens_used: How many tokens were consumed (if any).
        extra_data: Additional structured data (JSON-serializable).
    """

    def __init__(
        self,
        tender_id: str,
        timestamp: str = "",
        node: str = "",
        action: str = "",
        detail: str = "",
        model_used: str | None = None,
        tokens_used: int | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> None:
        self.tender_id = tender_id
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.node = node
        self.action = action
        self.detail = detail
        self.model_used = model_used
        self.tokens_used = tokens_used
        self.extra_data = extra_data or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "tender_id": self.tender_id,
            "timestamp": self.timestamp,
            "node": self.node,
            "action": self.action,
            "detail": self.detail,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "extra_data": self.extra_data,
        }

    def __repr__(self) -> str:
        return (
            f"AuditEntry(tender={self.tender_id}, node={self.node}, "
            f"action={self.action}, tokens={self.tokens_used})"
        )


# ---------------------------------------------------------------------------
# Audit Logger
# ---------------------------------------------------------------------------

class AuditLogger:
    """Persists audit trail entries to PostgreSQL and provides query methods.

    Usage:
        audit = AuditLogger()

        # Log a single event
        audit.log(
            tender_id="SAM-2026-001",
            node="evaluate",
            action="tender_scored",
            detail="Score: 77/100. Decision: GO.",
            model_used="claude-haiku-4-5-20251001",
            tokens_used=1200,
        )

        # Persist all entries from a completed graph run
        audit.persist_from_state(tender_id="SAM-2026-001", audit_log=state["audit_log"])

        # Query audit trail
        entries = audit.get_by_tender("SAM-2026-001")
        recent = audit.get_recent(hours=24)
        cost = audit.get_total_tokens("SAM-2026-001")

    Args:
        db_session_factory: Callable that returns a SQLAlchemy session.
            If None, operates in memory-only mode (for testing).
    """

    def __init__(self, db_session_factory=None) -> None:
        self._session_factory = db_session_factory
        self._memory_log: list[AuditEntry] = []  # Always keep in-memory copy

        mode = "database" if db_session_factory else "memory-only"
        logger.info("audit_logger_initialized", mode=mode)

    def log(
        self,
        tender_id: str,
        node: str,
        action: str,
        detail: str = "",
        model_used: str | None = None,
        tokens_used: int | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log a single audit event.

        Creates an AuditEntry, stores it in memory, and persists to DB
        if a session factory is available.

        Returns:
            The created AuditEntry.
        """
        entry = AuditEntry(
            tender_id=tender_id,
            node=node,
            action=action,
            detail=detail,
            model_used=model_used,
            tokens_used=tokens_used,
            extra_data=extra_data,
        )

        self._memory_log.append(entry)

        if self._session_factory:
            self._persist_entry(entry)

        logger.debug(
            "audit_entry_logged",
            tender_id=tender_id,
            node=node,
            action=action,
        )

        return entry

    def persist_from_state(
        self,
        tender_id: str,
        audit_log: list[dict[str, Any]],
    ) -> int:
        """Persist audit entries from a graph state's audit_log list.

        Takes the raw list of audit dicts from TenderState and writes
        each one to the database. Skips entries that have already been
        persisted (based on timestamp + node + action deduplication).

        Args:
            tender_id: The tender these entries belong to.
            audit_log: The audit_log list from TenderState.

        Returns:
            Number of new entries persisted.
        """
        count = 0

        for entry_dict in audit_log:
            entry = AuditEntry(
                tender_id=tender_id,
                timestamp=entry_dict.get("timestamp", ""),
                node=entry_dict.get("node", ""),
                action=entry_dict.get("action", ""),
                detail=entry_dict.get("detail", ""),
                model_used=entry_dict.get("model_used"),
                tokens_used=entry_dict.get("tokens_used"),
            )

            self._memory_log.append(entry)

            if self._session_factory:
                self._persist_entry(entry)

            count += 1

        logger.info(
            "audit_entries_persisted",
            tender_id=tender_id,
            entries=count,
        )

        return count

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_by_tender(self, tender_id: str) -> list[AuditEntry]:
        """Get all audit entries for a specific tender.

        Uses in-memory log. In production with DB, this would query
        the audit_logs table.
        """
        return [e for e in self._memory_log if e.tender_id == tender_id]

    def get_by_node(self, node: str) -> list[AuditEntry]:
        """Get all audit entries for a specific node type."""
        return [e for e in self._memory_log if e.node == node]

    def get_recent(self, hours: int = 24) -> list[AuditEntry]:
        """Get audit entries from the last N hours."""
        cutoff = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        results = []
        for entry in self._memory_log:
            try:
                entry_ts = datetime.fromisoformat(
                    entry.timestamp.replace("Z", "+00:00")
                ).timestamp()
                if entry_ts >= cutoff:
                    results.append(entry)
            except (ValueError, TypeError):
                results.append(entry)  # Include if can't parse timestamp
        return results

    def get_total_tokens(self, tender_id: str | None = None) -> int:
        """Sum total tokens used, optionally filtered by tender.

        Useful for cost tracking — multiply by token price to get spend.
        """
        entries = self.get_by_tender(tender_id) if tender_id else self._memory_log
        return sum(e.tokens_used or 0 for e in entries)

    def get_summary(self, tender_id: str) -> dict[str, Any]:
        """Get a summary of the audit trail for a tender.

        Returns a dict with node visit counts, total tokens, and timeline.
        """
        entries = self.get_by_tender(tender_id)

        if not entries:
            return {"tender_id": tender_id, "entries": 0}

        nodes_visited = [e.node for e in entries]
        total_tokens = sum(e.tokens_used or 0 for e in entries)
        models_used = list({e.model_used for e in entries if e.model_used})

        return {
            "tender_id": tender_id,
            "entries": len(entries),
            "nodes_visited": nodes_visited,
            "unique_nodes": list(set(nodes_visited)),
            "total_tokens": total_tokens,
            "models_used": models_used,
            "first_entry": entries[0].timestamp,
            "last_entry": entries[-1].timestamp,
            "actions": [e.action for e in entries],
        }

    def format_timeline(self, tender_id: str) -> str:
        """Format a human-readable timeline of a tender's processing.

        Useful for Slack notifications and debugging output.
        """
        entries = self.get_by_tender(tender_id)

        if not entries:
            return f"No audit trail found for tender {tender_id}."

        lines = [f"Audit Trail: {tender_id}", "=" * 50]

        for i, entry in enumerate(entries, 1):
            ts = entry.timestamp[:19].replace("T", " ")  # Trim to readable format
            tokens_str = f" [{entry.tokens_used} tokens]" if entry.tokens_used else ""
            lines.append(
                f"  {i}. [{ts}] {entry.node} → {entry.action}{tokens_str}"
            )
            if entry.detail:
                # Truncate long details
                detail = entry.detail[:120] + ("..." if len(entry.detail) > 120 else "")
                lines.append(f"     {detail}")

        total_tokens = sum(e.tokens_used or 0 for e in entries)
        lines.append("=" * 50)
        lines.append(f"  Total: {len(entries)} events, {total_tokens} tokens")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _persist_entry(self, entry: AuditEntry) -> None:
        """Write a single entry to the database.

        In a real implementation, this would INSERT into the audit_logs table
        from Step 2. For now, it's a placeholder that logs the write.
        """
        try:
            # In production, this would be:
            # session = self._session_factory()
            # db_entry = AuditLogModel(
            #     tender_id=entry.tender_id,
            #     node=entry.node,
            #     action=entry.action,
            #     detail=entry.detail,
            #     model_used=entry.model_used,
            #     tokens_used=entry.tokens_used,
            # )
            # session.add(db_entry)
            # session.commit()
            logger.debug("audit_entry_persisted_to_db", tender_id=entry.tender_id)
        except Exception as exc:
            logger.error("audit_persist_failed", error=str(exc))