"""
Cost Tracking Dashboard — CLI reporting for agent activity and costs.

WHAT THIS PROVIDES:
1. COST SUMMARY — Total spend, spend by model, spend by tender
2. PIPELINE STATUS — How many tenders discovered, evaluated, submitted
3. BUDGET MONITOR — Current spend vs monthly budget with warnings
4. ACTIVITY LOG — Recent actions across all tenders

HOW IT INTEGRATES:
After each graph run, call dashboard.record_run() with the completed state.
The dashboard accumulates data across runs and generates reports on demand.

In production, this can also be called by a cron job to generate daily/weekly
email reports, or exposed as a simple web endpoint.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from src.utils.audit_logger import AuditLogger
from src.utils.langsmith_config import CostTracker

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline statistics
# ---------------------------------------------------------------------------

@dataclass
class PipelineStats:
    """Aggregate statistics across all tenders."""
    total_tenders: int = 0
    tenders_by_status: dict[str, int] = field(default_factory=dict)
    total_sections_drafted: int = 0
    total_gaps_found: int = 0
    total_escalations: int = 0
    avg_eval_score: float = 0.0
    models_used: Counter = field(default_factory=Counter)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class CostDashboard:
    """CLI dashboard for monitoring agent activity and costs.

    Usage:
        dashboard = CostDashboard()

        # Record a completed graph run
        dashboard.record_run(result_state)

        # Print reports
        print(dashboard.format_summary())
        print(dashboard.format_cost_report())
        print(dashboard.format_pipeline_report())

        # Check budget
        status = dashboard.get_budget_status()
        if status["warning"]:
            print("Budget warning!")

    Args:
        monthly_budget: Monthly LLM spend limit in USD.
    """

    def __init__(self, monthly_budget: float = 300.0) -> None:
        self.monthly_budget = monthly_budget
        self.audit = AuditLogger()
        self.costs = CostTracker()
        self._runs: list[dict[str, Any]] = []

        logger.info("cost_dashboard_initialized", budget=monthly_budget)

    def record_run(self, state: dict[str, Any]) -> None:
        """Record a completed graph run for reporting.

        Call this after each graph.invoke() with the result state.

        Args:
            state: The completed TenderState dict from graph.invoke().
        """
        tender_id = state.get("tender_id", "unknown")
        audit_log = state.get("audit_log", [])

        # Persist audit entries
        self.audit.persist_from_state(tender_id, audit_log)

        # Extract cost data
        self.costs.record_from_audit(tender_id, audit_log)

        # Store run summary
        self._runs.append({
            "tender_id": tender_id,
            "tender_title": state.get("tender_title", ""),
            "status": state.get("status", "unknown"),
            "eval_score": state.get("eval_score", 0),
            "eval_decision": state.get("eval_decision", ""),
            "sections_drafted": len(state.get("drafted_sections", [])),
            "gaps_found": len(state.get("gaps", [])),
            "escalation_count": state.get("escalation_count", 0),
            "submission_method": state.get("submission_method", ""),
            "submission_confirmation": state.get("submission_confirmation", ""),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        })

        logger.info(
            "run_recorded",
            tender_id=tender_id,
            status=state.get("status"),
        )

    def get_pipeline_stats(self) -> PipelineStats:
        """Calculate aggregate pipeline statistics."""
        stats = PipelineStats()
        stats.total_tenders = len(self._runs)

        scores = []
        for run in self._runs:
            status = run.get("status", "unknown")
            stats.tenders_by_status[status] = stats.tenders_by_status.get(status, 0) + 1
            stats.total_sections_drafted += run.get("sections_drafted", 0)
            stats.total_gaps_found += run.get("gaps_found", 0)
            stats.total_escalations += run.get("escalation_count", 0)

            score = run.get("eval_score", 0)
            if score > 0:
                scores.append(score)

        stats.avg_eval_score = sum(scores) / len(scores) if scores else 0.0

        # Count models from audit entries
        for entry in self.audit._memory_log:
            if entry.model_used:
                stats.models_used[entry.model_used] += 1

        return stats

    def get_budget_status(self) -> dict[str, Any]:
        """Get current budget status."""
        return self.costs.check_budget(self.monthly_budget)

    # ------------------------------------------------------------------
    # Formatted reports
    # ------------------------------------------------------------------

    def format_summary(self) -> str:
        """Format a one-page executive summary."""
        stats = self.get_pipeline_stats()
        budget = self.get_budget_status()
        cost_report = self.costs.get_report()

        lines = [
            "",
            "=" * 60,
            "  AI TENDER AGENT — DASHBOARD SUMMARY",
            "=" * 60,
            "",
            f"  Tenders Processed:  {stats.total_tenders}",
            f"  Submitted:          {stats.tenders_by_status.get('submitted', 0)}",
            f"  Rejected:           {stats.tenders_by_status.get('rejected', 0)}",
            f"  Failed:             {stats.tenders_by_status.get('submission_failed', 0)}",
            f"  Avg Eval Score:     {stats.avg_eval_score:.0f}/100",
            "",
            f"  Sections Drafted:   {stats.total_sections_drafted}",
            f"  Gaps Found:         {stats.total_gaps_found}",
            f"  Slack Escalations:  {stats.total_escalations}",
            "",
            "  --- COST ---",
            f"  Total Spend:        ${budget['spent']:.4f}",
            f"  Budget:             ${budget['budget']:.2f}/month",
            f"  Remaining:          ${budget['remaining']:.4f}",
            f"  Used:               {budget['percent_used']:.1f}%",
        ]

        if budget["warning"]:
            lines.append(f"  ⚠️  BUDGET WARNING: {budget['percent_used']:.0f}% used!")
        if budget["exceeded"]:
            lines.append(f"  🔴 BUDGET EXCEEDED by ${abs(budget['remaining']):.2f}!")

        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def format_cost_report(self) -> str:
        """Format a detailed cost breakdown."""
        return self.costs.format_report()

    def format_pipeline_report(self) -> str:
        """Format a per-tender pipeline report."""
        if not self._runs:
            return "No tenders processed yet."

        lines = [
            "",
            "Tender Pipeline Report",
            "=" * 70,
            f"{'Tender ID':<20} {'Score':>5} {'Decision':>8} {'Sections':>8} "
            f"{'Gaps':>4} {'Status':>12} {'Cost':>10}",
            "-" * 70,
        ]

        for run in self._runs:
            tid = run["tender_id"][:18]
            score = run.get("eval_score", 0)
            decision = run.get("eval_decision", "?")
            sections = run.get("sections_drafted", 0)
            gaps = run.get("gaps_found", 0)
            status = run.get("status", "?")
            cost = self.costs.get_tender_cost(run["tender_id"])

            lines.append(
                f"  {tid:<20} {score:>5} {decision:>8} {sections:>8} "
                f"{gaps:>4} {status:>12} ${cost:>9.4f}"
            )

        lines.extend([
            "-" * 70,
            f"  Total: {len(self._runs)} tenders, "
            f"${self.costs.get_report().total_cost:.4f} total cost",
            "=" * 70,
        ])

        return "\n".join(lines)

    def format_recent_activity(self, max_entries: int = 20) -> str:
        """Format recent audit activity across all tenders."""
        recent = self.audit.get_recent(hours=24)[-max_entries:]

        if not recent:
            return "No activity in the last 24 hours."

        lines = ["Recent Activity (last 24h)", "-" * 50]

        for entry in recent:
            ts = entry.timestamp[:16].replace("T", " ")
            tokens_str = f" [{entry.tokens_used}t]" if entry.tokens_used else ""
            lines.append(
                f"  [{ts}] {entry.tender_id} / {entry.node} → {entry.action}{tokens_str}"
            )

        lines.append(f"\n  Total: {len(recent)} events")
        return "\n".join(lines)