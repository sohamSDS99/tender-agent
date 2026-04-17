"""
LangSmith Integration — LLM tracing and cost tracking.

WHAT LANGSMITH DOES:
LangSmith records every LLM call in the pipeline, capturing:
- The exact prompt sent (system + user messages)
- The model's response
- Latency (how long the call took)
- Token counts (input + output)
- Cost per call
- The full chain/graph context (which node triggered the call)

This data appears in a web dashboard at smith.langchain.com where you can:
- Debug bad responses (see exactly what the model saw)
- Track costs per tender, per node, per model
- Monitor latency trends
- Compare model performance across tenders

HOW TRACING WORKS:
LangSmith tracing is configured via environment variables. When these are set,
LangChain/LangGraph automatically sends trace data to LangSmith — no code
changes needed in the node implementations. This module handles:

1. CONFIGURATION — Sets the right env vars and validates the setup
2. COST TRACKING — Aggregates token usage and estimates costs per model
3. REPORTING — Generates cost summaries for budgeting

SETUP:
1. Create a free account at https://smith.langchain.com
2. Create a new project (e.g., "tender-agent")
3. Get your API key from Settings → API Keys
4. Set in .env:
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=ls-...
   LANGCHAIN_PROJECT=tender-agent

DRY-RUN MODE:
Tracing is disabled (LANGCHAIN_TRACING_V2=false). The cost tracker still
works using in-memory data from the audit logger.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Model pricing (per 1M tokens, as of 2026)
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, dict[str, float]] = {
    "qwen3.5-flash": {"input": 0.07, "output": 0.26},
    "qwen3.5-plus":  {"input": 0.26, "output": 1.56},
    "qwen3-max":     {"input": 0.78, "output": 3.90},
    "voyage-3-large": {"input": 0.06, "output": 0.0},
    # Dry-run variants (same pricing for estimation)
    "qwen3.5-flash-dry-run": {"input": 0.07, "output": 0.26},
    "qwen3.5-plus-dry-run":  {"input": 0.26, "output": 1.56},
    "qwen3-max-dry-run":     {"input": 0.78, "output": 3.90},
    "voyage-3-large-dry-run": {"input": 0.06, "output": 0.0},
}

# Default pricing for unknown models
DEFAULT_PRICING: dict[str, float] = {"input": 3.00, "output": 15.00}


# ---------------------------------------------------------------------------
# LangSmith configuration
# ---------------------------------------------------------------------------

def configure_langsmith(
    api_key: str | None = None,
    project: str | None = None,
    enabled: bool | None = None,
) -> dict[str, str]:
    """Configure LangSmith tracing via environment variables.

    LangChain/LangGraph reads these env vars automatically — once set,
    all LLM calls are traced without any code changes in the nodes.

    Args:
        api_key: LangSmith API key. Reads from LANGCHAIN_API_KEY if None.
        project: Project name. Reads from LANGCHAIN_PROJECT if None.
        enabled: Whether to enable tracing. If None, reads from
            LANGCHAIN_TRACING_V2 env var. Defaults to False if DRY_RUN=true.

    Returns:
        Dict of the env vars that were set (for logging/verification).
    """
    dry_run = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")

    # Resolve enabled state
    if enabled is None:
        env_val = os.getenv("LANGCHAIN_TRACING_V2", "")
        if env_val:
            enabled = env_val.lower() in ("true", "1", "yes")
        else:
            enabled = not dry_run  # Auto-enable when not in dry-run

    # Resolve API key and project
    resolved_key = api_key or os.getenv("LANGCHAIN_API_KEY", "")
    resolved_project = project or os.getenv("LANGCHAIN_PROJECT", "tender-agent")

    # Validate
    if enabled and not resolved_key:
        logger.warning(
            "langsmith_no_api_key",
            message="LANGCHAIN_API_KEY not set — disabling tracing.",
        )
        enabled = False

    # Set environment variables
    config = {
        "LANGCHAIN_TRACING_V2": str(enabled).lower(),
        "LANGCHAIN_PROJECT": resolved_project,
    }

    if resolved_key:
        config["LANGCHAIN_API_KEY"] = resolved_key

    # Optional: Set the endpoint (default is correct for most users)
    config["LANGCHAIN_ENDPOINT"] = os.getenv(
        "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
    )

    for key, value in config.items():
        os.environ[key] = value

    logger.info(
        "langsmith_configured",
        enabled=enabled,
        project=resolved_project,
        has_api_key=bool(resolved_key),
    )

    return config


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

@dataclass
class ModelUsage:
    """Token usage for a single model."""
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost(self) -> float:
        pricing = MODEL_PRICING.get(self.model, DEFAULT_PRICING)
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


@dataclass
class CostReport:
    """Cost summary across all models and tenders."""
    total_cost: float = 0.0
    total_tokens: int = 0
    total_calls: int = 0
    by_model: dict[str, ModelUsage] = field(default_factory=dict)
    by_tender: dict[str, float] = field(default_factory=dict)
    period: str = ""


class CostTracker:
    """Tracks LLM token usage and estimates costs.

    Usage:
        tracker = CostTracker()

        # Record usage from a node
        tracker.record(
            tender_id="SAM-2026-001",
            model="qwen3.5-plus",
            input_tokens=1500,
            output_tokens=800,
        )

        # Get cost report
        report = tracker.get_report()
        print(f"Total cost: ${report.total_cost:.4f}")

        # Get cost for a specific tender
        tender_cost = tracker.get_tender_cost("SAM-2026-001")

    Also supports extracting usage from audit log entries.
    """

    def __init__(self) -> None:
        self._usage: list[dict[str, Any]] = []
        logger.info("cost_tracker_initialized")

    def record(
        self,
        tender_id: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> float:
        """Record a single LLM call's token usage.

        Args:
            tender_id: Which tender this call was for.
            model: Model identifier.
            input_tokens: Input token count.
            output_tokens: Output token count.

        Returns:
            Estimated cost of this single call in USD.
        """
        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
        cost = (
            (input_tokens / 1_000_000) * pricing["input"]
            + (output_tokens / 1_000_000) * pricing["output"]
        )

        self._usage.append({
            "tender_id": tender_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return cost

    def record_from_audit(
        self,
        tender_id: str,
        audit_log: list[dict[str, Any]],
    ) -> float:
        """Extract and record token usage from audit log entries.

        Scans audit entries for tokens_used and model_used fields.
        Estimates a 70/30 input/output split when only total tokens
        are available (this is typical for most LLM calls).

        Args:
            tender_id: Tender ID.
            audit_log: The audit_log list from TenderState.

        Returns:
            Total estimated cost from these entries.
        """
        total_cost = 0.0

        for entry in audit_log:
            tokens = entry.get("tokens_used") or 0
            model = entry.get("model_used") or ""

            if tokens > 0 and model:
                # Estimate 70% input, 30% output
                input_t = int(tokens * 0.7)
                output_t = tokens - input_t
                cost = self.record(tender_id, model, input_t, output_t)
                total_cost += cost

        return total_cost

    def get_tender_cost(self, tender_id: str) -> float:
        """Get total estimated cost for a specific tender."""
        return sum(u["cost"] for u in self._usage if u["tender_id"] == tender_id)

    def get_report(self) -> CostReport:
        """Generate a comprehensive cost report.

        Returns:
            CostReport with totals by model and by tender.
        """
        report = CostReport()
        report.period = f"All time ({len(self._usage)} calls)"

        for u in self._usage:
            model = u["model"]
            tender = u["tender_id"]

            report.total_cost += u["cost"]
            report.total_tokens += u["input_tokens"] + u["output_tokens"]
            report.total_calls += 1

            if model not in report.by_model:
                report.by_model[model] = ModelUsage(model=model)
            report.by_model[model].input_tokens += u["input_tokens"]
            report.by_model[model].output_tokens += u["output_tokens"]
            report.by_model[model].call_count += 1

            report.by_tender[tender] = report.by_tender.get(tender, 0.0) + u["cost"]

        return report

    def format_report(self) -> str:
        """Format a human-readable cost report."""
        report = self.get_report()

        lines = [
            "Cost Tracking Report",
            "=" * 50,
            f"Total calls: {report.total_calls}",
            f"Total tokens: {report.total_tokens:,}",
            f"Total cost: ${report.total_cost:.4f}",
            "",
            "By Model:",
        ]

        for model, usage in sorted(report.by_model.items()):
            lines.append(
                f"  {model}: {usage.call_count} calls, "
                f"{usage.total_tokens:,} tokens, ${usage.estimated_cost:.4f}"
            )

        lines.append("")
        lines.append("By Tender:")
        for tender, cost in sorted(report.by_tender.items()):
            lines.append(f"  {tender}: ${cost:.4f}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def check_budget(self, monthly_budget: float = 300.0) -> dict[str, Any]:
        """Check if current spend is within budget.

        Args:
            monthly_budget: Monthly LLM budget in USD. Default $300.

        Returns:
            Dict with budget status, remaining amount, and warning flag.
        """
        report = self.get_report()
        remaining = monthly_budget - report.total_cost
        pct_used = (report.total_cost / monthly_budget * 100) if monthly_budget else 0

        status = {
            "budget": monthly_budget,
            "spent": round(report.total_cost, 4),
            "remaining": round(remaining, 4),
            "percent_used": round(pct_used, 1),
            "warning": pct_used >= 80,
            "exceeded": pct_used >= 100,
        }

        if status["warning"]:
            logger.warning("budget_warning", **status)

        return status