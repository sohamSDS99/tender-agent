"""
Pydantic models for Nexus SDK
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration payload sent during agent registration."""

    name: str = Field(..., description="Machine name (e.g. 'tender-agent')")
    display_name: str = Field(..., description="Human-readable name")
    description: Optional[str] = None
    version: str = "1.0.0"
    python_version: Optional[str] = None
    langgraph_version: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_models: Optional[dict[str, Any]] = None
    llm_pricing: Optional[dict[str, Any]] = None
    embedding_model: Optional[str] = None
    embedding_dimensions: Optional[int] = None
    state_fields_count: Optional[int] = None
    node_names: Optional[list[str]] = None
    tools: Optional[list[str]] = None
    health_endpoint: Optional[str] = None
    webhook_url: Optional[str] = None
    slack_channels: Optional[list[str]] = None
    db_connection: Optional[str] = None
    s3_bucket: Optional[str] = None
    env_vars_count: Optional[int] = None
    dry_run: bool = False
    budget_monthly_usd: Optional[float] = None
    tags: list[str] = Field(default_factory=list)
    changelog: Optional[str] = None


class MetricEvent(BaseModel):
    """A single metric data point."""

    agent_id: str
    metric_type: str = Field(
        ...,
        description=(
            "One of: token_usage, cost, latency, throughput, error_rate, "
            "task_completion, conversation_volume, tool_calls, model_distribution, "
            "uptime, queue_depth, user_satisfaction"
        ),
    )
    value: float
    metadata: Optional[dict[str, Any]] = None
    timestamp: Optional[datetime] = None


class HeartbeatPayload(BaseModel):
    """Payload for heartbeat calls."""

    agent_id: str
    status: Optional[str] = None  # running | degraded | stopped | error


class ChatMessage(BaseModel):
    """A chat message sent from agent to a Nexus thread."""

    thread_id: str
    content: str
    metadata: Optional[dict[str, Any]] = None


class RegisterResponse(BaseModel):
    """Response from agent registration."""

    agent_id: str = Field(..., alias="agentId")
    name: str
    version: str

    model_config = {"populate_by_name": True}


class IngestResponse(BaseModel):
    """Response from metric ingestion."""

    inserted: int
