"""
Tender database model.

Represents a discovered tender and tracks it through the entire pipeline:
discovery → evaluation → drafting → gap check → assembly → submission.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    Integer,
    String,
    Text,
    Boolean,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin


class TenderStatus(PyEnum):
    """
    Tracks where a tender is in the pipeline.

    The agent moves tenders through these stages sequentially.
    If a tender is rejected at evaluation, it goes to ARCHIVED.
    """

    DISCOVERED = "discovered"
    EVALUATING = "evaluating"
    ELIGIBLE = "eligible"
    ARCHIVED = "archived"
    DRAFTING = "drafting"
    GAP_CHECK = "gap_check"
    ESCALATED = "escalated"
    ASSEMBLING = "assembling"
    READY_TO_SUBMIT = "ready_to_submit"
    SUBMITTED = "submitted"
    FAILED = "failed"


class Tender(TimestampMixin, Base):
    """
    Core tender table. One row per tender.
    """

    __tablename__ = "tenders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # --- Identity ---
    external_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    url: Mapped[str | None] = mapped_column(String(2000), nullable=True)

    # --- Status ---
    status: Mapped[TenderStatus] = mapped_column(
        Enum(TenderStatus, name="tender_status"),
        default=TenderStatus.DISCOVERED,
        nullable=False,
    )

    # --- Deadlines ---
    submission_deadline: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # --- Evaluation ---
    evaluation_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    evaluation_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    evaluation_dimensions: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # --- Content ---
    raw_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    requirements: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # --- Drafting ---
    drafted_sections: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    gaps: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # --- Submission ---
    submission_method: Mapped[str | None] = mapped_column(String(50), nullable=True)
    submission_confirmation: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # --- Metadata ---
    estimated_value: Mapped[str | None] = mapped_column(String(100), nullable=True)
    agency: Mapped[str | None] = mapped_column(String(500), nullable=True)
    category: Mapped[str | None] = mapped_column(String(255), nullable=True)
    naics_code: Mapped[str | None] = mapped_column(String(10), nullable=True)

    # --- Cost Tracking ---
    total_tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    def __repr__(self) -> str:
        return (
            f"<Tender(id={self.id}, external_id='{self.external_id}', "
            f"title='{self.title[:50]}...', status={self.status.value})>"
        )


class AuditLog(TimestampMixin, Base):
    """
    Immutable audit trail for every agent action.
    Rows are INSERT-only. Never update or delete audit logs.
    """

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    tender_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    node_name: Mapped[str] = mapped_column(String(50), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False)

    input_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    decision: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    model_used: Mapped[str | None] = mapped_column(String(100), nullable=True)
    tokens_input: Mapped[int] = mapped_column(Integer, default=0)
    tokens_output: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)

    success: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<AuditLog(id={self.id}, node='{self.node_name}', "
            f"action='{self.action}', tender_id={self.tender_id})>"
        )