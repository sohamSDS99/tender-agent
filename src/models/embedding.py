"""
Document embedding model for the RAG knowledge base.

Each row represents a chunk of a company document that has been
embedded as a vector for similarity search.
"""

from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin


EMBEDDING_DIMENSION = 1024  # Voyage AI voyage-3-large produces 1024-dimensional vectors


class DocumentChunk(TimestampMixin, Base):
    """
    A chunk of a company document with its vector embedding.

    The RAG pipeline:
    1. Company docs are split into chunks
    2. Each chunk is embedded into a 1024-dim vector using Voyage AI
    3. When drafting, the tender requirement is converted to a vector
    4. pgvector finds the closest chunks
    5. Those chunks are fed to Sonnet/Opus as context
    """

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    source_file: Mapped[str] = mapped_column(String(500), nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    embedding: Mapped[list] = mapped_column(
        Vector(EMBEDDING_DIMENSION), nullable=False
    )

    chunk_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk(id={self.id}, source='{self.source_file}', "
            f"chunk={self.chunk_index}, tokens={self.token_count})>"
        )