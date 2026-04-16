"""
Embedding Pipeline — Reads unembedded chunks from PostgreSQL, computes vectors, stores them.

WHY THIS IS SEPARATE FROM THE INGESTION PIPELINE:
Step 5's IngestionPipeline handles parse → chunk → store (text only).
This pipeline handles the second pass: read stored chunks → embed → update with vectors.

Keeping them separate gives us important flexibility:
1. Re-embedding: If we switch from voyage-3-large to a newer model, we re-run just this
   pipeline — no need to re-parse documents.
2. Cost control: Embedding costs money ($0.06/1M tokens). Running it as a separate step
   lets you verify chunks look correct before spending money on embeddings.
3. Resumability: If embedding crashes halfway (API timeout, rate limit), the pipeline
   picks up where it left off — it only processes chunks where `embedding IS NULL`.

HOW IT WORKS:
1. Query DocumentChunk table for rows where embedding is NULL.
2. Batch those chunk texts (128 at a time for Voyage AI efficiency).
3. Call VoyageEmbedder to get vectors.
4. Update each row with its embedding vector.
5. Commit in batches to avoid long transactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.ingestion.embedder import EMBEDDING_DIMENSIONS, VoyageEmbedder
from src.models.embedding import DocumentChunk

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class EmbedPipelineResult:
    """Tracks what the embedding pipeline did.

    Attributes:
        chunks_found: Number of unembedded chunks in the database.
        chunks_embedded: Number of chunks that got embeddings.
        chunks_failed: Number of chunks that failed to embed.
        total_tokens: Total tokens sent to the embedding API.
        estimated_cost_usd: Estimated API cost.
        errors: List of error descriptions.
    """
    chunks_found: int = 0
    chunks_embedded: int = 0
    chunks_failed: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks_found": self.chunks_found,
            "chunks_embedded": self.chunks_embedded,
            "chunks_failed": self.chunks_failed,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "errors": self.errors,
        }

    def __repr__(self) -> str:
        return (
            f"EmbedPipelineResult(found={self.chunks_found}, "
            f"embedded={self.chunks_embedded}, "
            f"cost=${self.estimated_cost_usd:.4f})"
        )


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class EmbeddingPipeline:
    """Embeds all unembedded chunks in the database.

    Usage:
        from src.models.base import SessionLocal
        from src.ingestion.embed_pipeline import EmbeddingPipeline

        pipeline = EmbeddingPipeline()  # uses DRY_RUN from env

        with SessionLocal() as session:
            result = pipeline.embed_all(session)
            print(result)

    Args:
        embedder: A VoyageEmbedder instance. If None, creates one using env vars.
        batch_size: Number of chunks to embed per API call. Default 64.
            We use 64 instead of 128 (the Voyage AI max) to keep memory
            usage and request payloads manageable. Each chunk might be
            2000 characters, so 64 chunks = ~128KB payload — comfortable.
    """

    def __init__(
        self,
        embedder: VoyageEmbedder | None = None,
        batch_size: int = 64,
    ) -> None:
        self.embedder = embedder or VoyageEmbedder()
        self.batch_size = batch_size

    def embed_all(self, session: Session) -> EmbedPipelineResult:
        """Find all unembedded chunks and compute their embeddings.

        Only processes chunks where the `embedding` column is NULL.
        This makes the pipeline idempotent — running it twice is safe
        because already-embedded chunks are skipped.

        Args:
            session: SQLAlchemy session (caller manages lifecycle).

        Returns:
            EmbedPipelineResult summarising what happened.
        """
        result = EmbedPipelineResult()

        # Step 1: Find all chunks that need embeddings
        unembedded_chunks = session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.embedding.is_(None))
            .order_by(DocumentChunk.id)
        ).scalars().all()

        result.chunks_found = len(unembedded_chunks)

        if result.chunks_found == 0:
            logger.info("no_unembedded_chunks", message="All chunks already have embeddings.")
            return result

        logger.info(
            "embedding_pipeline_start",
            chunks_to_embed=result.chunks_found,
            batch_size=self.batch_size,
        )

        # Step 2: Process in batches
        for batch_start in range(0, len(unembedded_chunks), self.batch_size):
            batch_chunks = unembedded_chunks[batch_start : batch_start + self.batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]

            try:
                # Step 3: Get embeddings from Voyage AI (or dry-run)
                embed_result = self.embedder.embed_texts(batch_texts)

                # Step 4: Update each chunk with its embedding
                for chunk, embedding in zip(batch_chunks, embed_result.embeddings):
                    chunk.embedding = embedding

                # Step 5: Commit this batch
                session.commit()

                result.chunks_embedded += len(batch_chunks)
                result.total_tokens += embed_result.token_count

                logger.info(
                    "batch_embedded",
                    batch_start=batch_start,
                    batch_size=len(batch_chunks),
                    tokens=embed_result.token_count,
                    progress=f"{result.chunks_embedded}/{result.chunks_found}",
                )

            except Exception as exc:
                # Roll back the failed batch but continue with the next one
                session.rollback()
                result.chunks_failed += len(batch_chunks)
                error_msg = f"Batch {batch_start}: {exc}"
                result.errors.append(error_msg)
                logger.error(
                    "batch_embedding_failed",
                    batch_start=batch_start,
                    error=str(exc),
                )

        # Calculate cost estimate
        result.estimated_cost_usd = self.embedder.get_cost_estimate()

        logger.info("embedding_pipeline_complete", **result.to_dict())
        return result

    def embed_texts_standalone(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts without database interaction.

        Useful for embedding search queries at retrieval time (Step 7).
        The RAG pipeline will call this to convert a user's question into
        a vector before searching pgvector.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (same order as input).
        """
        result = self.embedder.embed_texts(texts)
        return result.embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query.

        Convenience method for Step 7 (RAG retrieval). Embeds a query
        string and returns a single vector.

        Args:
            query: The search query text.

        Returns:
            A single 1024-dimensional embedding vector.
        """
        return self.embedder.embed_single(query)