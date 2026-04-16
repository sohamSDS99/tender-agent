"""
RAG Retriever — Searches the knowledge base for chunks relevant to a query.

WHY THIS EXISTS:
When the tender drafting node encounters a requirement like "Describe your company's
experience with GHS classification," it needs to find the specific chunks of company
data that answer that question. This module takes the query, converts it to a vector
(using the same embedder from Step 6), and uses pgvector's cosine similarity search
to find the closest matching chunks in the database.

HOW PGVECTOR SIMILARITY SEARCH WORKS:
pgvector adds a special column type (vector) and operators to PostgreSQL. The key
operator is `<=>` (cosine distance), which computes how "far apart" two vectors are.
Cosine distance = 1 - cosine_similarity, so:
  - Distance 0.0 = identical meaning (perfect match)
  - Distance 1.0 = completely unrelated
  - Distance 2.0 = opposite meaning

We order results by ascending distance (closest first) and return the top K matches.
pgvector uses an IVFFlat or HNSW index for fast approximate nearest neighbour search —
even with 100,000 chunks, queries return in milliseconds.

KEY DESIGN DECISIONS:
- Lives in `src/knowledge/` (not `src/ingestion/`) because retrieval is a separate
  concern from ingestion. Ingestion puts data IN; retrieval gets data OUT.
- Returns RetrievedChunk objects with both the text AND similarity score, so the
  drafting LLM can weight high-confidence results more heavily.
- Supports metadata filtering (e.g., "only search company_profile documents" or
  "only search past_tender responses") to narrow results for specific tender sections.
- Has a dry-run mode that works without a database by searching in-memory chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from src.ingestion.embedder import VoyageEmbedder
from src.models.embedding import DocumentChunk

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """A single search result from the knowledge base.

    Attributes:
        content: The chunk text.
        source_file: Original document filename.
        page_number: Page/section number in the source document.
        section_heading: Heading of the section (if available).
        chunk_metadata: Additional metadata dict.
        similarity_score: Cosine similarity (0 to 1). Higher = more relevant.
            1.0 = perfect match, 0.5 = moderate relevance, <0.3 = weak match.
        chunk_id: Database ID of the chunk (for audit trail).
    """
    content: str
    source_file: str
    page_number: int
    section_heading: str | None
    chunk_metadata: dict[str, Any]
    similarity_score: float
    chunk_id: int | None = None

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return (
            f"RetrievedChunk(score={self.similarity_score:.3f}, "
            f"source='{self.source_file}', preview='{preview}...')"
        )


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class KnowledgeRetriever:
    """Searches the pgvector knowledge base for relevant chunks.

    Usage:
        from src.models.base import SessionLocal
        from src.knowledge.retriever import KnowledgeRetriever

        retriever = KnowledgeRetriever()

        with SessionLocal() as session:
            results = retriever.search(
                query="What ISO certifications does the company hold?",
                session=session,
                top_k=5,
            )
            for chunk in results:
                print(f"[{chunk.similarity_score:.3f}] {chunk.source_file}: {chunk.content[:100]}")

    Args:
        embedder: A VoyageEmbedder instance for converting queries to vectors.
            If None, creates one using env vars (respects DRY_RUN).
        default_top_k: Default number of results to return. Can be overridden
            per query. Default 5 — enough context for the LLM without
            overwhelming it with too many chunks.
        min_similarity: Minimum similarity score to include in results.
            Chunks below this threshold are filtered out even if they're
            in the top K. Default 0.3 — a reasonable cutoff that excludes
            clearly irrelevant results. Set to 0.0 to disable filtering.
    """

    def __init__(
        self,
        embedder: VoyageEmbedder | None = None,
        default_top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> None:
        self.embedder = embedder or VoyageEmbedder()
        self.default_top_k = default_top_k
        self.min_similarity = min_similarity

        logger.info(
            "retriever_initialized",
            default_top_k=self.default_top_k,
            min_similarity=self.min_similarity,
            dry_run=self.embedder.dry_run,
        )

    def search(
        self,
        query: str,
        session: Session,
        top_k: int | None = None,
        min_similarity: float | None = None,
        source_filter: str | None = None,
        category_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Search the knowledge base for chunks relevant to a query.

        HOW IT WORKS:
        1. Convert the query text to a 1024-dim vector using VoyageEmbedder.
        2. Run a pgvector cosine similarity search against the document_chunks table.
        3. Filter by minimum similarity and optional metadata filters.
        4. Return the top K results as RetrievedChunk objects.

        Args:
            query: The search query (e.g., a tender requirement).
            session: SQLAlchemy session.
            top_k: Number of results to return. Overrides default_top_k.
            min_similarity: Minimum similarity. Overrides instance default.
            source_filter: If set, only return chunks from this source file.
                Example: "company_profile.pdf"
            category_filter: If set, only return chunks with this category
                in their metadata. Example: "certification"

        Returns:
            List of RetrievedChunk objects, sorted by similarity (highest first).
        """
        k = top_k or self.default_top_k
        threshold = min_similarity if min_similarity is not None else self.min_similarity

        logger.info(
            "search_start",
            query_preview=query[:100],
            top_k=k,
            min_similarity=threshold,
            source_filter=source_filter,
            category_filter=category_filter,
        )

        # Step 1: Embed the query
        query_vector = self.embedder.embed_single(query)

        # Step 2: Build the pgvector similarity query
        # We use raw SQL here because SQLAlchemy's ORM doesn't natively
        # support pgvector's <=> (cosine distance) operator. The text()
        # construct lets us write the query directly while still using
        # parameterised inputs (safe from SQL injection).
        #
        # COSINE DISTANCE vs COSINE SIMILARITY:
        # pgvector's <=> returns cosine DISTANCE (0 = identical, 2 = opposite).
        # We convert to SIMILARITY: similarity = 1 - distance.
        # This makes results more intuitive: higher = better.

        # Build WHERE clauses dynamically based on filters
        where_clauses = ["embedding IS NOT NULL"]
        params: dict[str, Any] = {
            "query_vector": str(query_vector),
            "limit": k * 2,  # Fetch extra to account for post-filtering
        }

        if source_filter:
            where_clauses.append("source_file = :source_filter")
            params["source_filter"] = source_filter

        if category_filter:
            where_clauses.append("chunk_metadata->>'category' = :category_filter")
            params["category_filter"] = category_filter

        where_sql = " AND ".join(where_clauses)

        sql = text(f"""
            SELECT
                id,
                content,
                source_file,
                page_number,
                chunk_metadata,
                1 - (embedding <=> :query_vector::vector) AS similarity
            FROM document_chunks
            WHERE {where_sql}
            ORDER BY embedding <=> :query_vector::vector ASC
            LIMIT :limit
        """)

        rows = session.execute(sql, params).fetchall()

        # Step 3: Convert to RetrievedChunk objects and filter by threshold
        results: list[RetrievedChunk] = []
        for row in rows:
            similarity = float(row.similarity)
            if similarity < threshold:
                continue

            metadata = row.chunk_metadata or {}
            results.append(
                RetrievedChunk(
                    content=row.content,
                    source_file=row.source_file,
                    page_number=row.page_number,
                    section_heading=metadata.get("section_heading"),
                    chunk_metadata=metadata,
                    similarity_score=round(similarity, 4),
                    chunk_id=row.id,
                )
            )

            # Stop once we have enough results
            if len(results) >= k:
                break

        logger.info(
            "search_complete",
            query_preview=query[:60],
            results_found=len(results),
            top_score=results[0].similarity_score if results else 0.0,
        )

        return results

    def search_multi(
        self,
        queries: list[str],
        session: Session,
        top_k: int | None = None,
        deduplicate: bool = True,
    ) -> list[RetrievedChunk]:
        """Search with multiple queries and combine results.

        WHY MULTI-QUERY:
        A single tender requirement might need information from different
        angles. For example, "Describe your experience with OSHA compliance
        for manufacturing clients" could benefit from searching:
          - "OSHA compliance experience"
          - "manufacturing industry clients"
          - "regulatory compliance case studies"

        This method runs each query, combines the results, deduplicates by
        chunk ID, and re-ranks by the highest similarity score.

        Args:
            queries: List of query strings.
            session: SQLAlchemy session.
            top_k: Total results to return (after deduplication).
            deduplicate: If True, remove duplicate chunks (same chunk_id).
                Keeps the instance with the highest similarity score.

        Returns:
            Combined, deduplicated, and re-ranked list of RetrievedChunk objects.
        """
        k = top_k or self.default_top_k
        all_results: list[RetrievedChunk] = []

        for query in queries:
            results = self.search(query, session, top_k=k)
            all_results.extend(results)

        if deduplicate and all_results:
            # Keep the highest-scoring instance of each chunk
            best_by_id: dict[int | None, RetrievedChunk] = {}
            for chunk in all_results:
                key = chunk.chunk_id
                if key not in best_by_id or chunk.similarity_score > best_by_id[key].similarity_score:
                    best_by_id[key] = chunk
            all_results = list(best_by_id.values())

        # Sort by similarity (highest first) and take top K
        all_results.sort(key=lambda c: c.similarity_score, reverse=True)
        return all_results[:k]

    def format_context(
        self,
        chunks: list[RetrievedChunk],
        max_chars: int = 8000,
    ) -> str:
        """Format retrieved chunks into a context string for the LLM.

        This is what gets injected into the drafting prompt. Each chunk is
        wrapped with source attribution so the LLM can reference where
        information came from.

        Args:
            chunks: List of RetrievedChunk objects from search().
            max_chars: Maximum total characters in the context string.
                Default 8000 (~2000 tokens) — fits comfortably in the LLM
                prompt alongside the tender requirement and system instructions.

        Returns:
            Formatted context string ready to paste into an LLM prompt.
        """
        if not chunks:
            return "[No relevant information found in the knowledge base.]"

        parts: list[str] = []
        total_chars = 0

        for i, chunk in enumerate(chunks, start=1):
            source_label = chunk.source_file
            if chunk.section_heading:
                source_label += f" > {chunk.section_heading}"

            header = f"[Source {i}: {source_label} | Relevance: {chunk.similarity_score:.0%}]"
            entry = f"{header}\n{chunk.content}"

            # Check if adding this chunk would exceed the limit
            if total_chars + len(entry) + 2 > max_chars:
                # Add a truncation notice
                parts.append(f"\n[... {len(chunks) - i + 1} more results truncated for brevity]")
                break

            parts.append(entry)
            total_chars += len(entry) + 2  # +2 for the separator newlines

        return "\n\n".join(parts)