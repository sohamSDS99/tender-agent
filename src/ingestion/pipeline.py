"""
Ingestion Pipeline — Orchestrates document parsing, chunking, and database storage.

WHY THIS EXISTS:
This is the glue between the parser, chunker, and database. It provides a single
entry point: give it a file or a directory, and it handles everything — parsing the
documents, splitting them into chunks, and storing those chunks in the PostgreSQL
`document_chunks` table (the DocumentChunk model from Step 2).

The chunks stored here will later get embeddings added by Step 6 (Embedding & Vector
Storage). The pipeline deliberately does NOT compute embeddings — that's a separate
concern because:
1. Embedding requires API calls (Voyage AI) which cost money and need rate limiting.
2. Separating ingestion from embedding lets us re-embed existing chunks if we switch
   embedding models, without re-parsing all documents.
3. We can ingest a batch of documents quickly, then embed them asynchronously.

KEY FEATURES:
- Deduplication: Uses content hashes to skip chunks that already exist in the DB.
  Re-running ingestion on the same directory is safe — it only adds new/changed content.
- Batch storage: Commits chunks in batches (default 50) to avoid holding open a
  transaction for thousands of rows.
- Directory scanning: Recursively finds all supported files in a directory.
- Dry-run mode: Parses and chunks without writing to DB — useful for testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from src.ingestion.chunker import TextChunk, TextChunker
from src.ingestion.parser import SUPPORTED_EXTENSIONS, DocumentParser, ParsedPage
from src.models.embedding import DocumentChunk

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Ingestion result tracking
# ---------------------------------------------------------------------------

class IngestionResult:
    """Tracks the outcome of an ingestion run.

    Provides a clear summary of what happened: how many files were processed,
    how many chunks were created, how many were new vs. duplicates, and any
    errors encountered. This is important for auditing — the tender agent's
    knowledge base needs to be traceable.
    """

    def __init__(self) -> None:
        self.files_processed: int = 0
        self.files_failed: int = 0
        self.pages_extracted: int = 0
        self.chunks_created: int = 0
        self.chunks_stored: int = 0
        self.chunks_skipped_duplicate: int = 0
        self.errors: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary summary for logging/display."""
        return {
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "pages_extracted": self.pages_extracted,
            "chunks_created": self.chunks_created,
            "chunks_stored": self.chunks_stored,
            "chunks_skipped_duplicate": self.chunks_skipped_duplicate,
            "errors": self.errors,
        }

    def __repr__(self) -> str:
        return (
            f"IngestionResult(files={self.files_processed}, "
            f"chunks_stored={self.chunks_stored}, "
            f"duplicates_skipped={self.chunks_skipped_duplicate}, "
            f"errors={len(self.errors)})"
        )


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class IngestionPipeline:
    """End-to-end document ingestion: parse → chunk → store.

    Usage:
        from src.models.base import SessionLocal
        from src.ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline()

        # Ingest a single file
        with SessionLocal() as session:
            result = pipeline.ingest_file("data/knowledge_base/overview.pdf", session)
            print(result)

        # Ingest an entire directory
        with SessionLocal() as session:
            result = pipeline.ingest_directory("data/knowledge_base", session)
            print(result)

        # Dry run (no database writes) — useful for testing
        result = pipeline.ingest_directory_dry_run("data/knowledge_base")
        print(result)

    Args:
        chunk_size: Target chunk size in characters. Default 2000.
        chunk_overlap: Overlap between chunks in characters. Default 200.
        batch_size: Number of chunks to commit per batch. Default 50.
        document_category: Optional category tag applied to all chunks
            (e.g., "company_profile", "past_tender", "certification").
            Stored in chunk metadata for filtered retrieval.
    """

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        batch_size: int = 50,
        document_category: str | None = None,
    ) -> None:
        self.parser = DocumentParser()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.batch_size = batch_size
        self.document_category = document_category

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_file(
        self,
        file_path: str | Path,
        session: Session,
    ) -> IngestionResult:
        """Ingest a single document file into the database.

        Args:
            file_path: Path to the document.
            session: SQLAlchemy session (caller manages the session lifecycle).

        Returns:
            IngestionResult with counts of what was processed.
        """
        result = IngestionResult()
        path = Path(file_path)

        try:
            # Step 1: Parse the document into pages
            pages = self.parser.parse(path)
            result.files_processed = 1
            result.pages_extracted = len(pages)

            # Step 2: Chunk the pages
            chunks = self.chunker.chunk_pages(pages)
            result.chunks_created = len(chunks)

            # Step 3: Store chunks in database
            stored, skipped = self._store_chunks(chunks, session)
            result.chunks_stored = stored
            result.chunks_skipped_duplicate = skipped

        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            result.files_failed = 1
            result.errors.append(f"{path.name}: {exc}")
            logger.error("ingestion_file_failed", file=path.name, error=str(exc))

        logger.info("ingestion_file_complete", file=path.name, **result.to_dict())
        return result

    def ingest_directory(
        self,
        directory: str | Path,
        session: Session,
    ) -> IngestionResult:
        """Ingest all supported documents in a directory (recursively).

        Scans the directory for .pdf, .docx, .txt, and .md files, then
        ingests each one. Files that fail are logged and skipped — they
        don't stop the batch.

        Args:
            directory: Path to the directory to scan.
            session: SQLAlchemy session.

        Returns:
            Combined IngestionResult for all files.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        # Find all supported files recursively
        files = sorted(
            f for f in dir_path.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        logger.info(
            "ingestion_directory_scan",
            directory=str(dir_path),
            files_found=len(files),
        )

        if not files:
            logger.warning("no_supported_files", directory=str(dir_path))
            return IngestionResult()

        # Ingest each file and combine results
        combined = IngestionResult()

        for file_path in files:
            file_result = self.ingest_file(file_path, session)
            combined.files_processed += file_result.files_processed
            combined.files_failed += file_result.files_failed
            combined.pages_extracted += file_result.pages_extracted
            combined.chunks_created += file_result.chunks_created
            combined.chunks_stored += file_result.chunks_stored
            combined.chunks_skipped_duplicate += file_result.chunks_skipped_duplicate
            combined.errors.extend(file_result.errors)

        logger.info("ingestion_directory_complete", **combined.to_dict())
        return combined

    def ingest_directory_dry_run(
        self,
        directory: str | Path,
    ) -> IngestionResult:
        """Parse and chunk all documents without writing to database.

        Useful for testing your knowledge base documents: see how many
        chunks would be created, what sizes they are, and whether any
        files fail to parse — all without touching the database.

        Args:
            directory: Path to the directory to scan.

        Returns:
            IngestionResult (chunks_stored will be 0 since nothing is written).
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        files = sorted(
            f for f in dir_path.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        result = IngestionResult()

        for file_path in files:
            try:
                pages = self.parser.parse(file_path)
                chunks = self.chunker.chunk_pages(pages)
                result.files_processed += 1
                result.pages_extracted += len(pages)
                result.chunks_created += len(chunks)
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                result.files_failed += 1
                result.errors.append(f"{file_path.name}: {exc}")

        logger.info("dry_run_complete", **result.to_dict())
        return result

    # ------------------------------------------------------------------
    # Private storage methods
    # ------------------------------------------------------------------

    def _store_chunks(
        self,
        chunks: list[TextChunk],
        session: Session,
    ) -> tuple[int, int]:
        """Store chunks in the DocumentChunk table, deduplicating by content hash.

        HOW DEDUPLICATION WORKS:
        Before inserting, we check if a chunk with the same content_hash already
        exists in the database. If it does, we skip it. This means you can safely
        re-run ingestion on the same directory — only new or changed content gets
        added.

        WHY content_hash AND NOT (source_file + chunk_index)?
        Because the same text might appear in different documents (e.g., a standard
        disclaimer copied across multiple tender responses). We want to store it once
        and avoid inflating the vector database with duplicate embeddings.

        Args:
            chunks: List of TextChunk objects to store.
            session: SQLAlchemy session.

        Returns:
            Tuple of (chunks_stored, chunks_skipped).
        """
        stored = 0
        skipped = 0

        # Get existing hashes in one query to avoid N+1 queries
        chunk_hashes = [c.content_hash for c in chunks]
        existing_hashes = set(
            row[0]
            for row in session.execute(
                select(DocumentChunk.content_hash).where(
                    DocumentChunk.content_hash.in_(chunk_hashes)
                )
            ).all()
        )

        batch: list[DocumentChunk] = []

        for chunk in chunks:
            if chunk.content_hash in existing_hashes:
                skipped += 1
                continue

            # Build metadata dict
            meta = dict(chunk.metadata)
            if self.document_category:
                meta["category"] = self.document_category
            if chunk.section_heading:
                meta["section_heading"] = chunk.section_heading

            db_chunk = DocumentChunk(
                source_file=chunk.source_file,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                content=chunk.text,
                content_hash=chunk.content_hash,
                chunk_metadata=meta,
                # embedding is left as None — Step 6 will populate it
            )

            batch.append(db_chunk)
            # Track this hash so we don't double-insert within the same batch
            existing_hashes.add(chunk.content_hash)
            stored += 1

            # Commit in batches to keep transactions manageable
            if len(batch) >= self.batch_size:
                session.add_all(batch)
                session.commit()
                logger.debug("batch_committed", batch_size=len(batch))
                batch = []

        # Commit any remaining chunks
        if batch:
            session.add_all(batch)
            session.commit()
            logger.debug("final_batch_committed", batch_size=len(batch))

        return stored, skipped