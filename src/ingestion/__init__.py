"""
Document ingestion pipeline for the AI Tender Agent.

This package handles:
- Parsing raw documents (PDF, DOCX, TXT, MD) into structured text
- Chunking text into overlapping segments suitable for embedding
- Computing vector embeddings via Voyage AI (or dry-run fake vectors)
- Storing chunks and embeddings in PostgreSQL for RAG retrieval

Usage:
    from src.ingestion.pipeline import IngestionPipeline
    from src.ingestion.embed_pipeline import EmbeddingPipeline

    # Step 1: Ingest documents (parse + chunk + store)
    ingestion = IngestionPipeline()
    result = ingestion.ingest_directory("data/knowledge_base", session)

    # Step 2: Embed stored chunks (compute vectors + update DB)
    embedding = EmbeddingPipeline()
    result = embedding.embed_all(session)
"""

from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import TextChunker
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.embedder import VoyageEmbedder
from src.ingestion.embed_pipeline import EmbeddingPipeline

__all__ = [
    "DocumentParser",
    "TextChunker",
    "IngestionPipeline",
    "VoyageEmbedder",
    "EmbeddingPipeline",
]