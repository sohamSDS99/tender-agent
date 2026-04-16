"""
Document ingestion pipeline for the AI Tender Agent.

This package handles:
- Parsing raw documents (PDF, DOCX, TXT, MD) into structured text
- Chunking text into overlapping segments suitable for embedding
- Storing chunks in PostgreSQL (DocumentChunk model) for later vectorisation

Usage:
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    results = pipeline.ingest_directory("data/knowledge_base")
"""

from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import TextChunker
from src.ingestion.pipeline import IngestionPipeline

__all__ = ["DocumentParser", "TextChunker", "IngestionPipeline"]