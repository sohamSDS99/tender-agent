"""
Knowledge retrieval module for the AI Tender Agent.

This package handles searching the company knowledge base (pgvector)
for information relevant to tender requirements.

Usage:
    from src.knowledge.retriever import KnowledgeRetriever

    retriever = KnowledgeRetriever()
    results = retriever.search("ISO certifications", session, top_k=5)
    context = retriever.format_context(results)
"""

from src.knowledge.retriever import KnowledgeRetriever, RetrievedChunk

__all__ = ["KnowledgeRetriever", "RetrievedChunk"]