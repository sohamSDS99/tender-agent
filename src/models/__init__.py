"""
Database models package.

Import all models here so that:
1. They can be imported from one place: from src.models import Tender, AuditLog
2. SQLAlchemy's Base.metadata knows about all tables when we create them
"""

from src.models.base import Base, TimestampMixin, engine, SessionLocal, get_db
from src.models.tender import Tender, TenderStatus, AuditLog
from src.models.embedding import DocumentChunk, EMBEDDING_DIMENSION

__all__ = [
    "Base",
    "TimestampMixin",
    "engine",
    "SessionLocal",
    "get_db",
    "Tender",
    "TenderStatus",
    "AuditLog",
    "DocumentChunk",
    "EMBEDDING_DIMENSION",
]