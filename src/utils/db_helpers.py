"""
Database helper utilities.

Provides connection string conversion and other DB-related helpers
used across the application.
"""

from __future__ import annotations

from src.utils.config import settings


def get_checkpointer_db_uri() -> str:
    """
    Returns the database URI in the format the LangGraph
    PostgreSQL checkpointer expects.

    Our .env stores: postgresql+psycopg://user:pass@host:port/db
    The checkpointer needs: postgresql://user:pass@host:port/db

    This function strips the '+psycopg' SQLAlchemy dialect prefix.
    """
    return settings.database_url.replace("+psycopg", "")