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

    SQLAlchemy format:  postgresql+psycopg://user:pass@host:port/db?sslmode=require
    Checkpointer needs: postgresql://user:pass@host:port/db?sslmode=require

    This function strips the '+psycopg' SQLAlchemy dialect prefix.
    SSL query params (sslmode=require) are preserved so Supabase
    connections stay encrypted end-to-end.

    NOTE: Use the Supavisor session pooler (port 5432) — host format:
    aws-0-[region].pooler.supabase.com. Session mode supports prepared
    statements (required for LangGraph checkpointer). Do NOT use port
    6543 (transaction pooler) — it does not support prepared statements.
    """
    return settings.database_url.replace("+psycopg", "")