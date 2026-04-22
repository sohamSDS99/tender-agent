"""
Creates all database tables defined in src/models.

Works with both local PostgreSQL and Supabase.

For Supabase: the vector extension is usually already enabled via the
Supabase dashboard (Database → Extensions → vector). This script will
attempt to enable it anyway — if the DB user lacks superuser privileges
the step is skipped gracefully and you should enable it via the dashboard.

Usage:
    python scripts/create_tables.py
"""

from sqlalchemy import text

from src.models import Base, engine
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def enable_pgvector(conn) -> None:
    """Enable the pgvector extension if not already enabled."""
    try:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
        logger.info("pgvector_extension_enabled")
    except Exception as exc:
        # On Supabase, non-superuser roles cannot CREATE EXTENSION.
        # Enable it via: Supabase dashboard → Database → Extensions → vector
        logger.warning(
            "pgvector_extension_skipped",
            reason=str(exc)[:120],
            hint="Enable via Supabase dashboard: Database → Extensions → vector",
        )


def create_all_tables() -> None:
    """Create all tables in the database."""
    logger.info("creating_tables", database=str(engine.url))

    with engine.connect() as conn:
        enable_pgvector(conn)

    Base.metadata.create_all(bind=engine)

    logger.info("tables_created_successfully")

    from sqlalchemy import inspect

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    for table in tables:
        columns = [col["name"] for col in inspector.get_columns(table)]
        logger.info("table_exists", table=table, columns=columns)


if __name__ == "__main__":
    create_all_tables()
