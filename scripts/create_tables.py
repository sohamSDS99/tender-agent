"""
Creates all database tables defined in src/models.

Usage:
    python scripts/create_tables.py
"""

from src.models import Base, engine
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def create_all_tables() -> None:
    """Create all tables in the database."""
    logger.info("creating_tables", database=str(engine.url))

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