"""
SQLAlchemy base configuration.

All database models inherit from Base. This module also provides
the engine and session factory used throughout the application.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.utils.config import settings


# Naming convention for constraints — ensures consistent, predictable names
# for indexes, foreign keys, etc. across all databases. This matters when
# running migrations with Alembic later.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy models.

    Provides:
    - A consistent naming convention for database constraints
    - Common column definitions (id, created_at, updated_at) via mixins
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampMixin:
    """
    Mixin that adds created_at and updated_at columns to any model.

    Usage:
        class Tender(TimestampMixin, Base):
            __tablename__ = "tenders"
            ...

    Every row will automatically get:
    - created_at: set to the current UTC time when the row is first inserted
    - updated_at: set to the current UTC time on every update
    """

    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


# --- Database Engine & Session ---

# Supabase (and any hosted PostgreSQL) requires SSL and has connection limits.
# Free tier: 60 connections max. pool_size + max_overflow must stay well under that.
_connect_args: dict = {}
if "supabase.co" in settings.database_url:
    _connect_args["sslmode"] = "require"

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,   # Verify connections are alive before checkout
    pool_size=5,          # Base connections kept open
    max_overflow=10,      # Extra connections allowed under load
    pool_timeout=30,      # Seconds to wait for a connection from the pool
    pool_recycle=1800,    # Recycle connections every 30 min (avoids stale connections)
    connect_args=_connect_args,
    echo=False,  # Set to True to see all SQL queries in the console
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    """
    Dependency that provides a database session.

    Usage:
        db = next(get_db())
        try:
            # ... use db ...
            db.commit()
        finally:
            db.close()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()