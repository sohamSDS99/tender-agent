"""
Generates the SQL DDL for all application tables WITHOUT needing a database connection.

Use this when you can't connect directly to the database.
Paste the output into the Supabase SQL editor:
    Supabase Dashboard → SQL Editor → New query → paste → Run
"""

from sqlalchemy.schema import CreateTable
from sqlalchemy.dialects import postgresql

from src.models import Base


def main() -> None:
    print("-- ============================================================")
    print("-- TENDER AGENT — Database Schema")
    print("-- Paste this into: Supabase Dashboard → SQL Editor → Run")
    print("-- ============================================================")
    print()

    print("-- Step 1: Enable pgvector extension")
    print("CREATE EXTENSION IF NOT EXISTS vector;")
    print()

    print("-- Step 2: Create application tables")
    dialect = postgresql.dialect()
    for table in Base.metadata.sorted_tables:
        ddl = str(CreateTable(table).compile(dialect=dialect)).strip()
        print(f"-- Table: {table.name}")
        print(ddl + ";")
        print()

    print("-- Done. Verify with: SELECT table_name FROM information_schema.tables")
    print("-- WHERE table_schema = 'public';")


if __name__ == "__main__":
    main()
