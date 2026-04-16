"""
Tests the database connection and basic CRUD operations.

Usage:
    python scripts/test_db_connection.py
"""

from datetime import datetime, timezone
import random

from sqlalchemy import text

from src.models import SessionLocal, Tender, TenderStatus, AuditLog, DocumentChunk
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def test_connection() -> None:
    """Test basic database connectivity."""
    db = SessionLocal()
    try:
        # Test 1: Raw SQL connection
        result = db.execute(text("SELECT 1"))
        assert result.scalar() == 1
        logger.info("test_passed", test="raw_sql_connection")

        # Test 2: pgvector extension is available
        result = db.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        )
        row = result.fetchone()
        assert row is not None, "pgvector extension not installed!"
        logger.info("test_passed", test="pgvector_extension")

        # Test 3: Insert a test tender
        test_tender = Tender(
            external_id="TEST-001-VERIFY",
            title="Test Tender - Database Verification",
            source="test_script",
            status=TenderStatus.DISCOVERED,
            submission_deadline=datetime(2026, 12, 31, tzinfo=timezone.utc),
            agency="Test Agency",
        )
        db.add(test_tender)
        db.commit()
        logger.info("test_passed", test="tender_insert", tender_id=test_tender.id)

        # Test 4: Query it back
        queried = db.query(Tender).filter_by(external_id="TEST-001-VERIFY").first()
        assert queried is not None
        assert queried.title == "Test Tender - Database Verification"
        assert queried.status == TenderStatus.DISCOVERED
        logger.info("test_passed", test="tender_query", title=queried.title)

        # Test 5: Insert an audit log
        log_entry = AuditLog(
            tender_id=test_tender.id,
            node_name="test",
            action="database_verification",
            decision="all_tests_passed",
            success=True,
        )
        db.add(log_entry)
        db.commit()
        logger.info("test_passed", test="audit_log_insert", log_id=log_entry.id)

        # Test 6: Insert a vector embedding
        dummy_embedding = [random.uniform(-1, 1) for _ in range(1024)]
        test_chunk = DocumentChunk(
            source_file="test_document.pdf",
            source_type="test",
            chunk_index=0,
            content="This is a test chunk for database verification.",
            embedding=dummy_embedding,
            token_count=10,
        )
        db.add(test_chunk)
        db.commit()
        logger.info("test_passed", test="vector_insert", chunk_id=test_chunk.id)

        # Test 7: Vector similarity search works
        result = db.execute(
            text("""
                SELECT id, content, embedding <=> :query_vec AS distance
                FROM document_chunks
                ORDER BY embedding <=> :query_vec
                LIMIT 1
            """),
            {"query_vec": str(dummy_embedding)},
        )
        row = result.fetchone()
        assert row is not None
        assert row[2] == 0.0  # Distance to itself should be 0
        logger.info("test_passed", test="vector_similarity_search", distance=row[2])

        # Cleanup: Remove test data
        db.delete(test_chunk)
        db.delete(log_entry)
        db.delete(test_tender)
        db.commit()
        logger.info("cleanup_complete", message="All test data removed")

        print()
        print("=" * 60)
        print("  ALL 7 DATABASE TESTS PASSED")
        print("  PostgreSQL + pgvector + SQLAlchemy working correctly")
        print("=" * 60)
        print()

    except Exception as e:
        db.rollback()
        logger.error("test_failed", error=str(e))
        raise
    finally:
        db.close()


if __name__ == "__main__":
    test_connection()