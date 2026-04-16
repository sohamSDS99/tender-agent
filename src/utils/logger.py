"""
Structured logging configuration using structlog.

Every log entry includes:
- timestamp (ISO 8601)
- log level
- module/function that emitted it
- any key-value context you attach

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("tender_evaluated", tender_id="T-2026-001", score=72, decision="advance")
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog for the entire application.

    Call this once at application startup (in main.py or equivalent).
    After this call, all loggers created via get_logger() will use
    the configured format.

    Args:
        log_level: Minimum log level to emit. Default "INFO".
                   Use "DEBUG" during development for maximum verbosity.
    """
    # Configure the standard library logging (structlog wraps this)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Configure structlog processors
    # Processors are a pipeline: each one transforms the log event dict
    # before passing it to the next one
    structlog.configure(
        processors=[
            # Add the log level name (info, warning, error, etc.)
            structlog.stdlib.add_log_level,
            # Add a timestamp in ISO 8601 format
            structlog.processors.TimeStamper(fmt="iso"),
            # If the log entry has a stack trace (from an exception), format it
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # In development: pretty, coloured output to the terminal
            # In production: JSON output for log aggregation tools
            structlog.dev.ConsoleRenderer(),
        ],
        # Use standard library logging as the backend
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger for a module.

    Args:
        name: Typically __name__, so logs show which module emitted them.

    Returns:
        A structlog BoundLogger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("tender_discovered", source="sam_gov", count=5)
        logger.error("scraper_failed", url="https://sam.gov/...", error=str(e))
    """
    return structlog.get_logger(name)