"""
Centralised configuration management.

Loads environment variables from .env, validates required values,
and exposes them as typed attributes on a Settings object.

Usage:
    from src.utils.config import settings
    print(settings.dashscope_api_key)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
# find_dotenv() walks up the directory tree, but explicit is better than implicit
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")


def _require_env(name: str) -> str:
    """Get a required environment variable or crash with a clear message."""
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set. "
            f"Check your .env file. See .env.example for reference."
        )
    return value


def _optional_env(name: str, default: str = "") -> str:
    """Get an optional environment variable with a fallback default."""
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    """
    Immutable application settings loaded from environment variables.

    frozen=True means these values can't be changed after creation,
    which prevents bugs from code accidentally modifying config at runtime.
    """

    # --- Qwen / DashScope API ---
    dashscope_api_key: str = field(default_factory=lambda: _optional_env("DASHSCOPE_API_KEY"))
    qwen_base_url: str = field(
        default_factory=lambda: _optional_env(
            "QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
    )

    # --- Voyage AI ---
    voyage_api_key: str = field(default_factory=lambda: _require_env("VOYAGE_API_KEY"))

    # --- Database ---
    database_url: str = field(
        default_factory=lambda: _require_env("DATABASE_URL")
    )

    # --- Slack ---
    slack_bot_token: str = field(default_factory=lambda: _optional_env("SLACK_BOT_TOKEN"))
    slack_app_token: str = field(default_factory=lambda: _optional_env("SLACK_APP_TOKEN"))
    slack_channel_id: str = field(default_factory=lambda: _optional_env("SLACK_CHANNEL_ID"))

    # --- AWS ---
    aws_access_key_id: str = field(default_factory=lambda: _optional_env("AWS_ACCESS_KEY_ID"))
    aws_secret_access_key: str = field(
        default_factory=lambda: _optional_env("AWS_SECRET_ACCESS_KEY")
    )
    aws_region: str = field(default_factory=lambda: _optional_env("AWS_REGION", "us-east-1"))
    s3_bucket_name: str = field(
        default_factory=lambda: _optional_env("S3_BUCKET_NAME", "tender-agent-docs")
    )

    # --- Email ---
    imap_server: str = field(default_factory=lambda: _optional_env("IMAP_SERVER"))
    imap_email: str = field(default_factory=lambda: _optional_env("IMAP_EMAIL"))
    imap_password: str = field(default_factory=lambda: _optional_env("IMAP_PASSWORD"))

    # --- LangSmith ---
    langchain_api_key: str = field(default_factory=lambda: _optional_env("LANGCHAIN_API_KEY"))
    langchain_project: str = field(
        default_factory=lambda: _optional_env("LANGCHAIN_PROJECT", "tender-agent")
    )

    # --- Agent Behaviour ---
    tender_score_threshold: int = field(
        default_factory=lambda: int(_optional_env("TENDER_SCORE_THRESHOLD", "60"))
    )
    slack_timeout_hours: int = field(
        default_factory=lambda: int(_optional_env("SLACK_TIMEOUT_HOURS", "48"))
    )
    max_assembly_retries: int = field(
        default_factory=lambda: int(_optional_env("MAX_ASSEMBLY_RETRIES", "3"))
    )
    dry_run: bool = field(
        default_factory=lambda: _optional_env("DRY_RUN", "true").lower() == "true"
    )

    # --- Derived Paths ---
    project_root: Path = field(default_factory=lambda: _project_root)
    data_dir: Path = field(default_factory=lambda: _project_root / "data")
    templates_dir: Path = field(default_factory=lambda: _project_root / "templates")


# Singleton instance — import this everywhere
settings = Settings()