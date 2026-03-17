"""Pydantic models for data sync configuration (ISO 27034).

ISO Compliance:
- ISO/IEC 27034:2011 - Secure Coding (Pydantic validation)
- ISO/IEC 5055:2021 - Code Quality (<50 lines per function)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class SyncConfig(BaseModel):
    """Validated CLI configuration for sync operations."""

    source: Literal["local", "hf"] = "local"
    source_dir: Path | None = None
    push: bool = False
    dry_run: bool = False
    hf_repo_id: str = Field(
        default="Pierrax/ffe-history",
        pattern=r"^[\w-]+/[\w.-]+$",
    )


class SourceStatus(BaseModel):
    """Status of a data source directory."""

    path: Path
    accessible: bool
    file_count: int = 0
    latest_mtime: datetime | None = None


class FreshnessReport(BaseModel):
    """Comparison between source and local parquet freshness."""

    source_mtime: datetime | None = None
    parquet_mtime: datetime | None = None
    is_stale: bool = True
    days_behind: float = 0.0
