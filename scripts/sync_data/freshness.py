"""Freshness checking for data sources (ISO 5259).

Compares modification times between source HTML and local parquets.

ISO Compliance:
- ISO/IEC 5259:2024 - Data Quality for ML (Lineage)
- ISO/IEC 5055:2021 - Code Quality (<50 lines per function)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from scripts.sync_data.types import FreshnessReport, SourceStatus

logger = logging.getLogger(__name__)


def check_source(source_dir: Path) -> SourceStatus:
    """Check accessibility and stats of a data source directory."""
    if not source_dir.exists():
        return SourceStatus(path=source_dir, accessible=False)

    html_files = list(source_dir.rglob("*.html"))
    latest_mtime = None
    if html_files:
        latest_ts = max(f.stat().st_mtime for f in html_files)
        latest_mtime = datetime.fromtimestamp(latest_ts, tz=UTC)

    return SourceStatus(
        path=source_dir,
        accessible=True,
        file_count=len(html_files),
        latest_mtime=latest_mtime,
    )


def check_freshness(source_dir: Path, parquet_dir: Path) -> FreshnessReport:
    """Compare source HTML freshness vs local parquet files."""
    source_status = check_source(source_dir)
    parquet_files = list(parquet_dir.glob("*.parquet"))

    if not parquet_files:
        return FreshnessReport(source_mtime=source_status.latest_mtime, is_stale=True)

    latest_pq_ts = max(f.stat().st_mtime for f in parquet_files)
    parquet_mtime = datetime.fromtimestamp(latest_pq_ts, tz=UTC)

    is_stale = source_status.latest_mtime is not None and source_status.latest_mtime > parquet_mtime
    days_behind = 0.0
    if is_stale and source_status.latest_mtime:
        delta = source_status.latest_mtime - parquet_mtime
        days_behind = delta.total_seconds() / 86400

    return FreshnessReport(
        source_mtime=source_status.latest_mtime,
        parquet_mtime=parquet_mtime,
        is_stale=is_stale,
        days_behind=days_behind,
    )
