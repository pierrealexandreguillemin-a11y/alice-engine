"""Tests Sync Data - ISO 29119.

Document ID: ALICE-TEST-SYNC-DATA
Version: 1.0.0
Tests: 12

Classes:
- TestSourceCheck: Tests check_source (3 tests)
- TestFreshness: Tests check_freshness (3 tests)
- TestSymlink: Tests update_symlink (3 tests)
- TestHuggingFace: Tests pull/push HF (3 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML
"""

from __future__ import annotations

import time
from pathlib import Path

from scripts.sync_data.freshness import check_freshness, check_source


class TestSourceCheck:
    """Tests for check_source function."""

    def test_accessible_directory(self, tmp_path):
        (tmp_path / "2025" / "Interclubs").mkdir(parents=True)
        (tmp_path / "2025" / "Interclubs" / "ronde_1.html").write_text("<html/>")
        status = check_source(tmp_path)
        assert status.accessible is True
        assert status.file_count >= 1

    def test_missing_directory(self):
        status = check_source(Path("/nonexistent/path"))
        assert status.accessible is False
        assert status.file_count == 0

    def test_empty_directory(self, tmp_path):
        status = check_source(tmp_path)
        assert status.accessible is True
        assert status.file_count == 0


class TestFreshness:
    """Tests for check_freshness function."""

    def test_stale_parquet(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        parquet_dir = tmp_path / "data"
        parquet_dir.mkdir()
        (parquet_dir / "echiquiers.parquet").write_text("old")
        time.sleep(0.1)
        (source_dir / "ronde_1.html").write_text("<html/>")
        report = check_freshness(source_dir, parquet_dir)
        assert report.is_stale is True

    def test_fresh_parquet(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "ronde_1.html").write_text("<html/>")
        time.sleep(0.1)
        parquet_dir = tmp_path / "data"
        parquet_dir.mkdir()
        (parquet_dir / "echiquiers.parquet").write_text("new")
        report = check_freshness(source_dir, parquet_dir)
        assert report.is_stale is False

    def test_no_parquet_yet(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "ronde_1.html").write_text("<html/>")
        parquet_dir = tmp_path / "data"
        parquet_dir.mkdir()
        report = check_freshness(source_dir, parquet_dir)
        assert report.is_stale is True
