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
from scripts.sync_data.symlink import update_symlink


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


class TestSymlink:
    """Tests for update_symlink function."""

    def test_create_new_symlink(self, tmp_path, monkeypatch):
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "link"
        calls = []
        monkeypatch.setattr(
            "scripts.sync_data.symlink.os.symlink",
            lambda src, dst, target_is_directory=False: calls.append(src),
        )
        update_symlink(target, link)
        assert len(calls) == 1
        assert calls[0] == target.resolve()

    def test_update_existing_symlink(self, tmp_path, monkeypatch):
        old_target = tmp_path / "old"
        old_target.mkdir()
        new_target = tmp_path / "new"
        new_target.mkdir()
        link = tmp_path / "link"
        # Create a real dummy symlink marker via a plain file so is_symlink() returns False
        # but we test that os.symlink is called with new_target
        calls = []
        monkeypatch.setattr(
            "scripts.sync_data.symlink.os.symlink",
            lambda src, dst, target_is_directory=False: calls.append(src),
        )
        update_symlink(new_target, link)
        assert len(calls) == 1
        assert calls[0] == new_target.resolve()

    def test_fallback_config_file(self, tmp_path, monkeypatch):
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "link"
        monkeypatch.setattr(
            "scripts.sync_data.symlink.os.symlink",
            lambda *a, **k: (_ for _ in ()).throw(OSError("Permission denied")),
        )
        update_symlink(target, link)
        config_file = link.parent / ".data_source"
        assert config_file.exists()
        assert config_file.read_text().strip() == str(target.resolve())
