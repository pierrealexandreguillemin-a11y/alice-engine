"""Tests Sync Data - ISO 29119.

Document ID: ALICE-TEST-SYNC-DATA
Version: 1.0.0
Tests: 15

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
from unittest.mock import MagicMock, patch

import pytest

from scripts.sync_data.freshness import check_freshness, check_source
from scripts.sync_data.huggingface import pull_from_hf, push_to_hf
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
        """Test creating a new symlink/junction."""
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "link"
        calls = []
        monkeypatch.setattr(
            "scripts.sync_data.symlink._create_junction",
            lambda t, _link: calls.append(t),
        )
        monkeypatch.setattr(
            "scripts.sync_data.symlink.os.symlink",
            lambda src, dst, target_is_directory=False: calls.append(src),
        )
        update_symlink(target, link)
        assert len(calls) == 1

    def test_update_existing_junction(self, tmp_path, monkeypatch):
        """Test updating an existing junction to new target."""
        new_target = tmp_path / "new"
        new_target.mkdir()
        link = tmp_path / "link"
        # Create a file to simulate an existing link (will be unlinked)
        link.write_text("placeholder")
        calls = []
        monkeypatch.setattr(
            "scripts.sync_data.symlink._is_link_or_junction",
            lambda p: p == link,
        )
        monkeypatch.setattr(
            "scripts.sync_data.symlink._create_junction",
            lambda t, _link: calls.append(t),
        )
        monkeypatch.setattr(
            "scripts.sync_data.symlink.os.symlink",
            lambda src, dst, target_is_directory=False: calls.append(src),
        )
        update_symlink(new_target, link)
        assert len(calls) == 1

    def test_target_not_directory_raises(self, tmp_path):
        """Test ValueError when target is a file, not a directory."""
        target = tmp_path / "file.txt"
        target.write_text("not a dir")
        link = tmp_path / "link"
        with pytest.raises(ValueError, match="not a directory"):
            update_symlink(target, link)

    def test_link_exists_not_symlink_raises(self, tmp_path):
        """Test ValueError when link path is a regular file."""
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "link"
        link.write_text("regular file")
        with pytest.raises(ValueError, match="not a symlink"):
            update_symlink(target, link)

    def test_fallback_config_file(self, tmp_path, monkeypatch):
        """Test fallback to .data_source when symlink/junction fails."""
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "link"

        def raise_os_error(*_args, **_kwargs):
            raise OSError("Permission denied")

        monkeypatch.setattr("scripts.sync_data.symlink._create_junction", raise_os_error)
        monkeypatch.setattr("scripts.sync_data.symlink.os.symlink", raise_os_error)
        update_symlink(target, link)
        config_file = link.parent / ".data_source"
        assert config_file.exists()
        assert config_file.read_text().strip() == str(target.resolve())


class TestHuggingFace:
    """Tests for pull_from_hf and push_to_hf functions."""

    @patch("scripts.sync_data.huggingface.shutil.copy2")
    @patch("scripts.sync_data.huggingface.hf_hub_download")
    def test_pull_downloads_parquets(self, mock_download, mock_copy, tmp_path):
        mock_download.return_value = str(tmp_path / "file.parquet")
        pull_from_hf("Pierrax/ffe-history", tmp_path)
        assert mock_download.call_count == 2

    @patch("scripts.sync_data.huggingface.HfApi")
    def test_push_uploads_parquets(self, mock_api_class, tmp_path):
        (tmp_path / "echiquiers.parquet").write_text("data")
        (tmp_path / "joueurs.parquet").write_text("data")
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        push_to_hf(tmp_path, "Pierrax/ffe-history")
        assert mock_api.upload_file.call_count == 2

    @patch("scripts.sync_data.huggingface.hf_hub_download")
    def test_pull_network_error_propagates(self, mock_download, tmp_path):
        """Test network failure propagates from pull."""
        mock_download.side_effect = ConnectionError("Network unreachable")
        with pytest.raises(ConnectionError, match="Network unreachable"):
            pull_from_hf("Pierrax/ffe-history", tmp_path)

    @patch("scripts.sync_data.huggingface.HfApi")
    def test_push_missing_token_raises(self, mock_api_class, tmp_path):
        (tmp_path / "echiquiers.parquet").write_text("data")
        (tmp_path / "joueurs.parquet").write_text("data")
        mock_api = MagicMock()
        mock_api.upload_file.side_effect = Exception("Invalid token")
        mock_api_class.return_value = mock_api
        with pytest.raises(Exception, match="Invalid token"):
            push_to_hf(tmp_path, "Pierrax/ffe-history")
