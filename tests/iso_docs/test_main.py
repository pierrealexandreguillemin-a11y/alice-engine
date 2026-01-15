"""Tests Main Function - ISO 29119.

Document ID: ALICE-TEST-ISO-MAIN
Version: 1.0.0
Tests: 4

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<100 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.update_iso_docs import main


class TestMain:
    """Tests pour la fonction main."""

    @patch("scripts.update_iso_docs.generate_report")
    @patch("scripts.update_iso_docs.check_installed")
    def test_main_creates_directory(
        self, mock_installed: MagicMock, mock_report: MagicMock, tmp_path: Path
    ) -> None:
        """Test que main cree le repertoire docs/iso."""
        mock_report.return_value = "# Test Report"
        mock_installed.return_value = (True, "1.0.0")
        with patch("scripts.update_iso_docs.DOCS_ISO", tmp_path / "docs" / "iso"):
            with patch("scripts.update_iso_docs.ROOT", tmp_path):
                result = main()
        assert result == 0
        assert (tmp_path / "docs" / "iso").exists()

    @patch("scripts.update_iso_docs.generate_report")
    @patch("scripts.update_iso_docs.check_installed")
    def test_main_writes_report(
        self, mock_installed: MagicMock, mock_report: MagicMock, tmp_path: Path
    ) -> None:
        """Test que main ecrit le rapport."""
        mock_report.return_value = "# Test Report Content"
        mock_installed.return_value = (True, "1.0.0")
        docs_iso = tmp_path / "docs" / "iso"
        docs_iso.mkdir(parents=True)
        with patch("scripts.update_iso_docs.DOCS_ISO", docs_iso):
            with patch("scripts.update_iso_docs.ROOT", tmp_path):
                result = main()
        assert result == 0
        assert (docs_iso / "IMPLEMENTATION_STATUS.md").read_text() == "# Test Report Content"

    @patch("scripts.update_iso_docs.generate_report")
    @patch("scripts.update_iso_docs.check_installed")
    def test_main_returns_zero(
        self, mock_installed: MagicMock, mock_report: MagicMock, tmp_path: Path
    ) -> None:
        """Test que main retourne 0."""
        mock_report.return_value = "# Test"
        mock_installed.return_value = (True, "1.0.0")
        with patch("scripts.update_iso_docs.DOCS_ISO", tmp_path / "docs" / "iso"):
            with patch("scripts.update_iso_docs.ROOT", tmp_path):
                result = main()
        assert result == 0

    @patch("scripts.update_iso_docs.generate_report")
    @patch("scripts.update_iso_docs.check_installed")
    def test_main_prints_score(
        self,
        mock_installed: MagicMock,
        mock_report: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test que main affiche le score."""
        mock_report.return_value = "# Test"
        mock_installed.return_value = (True, "1.0.0")
        with patch("scripts.update_iso_docs.DOCS_ISO", tmp_path / "docs" / "iso"):
            with patch("scripts.update_iso_docs.ROOT", tmp_path):
                main()
        captured = capsys.readouterr()
        assert "Score DevOps:" in captured.out
