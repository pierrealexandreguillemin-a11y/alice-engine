"""Tests Graphs Complexity - ISO 29119.

Document ID: ALICE-TEST-GRAPHS-COMPLEXITY
Version: 1.0.0

Tests pour complexity.py.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path
from unittest.mock import MagicMock, patch


class TestGenerateComplexityReport:
    """Tests pour generate_complexity_report."""

    @patch("scripts.graphs.complexity.run_cmd")
    def test_radon_not_installed(self, mock_run_cmd: MagicMock) -> None:
        """Test quand radon n'est pas installé."""
        from scripts.graphs.complexity import generate_complexity_report

        mock_run_cmd.return_value = (1, "")

        result = generate_complexity_report()

        assert result is False

    @patch("scripts.graphs.complexity.run_cmd")
    @patch("scripts.graphs.complexity.generate_complexity_html")
    def test_successful_report(
        self, mock_html: MagicMock, mock_run_cmd: MagicMock, tmp_path: Path
    ) -> None:
        """Test génération rapport réussie."""
        from scripts.graphs.complexity import generate_complexity_report

        mock_run_cmd.return_value = (0, "complexity output")

        with (
            patch("scripts.graphs.complexity.REPORTS_DIR", tmp_path),
            patch("scripts.graphs.complexity.ROOT", tmp_path),
        ):
            result = generate_complexity_report()

        assert result is True
        mock_html.assert_called_once()


class TestGenerateComplexityHtml:
    """Tests pour generate_complexity_html."""

    def test_generate_html(self, tmp_path: Path) -> None:
        """Test génération HTML."""
        from scripts.graphs.complexity import generate_complexity_html

        (tmp_path / "cyclomatic.txt").write_text("app/main.py - A (5)")
        (tmp_path / "maintainability.txt").write_text("app/main.py - A (85)")

        output = tmp_path / "index.html"

        with patch("scripts.graphs.complexity.ROOT", tmp_path):
            generate_complexity_html(tmp_path, output)

        assert output.exists()
        content = output.read_text()
        assert "Rapport Complexite" in content
        assert "A (5)" in content
        assert "ISO 25010" in content

    def test_generate_html_missing_files(self, tmp_path: Path) -> None:
        """Test HTML avec fichiers manquants."""
        from scripts.graphs.complexity import generate_complexity_html

        output = tmp_path / "index.html"

        with patch("scripts.graphs.complexity.ROOT", tmp_path):
            generate_complexity_html(tmp_path, output)

        assert output.exists()
        content = output.read_text()
        assert "N/A" in content


class TestCheckDuplication:
    """Tests pour check_duplication."""

    @patch("scripts.graphs.complexity.run_cmd")
    def test_check_duplication(self, mock_run_cmd: MagicMock, tmp_path: Path) -> None:
        """Test détection duplication."""
        from scripts.graphs.complexity import check_duplication

        mock_run_cmd.return_value = (0, "No duplicate code found")

        with (
            patch("scripts.graphs.complexity.REPORTS_DIR", tmp_path),
            patch("scripts.graphs.complexity.ROOT", tmp_path),
        ):
            result = check_duplication()

        assert result is True

    @patch("scripts.graphs.complexity.run_cmd")
    def test_check_duplication_found(self, mock_run_cmd: MagicMock, tmp_path: Path) -> None:
        """Test duplication détectée."""
        from scripts.graphs.complexity import check_duplication

        mock_run_cmd.return_value = (0, "Similar lines in 2 files")

        with (
            patch("scripts.graphs.complexity.REPORTS_DIR", tmp_path),
            patch("scripts.graphs.complexity.ROOT", tmp_path),
        ):
            result = check_duplication()

        assert result is True
