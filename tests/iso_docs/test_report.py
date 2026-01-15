"""Tests Generate Report - ISO 29119.

Document ID: ALICE-TEST-ISO-REPORT
Version: 1.0.0
Tests: 6

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<150 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from unittest.mock import MagicMock, patch

from scripts.update_iso_docs import MODULES, generate_report


class TestGenerateReport:
    """Tests pour generate_report."""

    @patch("scripts.iso_docs.templates.check_installed")
    @patch("scripts.iso_docs.templates.get_test_coverage")
    @patch("scripts.iso_docs.templates.get_complexity_avg")
    def test_generate_report_structure(
        self, mock_complexity: MagicMock, mock_coverage: MagicMock, mock_installed: MagicMock
    ) -> None:
        """Test structure du rapport genere."""
        mock_installed.return_value = (True, "1.0.0")
        mock_coverage.return_value = "80%"
        mock_complexity.return_value = "A (1.5)"
        report = generate_report()
        assert "# IMPLEMENTATION DEVOPS - STATUT ISO" in report
        assert "## SCORE DEVOPS GLOBAL" in report
        assert "## CONFORMITE PAR NORME ISO" in report

    @patch("scripts.iso_docs.templates.check_installed")
    @patch("scripts.iso_docs.templates.get_test_coverage")
    @patch("scripts.iso_docs.templates.get_complexity_avg")
    def test_generate_report_score_calculation(
        self, mock_complexity: MagicMock, mock_coverage: MagicMock, mock_installed: MagicMock
    ) -> None:
        """Test calcul du score DevOps."""
        mock_installed.return_value = (True, "1.0.0")
        mock_coverage.return_value = "80%"
        mock_complexity.return_value = "A (1.5)"
        report = generate_report()
        assert "100/100" in report or "Score actuel: 100" in report

    @patch("scripts.iso_docs.templates.check_installed")
    @patch("scripts.iso_docs.templates.get_test_coverage")
    @patch("scripts.iso_docs.templates.get_complexity_avg")
    def test_generate_report_partial_installation(
        self, mock_complexity: MagicMock, mock_coverage: MagicMock, mock_installed: MagicMock
    ) -> None:
        """Test rapport avec installation partielle."""
        mock_installed.side_effect = [
            (True, "1.0.0") if i % 2 == 0 else (False, "non installe") for i in range(len(MODULES))
        ]
        mock_coverage.return_value = "50%"
        mock_complexity.return_value = "C (5.0)"
        report = generate_report()
        assert "MODULES MANQUANTS" in report

    @patch("scripts.iso_docs.templates.check_installed")
    @patch("scripts.iso_docs.templates.get_test_coverage")
    @patch("scripts.iso_docs.templates.get_complexity_avg")
    def test_generate_report_timestamp(
        self, mock_complexity: MagicMock, mock_coverage: MagicMock, mock_installed: MagicMock
    ) -> None:
        """Test presence du timestamp."""
        mock_installed.return_value = (True, "1.0.0")
        mock_coverage.return_value = "80%"
        mock_complexity.return_value = "A (1.5)"
        report = generate_report()
        assert "Date de mise a jour:" in report

    @patch("scripts.iso_docs.templates.check_installed")
    @patch("scripts.iso_docs.templates.get_test_coverage")
    @patch("scripts.iso_docs.templates.get_complexity_avg")
    def test_generate_report_coverage_status(
        self, mock_complexity: MagicMock, mock_coverage: MagicMock, mock_installed: MagicMock
    ) -> None:
        """Test statut coverage dans le rapport."""
        mock_installed.return_value = (True, "1.0.0")
        mock_coverage.return_value = "85%"
        mock_complexity.return_value = "A"
        report = generate_report()
        assert "85%" in report

    @patch("scripts.iso_docs.templates.check_installed")
    @patch("scripts.iso_docs.templates.get_test_coverage")
    @patch("scripts.iso_docs.templates.get_complexity_avg")
    def test_generate_report_all_installed(
        self, mock_complexity: MagicMock, mock_coverage: MagicMock, mock_installed: MagicMock
    ) -> None:
        """Test rapport sans modules manquants."""
        mock_installed.return_value = (True, "1.0.0")
        mock_coverage.return_value = "80%"
        mock_complexity.return_value = "A"
        report = generate_report()
        assert "Tous les modules recommandes sont installes" in report
