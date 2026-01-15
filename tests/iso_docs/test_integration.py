"""Tests Integration & Edge Cases - ISO 29119.

Document ID: ALICE-TEST-ISO-INTEGRATION
Version: 1.0.0
Tests: 5

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<100 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from unittest.mock import MagicMock, patch

from scripts.update_iso_docs import check_installed, generate_report


class TestIntegration:
    """Tests d'integration."""

    @patch("scripts.iso_docs.templates.check_installed")
    @patch("scripts.iso_docs.templates.get_test_coverage")
    @patch("scripts.iso_docs.templates.get_complexity_avg")
    def test_report_generation_mocked(
        self, mock_complexity: MagicMock, mock_coverage: MagicMock, mock_installed: MagicMock
    ) -> None:
        """Test generation rapport avec mocks."""
        mock_installed.return_value = (True, "1.0.0")
        mock_coverage.return_value = "85%"
        mock_complexity.return_value = "A (2.5)"
        report = generate_report()
        assert "# IMPLEMENTATION DEVOPS" in report
        assert "85%" in report

    @patch("scripts.iso_docs.templates.check_installed")
    def test_coverage_badge_good(self, mock_installed: MagicMock) -> None:
        """Test badge coverage vert pour >80%."""
        mock_installed.return_value = (True, "1.0.0")
        with patch("scripts.iso_docs.templates.get_test_coverage", return_value="85%"):
            with patch("scripts.iso_docs.templates.get_complexity_avg", return_value="A"):
                report = generate_report()
        assert "85%" in report


class TestEdgeCases:
    """Tests edge cases."""

    def test_check_installed_multiline_version(self) -> None:
        """Test version sur plusieurs lignes."""
        with patch("scripts.iso_docs.checkers.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "Version 1.0.0\nBuild 12345"
            mock_result.stderr = ""
            mock_run.return_value = mock_result
            installed, version = check_installed("tool --version")
            assert "\n" not in version

    @patch("scripts.iso_docs.templates.check_installed")
    @patch("scripts.iso_docs.templates.get_test_coverage")
    @patch("scripts.iso_docs.templates.get_complexity_avg")
    def test_report_special_characters(
        self, mock_complexity: MagicMock, mock_coverage: MagicMock, mock_installed: MagicMock
    ) -> None:
        """Test rapport avec caracteres speciaux."""
        mock_installed.return_value = (True, "1.0.0-beta+build.123")
        mock_coverage.return_value = "N/A"
        mock_complexity.return_value = "N/A"
        report = generate_report()
        assert isinstance(report, str)

    @patch("scripts.iso_docs.templates.check_installed")
    @patch("scripts.iso_docs.templates.get_test_coverage")
    @patch("scripts.iso_docs.templates.get_complexity_avg")
    def test_report_na_metrics(
        self, mock_complexity: MagicMock, mock_coverage: MagicMock, mock_installed: MagicMock
    ) -> None:
        """Test rapport avec metriques N/A."""
        mock_installed.return_value = (False, "non installe")
        mock_coverage.return_value = "N/A"
        mock_complexity.return_value = "N/A"
        report = generate_report()
        assert "N/A" in report
