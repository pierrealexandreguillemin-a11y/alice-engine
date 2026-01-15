"""Tests Checkers - ISO 29119.

Document ID: ALICE-TEST-ISO-CHECKERS
Version: 1.0.0
Tests: 11

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<150 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from unittest.mock import MagicMock, patch

from scripts.update_iso_docs import check_installed, get_complexity_avg, get_test_coverage


class TestCheckInstalled:
    """Tests pour check_installed."""

    def test_check_installed_python(self) -> None:
        """Test verification Python installe."""
        installed, version = check_installed("python --version")
        assert installed is True
        assert "Python" in version or "python" in version.lower()

    def test_check_installed_nonexistent(self) -> None:
        """Test verification outil inexistant."""
        installed, version = check_installed("nonexistent_tool_12345 --version")
        assert installed is False
        assert version == "non installe"

    def test_check_installed_timeout(self) -> None:
        """Test timeout gere correctement."""
        with patch("scripts.iso_docs.checkers.subprocess.run") as mock_run:
            import subprocess

            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 5)
            installed, version = check_installed("slow_command")
            assert installed is False
            assert version == "non installe"

    def test_check_installed_return_code_nonzero(self) -> None:
        """Test commande avec code retour non-zero."""
        with patch("scripts.iso_docs.checkers.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_result.stderr = "error"
            mock_run.return_value = mock_result
            installed, version = check_installed("failing_command")
            assert installed is False

    def test_check_installed_version_from_stdout(self) -> None:
        """Test extraction version depuis stdout."""
        with patch("scripts.iso_docs.checkers.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "tool 1.2.3\n"
            mock_result.stderr = ""
            mock_run.return_value = mock_result
            installed, version = check_installed("tool --version")
            assert installed is True
            assert "1.2.3" in version

    def test_check_installed_version_from_stderr(self) -> None:
        """Test extraction version depuis stderr."""
        with patch("scripts.iso_docs.checkers.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = "version 2.0.0"
            mock_run.return_value = mock_result
            installed, version = check_installed("tool --version")
            assert installed is True
            assert "2.0.0" in version

    def test_check_installed_empty_version(self) -> None:
        """Test version vide retourne 'installed'."""
        with patch("scripts.iso_docs.checkers.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            mock_run.return_value = mock_result
            installed, version = check_installed("tool --version")
            assert installed is True
            assert version == "installed"

    def test_check_installed_long_version_truncated(self) -> None:
        """Test version longue tronquee a 50 caracteres."""
        with patch("scripts.iso_docs.checkers.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "a" * 100 + "\n"
            mock_result.stderr = ""
            mock_run.return_value = mock_result
            installed, version = check_installed("tool --version")
            assert installed is True
            assert len(version) <= 50


class TestGetTestCoverage:
    """Tests pour get_test_coverage."""

    def test_get_test_coverage_with_total(self) -> None:
        """Test extraction coverage avec ligne TOTAL."""
        with patch("scripts.iso_docs.checkers.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = """
            Name                 Stmts   Miss  Cover
            ----------------------------------------
            TOTAL                   55     10    82%
            """
            mock_run.return_value = mock_result
            coverage = get_test_coverage()
            assert coverage == "82%"

    def test_get_test_coverage_no_total(self) -> None:
        """Test extraction coverage sans ligne TOTAL."""
        with patch("scripts.iso_docs.checkers.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "No tests found"
            mock_run.return_value = mock_result
            coverage = get_test_coverage()
            assert coverage == "N/A"

    def test_get_test_coverage_exception(self) -> None:
        """Test gestion exception."""
        with patch("scripts.iso_docs.checkers.subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Error")
            coverage = get_test_coverage()
            assert coverage == "N/A"


class TestGetComplexityAvg:
    """Tests pour get_complexity_avg."""

    def test_get_complexity_avg_found(self) -> None:
        """Test extraction complexite moyenne."""
        with patch("scripts.iso_docs.checkers.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "Average complexity: A (1.5)"
            mock_run.return_value = mock_result
            complexity = get_complexity_avg()
            assert "A" in complexity

    def test_get_complexity_avg_not_found(self) -> None:
        """Test extraction sans ligne Average complexity."""
        with patch("scripts.iso_docs.checkers.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "No complexity data"
            mock_run.return_value = mock_result
            complexity = get_complexity_avg()
            assert complexity == "N/A"

    def test_get_complexity_avg_exception(self) -> None:
        """Test gestion exception."""
        with patch("scripts.iso_docs.checkers.subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Error")
            complexity = get_complexity_avg()
            assert complexity == "N/A"
