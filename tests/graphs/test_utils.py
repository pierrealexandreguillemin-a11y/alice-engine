"""Tests Graphs Utils - ISO 29119.

Document ID: ALICE-TEST-GRAPHS-UTILS
Version: 1.0.0

Tests pour utils.py: run_cmd, print_header, constantes.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from unittest.mock import MagicMock, patch

import pytest

from scripts.graphs.utils import GRAPHS_DIR, REPORTS_DIR, ROOT, print_header, run_cmd


class TestRunCmd:
    """Tests pour run_cmd."""

    def test_run_cmd_success(self) -> None:
        """Test commande réussie."""
        code, output = run_cmd(["python", "--version"])
        assert code == 0
        assert "Python" in output or "python" in output.lower()

    def test_run_cmd_failure(self) -> None:
        """Test commande échouée."""
        code, output = run_cmd(["python", "-c", "import sys; sys.exit(1)"])
        assert code == 1

    def test_run_cmd_not_found(self) -> None:
        """Test commande inexistante."""
        code, output = run_cmd(["nonexistent_command_xyz"])
        assert code == -1
        assert "not found" in output.lower()

    def test_run_cmd_python_m_conversion(self) -> None:
        """Test conversion commande vers python -m."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            run_cmd(["pydeps", "--version"])

            called_cmd = mock_run.call_args[0][0]
            assert "-m" in called_cmd
            assert "pydeps" in called_cmd

    def test_run_cmd_radon_conversion(self) -> None:
        """Test conversion radon vers python -m."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="out", stderr="err")
            code, output = run_cmd(["radon", "cc", "app"])

            called_cmd = mock_run.call_args[0][0]
            assert "-m" in called_cmd


class TestPrintHeader:
    """Tests pour print_header."""

    def test_print_header(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test affichage header."""
        print_header("Test Title")
        captured = capsys.readouterr()
        assert "Test Title" in captured.out
        assert "=" * 60 in captured.out


class TestConstants:
    """Tests pour les constantes."""

    def test_root_exists(self) -> None:
        """Test que ROOT existe."""
        assert ROOT.exists()

    def test_graphs_dir_path(self) -> None:
        """Test chemin GRAPHS_DIR."""
        assert GRAPHS_DIR == ROOT / "graphs"

    def test_reports_dir_path(self) -> None:
        """Test chemin REPORTS_DIR."""
        assert REPORTS_DIR == ROOT / "reports"
