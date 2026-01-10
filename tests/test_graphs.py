"""Tests pour scripts/graphs - ISO 29119.

Ce module teste les fonctionnalités de génération de graphs:
- utils.py: run_cmd, print_header
- complexity.py: generate_complexity_report, generate_complexity_html
- dependencies.py: generate_dependency_graph, check_circular_imports

ISO 29119: Test coverage pour les outils de visualisation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.graphs.utils import GRAPHS_DIR, REPORTS_DIR, ROOT, print_header, run_cmd

# =============================================================================
# Tests pour utils.py
# =============================================================================


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
        # pydeps should be converted to python -m pydeps
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            run_cmd(["pydeps", "--version"])

            # Vérifier que la commande a été convertie
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


# =============================================================================
# Tests pour complexity.py
# =============================================================================


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

        # Mock run_cmd pour retourner succès
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

        # Créer fichiers source
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


# =============================================================================
# Tests pour dependencies.py
# =============================================================================


class TestGenerateDependencyGraph:
    """Tests pour generate_dependency_graph."""

    @patch("scripts.graphs.dependencies.run_cmd")
    def test_pydeps_not_installed(self, mock_run_cmd: MagicMock) -> None:
        """Test quand pydeps n'est pas installé."""
        from scripts.graphs.dependencies import generate_dependency_graph

        mock_run_cmd.return_value = (1, "")

        result = generate_dependency_graph()

        assert result is False

    @patch("scripts.graphs.dependencies.run_cmd")
    def test_graphviz_not_installed(self, mock_run_cmd: MagicMock) -> None:
        """Test quand graphviz n'est pas installé."""
        from scripts.graphs.dependencies import generate_dependency_graph

        # pydeps OK, mais dot (graphviz) échoue
        mock_run_cmd.side_effect = [(0, ""), (1, "")]

        result = generate_dependency_graph()

        assert result is False

    @patch("scripts.graphs.dependencies.run_cmd")
    def test_successful_generation(self, mock_run_cmd: MagicMock, tmp_path: Path) -> None:
        """Test génération réussie."""
        from scripts.graphs.dependencies import generate_dependency_graph

        # Créer le fichier de sortie simulé
        output_file = tmp_path / "dependencies.svg"
        output_file.write_text("<svg>graph</svg>")

        mock_run_cmd.side_effect = [
            (0, "pydeps version"),  # pydeps --version
            (0, "dot version"),  # dot -V
            (0, "generated"),  # pydeps app...
        ]

        with patch("scripts.graphs.dependencies.GRAPHS_DIR", tmp_path):
            with patch("scripts.graphs.dependencies.ROOT", tmp_path):
                result = generate_dependency_graph()

        assert result is True


class TestGenerateImportsGraph:
    """Tests pour generate_imports_graph."""

    @patch("scripts.graphs.dependencies.run_cmd")
    def test_successful_generation(self, mock_run_cmd: MagicMock, tmp_path: Path) -> None:
        """Test génération imports graph."""
        from scripts.graphs.dependencies import generate_imports_graph

        output_file = tmp_path / "imports.svg"
        output_file.write_text("<svg>imports</svg>")

        mock_run_cmd.return_value = (0, "generated")

        with patch("scripts.graphs.dependencies.GRAPHS_DIR", tmp_path):
            with patch("scripts.graphs.dependencies.ROOT", tmp_path):
                result = generate_imports_graph()

        assert result is True

    @patch("scripts.graphs.dependencies.run_cmd")
    def test_failed_generation(self, mock_run_cmd: MagicMock, tmp_path: Path) -> None:
        """Test échec génération."""
        from scripts.graphs.dependencies import generate_imports_graph

        mock_run_cmd.return_value = (1, "error")

        with patch("scripts.graphs.dependencies.GRAPHS_DIR", tmp_path):
            result = generate_imports_graph()

        assert result is False


class TestCheckCircularImports:
    """Tests pour check_circular_imports."""

    @patch("scripts.graphs.dependencies.run_cmd")
    def test_no_circular_imports(self, mock_run_cmd: MagicMock, tmp_path: Path) -> None:
        """Test aucun import circulaire."""
        from scripts.graphs.dependencies import check_circular_imports

        mock_run_cmd.return_value = (0, "No circular imports")

        with patch("scripts.graphs.dependencies.GRAPHS_DIR", tmp_path):
            with patch("scripts.graphs.dependencies.ROOT", tmp_path):
                result = check_circular_imports()

        assert result is True
        assert (tmp_path / "circular-imports.txt").exists()

    @patch("scripts.graphs.dependencies.run_cmd")
    def test_circular_imports_found(self, mock_run_cmd: MagicMock, tmp_path: Path) -> None:
        """Test imports circulaires détectés."""
        from scripts.graphs.dependencies import check_circular_imports

        mock_run_cmd.return_value = (1, "Cycle: a -> b -> a")

        with patch("scripts.graphs.dependencies.GRAPHS_DIR", tmp_path):
            with patch("scripts.graphs.dependencies.ROOT", tmp_path):
                result = check_circular_imports()

        assert result is False
