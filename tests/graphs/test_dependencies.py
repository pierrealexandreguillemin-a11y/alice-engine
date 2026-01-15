"""Tests Graphs Dependencies - ISO 29119.

Document ID: ALICE-TEST-GRAPHS-DEPENDENCIES
Version: 1.0.0

Tests pour dependencies.py.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path
from unittest.mock import MagicMock, patch


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

        mock_run_cmd.side_effect = [(0, ""), (1, "")]

        result = generate_dependency_graph()

        assert result is False

    @patch("scripts.graphs.dependencies.run_cmd")
    def test_successful_generation(self, mock_run_cmd: MagicMock, tmp_path: Path) -> None:
        """Test génération réussie."""
        from scripts.graphs.dependencies import generate_dependency_graph

        output_file = tmp_path / "dependencies.svg"
        output_file.write_text("<svg>graph</svg>")

        mock_run_cmd.side_effect = [
            (0, "pydeps version"),
            (0, "dot version"),
            (0, "generated"),
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
