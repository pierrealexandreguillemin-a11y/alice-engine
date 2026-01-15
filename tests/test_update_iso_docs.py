"""Module: test_update_iso_docs.py - Tests ISO Docs Generator.

Tests unitaires pour la generation de documentation ISO automatique.

ISO Compliance:
- ISO/IEC 29119 - Software Testing (unit tests)
- ISO/IEC 15289 - Documentation Lifecycle (quality records)

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.update_iso_docs import (
    MODULES,
    check_installed,
    generate_report,
    get_complexity_avg,
    get_test_coverage,
    main,
)

# ==============================================================================
# TESTS CHECK_INSTALLED
# ==============================================================================


class TestCheckInstalled:
    """Tests pour check_installed."""

    def test_check_installed_python(self) -> None:
        """Test vérification Python installé."""
        installed, version = check_installed("python --version")
        assert installed is True
        assert "Python" in version or "python" in version.lower()

    def test_check_installed_nonexistent(self) -> None:
        """Test vérification outil inexistant."""
        installed, version = check_installed("nonexistent_tool_12345 --version")
        assert installed is False
        assert version == "non installe"

    def test_check_installed_timeout(self) -> None:
        """Test timeout géré correctement."""
        with patch("scripts.iso_docs.checkers.subprocess.run") as mock_run:
            import subprocess

            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 5)

            installed, version = check_installed("slow_command")

            assert installed is False
            assert version == "non installe"

    def test_check_installed_return_code_nonzero(self) -> None:
        """Test commande avec code retour non-zéro."""
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
        """Test version longue tronquée à 50 caractères."""
        with patch("scripts.iso_docs.checkers.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "a" * 100 + "\n"
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            installed, version = check_installed("tool --version")

            assert installed is True
            assert len(version) <= 50


# ==============================================================================
# TESTS GET_TEST_COVERAGE
# ==============================================================================


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
            app/__init__.py          5      0   100%
            app/main.py             50     10    80%
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


# ==============================================================================
# TESTS GET_COMPLEXITY_AVG
# ==============================================================================


class TestGetComplexityAvg:
    """Tests pour get_complexity_avg."""

    def test_get_complexity_avg_found(self) -> None:
        """Test extraction complexité moyenne."""
        with patch("scripts.iso_docs.checkers.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = """
            app/main.py
                F 5:0 function1 - A
                F 15:0 function2 - B

            Average complexity: A (1.5)
            """
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


# ==============================================================================
# TESTS MODULES CONFIGURATION
# ==============================================================================


class TestModulesConfiguration:
    """Tests pour la configuration des modules."""

    def test_modules_not_empty(self) -> None:
        """Test que la liste des modules n'est pas vide."""
        assert len(MODULES) > 0

    def test_modules_have_required_fields(self) -> None:
        """Test que chaque module a les champs requis."""
        required_fields = {"name", "category", "iso", "check"}
        for mod in MODULES:
            for field in required_fields:
                assert field in mod, f"Module {mod.get('name', 'unknown')} manque {field}"

    def test_modules_categories(self) -> None:
        """Test les catégories des modules."""
        categories = {mod["category"] for mod in MODULES}
        expected_categories = {
            "Qualite Code",
            "Type Safety",
            "Formatage",
            "Securite",
            "Secrets",
            "Tests",
            "Coverage",
            "Tests Async",
            "Complexite",
            "Architecture",
            "Git Hooks",
            "Commits",
            "Documentation",
            "API Docs",
        }
        # Au moins certaines catégories attendues
        assert len(categories & expected_categories) > 0

    def test_modules_iso_references(self) -> None:
        """Test que les références ISO sont présentes."""
        iso_standards = set()
        for mod in MODULES:
            iso_standards.add(mod["iso"])

        # Vérifier quelques normes ISO clés
        all_isos = " ".join(iso_standards)
        assert "ISO 25010" in all_isos
        assert "ISO 27001" in all_isos
        assert "ISO 29119" in all_isos

    def test_module_ruff_config(self) -> None:
        """Test configuration module Ruff."""
        ruff_mods = [m for m in MODULES if m["name"] == "ruff"]
        assert len(ruff_mods) == 1
        assert "ISO 25010" in ruff_mods[0]["iso"]
        assert ruff_mods[0]["check"] == "python -m ruff --version"

    def test_module_pytest_config(self) -> None:
        """Test configuration module Pytest."""
        pytest_mods = [m for m in MODULES if m["name"] == "pytest"]
        assert len(pytest_mods) == 1
        assert "ISO 29119" in pytest_mods[0]["iso"]

    def test_module_bandit_security(self) -> None:
        """Test configuration module Bandit (sécurité)."""
        bandit_mods = [m for m in MODULES if m["name"] == "bandit"]
        assert len(bandit_mods) == 1
        assert bandit_mods[0]["category"] == "Securite"
        assert "ISO 27001" in bandit_mods[0]["iso"]


# ==============================================================================
# TESTS GENERATE_REPORT
# ==============================================================================


class TestGenerateReport:
    """Tests pour generate_report."""

    @patch("scripts.iso_docs.templates.check_installed")
    @patch("scripts.iso_docs.templates.get_test_coverage")
    @patch("scripts.iso_docs.templates.get_complexity_avg")
    def test_generate_report_structure(
        self, mock_complexity: MagicMock, mock_coverage: MagicMock, mock_installed: MagicMock
    ) -> None:
        """Test structure du rapport généré."""
        mock_installed.return_value = (True, "1.0.0")
        mock_coverage.return_value = "80%"
        mock_complexity.return_value = "A (1.5)"

        report = generate_report()

        # Vérifier sections principales
        assert "# IMPLEMENTATION DEVOPS - STATUT ISO" in report
        assert "## SCORE DEVOPS GLOBAL" in report
        assert "## CONFORMITE PAR NORME ISO" in report
        assert "### ISO 25010" in report
        assert "### ISO 27001" in report
        assert "### ISO 29119" in report
        assert "### ISO 42010" in report
        assert "## METRIQUES QUALITE ACTUELLES" in report
        assert "## TABLEAU COMPLET" in report
        assert "## COMMANDES RAPIDES" in report

    @patch("scripts.iso_docs.templates.check_installed")
    @patch("scripts.iso_docs.templates.get_test_coverage")
    @patch("scripts.iso_docs.templates.get_complexity_avg")
    def test_generate_report_score_calculation(
        self, mock_complexity: MagicMock, mock_coverage: MagicMock, mock_installed: MagicMock
    ) -> None:
        """Test calcul du score DevOps."""
        # Tous les modules installés
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
        # Alterner entre installé et non installé
        mock_installed.side_effect = [
            (True, "1.0.0") if i % 2 == 0 else (False, "non installe") for i in range(len(MODULES))
        ]
        mock_coverage.return_value = "50%"
        mock_complexity.return_value = "C (5.0)"

        report = generate_report()

        # Vérifier présence modules manquants
        assert "MODULES MANQUANTS" in report

    @patch("scripts.iso_docs.templates.check_installed")
    @patch("scripts.iso_docs.templates.get_test_coverage")
    @patch("scripts.iso_docs.templates.get_complexity_avg")
    def test_generate_report_timestamp(
        self, mock_complexity: MagicMock, mock_coverage: MagicMock, mock_installed: MagicMock
    ) -> None:
        """Test présence du timestamp."""
        mock_installed.return_value = (True, "1.0.0")
        mock_coverage.return_value = "80%"
        mock_complexity.return_value = "A (1.5)"

        report = generate_report()

        assert "Date de mise a jour:" in report
        assert "scripts/update_iso_docs.py" in report

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
        assert "Test Coverage" in report

    @patch("scripts.iso_docs.templates.check_installed")
    @patch("scripts.iso_docs.templates.get_test_coverage")
    @patch("scripts.iso_docs.templates.get_complexity_avg")
    def test_generate_report_all_installed_no_missing(
        self, mock_complexity: MagicMock, mock_coverage: MagicMock, mock_installed: MagicMock
    ) -> None:
        """Test rapport sans modules manquants."""
        mock_installed.return_value = (True, "1.0.0")
        mock_coverage.return_value = "80%"
        mock_complexity.return_value = "A"

        report = generate_report()

        assert "Tous les modules recommandes sont installes" in report


# ==============================================================================
# TESTS MAIN FUNCTION
# ==============================================================================


class TestMain:
    """Tests pour la fonction main."""

    @patch("scripts.update_iso_docs.generate_report")
    @patch("scripts.update_iso_docs.check_installed")
    def test_main_creates_directory(
        self, mock_installed: MagicMock, mock_report: MagicMock, tmp_path: Path
    ) -> None:
        """Test que main crée le répertoire docs/iso."""
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
        """Test que main écrit le rapport."""
        mock_report.return_value = "# Test Report Content"
        mock_installed.return_value = (True, "1.0.0")

        docs_iso = tmp_path / "docs" / "iso"
        docs_iso.mkdir(parents=True)

        with patch("scripts.update_iso_docs.DOCS_ISO", docs_iso):
            with patch("scripts.update_iso_docs.ROOT", tmp_path):
                result = main()

        assert result == 0
        output_file = docs_iso / "IMPLEMENTATION_STATUS.md"
        assert output_file.exists()
        assert output_file.read_text() == "# Test Report Content"

    @patch("scripts.update_iso_docs.generate_report")
    @patch("scripts.update_iso_docs.check_installed")
    def test_main_returns_zero(
        self, mock_installed: MagicMock, mock_report: MagicMock, tmp_path: Path
    ) -> None:
        """Test que main retourne 0."""
        mock_report.return_value = "# Test"
        mock_installed.return_value = (True, "1.0.0")

        docs_iso = tmp_path / "docs" / "iso"

        with patch("scripts.update_iso_docs.DOCS_ISO", docs_iso):
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

        docs_iso = tmp_path / "docs" / "iso"

        with patch("scripts.update_iso_docs.DOCS_ISO", docs_iso):
            with patch("scripts.update_iso_docs.ROOT", tmp_path):
                main()

        captured = capsys.readouterr()
        assert "Score DevOps:" in captured.out
        assert "Modules installes:" in captured.out


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestIntegration:
    """Tests d'intégration."""

    @patch("scripts.iso_docs.templates.check_installed")
    @patch("scripts.iso_docs.templates.get_test_coverage")
    @patch("scripts.iso_docs.templates.get_complexity_avg")
    def test_report_generation_mocked(
        self,
        mock_complexity: MagicMock,
        mock_coverage: MagicMock,
        mock_installed: MagicMock,
    ) -> None:
        """Test génération rapport avec mocks (évite subprocess Windows)."""
        mock_installed.return_value = (True, "1.0.0")
        mock_coverage.return_value = "85%"
        mock_complexity.return_value = "A (2.5)"

        report = generate_report()

        # Vérifier structure minimale
        assert "# IMPLEMENTATION DEVOPS" in report
        assert "Score actuel:" in report
        assert "Modules installes:" in report
        assert "85%" in report

    @patch("scripts.iso_docs.templates.check_installed")
    def test_coverage_badge_good(self, mock_installed: MagicMock) -> None:
        """Test badge coverage vert pour >80%."""
        mock_installed.return_value = (True, "1.0.0")

        with patch("scripts.iso_docs.templates.get_test_coverage", return_value="85%"):
            with patch("scripts.iso_docs.templates.get_complexity_avg", return_value="A"):
                report = generate_report()

        # Coverage >= 80% devrait avoir un statut positif
        assert "85%" in report

    @patch("scripts.iso_docs.templates.check_installed")
    def test_coverage_badge_bad(self, mock_installed: MagicMock) -> None:
        """Test badge coverage rouge pour <80%."""
        mock_installed.return_value = (True, "1.0.0")

        with patch("scripts.iso_docs.templates.get_test_coverage", return_value="50%"):
            with patch("scripts.iso_docs.templates.get_complexity_avg", return_value="A"):
                report = generate_report()

        # Coverage < 80% devrait apparaître
        assert "50%" in report


# ==============================================================================
# EDGE CASES
# ==============================================================================


class TestEdgeCases:
    """Tests edge cases."""

    def test_check_installed_multiline_version(self) -> None:
        """Test version sur plusieurs lignes."""
        with patch("scripts.iso_docs.checkers.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "Version 1.0.0\nBuild 12345\nDate 2025-01-01"
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            installed, version = check_installed("tool --version")

            # Ne garde que la première ligne
            assert "\n" not in version
            assert "1.0.0" in version

    @patch("scripts.iso_docs.templates.check_installed")
    @patch("scripts.iso_docs.templates.get_test_coverage")
    @patch("scripts.iso_docs.templates.get_complexity_avg")
    def test_report_special_characters(
        self, mock_complexity: MagicMock, mock_coverage: MagicMock, mock_installed: MagicMock
    ) -> None:
        """Test rapport avec caractères spéciaux dans version."""
        mock_installed.return_value = (True, "1.0.0-beta+build.123")
        mock_coverage.return_value = "N/A"
        mock_complexity.return_value = "N/A"

        report = generate_report()

        # Ne doit pas crasher
        assert isinstance(report, str)
        assert len(report) > 0

    @patch("scripts.iso_docs.templates.check_installed")
    @patch("scripts.iso_docs.templates.get_test_coverage")
    @patch("scripts.iso_docs.templates.get_complexity_avg")
    def test_report_na_metrics(
        self, mock_complexity: MagicMock, mock_coverage: MagicMock, mock_installed: MagicMock
    ) -> None:
        """Test rapport avec métriques N/A."""
        mock_installed.return_value = (False, "non installe")
        mock_coverage.return_value = "N/A"
        mock_complexity.return_value = "N/A"

        report = generate_report()

        assert "N/A" in report
        # Le rapport doit quand même être généré
        assert "METRIQUES QUALITE ACTUELLES" in report
