"""Tests Modules Configuration - ISO 29119.

Document ID: ALICE-TEST-ISO-MODULES
Version: 1.0.0
Tests: 7

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<100 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from scripts.update_iso_docs import MODULES


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
        """Test les categories des modules."""
        categories = {mod["category"] for mod in MODULES}
        expected = {"Qualite Code", "Securite", "Tests", "Coverage", "Complexite"}
        assert len(categories & expected) > 0

    def test_modules_iso_references(self) -> None:
        """Test que les references ISO sont presentes."""
        all_isos = " ".join(mod["iso"] for mod in MODULES)
        assert "ISO 25010" in all_isos
        assert "ISO 27001" in all_isos
        assert "ISO 29119" in all_isos

    def test_module_ruff_config(self) -> None:
        """Test configuration module Ruff."""
        ruff_mods = [m for m in MODULES if m["name"] == "ruff"]
        assert len(ruff_mods) == 1
        assert "ISO 25010" in ruff_mods[0]["iso"]

    def test_module_pytest_config(self) -> None:
        """Test configuration module Pytest."""
        pytest_mods = [m for m in MODULES if m["name"] == "pytest"]
        assert len(pytest_mods) == 1
        assert "ISO 29119" in pytest_mods[0]["iso"]

    def test_module_bandit_security(self) -> None:
        """Test configuration module Bandit (securite)."""
        bandit_mods = [m for m in MODULES if m["name"] == "bandit"]
        assert len(bandit_mods) == 1
        assert bandit_mods[0]["category"] == "Securite"
