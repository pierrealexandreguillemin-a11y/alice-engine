"""Tests Types Input - ISO 24029.

Document ID: ALICE-TEST-INPUT-TYPES
Version: 1.0.0
Tests: 7

Classes:
- TestOODEnums: Tests enums (2 tests)
- TestFeatureBounds: Tests dataclass bounds (3 tests)
- TestInputValidationResult: Tests dataclass result (2 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<100 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from scripts.model_registry.input_types import (
    DEFAULT_STD_TOLERANCE,
    OOD_REJECTION_THRESHOLD,
    InputValidationResult,
    OODAction,
    OODSeverity,
)

# Fixtures (categorical_bounds, numeric_bounds) are auto-loaded via pytest_plugins


class TestOODEnums:
    """Tests pour les enums OOD."""

    def test_ood_severity_values(self):
        """Vérifie les valeurs de sévérité OOD."""
        assert OODSeverity.VALID.value == "valid"
        assert OODSeverity.WARNING.value == "warning"
        assert OODSeverity.OUT_OF_BOUNDS.value == "out_of_bounds"
        assert OODSeverity.EXTREME.value == "extreme"

    def test_ood_action_values(self):
        """Vérifie les valeurs d'action OOD."""
        assert OODAction.ACCEPT.value == "accept"
        assert OODAction.WARN.value == "warn"
        assert OODAction.REJECT.value == "reject"
        assert OODAction.CLAMP.value == "clamp"


class TestFeatureBounds:
    """Tests pour FeatureBounds."""

    def test_check_valid_value(self, numeric_bounds):
        """Vérifie valeur dans les bornes."""
        severity, msg = numeric_bounds.check_value(50000)
        assert severity == OODSeverity.VALID
        assert msg == ""

    def test_check_warning_value(self, numeric_bounds):
        """Vérifie valeur en zone warning (hors p01-p99)."""
        severity, msg = numeric_bounds.check_value(21500)  # Below p01 but within min/max
        assert severity == OODSeverity.WARNING

    def test_check_categorical(self, categorical_bounds):
        """Vérifie catégorie valide et invalide."""
        sev_valid, _ = categorical_bounds.check_value("A")
        assert sev_valid == OODSeverity.VALID

        sev_invalid, msg = categorical_bounds.check_value("Z")
        assert sev_invalid == OODSeverity.OUT_OF_BOUNDS
        assert "Unknown category" in msg


class TestInputValidationResult:
    """Tests pour InputValidationResult."""

    def test_result_creation(self):
        """Vérifie création résultat."""
        result = InputValidationResult(is_valid=True, action=OODAction.ACCEPT)
        assert result.is_valid is True
        assert result.action == OODAction.ACCEPT
        assert result.ood_ratio == 0.0

    def test_result_to_dict(self):
        """Vérifie conversion dict."""
        result = InputValidationResult(is_valid=False, action=OODAction.REJECT, ood_ratio=0.5)
        d = result.to_dict()
        assert d["is_valid"] is False
        assert d["action"] == "reject"
        assert d["ood_ratio"] == 0.5


class TestConstants:
    """Tests pour les constantes ISO 24029."""

    def test_default_constants(self):
        """Vérifie les constantes par défaut."""
        assert DEFAULT_STD_TOLERANCE == 4.0
        assert OOD_REJECTION_THRESHOLD == 0.3
