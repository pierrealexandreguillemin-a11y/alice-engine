"""Tests Types Bias - ISO 24027.

Document ID: ALICE-TEST-BIAS-TYPES
Version: 1.0.0
Tests: 7

Classes:
- TestFairnessEnums: Tests enums (2 tests)
- TestBiasMetrics: Tests dataclass metrics (2 tests)
- TestBiasMonitorResult: Tests dataclass result (2 tests)
- TestBiasMonitorConfig: Tests config (1 test)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<100 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from scripts.monitoring.bias_types import (
    DEMOGRAPHIC_PARITY_THRESHOLD,
    DISPARATE_IMPACT_THRESHOLD,
    EQUALIZED_ODDS_THRESHOLD,
    BiasAlertLevel,
    BiasMetrics,
    BiasMonitorConfig,
    BiasMonitorResult,
    FairnessStatus,
)

# Fixtures are auto-loaded via pytest_plugins in conftest.py


class TestFairnessEnums:
    """Tests pour les enums de fairness."""

    def test_fairness_status_values(self):
        """Vérifie les valeurs de statut."""
        assert FairnessStatus.FAIR.value == "fair"
        assert FairnessStatus.CAUTION.value == "caution"
        assert FairnessStatus.BIASED.value == "biased"
        assert FairnessStatus.CRITICAL.value == "critical"

    def test_bias_alert_level_values(self):
        """Vérifie les niveaux d'alerte."""
        assert BiasAlertLevel.NONE.value == "none"
        assert BiasAlertLevel.INFO.value == "info"
        assert BiasAlertLevel.WARNING.value == "warning"
        assert BiasAlertLevel.CRITICAL.value == "critical"


class TestBiasMetrics:
    """Tests pour BiasMetrics."""

    def test_metrics_creation(self):
        """Création métriques de biais."""
        metrics = BiasMetrics(
            group_name="gender",
            group_value="F",
            n_samples=200,
            positive_rate=0.65,
            true_positive_rate=0.75,
            false_positive_rate=0.15,
        )
        assert metrics.group_name == "gender"
        assert metrics.positive_rate == 0.65

    def test_metrics_to_dict(self):
        """Conversion en dict."""
        metrics = BiasMetrics(
            group_name="age", group_value="25-34", n_samples=100, positive_rate=0.5
        )
        d = metrics.to_dict()
        assert d["group"] == "age=25-34"
        assert d["n_samples"] == 100


class TestBiasMonitorResult:
    """Tests pour BiasMonitorResult."""

    def test_result_creation(self):
        """Création résultat monitoring."""
        result = BiasMonitorResult(
            timestamp="2026-01-14",
            protected_attribute="gender",
            n_total_samples=500,
            demographic_parity=0.85,
        )
        assert result.overall_status == FairnessStatus.FAIR

    def test_result_to_dict(self):
        """Conversion résultat en dict."""
        result = BiasMonitorResult(
            timestamp="2026-01-14",
            protected_attribute="age",
            n_total_samples=1000,
            demographic_parity=0.75,
            overall_status=FairnessStatus.BIASED,
        )
        d = result.to_dict()
        assert d["status"] == "biased"
        assert d["metrics"]["demographic_parity"] == 0.75


class TestBiasMonitorConfig:
    """Tests pour BiasMonitorConfig."""

    def test_config_from_dict(self):
        """Reconstruction config depuis dict."""
        data = {
            "protected_attributes": ["gender"],
            "thresholds": {"demographic_parity": 0.85},
            "min_group_size": 50,
            "model_version": "v2.0",
        }
        config = BiasMonitorConfig.from_dict(data)
        assert config.demographic_parity_threshold == 0.85
        assert config.min_group_size == 50


class TestConstants:
    """Tests pour les constantes EEOC."""

    def test_eeoc_thresholds(self):
        """Vérifie les seuils EEOC 80% rule."""
        assert DEMOGRAPHIC_PARITY_THRESHOLD == 0.8
        assert DISPARATE_IMPACT_THRESHOLD == 0.8
        assert EQUALIZED_ODDS_THRESHOLD == 0.1
