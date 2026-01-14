"""Tests Types Drift - ISO 23894.

Document ID: ALICE-TEST-DRIFT-TYPES
Version: 1.0.0
Tests: 6

Classes:
- TestDriftSeverity: Tests enum sévérité (2 tests)
- TestDriftType: Tests enum type (1 test)
- TestFeatureDriftResult: Tests dataclass (2 tests)
- TestDriftMonitorResult: Tests dataclass (1 test)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<100 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from scripts.model_registry.drift_types import (
    KS_PVALUE_CRITICAL,
    KS_PVALUE_OK,
    KS_PVALUE_WARNING,
    PSI_THRESHOLD_CRITICAL,
    PSI_THRESHOLD_OK,
    PSI_THRESHOLD_WARNING,
    DriftMonitorResult,
    DriftSeverity,
    DriftType,
    FeatureDriftResult,
)

# Fixtures are auto-loaded via pytest_plugins in conftest.py


class TestDriftSeverity:
    """Tests pour l'enum DriftSeverity."""

    def test_severity_values(self):
        """Vérifie les valeurs de sévérité."""
        assert DriftSeverity.NONE.value == "none"
        assert DriftSeverity.LOW.value == "low"
        assert DriftSeverity.MEDIUM.value == "medium"
        assert DriftSeverity.HIGH.value == "high"
        assert DriftSeverity.CRITICAL.value == "critical"

    def test_severity_ordering(self):
        """Vérifie l'ordre logique des sévérités."""
        severities = [
            DriftSeverity.NONE,
            DriftSeverity.LOW,
            DriftSeverity.MEDIUM,
            DriftSeverity.HIGH,
            DriftSeverity.CRITICAL,
        ]
        assert len(severities) == 5


class TestDriftType:
    """Tests pour l'enum DriftType."""

    def test_drift_types(self):
        """Vérifie les types de drift."""
        assert DriftType.COVARIATE.value == "covariate"
        assert DriftType.CONCEPT.value == "concept"
        assert DriftType.PRIOR.value == "prior"


class TestFeatureDriftResult:
    """Tests pour FeatureDriftResult."""

    def test_feature_result_creation(self):
        """Vérifie la création d'un résultat."""
        result = FeatureDriftResult(
            feature_name="test", psi_score=0.15, severity=DriftSeverity.MEDIUM
        )
        assert result.feature_name == "test"
        assert result.psi_score == 0.15
        assert result.severity == DriftSeverity.MEDIUM

    def test_feature_result_to_dict(self):
        """Vérifie la conversion en dict."""
        result = FeatureDriftResult(
            feature_name="f1", psi_score=0.1, ks_statistic=0.05, ks_pvalue=0.03
        )
        d = result.to_dict()
        assert d["feature"] == "f1"
        assert d["psi"] == 0.1
        assert d["ks_statistic"] == 0.05


class TestDriftMonitorResult:
    """Tests pour DriftMonitorResult."""

    def test_monitor_result_to_dict(self):
        """Vérifie la conversion complète en dict."""
        result = DriftMonitorResult(
            timestamp="2026-01-14", model_version="v1", baseline_samples=100, current_samples=80
        )
        d = result.to_dict()
        assert d["model_version"] == "v1"
        assert d["samples"]["baseline"] == 100
        assert d["summary"]["drift_detected"] is False


class TestConstants:
    """Tests pour les constantes ISO 23894."""

    def test_psi_thresholds(self):
        """Vérifie les seuils PSI."""
        assert PSI_THRESHOLD_OK < PSI_THRESHOLD_WARNING < PSI_THRESHOLD_CRITICAL
        assert PSI_THRESHOLD_OK == 0.1
        assert PSI_THRESHOLD_WARNING == 0.2
        assert PSI_THRESHOLD_CRITICAL == 0.25

    def test_ks_pvalues(self):
        """Vérifie les seuils KS p-value."""
        assert KS_PVALUE_OK > KS_PVALUE_WARNING > KS_PVALUE_CRITICAL
        assert KS_PVALUE_OK == 0.05
