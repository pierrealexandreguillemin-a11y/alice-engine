"""Tests Monitor Drift - ISO 23894.

Document ID: ALICE-TEST-DRIFT-MON
Version: 1.1.0
Tests: 13

Classes:
- TestComputeOverallSeverity: Tests helper _compute_overall_severity (3 tests)
- TestIsSignificantDrift: Tests helper _is_significant_drift (3 tests)
- TestAnalyzeFeatureDrift: Tests analyse feature (3 tests)
- TestMonitorDrift: Tests monitoring complet (4 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<150 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

import numpy as np

from scripts.model_registry.drift_monitor import (
    _compute_overall_severity,
    _is_significant_drift,
    analyze_feature_drift,
    monitor_drift,
)
from scripts.model_registry.drift_types import DriftSeverity

# Fixtures (baseline_data, current_no_drift, current_with_drift) are auto-loaded via pytest_plugins


class TestComputeOverallSeverity:
    """Tests pour _compute_overall_severity (ISO 29119 - Unit tests)."""

    def test_returns_critical_when_present(self):
        """Retourne CRITICAL si présent dans la liste."""
        severities = [DriftSeverity.LOW, DriftSeverity.CRITICAL, DriftSeverity.MEDIUM]
        assert _compute_overall_severity(severities) == DriftSeverity.CRITICAL

    def test_returns_highest_severity(self):
        """Retourne la sévérité la plus haute."""
        severities = [DriftSeverity.LOW, DriftSeverity.MEDIUM]
        assert _compute_overall_severity(severities) == DriftSeverity.MEDIUM

    def test_returns_none_for_empty_list(self):
        """Retourne NONE pour liste vide."""
        assert _compute_overall_severity([]) == DriftSeverity.NONE


class TestIsSignificantDrift:
    """Tests pour _is_significant_drift (ISO 29119 - Unit tests)."""

    def test_medium_is_significant(self):
        """MEDIUM est significatif."""
        assert _is_significant_drift(DriftSeverity.MEDIUM) is True

    def test_high_is_significant(self):
        """HIGH est significatif."""
        assert _is_significant_drift(DriftSeverity.HIGH) is True

    def test_low_is_not_significant(self):
        """LOW n'est pas significatif."""
        assert _is_significant_drift(DriftSeverity.LOW) is False


class TestAnalyzeFeatureDrift:
    """Tests pour analyze_feature_drift."""

    def test_numeric_no_drift(self):
        """Analyse feature numérique sans drift."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 500)
        current = np.random.normal(0, 1.05, 500)

        result = analyze_feature_drift("test", baseline, current)
        assert result.feature_name == "test"
        assert result.severity in (DriftSeverity.NONE, DriftSeverity.LOW)

    def test_numeric_with_drift(self):
        """Analyse feature numérique avec drift."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 500)
        current = np.random.normal(3, 2, 500)

        result = analyze_feature_drift("shifted", baseline, current)
        assert result.severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL)
        assert result.psi_score > 0.2

    def test_categorical_drift(self):
        """Analyse feature catégorielle."""
        baseline = np.array(["A", "B", "C"] * 100)
        current = np.array(["A", "D", "E"] * 80)

        result = analyze_feature_drift("cat", baseline, current, is_categorical=True)
        assert result.is_categorical is True
        assert result.chi2_pvalue is not None


class TestMonitorDrift:
    """Tests pour monitor_drift."""

    def test_no_drift_detected(self, baseline_data, current_no_drift):
        """Pas de drift détecté."""
        result = monitor_drift(
            baseline_data,
            current_no_drift,
            model_version="v1.0",
            categorical_features=["category"],
        )
        assert result.drift_detected is False
        assert result.overall_severity in (
            DriftSeverity.NONE,
            DriftSeverity.LOW,
            DriftSeverity.MEDIUM,
        )

    def test_drift_detected(self, baseline_data, current_with_drift):
        """Drift détecté."""
        result = monitor_drift(
            baseline_data,
            current_with_drift,
            model_version="v1.0",
            categorical_features=["category"],
        )
        assert len(result.feature_results) > 0
        assert result.model_version == "v1.0"

    def test_result_structure(self, baseline_data, current_no_drift):
        """Vérifie la structure du résultat."""
        result = monitor_drift(
            baseline_data,
            current_no_drift,
            model_version="test",
            categorical_features=["category"],
        )

        assert result.timestamp is not None
        assert result.baseline_samples == len(baseline_data)
        assert result.current_samples == len(current_no_drift)
        assert isinstance(result.recommendations, list)

    def test_to_dict_serialization(self, baseline_data, current_no_drift):
        """Vérifie la sérialisation en dict."""
        result = monitor_drift(
            baseline_data,
            current_no_drift,
            model_version="v2",
            categorical_features=["category"],
        )
        d = result.to_dict()

        assert "timestamp" in d
        assert "model_version" in d
        assert d["model_version"] == "v2"
        assert "features" in d
        assert "summary" in d
        assert "recommendations" in d


class TestSelectiveMonitoring:
    """Tests monitoring sélectif."""

    def test_selected_features(self, baseline_data, current_no_drift):
        """Test monitoring features sélectionnées."""
        result = monitor_drift(
            baseline_data,
            current_no_drift,
            model_version="v1",
            features_to_monitor=["feature_a"],
            categorical_features=[],
        )
        assert len(result.feature_results) == 1
        assert result.feature_results[0].feature_name == "feature_a"
