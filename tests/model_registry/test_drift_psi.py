"""Tests Drift PSI & Metrics - ISO 29119.

Document ID: ALICE-TEST-MODEL-DRIFT-PSI
Version: 1.0.0
Tests: 2 classes (TestPSI, TestDriftMetrics)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)
- ISO/IEC 5259:2024 - Data Quality for ML (drift detection)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import numpy as np
import pandas as pd
import pytest

from scripts.model_registry import (
    PSI_THRESHOLD_WARNING,
    DriftMetrics,
    compute_drift_metrics,
    compute_psi,
)


class TestPSI:
    """Tests pour Population Stability Index."""

    def test_psi_identical_distributions(self) -> None:
        """Test PSI=0 pour distributions identiques."""
        baseline = pd.Series([1500] * 100 + [1600] * 100 + [1700] * 100)
        current = pd.Series([1500] * 100 + [1600] * 100 + [1700] * 100)
        psi = compute_psi(baseline, current)
        assert psi < 0.01

    def test_psi_slight_shift(self) -> None:
        """Test PSI faible pour leger decalage."""
        np.random.seed(42)
        baseline = pd.Series(np.random.normal(1500, 200, 1000))
        current = pd.Series(np.random.normal(1520, 200, 1000))
        psi = compute_psi(baseline, current)
        assert psi < PSI_THRESHOLD_WARNING

    def test_psi_significant_shift(self) -> None:
        """Test PSI eleve pour decalage significatif."""
        np.random.seed(42)
        baseline = pd.Series(np.random.normal(1500, 200, 1000))
        current = pd.Series(np.random.normal(1700, 200, 1000))
        psi = compute_psi(baseline, current)
        assert psi > PSI_THRESHOLD_WARNING


class TestDriftMetrics:
    """Tests pour calcul metriques de drift."""

    @pytest.fixture
    def sample_predictions(self) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Cree des donnees de prediction de test."""
        predictions = pd.DataFrame(
            {
                "predicted_proba": [0.7, 0.3, 0.8, 0.4, 0.6, 0.2, 0.9, 0.5],
                "elo_blanc": [1500, 1600, 1700, 1400, 1550, 1450, 1800, 1500],
                "elo_noir": [1450, 1550, 1650, 1500, 1500, 1550, 1700, 1550],
            }
        )
        actuals = pd.Series([1, 0, 1, 0, 1, 0, 1, 1])
        baseline_elo = pd.Series([1500, 1550, 1600, 1450, 1650, 1400, 1700, 1500] * 10)
        return predictions, actuals, baseline_elo

    def test_compute_drift_metrics(self, sample_predictions: tuple) -> None:
        """Test calcul metriques de drift."""
        predictions, actuals, baseline_elo = sample_predictions
        metrics = compute_drift_metrics(
            round_number=1,
            predictions=predictions,
            actuals=actuals,
            baseline_elo_mean=1550,
            baseline_elo_std=150,
            baseline_elo_distribution=baseline_elo,
        )
        assert isinstance(metrics, DriftMetrics)
        assert metrics.round_number == 1
        assert 0 <= metrics.accuracy <= 1
        assert metrics.predictions_count == 8

    def test_drift_metrics_to_dict(self, sample_predictions: tuple) -> None:
        """Test conversion metriques en dict."""
        predictions, actuals, baseline_elo = sample_predictions
        metrics = compute_drift_metrics(
            round_number=1,
            predictions=predictions,
            actuals=actuals,
            baseline_elo_mean=1550,
            baseline_elo_std=150,
            baseline_elo_distribution=baseline_elo,
        )
        metrics_dict = metrics.to_dict()
        assert "round" in metrics_dict
        assert "drift" in metrics_dict

    def test_drift_alert_on_elo_shift(self) -> None:
        """Test alerte sur shift ELO significatif."""
        predictions = pd.DataFrame(
            {
                "predicted_proba": [0.5] * 10,
                "elo_blanc": [1800] * 10,
                "elo_noir": [1750] * 10,
            }
        )
        actuals = pd.Series([1] * 5 + [0] * 5)
        baseline_elo = pd.Series([1500, 1450, 1550, 1400, 1600] * 20)
        metrics = compute_drift_metrics(
            round_number=1,
            predictions=predictions,
            actuals=actuals,
            baseline_elo_mean=1500,
            baseline_elo_std=150,
            baseline_elo_distribution=baseline_elo,
        )
        assert metrics.has_warning is True

    def test_missing_columns_raises_error(self) -> None:
        """Test erreur si colonnes manquantes."""
        predictions = pd.DataFrame(
            {
                "predicted_proba": [0.5, 0.6],
                "elo_blanc": [1500, 1600],
            }
        )
        actuals = pd.Series([1, 0])
        baseline_elo = pd.Series([1500] * 10)
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_drift_metrics(
                round_number=1,
                predictions=predictions,
                actuals=actuals,
                baseline_elo_mean=1500,
                baseline_elo_std=150,
                baseline_elo_distribution=baseline_elo,
            )
