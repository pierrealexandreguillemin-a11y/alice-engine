"""Tests Metrics - ISO 29119.

Document ID: ALICE-TEST-TRAIN-METRICS
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import numpy as np

from scripts.ml_types import ModelMetrics
from scripts.training import compute_all_metrics


class TestComputeAllMetrics:
    """Tests pour compute_all_metrics."""

    def test_perfect_predictions(self) -> None:
        """Test metriques predictions parfaites."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)

        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.auc_roc == 1.0

    def test_all_wrong_predictions(self) -> None:
        """Test metriques predictions inversees."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.2, 0.1])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)

        assert metrics.accuracy == 0.0
        assert metrics.recall == 0.0

    def test_metrics_return_type(self) -> None:
        """Test type retour ModelMetrics."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_proba = np.array([0.3, 0.7, 0.6, 0.4])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)

        assert isinstance(metrics, ModelMetrics)
        assert isinstance(metrics.auc_roc, float)
        assert isinstance(metrics.accuracy, float)

    def test_confusion_matrix_values(self) -> None:
        """Test valeurs matrice confusion."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([0.3, 0.6, 0.4, 0.7])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)

        assert metrics.true_negatives == 1
        assert metrics.false_positives == 1
        assert metrics.false_negatives == 1
        assert metrics.true_positives == 1

    def test_metrics_to_dict(self) -> None:
        """Test conversion metriques en dict."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([0.2, 0.8, 0.3, 0.9])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)
        metrics_dict = metrics.to_dict()

        assert "auc_roc" in metrics_dict
        assert "accuracy" in metrics_dict
        assert "f1_score" in metrics_dict
        assert isinstance(metrics_dict["auc_roc"], float)
