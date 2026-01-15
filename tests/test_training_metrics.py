"""Tests Training Metrics - ISO 29119/25010.

Document ID: ALICE-TEST-TRAINING-METRICS
Version: 1.0.0
Tests: 6

Classes:
- TestComputeAllMetrics: Tests calcul métriques (6 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 25010:2023 - Quality Model (performance metrics)
- ISO/IEC 5055:2021 - Code Quality (<80 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import numpy as np

from scripts.training.metrics import compute_all_metrics


class TestComputeAllMetrics:
    """Tests pour compute_all_metrics."""

    def test_perfect_predictions(self) -> None:
        """Métriques parfaites pour prédictions correctes."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)

        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.auc_roc > 0.9

    def test_random_predictions(self) -> None:
        """Métriques ~0.5 pour prédictions aléatoires."""
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_proba = np.array([0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)

        assert 0.3 <= metrics.accuracy <= 0.7
        assert metrics.auc_roc is not None

    def test_confusion_matrix_values(self) -> None:
        """Vérifie les valeurs de la matrice de confusion."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.9])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)

        assert metrics.true_negatives == 2
        assert metrics.false_positives == 1
        assert metrics.false_negatives == 1
        assert metrics.true_positives == 2

    def test_all_positive_predictions(self) -> None:
        """Gère cas où toutes prédictions sont positives."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        y_proba = np.array([0.6, 0.7, 0.8, 0.9])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)

        assert metrics.recall == 1.0
        assert metrics.precision == 0.5
        assert metrics.true_negatives == 0
        assert metrics.false_positives == 2

    def test_all_negative_predictions(self) -> None:
        """Gère cas où toutes prédictions sont négatives."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)

        assert metrics.recall == 0.0
        assert metrics.precision == 0.0  # zero_division=0
        assert metrics.true_positives == 0
        assert metrics.false_negatives == 2

    def test_log_loss_computed(self) -> None:
        """Vérifie que log_loss est calculé."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([0.2, 0.8, 0.3, 0.7])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)

        assert metrics.log_loss > 0
        assert metrics.log_loss < 1.0  # Bonnes probas = faible log_loss
