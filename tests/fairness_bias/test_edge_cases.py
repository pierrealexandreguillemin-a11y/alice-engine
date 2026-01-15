"""Tests Edge Cases Fairness Bias - ISO 24027.

Document ID: ALICE-TEST-FAIRNESS-BIAS-EDGE
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC TR 24027:2021 - Bias in AI systems
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import numpy as np
import pandas as pd

from scripts.fairness.bias_detection import (
    BiasLevel,
    BiasMetrics,
    compute_bias_metrics_by_group,
)


class TestBiasLevel:
    """Tests pour l'enum BiasLevel."""

    def test_bias_levels(self):
        """Vérifie les niveaux de biais."""
        assert BiasLevel.ACCEPTABLE.value == "acceptable"
        assert BiasLevel.WARNING.value == "warning"
        assert BiasLevel.CRITICAL.value == "critical"

    def test_bias_level_comparison(self):
        """Vérifie comparaison des niveaux."""
        metrics_acceptable = BiasMetrics(
            group_name="A",
            group_size=10,
            positive_rate=0.5,
            true_positive_rate=0.5,
            level=BiasLevel.ACCEPTABLE,
        )
        metrics_critical = BiasMetrics(
            group_name="B",
            group_size=10,
            positive_rate=0.5,
            true_positive_rate=0.5,
            level=BiasLevel.CRITICAL,
        )

        assert metrics_acceptable.level != metrics_critical.level


class TestEdgeCases:
    """Tests pour cas limites."""

    def test_single_group(self):
        """Un seul groupe = pas de biais possible."""
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 1])
        groups = np.array(["A", "A", "A"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups)

        assert len(metrics) == 1
        assert metrics[0].spd == 0.0

    def test_all_positive_predictions(self):
        """Toutes prédictions positives."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 1])
        groups = np.array(["A", "A", "B", "B"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups)

        for m in metrics:
            assert m.positive_rate == 1.0
            assert m.spd == 0.0

    def test_all_negative_predictions(self):
        """Toutes prédictions négatives."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 0, 0, 0])
        groups = np.array(["A", "A", "B", "B"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups)

        for m in metrics:
            assert m.positive_rate == 0.0

    def test_pandas_input(self):
        """Test avec entrées pandas."""
        y_true = pd.Series([1, 0, 1, 0])
        y_pred = pd.Series([1, 0, 1, 0])
        groups = pd.Series(["A", "A", "B", "B"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups)

        assert len(metrics) == 2

    def test_division_by_zero_protection(self):
        """Protection contre division par zéro dans DIR."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        groups = np.array(["A", "A", "B", "B"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups, reference_group="B")

        group_a = next(m for m in metrics if m.group_name == "A")
        assert group_a.dir == float("inf") or group_a.dir > 10
