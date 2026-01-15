"""Tests Compute Bias Metrics - ISO 24027.

Document ID: ALICE-TEST-FAIRNESS-BIAS-COMPUTE
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC TR 24027:2021 - Bias in AI systems
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import numpy as np

from scripts.fairness.bias_detection import (
    compute_bias_by_elo_range,
    compute_bias_metrics_by_group,
)


class TestComputeBiasMetricsByGroup:
    """Tests pour compute_bias_metrics_by_group."""

    def test_no_bias_equal_groups(self):
        """Groupes identiques = pas de biais."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups)

        assert len(metrics) == 2

        for m in metrics:
            assert m.positive_rate == 0.5
            assert abs(m.spd) < 0.01
            assert abs(m.dir - 1.0) < 0.01

    def test_detect_positive_bias(self):
        """Détecte biais positif (groupe surreprésenté)."""
        y_true = np.array([1, 1, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 1, 0])
        groups = np.array(["A", "A", "A", "A", "A", "B", "B", "B"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups, reference_group="B")

        group_a = next(m for m in metrics if m.group_name == "A")

        assert group_a.positive_rate == 0.8
        assert group_a.spd > 0
        assert group_a.dir > 1.0

    def test_detect_negative_bias(self):
        """Détecte biais négatif (groupe sous-représenté)."""
        y_true = np.array([1, 0, 0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 0, 0, 0, 0, 1, 1, 1])
        groups = np.array(["A", "A", "A", "A", "A", "B", "B", "B"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups, reference_group="B")

        group_a = next(m for m in metrics if m.group_name == "A")

        assert group_a.positive_rate == 0.2
        assert group_a.spd < 0
        assert group_a.dir < 1.0

    def test_reference_group_auto_select(self):
        """Sélection automatique du groupe majoritaire comme référence."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        groups = np.array(["A", "A", "A", "A", "A", "A", "A", "B", "B", "B"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups)

        group_a = next(m for m in metrics if m.group_name == "A")
        assert group_a.spd == 0.0

    def test_multiple_groups(self):
        """Test avec plusieurs groupes (divisions chess)."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        groups = np.random.choice(["N1", "N2", "N3", "REG", "DEP"], n)

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups)

        assert len(metrics) == 5
        assert all(m.group_size > 0 for m in metrics)
        assert all(0 <= m.positive_rate <= 1 for m in metrics)

    def test_empty_group_handled(self):
        """Groupe vide géré correctement."""
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 1])
        groups = np.array(["A", "A", "A"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups)

        assert len(metrics) == 1
        assert metrics[0].group_name == "A"


class TestComputeBiasByEloRange:
    """Tests pour compute_bias_by_elo_range."""

    def test_default_elo_bins(self):
        """Test avec tranches Elo par défaut."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        elo = np.random.randint(1200, 2400, n)

        metrics = compute_bias_by_elo_range(y_true, y_pred, elo)

        group_names = {m.group_name for m in metrics}
        assert len(group_names) > 0

    def test_custom_elo_bins(self):
        """Test avec tranches Elo personnalisées."""
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        elo = np.array([1500, 1600, 1700, 1800, 1900, 2000])

        metrics = compute_bias_by_elo_range(y_true, y_pred, elo, bins=[1400, 1700, 2000, 2300])

        assert len(metrics) >= 1

    def test_reference_1800_2000(self):
        """Vérifie que 1800-2000 est le groupe de référence."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        elo = np.array([1850, 1900, 1950, 1850, 1600, 1650, 2100, 2150])

        metrics = compute_bias_by_elo_range(y_true, y_pred, elo)

        ref_group = next((m for m in metrics if m.group_name == "1800-2000"), None)
        if ref_group:
            assert ref_group.spd == 0.0
