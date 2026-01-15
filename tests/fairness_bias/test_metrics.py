"""Tests Bias Metrics - ISO 24027.

Document ID: ALICE-TEST-FAIRNESS-BIAS-METRICS
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC TR 24027:2021 - Bias in AI systems
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from scripts.fairness.bias_detection import (
    BiasLevel,
    BiasMetrics,
    BiasThresholds,
)


class TestBiasMetrics:
    """Tests pour la dataclass BiasMetrics."""

    def test_bias_metrics_creation(self):
        """Vérifie création BiasMetrics avec valeurs par défaut."""
        metrics = BiasMetrics(
            group_name="test",
            group_size=100,
            positive_rate=0.5,
            true_positive_rate=0.6,
        )

        assert metrics.group_name == "test"
        assert metrics.group_size == 100
        assert metrics.positive_rate == 0.5
        assert metrics.true_positive_rate == 0.6
        assert metrics.spd == 0.0  # Défaut
        assert metrics.eod == 0.0  # Défaut
        assert metrics.dir == 1.0  # Défaut
        assert metrics.level == BiasLevel.ACCEPTABLE  # Défaut

    def test_bias_metrics_with_bias(self):
        """Vérifie BiasMetrics avec biais."""
        metrics = BiasMetrics(
            group_name="biased",
            group_size=50,
            positive_rate=0.3,
            true_positive_rate=0.4,
            spd=-0.2,
            eod=-0.15,
            dir=0.6,
            level=BiasLevel.CRITICAL,
        )

        assert metrics.spd == -0.2
        assert metrics.dir == 0.6
        assert metrics.level == BiasLevel.CRITICAL


class TestBiasThresholds:
    """Tests pour BiasThresholds."""

    def test_default_thresholds(self):
        """Vérifie seuils par défaut ISO 24027."""
        thresholds = BiasThresholds()

        assert thresholds.spd_warning == 0.1
        assert thresholds.spd_critical == 0.2
        assert thresholds.dir_min == 0.8  # EEOC 4/5 rule
        assert thresholds.dir_max == 1.25
        assert thresholds.eod_warning == 0.1
        assert thresholds.eod_critical == 0.2

    def test_custom_thresholds(self):
        """Vérifie seuils personnalisés."""
        thresholds = BiasThresholds(
            spd_warning=0.05,
            spd_critical=0.1,
            dir_min=0.9,
            dir_max=1.1,
        )

        assert thresholds.spd_warning == 0.05
        assert thresholds.dir_min == 0.9
