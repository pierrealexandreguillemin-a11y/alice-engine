"""Tests Check Bias Thresholds - ISO 24027.

Document ID: ALICE-TEST-FAIRNESS-BIAS-THRESHOLDS
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
    check_bias_thresholds,
)


class TestCheckBiasThresholds:
    """Tests pour check_bias_thresholds."""

    def test_acceptable_bias(self):
        """Biais acceptable = pas d'alertes."""
        metrics = [
            BiasMetrics(
                group_name="A",
                group_size=100,
                positive_rate=0.5,
                true_positive_rate=0.6,
                spd=0.05,
                eod=0.03,
                dir=1.05,
            ),
            BiasMetrics(
                group_name="B",
                group_size=100,
                positive_rate=0.48,
                true_positive_rate=0.58,
                spd=-0.02,
                eod=-0.02,
                dir=0.96,
            ),
        ]

        level, alerts = check_bias_thresholds(metrics)

        assert level == BiasLevel.ACCEPTABLE
        assert len(alerts) == 0

    def test_warning_spd(self):
        """SPD > 0.1 déclenche warning."""
        metrics = [
            BiasMetrics(
                group_name="biased",
                group_size=50,
                positive_rate=0.6,
                true_positive_rate=0.7,
                spd=0.15,
                eod=0.05,
                dir=1.1,
            )
        ]

        level, alerts = check_bias_thresholds(metrics)

        assert level == BiasLevel.WARNING
        assert len(alerts) == 1
        assert "SPD warning" in alerts[0]

    def test_critical_dir(self):
        """DIR < 0.6 déclenche critique."""
        metrics = [
            BiasMetrics(
                group_name="severely_biased",
                group_size=50,
                positive_rate=0.2,
                true_positive_rate=0.3,
                spd=-0.3,
                eod=-0.2,
                dir=0.5,
            )
        ]

        level, alerts = check_bias_thresholds(metrics)

        assert level == BiasLevel.CRITICAL
        assert len(alerts) >= 1

    def test_eeoc_4_5_rule(self):
        """DIR hors 0.8-1.25 déclenche warning (EEOC 4/5 rule)."""
        metrics = [
            BiasMetrics(
                group_name="underrepresented",
                group_size=50,
                positive_rate=0.35,
                true_positive_rate=0.4,
                spd=-0.05,
                eod=-0.05,
                dir=0.75,
            )
        ]

        level, alerts = check_bias_thresholds(metrics)

        assert level == BiasLevel.WARNING
        assert "DIR warning" in alerts[0]
