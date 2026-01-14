"""Tests types robustesse - ISO 24029.

Document ID: ALICE-TEST-ROBUST-TYPES
Version: 1.0.0
Tests: 4

Classes:
- TestRobustnessThresholds: Tests seuils (2 tests)
- TestRobustnessMetrics: Tests dataclass (2 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 24029-1:2021 - Neural Network Robustness
- ISO/IEC 5055:2021 - Code Quality (<100 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from scripts.robustness.adversarial_tests import (
    RobustnessLevel,
    RobustnessMetrics,
    RobustnessThresholds,
)


class TestRobustnessThresholds:
    """Tests pour RobustnessThresholds."""

    def test_default_thresholds(self):
        """Vérifie seuils par défaut ISO 24029."""
        thresholds = RobustnessThresholds()

        assert thresholds.degradation_acceptable == 0.03
        assert thresholds.degradation_warning == 0.05
        assert thresholds.degradation_critical == 0.10
        assert thresholds.stability_threshold == 0.95
        assert len(thresholds.noise_levels) == 3
        assert len(thresholds.perturbation_levels) == 3

    def test_custom_thresholds(self):
        """Vérifie seuils personnalisés."""
        thresholds = RobustnessThresholds(
            degradation_acceptable=0.02,
            degradation_warning=0.04,
            noise_levels=(0.01, 0.02),
        )

        assert thresholds.degradation_acceptable == 0.02
        assert len(thresholds.noise_levels) == 2


class TestRobustnessMetrics:
    """Tests pour la dataclass RobustnessMetrics."""

    def test_metrics_creation(self):
        """Vérifie création RobustnessMetrics."""
        metrics = RobustnessMetrics(
            test_name="test",
            original_score=0.95,
            perturbed_score=0.90,
            degradation=0.05,
            stability_ratio=0.92,
        )

        assert metrics.test_name == "test"
        assert metrics.original_score == 0.95
        assert metrics.degradation == 0.05
        assert metrics.level == RobustnessLevel.ACCEPTABLE

    def test_metrics_with_level(self):
        """Vérifie RobustnessMetrics avec niveau explicite."""
        metrics = RobustnessMetrics(
            test_name="fragile_test",
            original_score=0.95,
            perturbed_score=0.70,
            degradation=0.26,
            stability_ratio=0.60,
            level=RobustnessLevel.FRAGILE,
        )

        assert metrics.level == RobustnessLevel.FRAGILE


class TestRobustnessLevel:
    """Tests pour l'enum RobustnessLevel."""

    def test_robustness_levels(self):
        """Vérifie les niveaux de robustesse."""
        assert RobustnessLevel.ROBUST.value == "robust"
        assert RobustnessLevel.ACCEPTABLE.value == "acceptable"
        assert RobustnessLevel.WARNING.value == "warning"
        assert RobustnessLevel.FRAGILE.value == "fragile"
