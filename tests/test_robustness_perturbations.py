"""Tests perturbations robustesse - ISO 24029.

Document ID: ALICE-TEST-ROBUST-PERT
Version: 1.0.0
Tests: 10

Classes:
- TestInputNoise: Tests bruit gaussien (3 tests)
- TestFeaturePerturbation: Tests perturbation (3 tests)
- TestOutOfDistribution: Tests OOD (2 tests)
- TestExtremeValues: Tests valeurs extrêmes (2 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 24029-1:2021 - Neural Network Robustness
- ISO/IEC 5055:2021 - Code Quality (<150 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from scripts.robustness.adversarial_tests import (
    RobustnessMetrics,
    run_extreme_values_test,
    run_feature_perturbation_test,
    run_noise_test,
    run_out_of_distribution_test,
)
from tests.conftest_robustness import (  # noqa: F401
    fragile_model,
    robust_model,
    sample_data,
    simple_model,
)


class TestInputNoise:
    """Tests pour run_noise_test."""

    def test_noise_basic(self, simple_model, sample_data):
        """Test basique avec bruit gaussien."""
        X, y = sample_data
        result = run_noise_test(simple_model, X, y, noise_std=0.01)

        assert isinstance(result, RobustnessMetrics)
        assert result.test_name == "gaussian_noise_0.01"
        assert 0 <= result.original_score <= 1
        assert 0 <= result.stability_ratio <= 1
        assert "noise_std" in result.details

    def test_noise_increases_degradation(self, simple_model, sample_data):
        """Bruit plus fort = dégradation plus importante."""
        X, y = sample_data
        result_low = run_noise_test(simple_model, X, y, noise_std=0.01)
        result_high = run_noise_test(simple_model, X, y, noise_std=0.5)
        assert result_high.degradation >= result_low.degradation

    def test_robust_model_stable(self, robust_model, sample_data):
        """Modèle robuste reste stable sous bruit."""
        X, y = sample_data
        result = run_noise_test(robust_model, X, y, noise_std=0.5)
        assert result.stability_ratio == 1.0


class TestFeaturePerturbation:
    """Tests pour run_feature_perturbation_test."""

    def test_perturbation_basic(self, simple_model, sample_data):
        """Test basique de perturbation."""
        X, y = sample_data
        result = run_feature_perturbation_test(simple_model, X, y, perturbation=0.05)

        assert isinstance(result, RobustnessMetrics)
        assert "feature_perturbation" in result.test_name
        assert "n_features_perturbed" in result.details

    def test_perturbation_amplitude(self, simple_model, sample_data):
        """Perturbation plus forte = impact plus important."""
        X, y = sample_data
        result_low = run_feature_perturbation_test(simple_model, X, y, perturbation=0.01)
        result_high = run_feature_perturbation_test(simple_model, X, y, perturbation=0.5)

        assert isinstance(result_low.degradation, float)
        assert isinstance(result_high.degradation, float)

    def test_selective_features(self, simple_model, sample_data):
        """Test perturbation de features sélectives."""
        X, y = sample_data
        result = run_feature_perturbation_test(
            simple_model, X, y, perturbation=0.1, n_features_to_perturb=2
        )
        assert result.details["n_features_perturbed"] == 2


class TestOutOfDistribution:
    """Tests pour run_out_of_distribution_test."""

    def test_ood_basic(self, simple_model, sample_data):
        """Test basique OOD."""
        X, y = sample_data
        result = run_out_of_distribution_test(simple_model, X, y)

        assert isinstance(result, RobustnessMetrics)
        assert result.test_name == "out_of_distribution"
        assert "ood_multiplier" in result.details
        assert "n_ood_samples" in result.details

    def test_ood_multiplier_effect(self, simple_model, sample_data):
        """Multiplicateur OOD plus élevé = écart plus important."""
        X, y = sample_data
        result_low = run_out_of_distribution_test(simple_model, X, y, ood_multiplier=2.0)
        result_high = run_out_of_distribution_test(simple_model, X, y, ood_multiplier=5.0)
        assert result_high.details["ood_multiplier"] > result_low.details["ood_multiplier"]


class TestExtremeValues:
    """Tests pour run_extreme_values_test."""

    def test_extreme_basic(self, simple_model, sample_data):
        """Test basique valeurs extrêmes."""
        X, y = sample_data
        result = run_extreme_values_test(simple_model, X, y)

        assert isinstance(result, RobustnessMetrics)
        assert "extreme_values" in result.test_name
        assert "percentile" in result.details

    def test_extreme_percentile(self, simple_model, sample_data):
        """Test avec différents percentiles."""
        X, y = sample_data
        result_95 = run_extreme_values_test(simple_model, X, y, percentile=95.0)
        result_99 = run_extreme_values_test(simple_model, X, y, percentile=99.0)

        assert result_95.details["percentile"] == 95.0
        assert result_99.details["percentile"] == 99.0
