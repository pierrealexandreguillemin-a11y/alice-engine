"""Tests cas limites robustesse - ISO 24029.

Document ID: ALICE-TEST-ROBUST-EDGE
Version: 1.0.0
Tests: 7

Classes:
- TestEdgeCases: Tests cas limites (5 tests)
- TestIntegration: Tests intégration (2 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 24029-1:2021 - Neural Network Robustness
- ISO/IEC 5055:2021 - Code Quality (<120 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

import numpy as np
import pandas as pd

from scripts.robustness.adversarial_tests import (
    compute_robustness_metrics,
    generate_robustness_report,
    run_noise_test,
)
from tests.conftest_robustness import sample_data, simple_model  # noqa: F401


class TestEdgeCases:
    """Tests pour cas limites."""

    def test_small_dataset(self, simple_model):
        """Test avec petit dataset."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 1])
        metrics = compute_robustness_metrics(simple_model, X, y)

        assert len(metrics) == 8
        for m in metrics:
            assert m.original_score is not None

    def test_single_feature(self, simple_model):
        """Test avec une seule feature."""
        X = np.array([1, 2, 3, 4, 5])
        y = np.array([0, 0, 1, 1, 1])
        result = run_noise_test(simple_model, X, y, noise_std=0.1)
        assert result is not None

    def test_pandas_input(self, simple_model):
        """Test avec entrées pandas."""
        X = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
        y = pd.Series(np.random.randint(0, 2, 50))
        report = generate_robustness_report(simple_model, X, y)

        assert report is not None
        assert report.total_tests == 8

    def test_binary_predictions(self, sample_data):
        """Test avec prédictions binaires strictes."""
        X, y = sample_data

        def binary_model(x_input):
            x_input = np.asarray(x_input)
            return np.random.randint(0, 2, x_input.shape[0])

        result = run_noise_test(binary_model, X, y, noise_std=0.1)
        assert result is not None
        assert 0 <= result.stability_ratio <= 1

    def test_perfect_model(self, sample_data):
        """Test avec modèle parfait."""
        X, y = sample_data

        def perfect_model(x_input):  # noqa: ARG001
            return y

        result = run_noise_test(perfect_model, X, y, noise_std=0.01)
        assert result.original_score == 1.0


class TestIntegration:
    """Tests d'intégration."""

    def test_full_pipeline(self, simple_model, sample_data):
        """Test pipeline complet de robustesse."""
        X, y = sample_data
        report = generate_robustness_report(simple_model, X, y, model_name="IntegrationTest")

        assert report.model_name == "IntegrationTest"
        assert report.total_tests > 0
        assert len(report.metrics) == report.total_tests

        for m in report.metrics:
            assert 0 <= m.original_score <= 1
            assert 0 <= m.stability_ratio <= 1
            assert m.degradation >= -1

        assert len(report.recommendations) > 0
        assert report.iso_compliance["iso_24029_1"] is True
        assert report.iso_compliance["iso_24029_2"] is True

    def test_reproducibility(self, simple_model, sample_data):
        """Test reproductibilité avec seed fixé."""
        X, y = sample_data

        np.random.seed(42)
        report1 = generate_robustness_report(simple_model, X, y)

        np.random.seed(42)
        report2 = generate_robustness_report(simple_model, X, y)

        for m1, m2 in zip(report1.metrics, report2.metrics, strict=False):
            assert m1.test_name == m2.test_name
            assert m1.original_score == m2.original_score
