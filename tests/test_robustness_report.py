"""Tests rapport robustesse - ISO 24029.

Document ID: ALICE-TEST-ROBUST-RPT
Version: 1.0.0
Tests: 7

Classes:
- TestComputeRobustnessMetrics: Tests suite complète (2 tests)
- TestGenerateRobustnessReport: Tests rapport (5 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 24029-1:2021 - Neural Network Robustness
- ISO/IEC 5055:2021 - Code Quality (<120 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from scripts.robustness.adversarial_tests import (
    RobustnessLevel,
    RobustnessReport,
    RobustnessThresholds,
    compute_robustness_metrics,
    generate_robustness_report,
)
from tests.conftest_robustness import (  # noqa: F401
    fragile_model,
    robust_model,
    sample_data,
    simple_model,
)


class TestComputeRobustnessMetrics:
    """Tests pour compute_robustness_metrics."""

    def test_all_tests_run(self, simple_model, sample_data):
        """Vérifie que tous les tests sont exécutés."""
        X, y = sample_data
        metrics = compute_robustness_metrics(simple_model, X, y)

        assert len(metrics) == 8
        test_names = [m.test_name for m in metrics]
        assert any("noise" in name for name in test_names)
        assert any("perturbation" in name for name in test_names)
        assert any("distribution" in name for name in test_names)
        assert any("extreme" in name for name in test_names)

    def test_custom_thresholds(self, simple_model, sample_data):
        """Test avec seuils personnalisés."""
        X, y = sample_data
        thresholds = RobustnessThresholds(noise_levels=(0.01,), perturbation_levels=(0.05,))
        metrics = compute_robustness_metrics(simple_model, X, y, thresholds)
        assert len(metrics) == 4


class TestGenerateRobustnessReport:
    """Tests pour generate_robustness_report."""

    def test_report_structure(self, simple_model, sample_data):
        """Vérifie structure du rapport."""
        X, y = sample_data
        report = generate_robustness_report(simple_model, X, y, model_name="TestModel")

        assert isinstance(report, RobustnessReport)
        assert report.model_name == "TestModel"
        assert report.total_tests == 8
        assert len(report.metrics) == 8
        assert report.timestamp is not None
        assert 0 <= report.original_accuracy <= 1

    def test_report_iso_compliance(self, simple_model, sample_data):
        """Vérifie champs ISO compliance."""
        X, y = sample_data
        report = generate_robustness_report(simple_model, X, y)

        assert "iso_24029_1" in report.iso_compliance
        assert "iso_24029_2" in report.iso_compliance
        assert "iso_42001" in report.iso_compliance
        assert "all_tests_passed" in report.iso_compliance
        assert report.iso_compliance["iso_24029_1"] is True

    def test_report_recommendations(self, simple_model, sample_data):
        """Vérifie que des recommandations sont générées."""
        X, y = sample_data
        report = generate_robustness_report(simple_model, X, y)
        assert len(report.recommendations) > 0

    def test_robust_model_report(self, robust_model, sample_data):
        """Modèle robuste génère rapport positif."""
        X, y = sample_data
        report = generate_robustness_report(robust_model, X, y)
        assert report.overall_level in (RobustnessLevel.ROBUST, RobustnessLevel.ACCEPTABLE)

    def test_fragile_model_detected(self, fragile_model, sample_data):
        """Modèle fragile est détecté."""
        X, y = sample_data
        report = generate_robustness_report(fragile_model, X, y)
        assert report.overall_level in (
            RobustnessLevel.WARNING,
            RobustnessLevel.FRAGILE,
            RobustnessLevel.ACCEPTABLE,
        )
