"""Tests Generate Fairness Report - ISO 24027.

Document ID: ALICE-TEST-FAIRNESS-BIAS-REPORT
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
    BiasLevel,
    BiasReport,
    generate_fairness_report,
)


class TestGenerateFairnessReport:
    """Tests pour generate_fairness_report."""

    def test_report_structure(self):
        """Vérifie structure du rapport."""
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        groups = np.array(["A", "A", "A", "B", "B", "B"])

        report = generate_fairness_report(
            y_true,
            y_pred,
            groups,
            model_name="TestModel",
            feature_name="division",
        )

        assert isinstance(report, BiasReport)
        assert report.model_name == "TestModel"
        assert report.feature_analyzed == "division"
        assert report.total_samples == 6
        assert len(report.metrics_by_group) == 2
        assert report.timestamp is not None

    def test_report_iso_compliance(self):
        """Vérifie champs ISO compliance."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        groups = np.array(["A", "A", "B", "B"])

        report = generate_fairness_report(y_true, y_pred, groups)

        assert "iso_24027" in report.iso_compliance
        assert "iso_42001" in report.iso_compliance
        assert "eeoc_4_5_rule" in report.iso_compliance
        assert report.iso_compliance["iso_24027"] is True

    def test_report_recommendations_critical(self):
        """Vérifie recommandations pour biais critique."""
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        groups = np.array(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])

        report = generate_fairness_report(y_true, y_pred, groups, reference_group="B")

        assert report.overall_level == BiasLevel.CRITICAL
        assert len(report.recommendations) > 0
        assert any("URGENT" in r or "critique" in r.lower() for r in report.recommendations)

    def test_report_recommendations_acceptable(self):
        """Vérifie recommandations pour biais acceptable."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        report = generate_fairness_report(y_true, y_pred, groups)

        assert report.overall_level == BiasLevel.ACCEPTABLE
        assert any("Aucun biais" in r for r in report.recommendations)

    def test_report_reference_group(self):
        """Vérifie que le groupe de référence est inclus."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        groups = np.array(["X", "X", "Y", "Y"])

        report = generate_fairness_report(y_true, y_pred, groups, reference_group="X")

        assert report.reference_group == "X"
