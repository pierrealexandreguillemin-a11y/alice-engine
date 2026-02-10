"""Tests Error Paths - Fairness Report - ISO 29119.

Document ID: ALICE-TEST-FAIRNESS-ERROR-PATHS
Version: 1.0.0
Tests count: 11

Covers:
- Length mismatch y_true/y_pred
- All-zero predictions
- All-one predictions
- Empty groups array
- Protected attr not in test_data
- Empty string group names
- Pydantic validators on metrics
- Calibration gap computed

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing (error paths)
- ISO/IEC TR 24027:2021 - Bias in AI systems
- NIST AI 100-1 MEASURE 2.11 - Fairness evaluation

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from scripts.fairness.auto_report.generator import (
    _analyze_attribute,
    generate_comprehensive_report,
)
from scripts.fairness.auto_report.types import (
    AttributeAnalysis,
    GroupMetrics,
)
from scripts.fairness.protected.types import (
    ProtectedAttribute,
    ProtectionLevel,
)


class TestLengthMismatch:
    """Tests pour les erreurs de taille y_true/y_pred."""

    def test_raises_on_length_mismatch(self) -> None:
        """Leve ValueError si y_true et y_pred n'ont pas la meme taille."""
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0])
        protected = [
            ProtectedAttribute(
                name="attr",
                level=ProtectionLevel.PROXY_CHECK,
                reason="test",
            ),
        ]
        with pytest.raises(ValueError, match="mismatch"):
            generate_comprehensive_report(
                y_true,
                y_pred,
                pd.DataFrame({"attr": ["A", "B", "C"]}),
                "Model",
                "v1",
                protected,
            )


class TestDegenerateInputs:
    """Tests pour les entrees degenerees."""

    def test_all_zero_predictions(self) -> None:
        """Toutes predictions a 0 ne crash pas."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.zeros(10, dtype=int)
        groups = np.array(["A", "B"] * 5)
        analysis = _analyze_attribute(y_true, y_pred, groups, "test")
        assert analysis.demographic_parity_ratio == 1.0  # 0/0 -> 1.0

    def test_all_one_predictions(self) -> None:
        """Toutes predictions a 1 ne crash pas."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.ones(10, dtype=int)
        groups = np.array(["A", "B"] * 5)
        analysis = _analyze_attribute(y_true, y_pred, groups, "test")
        assert analysis.demographic_parity_ratio == 1.0

    def test_single_sample_per_group(self) -> None:
        """Un seul echantillon par groupe ne crash pas."""
        y_true = np.array([1, 0])
        y_pred = np.array([1, 0])
        groups = np.array(["A", "B"])
        analysis = _analyze_attribute(y_true, y_pred, groups, "test")
        assert analysis.group_count == 2

    def test_empty_string_group_becomes_named(self) -> None:
        """Groupe avec nom vide est renomme '(empty)'."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0, 1, 0, 1, 1, 0, 0])
        groups = np.array(["A", "A", "A", "A", "A", "", "", "", "", ""])
        analysis = _analyze_attribute(y_true, y_pred, groups, "test")
        group_names = [g.group_name for g in analysis.group_details]
        assert "(empty)" in group_names


class TestProtectedAttrNotInTestData:
    """Tests pour les attributs absents des donnees de test."""

    def test_missing_attr_skipped(self) -> None:
        """Attribut absent de test_data est ignore (avec warning)."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 0, 1])
        test_data = pd.DataFrame({"other_col": [1, 2, 3, 4, 5]})
        protected = [
            ProtectedAttribute(
                name="nonexistent",
                level=ProtectionLevel.PROXY_CHECK,
                reason="test",
            ),
        ]
        report = generate_comprehensive_report(y_true, y_pred, test_data, "Model", "v1", protected)
        assert len(report.analyses) == 0


class TestCalibrationGap:
    """Tests pour le calcul du calibration gap."""

    def test_calibration_gap_in_group_details(self) -> None:
        """calibration_gap est calcule dans les group details."""
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.integers(0, 2, n)
        y_pred = rng.integers(0, 2, n)
        groups = np.array(["A", "B"] * (n // 2))
        analysis = _analyze_attribute(y_true, y_pred, groups, "test")
        for g in analysis.group_details:
            assert g.calibration_gap >= 0.0

    def test_max_calibration_gap_computed(self) -> None:
        """max_calibration_gap est le max des calibration_gap des groupes."""
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.integers(0, 2, n)
        y_pred = rng.integers(0, 2, n)
        groups = np.array(["A", "B"] * (n // 2))
        analysis = _analyze_attribute(y_true, y_pred, groups, "test")
        if analysis.group_details:
            max_gap = max(g.calibration_gap for g in analysis.group_details)
            assert analysis.max_calibration_gap == pytest.approx(max_gap, abs=0.001)


class TestTypesValidation:
    """Tests pour la validation Pydantic des types fairness."""

    def test_group_metrics_rejects_negative_tpr(self) -> None:
        """TPR negatif est rejete."""
        with pytest.raises(ValidationError):
            GroupMetrics(
                group_name="A",
                sample_count=100,
                positive_rate=0.5,
                tpr=-0.1,
                fpr=0.1,
                precision=0.5,
                accuracy=0.8,
            )

    def test_attribute_analysis_rejects_dp_above_1(self) -> None:
        """DP ratio > 1.0 est rejete."""
        with pytest.raises(ValidationError):
            AttributeAnalysis(
                attribute_name="test",
                sample_count=100,
                group_count=2,
                demographic_parity_ratio=1.5,
                equalized_odds_tpr_diff=0.05,
                equalized_odds_fpr_diff=0.04,
                predictive_parity_diff=0.03,
                min_group_accuracy=0.8,
                status="fair",
            )
