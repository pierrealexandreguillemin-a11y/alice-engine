"""Tests Fairness Report Generator - ISO 29119.

Document ID: ALICE-TEST-FAIRNESS-AUTO-GENERATOR
Version: 1.1.0
Tests count: 19

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC TR 24027:2021 - Bias in AI systems
- NIST AI 100-1 MEASURE 2.11 - Fairness evaluation

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.fairness.auto_report.computations import bootstrap_dp_ci
from scripts.fairness.auto_report.generator import (
    _analyze_attribute,
    _determine_overall_status,
    _generate_recommendations,
    generate_comprehensive_report,
)
from scripts.fairness.auto_report.types import AttributeAnalysis
from scripts.fairness.protected.types import (
    ProtectedAttribute,
    ProtectionLevel,
)


class TestAnalyzeAttribute:
    """Tests pour l'analyse par attribut."""

    def test_computes_demographic_parity(
        self,
        sample_y_true: np.ndarray,
        sample_y_pred: np.ndarray,
    ) -> None:
        """Le demographic parity ratio est calcule."""
        groups = np.array(["A", "B"] * 250)
        analysis = _analyze_attribute(sample_y_true, sample_y_pred, groups, "test_attr")
        assert 0.0 <= analysis.demographic_parity_ratio <= 1.0

    def test_computes_equalized_odds(
        self,
        sample_y_true: np.ndarray,
        sample_y_pred: np.ndarray,
    ) -> None:
        """Les equalized odds sont calcules."""
        groups = np.array(["A", "B"] * 250)
        analysis = _analyze_attribute(sample_y_true, sample_y_pred, groups, "test_attr")
        assert analysis.equalized_odds_tpr_diff >= 0.0
        assert analysis.equalized_odds_fpr_diff >= 0.0

    def test_handles_single_group(
        self,
        sample_y_true: np.ndarray,
        sample_y_pred: np.ndarray,
    ) -> None:
        """Un seul groupe -> pas d'erreur, status fair."""
        groups = np.array(["A"] * 500)
        analysis = _analyze_attribute(sample_y_true, sample_y_pred, groups, "test_attr")
        assert analysis.status == "fair"
        assert analysis.group_count == 1

    def test_handles_empty_group(self) -> None:
        """Groupes avec peu de donnees -> pas d'erreur."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 0, 1])
        groups = np.array(["A", "A", "A", "B", "B"])
        analysis = _analyze_attribute(y_true, y_pred, groups, "test_attr")
        assert analysis.group_count == 2

    def test_group_details_populated(
        self,
        sample_y_true: np.ndarray,
        sample_y_pred: np.ndarray,
    ) -> None:
        """Les group_details sont remplis (EU AI Act Art.13)."""
        groups = np.array(["A", "B", "C"] * 166 + ["A", "B"])
        analysis = _analyze_attribute(sample_y_true, sample_y_pred, groups, "test_attr")
        assert len(analysis.group_details) >= 2
        for g in analysis.group_details:
            assert g.sample_count > 0
            assert 0.0 <= g.tpr <= 1.0
            assert 0.0 <= g.fpr <= 1.0

    def test_metrics_match_manual_calculation(self) -> None:
        """Metriques correspondent au calcul manuel (M27)."""
        # Groupe A: y_true=[1,1,0,0], y_pred=[1,0,1,0]
        #   positive_rate = 2/4 = 0.5, TPR = 1/2 = 0.5, FPR = 1/2 = 0.5
        #   precision = 1/2 = 0.5, accuracy = 2/4 = 0.5
        # Groupe B: y_true=[1,1,1,0,0,0], y_pred=[1,1,1,0,0,0]
        #   positive_rate = 3/6 = 0.5, TPR = 3/3 = 1.0, FPR = 0/3 = 0.0
        #   precision = 3/3 = 1.0, accuracy = 6/6 = 1.0
        # DP ratio = min(0.5,0.5) / max(0.5,0.5) = 1.0
        # TPR diff = |1.0 - 0.5| = 0.5
        # FPR diff = |0.5 - 0.0| = 0.5
        y_true = np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B", "B", "B"])

        import pytest

        analysis = _analyze_attribute(y_true, y_pred, groups, "manual")
        assert analysis.demographic_parity_ratio == pytest.approx(1.0, abs=0.01)
        assert analysis.equalized_odds_tpr_diff == pytest.approx(0.5, abs=0.01)
        assert analysis.equalized_odds_fpr_diff == pytest.approx(0.5, abs=0.01)
        # Verify group details
        grp_a = next(g for g in analysis.group_details if g.group_name == "A")
        grp_b = next(g for g in analysis.group_details if g.group_name == "B")
        assert grp_a.tpr == pytest.approx(0.5, abs=0.01)
        assert grp_b.tpr == pytest.approx(1.0, abs=0.01)
        assert grp_a.accuracy == pytest.approx(0.5, abs=0.01)
        assert grp_b.accuracy == pytest.approx(1.0, abs=0.01)

    def test_reuses_existing_bias_metrics(
        self,
        sample_y_true: np.ndarray,
        sample_y_pred: np.ndarray,
    ) -> None:
        """Reutilise compute_bias_metrics_by_group (ISO 5055)."""
        groups = np.array(["A", "B"] * 250)
        analysis = _analyze_attribute(sample_y_true, sample_y_pred, groups, "test_attr")
        # group_details populated means BiasMetrics were computed
        assert len(analysis.group_details) == 2
        assert all(g.group_name in ("A", "B") for g in analysis.group_details)


class TestBootstrapCI:
    """Tests pour les intervalles de confiance bootstrap."""

    def test_returns_ci_for_dp_ratio(
        self,
        sample_y_pred: np.ndarray,
    ) -> None:
        """Retourne CI pour demographic_parity_ratio."""
        groups = np.array(["A", "B"] * 250)
        ci = bootstrap_dp_ci(sample_y_pred, groups, np.array(["A", "B"]))
        assert "demographic_parity_ratio" in ci
        lo, hi = ci["demographic_parity_ratio"]
        assert lo <= hi
        assert 0.0 <= lo <= 1.0

    def test_ci_in_analysis(
        self,
        sample_y_true: np.ndarray,
        sample_y_pred: np.ndarray,
    ) -> None:
        """CI present dans l'analyse d'attribut."""
        groups = np.array(["A", "B"] * 250)
        analysis = _analyze_attribute(sample_y_true, sample_y_pred, groups, "test")
        assert "demographic_parity_ratio" in analysis.confidence_intervals


class TestDetermineOverallStatus:
    """Tests pour la determination du status global."""

    def test_all_fair_returns_fair(
        self,
        mock_analysis_fair: AttributeAnalysis,
    ) -> None:
        """Toutes les analyses fair -> status global fair."""
        result = _determine_overall_status([mock_analysis_fair, mock_analysis_fair])
        assert result == "fair"

    def test_one_caution_returns_caution(
        self,
        mock_analysis_fair: AttributeAnalysis,
    ) -> None:
        """Une analyse caution -> status global caution."""
        caution = AttributeAnalysis(
            attribute_name="test",
            sample_count=100,
            group_count=2,
            demographic_parity_ratio=0.75,
            equalized_odds_tpr_diff=0.12,
            equalized_odds_fpr_diff=0.10,
            predictive_parity_diff=0.08,
            min_group_accuracy=0.72,
            status="caution",
        )
        result = _determine_overall_status([mock_analysis_fair, caution])
        assert result == "caution"

    def test_one_critical_returns_critical(
        self,
        mock_analysis_fair: AttributeAnalysis,
        mock_analysis_critical: AttributeAnalysis,
    ) -> None:
        """Une analyse critical -> status global critical."""
        result = _determine_overall_status([mock_analysis_fair, mock_analysis_critical])
        assert result == "critical"


class TestGenerateRecommendations:
    """Tests pour la generation de recommandations."""

    def test_fair_no_recommendations(
        self,
        mock_analysis_fair: AttributeAnalysis,
    ) -> None:
        """Toutes fair -> recommandation de maintien."""
        recs = _generate_recommendations([mock_analysis_fair])
        assert len(recs) >= 1
        assert any("monitoring" in r.lower() or "maintenir" in r.lower() for r in recs)

    def test_caution_suggests_monitoring(self) -> None:
        """Caution -> recommande monitoring renforce."""
        caution = AttributeAnalysis(
            attribute_name="test",
            sample_count=100,
            group_count=2,
            demographic_parity_ratio=0.75,
            equalized_odds_tpr_diff=0.12,
            equalized_odds_fpr_diff=0.10,
            predictive_parity_diff=0.08,
            min_group_accuracy=0.72,
            status="caution",
        )
        recs = _generate_recommendations([caution])
        assert len(recs) >= 1

    def test_critical_suggests_mitigation(
        self,
        mock_analysis_critical: AttributeAnalysis,
    ) -> None:
        """Critical -> recommande mitigation."""
        recs = _generate_recommendations([mock_analysis_critical])
        assert len(recs) >= 1
        assert any("mitigation" in r.lower() or "urgent" in r.lower() for r in recs)


class TestGenerateComprehensiveReport:
    """Tests d'integration pour le rapport complet."""

    def test_analyzes_all_protected_attrs(
        self,
        sample_y_true: np.ndarray,
        sample_y_pred: np.ndarray,
        sample_test_data: pd.DataFrame,
    ) -> None:
        """Le rapport analyse tous les attributs proteges."""
        protected = [
            ProtectedAttribute(
                name="ligue_code",
                level=ProtectionLevel.PROXY_CHECK,
                reason="geo",
            ),
            ProtectedAttribute(
                name="blanc_titre",
                level=ProtectionLevel.PROXY_CHECK,
                reason="gender proxy",
            ),
        ]
        report = generate_comprehensive_report(
            sample_y_true,
            sample_y_pred,
            sample_test_data,
            "TestModel",
            "v1.0",
            protected,
        )
        assert len(report.analyses) == 2

    def test_model_agnostic_accepts_arrays(
        self,
        sample_y_true: np.ndarray,
        sample_y_pred: np.ndarray,
        sample_test_data: pd.DataFrame,
    ) -> None:
        """Accepte des numpy arrays (model-agnostic)."""
        protected = [
            ProtectedAttribute(
                name="ligue_code",
                level=ProtectionLevel.PROXY_CHECK,
                reason="geo",
            ),
        ]
        report = generate_comprehensive_report(
            sample_y_true,
            sample_y_pred,
            sample_test_data,
            "TestModel",
            "v1.0",
            protected,
        )
        assert report.total_samples == 500

    def test_report_serializable_to_json(
        self,
        sample_y_true: np.ndarray,
        sample_y_pred: np.ndarray,
        sample_test_data: pd.DataFrame,
    ) -> None:
        """Le rapport est serialisable en JSON."""
        protected = [
            ProtectedAttribute(
                name="ligue_code",
                level=ProtectionLevel.PROXY_CHECK,
                reason="geo",
            ),
        ]
        report = generate_comprehensive_report(
            sample_y_true,
            sample_y_pred,
            sample_test_data,
            "TestModel",
            "v1.0",
            protected,
        )
        json_str = json.dumps(report.to_dict())
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["model_name"] == "TestModel"

    def test_iso_compliance_section_present(
        self,
        sample_y_true: np.ndarray,
        sample_y_pred: np.ndarray,
        sample_test_data: pd.DataFrame,
    ) -> None:
        """La section ISO compliance est presente."""
        protected = [
            ProtectedAttribute(
                name="ligue_code",
                level=ProtectionLevel.PROXY_CHECK,
                reason="geo",
            ),
        ]
        report = generate_comprehensive_report(
            sample_y_true,
            sample_y_pred,
            sample_test_data,
            "TestModel",
            "v1.0",
            protected,
        )
        assert "iso_24027" in report.iso_compliance
        assert "nist_ai_100_1" in report.iso_compliance
