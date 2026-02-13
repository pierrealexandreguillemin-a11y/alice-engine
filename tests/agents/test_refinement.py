"""Tests Iterative Refinement - ISO 29119.

Document ID: ALICE-TEST-AGENTS-REFINEMENT
Version: 1.0.0
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scripts.agents.iterative_refinement import (
    IterativeRefinement,
    RefinementAction,
    RefinementResult,
)
from scripts.agents.refinement_helpers import (
    calculate_reweighting,
    should_reweight,
)
from scripts.agents.semantic_memory import ISOSemanticMemory


class TestRefinementAction:
    """Tests pour l'enum RefinementAction."""

    def test_action_values(self):
        """Vérifie les valeurs d'action."""
        assert RefinementAction.NONE.value == "none"
        assert RefinementAction.REWEIGHT.value == "reweight"
        assert RefinementAction.RESAMPLE.value == "resample"
        assert RefinementAction.ADJUST_THRESHOLD.value == "adjust_threshold"
        assert RefinementAction.RETRAIN.value == "retrain"
        assert RefinementAction.SCHEDULE_REVIEW.value == "schedule_review"
        assert RefinementAction.BLOCK_DEPLOYMENT.value == "block_deployment"

    def test_action_count(self):
        """Vérifie le nombre d'actions."""
        assert len(RefinementAction) == 7


class TestRefinementHelpers:
    """Tests pour les fonctions helper de raffinement."""

    def test_should_reweight_balanced(self):
        """Test should_reweight avec groupes équilibrés."""
        group_analyses = [
            {"group_name": "A", "sample_count": 100},
            {"group_name": "B", "sample_count": 110},
            {"group_name": "C", "sample_count": 95},
        ]
        assert should_reweight(group_analyses) is False

    def test_should_reweight_imbalanced(self):
        """Test should_reweight avec déséquilibre > 2x."""
        group_analyses = [
            {"group_name": "A", "sample_count": 100},
            {"group_name": "B", "sample_count": 40},
            {"group_name": "C", "sample_count": 90},
        ]
        # max=100, min=40 -> ratio=2.5 > 2
        assert should_reweight(group_analyses) is True

    def test_should_reweight_empty(self):
        """Test should_reweight avec liste vide."""
        assert should_reweight([]) is False

    def test_should_reweight_missing_counts(self):
        """Test should_reweight avec counts manquants - raises ValueError."""
        group_analyses = [
            {"group_name": "A"},
            {"group_name": "B", "sample_count": 0},
        ]
        # Edge case: all counts are 0 -> ValueError from min()
        with pytest.raises(ValueError):
            should_reweight(group_analyses)

    def test_calculate_reweighting_basic(self):
        """Test calculate_reweighting avec groupes basiques."""
        group_analyses = [
            {"group_name": "A", "sample_count": 100},
            {"group_name": "B", "sample_count": 50},
        ]
        weights = calculate_reweighting(group_analyses)

        assert "A" in weights
        assert "B" in weights
        # Total=150, n_groups=2, avg=75
        # A: 75/100 = 0.75
        # B: 75/50 = 1.50
        assert weights["A"] == pytest.approx(0.75)
        assert weights["B"] == pytest.approx(1.50)

    def test_calculate_reweighting_three_groups(self):
        """Test calculate_reweighting avec 3 groupes."""
        group_analyses = [
            {"group_name": "A", "sample_count": 100},
            {"group_name": "B", "sample_count": 200},
            {"group_name": "C", "sample_count": 50},
        ]
        weights = calculate_reweighting(group_analyses)

        # Total=350, n_groups=3, avg=116.67
        assert weights["A"] == pytest.approx(350 / 3 / 100, rel=1e-5)
        assert weights["B"] == pytest.approx(350 / 3 / 200, rel=1e-5)
        assert weights["C"] == pytest.approx(350 / 3 / 50, rel=1e-5)

    def test_calculate_reweighting_empty(self):
        """Test calculate_reweighting avec liste vide."""
        weights = calculate_reweighting([])
        assert weights == {}

    def test_calculate_reweighting_zero_count(self):
        """Test calculate_reweighting avec count=0."""
        group_analyses = [
            {"group_name": "A", "sample_count": 100},
            {"group_name": "B", "sample_count": 0},
        ]
        weights = calculate_reweighting(group_analyses)

        assert weights["A"] == pytest.approx(0.5)
        assert weights["B"] == 1.0  # Default weight for zero count


class TestIterativeRefinement:
    """Tests pour IterativeRefinement."""

    @pytest.fixture
    def memory(self) -> ISOSemanticMemory:
        """Fixture mémoire sémantique."""
        return ISOSemanticMemory()

    @pytest.fixture
    def refinement(self, memory: ISOSemanticMemory) -> IterativeRefinement:
        """Fixture module de raffinement."""
        return IterativeRefinement(memory=memory)

    def test_initialization(self, refinement: IterativeRefinement):
        """Test initialisation du module."""
        assert refinement.memory is not None
        assert refinement._iteration_count == 0
        assert refinement._max_iterations == 5

    def test_max_iterations_property(self, refinement: IterativeRefinement):
        """Test propriété max_iterations."""
        assert refinement.max_iterations == 5

    def test_refine_fairness_compliant(self, refinement: IterativeRefinement):
        """Test raffinement fairness avec statut compliant."""
        report = {
            "demographic_parity_ratio": 0.85,
        }
        result = refinement.refine_fairness(report)

        assert result.action == RefinementAction.NONE
        assert result.success is True
        assert "reason" in result.details

    def test_refine_fairness_caution(self, refinement: IterativeRefinement):
        """Test raffinement fairness avec statut caution."""
        report = {
            "demographic_parity_ratio": 0.70,
        }
        result = refinement.refine_fairness(report)

        assert result.action == RefinementAction.SCHEDULE_REVIEW
        assert result.success is True
        assert "metric" in result.details
        assert "review_deadline" in result.details
        assert len(result.next_steps) > 0

    def test_refine_fairness_critical_with_reweight(self, refinement: IterativeRefinement):
        """Test raffinement fairness critique avec reweighting."""
        report = {
            "demographic_parity_ratio": 0.55,
            "disadvantaged_groups": ["PACA"],
            "group_analyses": [
                {"group_name": "IDF", "sample_count": 200},
                {"group_name": "PACA", "sample_count": 50},
            ],
        }
        result = refinement.refine_fairness(report, auto_apply=False)

        assert result.action == RefinementAction.REWEIGHT
        assert result.success is True
        assert "weights" in result.details
        assert "disadvantaged_groups" in result.details
        assert result.details["auto_applied"] is False
        assert len(result.next_steps) > 0

    def test_refine_fairness_critical_auto_apply(self, refinement: IterativeRefinement):
        """Test raffinement fairness avec auto_apply."""
        report = {
            "demographic_parity_ratio": 0.55,
            "disadvantaged_groups": ["PACA"],
            "group_analyses": [
                {"group_name": "IDF", "sample_count": 200},
                {"group_name": "PACA", "sample_count": 50},
            ],
        }
        result = refinement.refine_fairness(report, auto_apply=True)

        assert result.action == RefinementAction.REWEIGHT
        assert result.success is True
        assert result.details["auto_applied"] is True

    def test_refine_fairness_critical_no_reweight(self, refinement: IterativeRefinement):
        """Test raffinement critique sans possibilité de reweight."""
        report = {
            "demographic_parity_ratio": 0.55,
            "disadvantaged_groups": ["PACA"],
            "group_analyses": [
                {"group_name": "IDF", "sample_count": 100},
                {"group_name": "PACA", "sample_count": 95},
            ],
        }
        result = refinement.refine_fairness(report)

        assert result.action == RefinementAction.BLOCK_DEPLOYMENT
        assert result.success is False
        assert "reason" in result.details
        assert len(result.next_steps) > 0

    def test_refine_robustness_compliant(self, refinement: IterativeRefinement):
        """Test raffinement robustness compliant."""
        report = {
            "noise_tolerance": 0.96,
        }
        result = refinement.refine_robustness(report)

        assert result.action == RefinementAction.NONE
        assert result.success is True

    def test_refine_robustness_with_critical_features(self, refinement: IterativeRefinement):
        """Test raffinement robustness avec features critiques."""
        report = {
            "noise_tolerance": 0.87,
            "critical_features": ["feature_A", "feature_B"],
        }
        result = refinement.refine_robustness(report)

        assert result.action == RefinementAction.RETRAIN
        assert result.success is True
        assert "critical_features" in result.details
        assert result.details["critical_features"] == ["feature_A", "feature_B"]
        assert len(result.next_steps) > 0

    def test_refine_robustness_caution(self, refinement: IterativeRefinement):
        """Test raffinement robustness avec avertissement."""
        report = {
            "noise_tolerance": 0.92,
            "critical_features": [],
        }
        result = refinement.refine_robustness(report)

        assert result.action == RefinementAction.SCHEDULE_REVIEW
        assert result.success is True

    def test_handle_critical_fairness_delegates(
        self, refinement: IterativeRefinement, monkeypatch: pytest.MonkeyPatch
    ):
        """Test que _handle_critical_fairness délègue aux helpers."""
        mock_should_reweight = MagicMock(return_value=True)
        mock_calculate = MagicMock(return_value={"A": 1.0})

        monkeypatch.setattr(
            "scripts.agents.refinement_helpers.should_reweight", mock_should_reweight
        )
        monkeypatch.setattr(
            "scripts.agents.refinement_helpers.calculate_reweighting", mock_calculate
        )

        report = {
            "demographic_parity_ratio": 0.55,
            "group_analyses": [{"group_name": "A", "sample_count": 100}],
        }
        result = refinement.refine_fairness(report)

        assert result.action == RefinementAction.REWEIGHT
        mock_should_reweight.assert_called_once()
        mock_calculate.assert_called_once()

    def test_refinement_result_structure(self, refinement: IterativeRefinement):
        """Test structure de RefinementResult."""
        report = {"demographic_parity_ratio": 0.85}
        result = refinement.refine_fairness(report)

        assert isinstance(result, RefinementResult)
        assert isinstance(result.action, RefinementAction)
        assert isinstance(result.success, bool)
        assert isinstance(result.details, dict)
        assert isinstance(result.next_steps, list)

    def test_refinement_with_missing_fields(self, refinement: IterativeRefinement):
        """Test raffinement avec champs manquants defaut compliant."""
        result = refinement.refine_fairness({})
        assert result.action == RefinementAction.NONE
        assert result.success is True
