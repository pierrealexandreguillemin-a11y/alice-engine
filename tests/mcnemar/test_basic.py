"""Tests McNemar Basic - ISO 29119.

Document ID: ALICE-TEST-MCNEMAR-BASIC
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import numpy as np

from scripts.comparison.mcnemar_test import (
    McNemarResult,
    bootstrap_confidence_interval,
    mcnemar_5x2cv_test,
    mcnemar_simple_test,
)


class TestMcNemarSimpleTest:
    """Tests pour mcnemar_simple_test."""

    def test_identical_predictions_not_significant(
        self,
        identical_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test que predictions identiques = pas significatif."""
        y_true, y_pred_a, y_pred_b = identical_predictions

        result = mcnemar_simple_test(y_true, y_pred_a, y_pred_b)

        assert not result.significant
        assert result.p_value == 1.0
        assert result.winner is None

    def test_different_predictions_significant(
        self,
        different_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test que predictions differentes = significatif."""
        y_true, y_pred_a, y_pred_b = different_predictions

        result = mcnemar_simple_test(y_true, y_pred_a, y_pred_b)

        assert result.significant
        assert result.p_value < 0.05
        assert result.winner == "model_a"

    def test_result_has_all_fields(
        self,
        identical_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test que le resultat a tous les champs."""
        y_true, y_pred_a, y_pred_b = identical_predictions

        result = mcnemar_simple_test(y_true, y_pred_a, y_pred_b)

        assert isinstance(result, McNemarResult)
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.significant, bool)
        assert isinstance(result.effect_size, float)
        assert isinstance(result.confidence_interval, tuple)

    def test_confidence_interval_contains_zero_when_no_diff(
        self,
        identical_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test que l'intervalle contient 0 quand pas de difference."""
        y_true, y_pred_a, y_pred_b = identical_predictions

        result = mcnemar_simple_test(y_true, y_pred_a, y_pred_b)

        lower, upper = result.confidence_interval
        assert lower <= 0 <= upper

    def test_effect_size_near_zero_when_no_diff(
        self,
        identical_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test que effect size proche de 0 quand pas de difference."""
        y_true, y_pred_a, y_pred_b = identical_predictions

        result = mcnemar_simple_test(y_true, y_pred_a, y_pred_b)

        assert abs(result.effect_size) < 0.1


class TestMcNemar5x2cvTest:
    """Tests pour mcnemar_5x2cv_test."""

    def test_basic_execution(
        self,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test execution basique."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier

        X, y = classification_data

        model_a = LogisticRegression(random_state=42)
        model_b = DecisionTreeClassifier(random_state=42)

        result = mcnemar_5x2cv_test(
            model_a_fit=model_a.fit,
            model_b_fit=model_b.fit,
            model_a_predict=model_a.predict,
            model_b_predict=model_b.predict,
            X=X,
            y=y,
            n_iterations=2,
        )

        assert isinstance(result, McNemarResult)
        assert 0 <= result.p_value <= 1


class TestBootstrapConfidenceInterval:
    """Tests pour bootstrap_confidence_interval."""

    def test_basic_execution(self) -> None:
        """Test execution basique."""
        accuracies_a = [0.8, 0.82, 0.79, 0.81, 0.80]
        accuracies_b = [0.75, 0.77, 0.76, 0.74, 0.75]

        lower, upper = bootstrap_confidence_interval(
            accuracies_a,
            accuracies_b,
            n_bootstrap=100,
        )

        assert lower < upper
        assert lower > 0

    def test_interval_contains_zero_when_similar(self) -> None:
        """Test que l'intervalle contient 0 quand modeles similaires."""
        accuracies_a = [0.80, 0.79, 0.81, 0.80, 0.80]
        accuracies_b = [0.80, 0.81, 0.79, 0.80, 0.80]

        lower, upper = bootstrap_confidence_interval(
            accuracies_a,
            accuracies_b,
            n_bootstrap=100,
        )

        assert lower <= 0.05
        assert upper >= -0.05
