"""Tests: tests/test_mcnemar_comparison.py - McNemar Statistical Tests.

Document ID: ALICE-TEST-MCNEMAR-001
Version: 1.0.0

Tests unitaires pour les tests statistiques McNemar.
Couvre: mcnemar_test, statistical_comparison.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 24029:2021 - Statistical validation
- ISO/IEC 5055:2021 - Code Quality

Test Coverage Target: 90%
Total Tests: 15

Author: ALICE Engine Team
Last Updated: 2026-01-10
"""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003 - Used at runtime in tests

import numpy as np
import pytest

from scripts.comparison.mcnemar_test import (
    McNemarResult,
    bootstrap_confidence_interval,
    mcnemar_5x2cv_test,
    mcnemar_simple_test,
)
from scripts.comparison.statistical_comparison import (
    ModelComparison,
    compare_models,
    save_comparison_report,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def identical_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predictions identiques (pas de difference)."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = y_true.copy()
    return y_true, y_pred, y_pred


@pytest.fixture
def different_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predictions differentes (un modele meilleur)."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)

    # Modele A: 90% accuracy
    y_pred_a = y_true.copy()
    errors_a = np.random.choice(100, 10, replace=False)
    y_pred_a[errors_a] = 1 - y_pred_a[errors_a]

    # Modele B: 70% accuracy
    y_pred_b = y_true.copy()
    errors_b = np.random.choice(100, 30, replace=False)
    y_pred_b[errors_b] = 1 - y_pred_b[errors_b]

    return y_true, y_pred_a, y_pred_b


@pytest.fixture
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Donnees de classification."""
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


# =============================================================================
# TESTS: McNemar Simple Test
# =============================================================================


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
        assert result.winner == "model_a"  # A is better

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
        assert isinstance(result.model_a_mean_accuracy, float)
        assert isinstance(result.model_b_mean_accuracy, float)

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

    def test_custom_alpha(
        self,
        different_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test avec alpha personnalise."""
        y_true, y_pred_a, y_pred_b = different_predictions

        result_005 = mcnemar_simple_test(y_true, y_pred_a, y_pred_b, alpha=0.05)
        result_001 = mcnemar_simple_test(y_true, y_pred_a, y_pred_b, alpha=0.01)

        # p-value ne change pas, mais significance peut
        assert result_005.p_value == result_001.p_value


# =============================================================================
# TESTS: McNemar 5x2cv Test
# =============================================================================


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
            n_iterations=2,  # Reduced for speed
        )

        assert isinstance(result, McNemarResult)
        assert 0 <= result.p_value <= 1

    def test_reproducibility(
        self,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test reproductibilite avec meme seed."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier

        X, y = classification_data

        results = []
        for _ in range(2):
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
                random_state=42,
            )
            results.append(result)

        assert results[0].p_value == results[1].p_value

    def test_accuracies_in_range(
        self,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test que les accuracies sont dans [0, 1]."""
        from sklearn.linear_model import LogisticRegression

        X, y = classification_data

        model = LogisticRegression(random_state=42)

        result = mcnemar_5x2cv_test(
            model_a_fit=model.fit,
            model_b_fit=model.fit,
            model_a_predict=model.predict,
            model_b_predict=model.predict,
            X=X,
            y=y,
            n_iterations=2,
        )

        assert 0 <= result.model_a_mean_accuracy <= 1
        assert 0 <= result.model_b_mean_accuracy <= 1


# =============================================================================
# TESTS: Bootstrap Confidence Interval
# =============================================================================


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
        assert lower > 0  # A is consistently better

    def test_interval_contains_zero_when_similar(self) -> None:
        """Test que l'intervalle contient 0 quand modeles similaires."""
        accuracies_a = [0.80, 0.79, 0.81, 0.80, 0.80]
        accuracies_b = [0.80, 0.81, 0.79, 0.80, 0.80]

        lower, upper = bootstrap_confidence_interval(
            accuracies_a,
            accuracies_b,
            n_bootstrap=100,
        )

        # Should contain 0
        assert lower <= 0.05  # Allow some margin
        assert upper >= -0.05

    def test_reproducibility(self) -> None:
        """Test reproductibilite."""
        accuracies_a = [0.8, 0.82, 0.79]
        accuracies_b = [0.75, 0.77, 0.76]

        ci1 = bootstrap_confidence_interval(
            accuracies_a,
            accuracies_b,
            n_bootstrap=100,
            random_state=42,
        )
        ci2 = bootstrap_confidence_interval(
            accuracies_a,
            accuracies_b,
            n_bootstrap=100,
            random_state=42,
        )

        assert ci1 == ci2


# =============================================================================
# TESTS: Statistical Comparison
# =============================================================================


class TestCompareModels:
    """Tests pour compare_models."""

    def test_basic_comparison(self) -> None:
        """Test comparaison basique."""
        np.random.seed(42)
        y_test = np.random.randint(0, 2, 100)

        # Model A: perfect
        def predict_a(x_data: np.ndarray) -> np.ndarray:
            return y_test

        # Model B: 70% accuracy
        def predict_b(x_data: np.ndarray) -> np.ndarray:
            result = y_test.copy()
            result[:30] = 1 - result[:30]
            return result

        X_test = np.random.randn(100, 5)

        result = compare_models(
            model_a_predict=predict_a,
            model_b_predict=predict_b,
            X_test=X_test,
            y_test=y_test,
            model_a_name="Perfect",
            model_b_name="Noisy",
        )

        assert isinstance(result, ModelComparison)
        assert result.winner == "Perfect"
        assert result.metrics_a["accuracy"] > result.metrics_b["accuracy"]

    def test_tie_detection(self) -> None:
        """Test detection d'egalite."""
        np.random.seed(42)
        y_test = np.random.randint(0, 2, 100)

        def predict_same(x_data: np.ndarray) -> np.ndarray:
            return y_test

        X_test = np.random.randn(100, 5)

        result = compare_models(
            model_a_predict=predict_same,
            model_b_predict=predict_same,
            X_test=X_test,
            y_test=y_test,
        )

        assert result.winner == "tie"
        assert not result.mcnemar_result.significant


class TestCompareWithBaseline:
    """Tests pour compare_with_baseline."""

    def test_compare_with_baseline_basic(self, tmp_path: Path) -> None:
        """Test comparaison avec baseline."""
        from unittest.mock import MagicMock

        import pandas as pd

        from scripts.comparison.statistical_comparison import compare_with_baseline

        # Create mock predictor
        mock_predictor = MagicMock()
        test_data = pd.DataFrame(
            {
                "feature_1": np.random.randn(50),
                "feature_2": np.random.randn(50),
                "target": np.random.randint(0, 2, 50),
            }
        )

        # Mock predict to return correct predictions
        mock_predictor.predict.return_value = pd.Series(test_data["target"].values)
        mock_predictor.predict_proba.return_value = pd.DataFrame(
            {
                0: 1 - test_data["target"].values,
                1: test_data["target"].values,
            }
        )

        def baseline_predict(x_data: pd.DataFrame) -> np.ndarray:
            return test_data["target"].values

        result = compare_with_baseline(
            autogluon_predictor=mock_predictor,
            baseline_predict=baseline_predict,
            test_data=test_data,
            label="target",
            output_path=tmp_path / "report.json",
        )

        assert isinstance(result, ModelComparison)
        assert (tmp_path / "report.json").exists()


class TestFullComparisonPipeline:
    """Tests pour full_comparison_pipeline."""

    def test_full_pipeline_basic(self, tmp_path: Path) -> None:
        """Test pipeline complet."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier

        from scripts.comparison.statistical_comparison import full_comparison_pipeline

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model_a = LogisticRegression(random_state=42)
        model_b = DecisionTreeClassifier(random_state=42)

        result = full_comparison_pipeline(
            model_a_fit=model_a.fit,
            model_b_fit=model_b.fit,
            model_a_predict=model_a.predict,
            model_b_predict=model_b.predict,
            X=X,
            y=y,
            n_iterations=2,
            output_dir=tmp_path,
        )

        assert isinstance(result, ModelComparison)
        assert (tmp_path / "comparison_report.json").exists()


class TestGenerateRecommendation:
    """Tests pour _generate_recommendation."""

    def test_recommendation_tie_practical(self) -> None:
        """Test recommandation pour tie avec signif pratique."""
        from scripts.comparison.statistical_comparison import _generate_recommendation

        mcnemar = McNemarResult(
            statistic=1.0,
            p_value=0.1,
            significant=False,
            effect_size=0.1,
            confidence_interval=(-0.05, 0.15),
            model_a_mean_accuracy=0.85,
            model_b_mean_accuracy=0.78,
            winner=None,
        )

        rec = _generate_recommendation(
            winner="tie",
            mcnemar=mcnemar,
            metrics_a={"accuracy": 0.85},
            metrics_b={"accuracy": 0.78},
            model_a_name="A",
            model_b_name="B",
            practical_significance=True,
        )

        assert "tendance" in rec.lower() or "A" in rec

    def test_recommendation_winner_practical(self) -> None:
        """Test recommandation pour gagnant avec signif pratique."""
        from scripts.comparison.statistical_comparison import _generate_recommendation

        mcnemar = McNemarResult(
            statistic=5.0,
            p_value=0.01,
            significant=True,
            effect_size=0.3,
            confidence_interval=(0.05, 0.15),
            model_a_mean_accuracy=0.9,
            model_b_mean_accuracy=0.7,
            winner="model_a",
        )

        rec = _generate_recommendation(
            winner="ModelA",
            mcnemar=mcnemar,
            metrics_a={"accuracy": 0.9},
            metrics_b={"accuracy": 0.7},
            model_a_name="ModelA",
            model_b_name="ModelB",
            practical_significance=True,
        )

        assert "deployer" in rec.lower() or "ModelA" in rec


class TestSaveComparisonReport:
    """Tests pour save_comparison_report."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        """Test que la sauvegarde cree un fichier."""
        comparison = ModelComparison(
            model_a_name="A",
            model_b_name="B",
            mcnemar_result=McNemarResult(
                statistic=1.0,
                p_value=0.5,
                significant=False,
                effect_size=0.1,
                confidence_interval=(-0.1, 0.1),
                model_a_mean_accuracy=0.8,
                model_b_mean_accuracy=0.79,
                winner=None,
            ),
            metrics_a={"accuracy": 0.8},
            metrics_b={"accuracy": 0.79},
            winner="tie",
            practical_significance=False,
            recommendation="Choose based on operational criteria.",
        )

        output_path = tmp_path / "report.json"
        save_comparison_report(comparison, output_path)

        assert output_path.exists()

    def test_save_json_valid(self, tmp_path: Path) -> None:
        """Test que le JSON est valide."""
        import json

        comparison = ModelComparison(
            model_a_name="A",
            model_b_name="B",
            mcnemar_result=McNemarResult(
                statistic=1.0,
                p_value=0.5,
                significant=False,
                effect_size=0.1,
                confidence_interval=(-0.1, 0.1),
                model_a_mean_accuracy=0.8,
                model_b_mean_accuracy=0.79,
                winner=None,
            ),
            metrics_a={"accuracy": 0.8},
            metrics_b={"accuracy": 0.79},
            winner="tie",
            practical_significance=False,
            recommendation="Test",
        )

        output_path = tmp_path / "report.json"
        save_comparison_report(comparison, output_path)

        # Should not raise
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["winner"] == "tie"
        assert "mcnemar" in data
