"""Tests McNemar Compare - ISO 29119.

Document ID: ALICE-TEST-MCNEMAR-COMPARE
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from scripts.comparison.statistical_comparison import ModelComparison, compare_models


class TestCompareModels:
    """Tests pour compare_models."""

    def test_basic_comparison(self) -> None:
        """Test comparaison basique."""
        np.random.seed(42)
        y_test = np.random.randint(0, 2, 100)

        def predict_a(x_data: np.ndarray) -> np.ndarray:
            return y_test

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
        from scripts.comparison.statistical_comparison import compare_with_baseline

        mock_predictor = MagicMock()
        test_data = pd.DataFrame(
            {
                "feature_1": np.random.randn(50),
                "feature_2": np.random.randn(50),
                "target": np.random.randint(0, 2, 50),
            }
        )

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
