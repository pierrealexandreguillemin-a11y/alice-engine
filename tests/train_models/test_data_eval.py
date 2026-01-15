"""Tests Data Loading and Evaluation - ISO 29119.

Document ID: ALICE-TEST-TRAIN-DATA-EVAL
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from tests.train_models.conftest import MockMetricsTPP, MockModelResultTPP


class TestLoadDatasets:
    """Tests pour _load_datasets."""

    def test_load_datasets(self, tmp_path: Path) -> None:
        """Test chargement des datasets."""
        from scripts.train_models_parallel import _load_datasets

        train_df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
        valid_df = pd.DataFrame({"a": [4, 5], "target": [1, 0]})
        test_df = pd.DataFrame({"a": [6, 7], "target": [0, 1]})

        train_df.to_parquet(tmp_path / "train.parquet")
        valid_df.to_parquet(tmp_path / "valid.parquet")
        test_df.to_parquet(tmp_path / "test.parquet")

        train, valid, test = _load_datasets(tmp_path)

        assert len(train) == 3
        assert len(valid) == 2
        assert len(test) == 2


class TestEvaluateOnTest:
    """Tests pour _evaluate_on_test."""

    def test_evaluate_models(self) -> None:
        """Test evaluation sur test set."""
        from scripts.train_models_parallel import _evaluate_on_test

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])

        results = {
            "CatBoost": MockModelResultTPP(model=mock_model),
        }

        X_test = pd.DataFrame({"a": [1, 2]})
        y_test = pd.Series([1, 0])

        with patch("scripts.train_models_parallel.compute_all_metrics") as mock_metrics:
            mock_metrics.return_value = MagicMock(auc_roc=0.90, accuracy=0.85, f1_score=0.82)
            _evaluate_on_test(results, X_test, y_test)

        assert results["CatBoost"].metrics.test_auc == 0.90
        assert results["CatBoost"].metrics.test_accuracy == 0.85

    def test_skip_none_model(self) -> None:
        """Test skip modele None."""
        from scripts.train_models_parallel import _evaluate_on_test

        results = {"Failed": MockModelResultTPP(model=None)}
        X_test = pd.DataFrame({"a": [1]})
        y_test = pd.Series([0])

        _evaluate_on_test(results, X_test, y_test)


class TestBuildSummary:
    """Tests pour _build_summary."""

    def test_build_summary(self, tmp_path: Path) -> None:
        """Test construction resume."""
        from scripts.train_models_parallel import _build_summary

        mock_model = MagicMock()
        results = {
            "CatBoost": MockModelResultTPP(
                model=mock_model,
                metrics=MockMetricsTPP(auc_roc=0.85, test_auc=0.82),
            ),
            "XGBoost": MockModelResultTPP(
                model=mock_model,
                metrics=MockMetricsTPP(auc_roc=0.83, test_auc=0.80),
            ),
        }

        version_dir = tmp_path / "v1"
        summary = _build_summary(results, version_dir)

        assert summary["best_model"] == "CatBoost"
        assert summary["best_auc"] == 0.82
        assert "version_dir" in summary

    def test_build_summary_with_none_models(self, tmp_path: Path) -> None:
        """Test resume avec modeles None."""
        from scripts.train_models_parallel import _build_summary

        results = {
            "Failed": MockModelResultTPP(model=None),
            "Success": MockModelResultTPP(
                model=MagicMock(),
                metrics=MockMetricsTPP(test_auc=0.75),
            ),
        }

        summary = _build_summary(results, tmp_path / "v1")

        assert summary["best_model"] == "Success"
        assert summary["best_auc"] == 0.75

    def test_build_summary_fallback_auc(self, tmp_path: Path) -> None:
        """Test resume utilise auc_roc si test_auc None."""
        from scripts.train_models_parallel import _build_summary

        results = {
            "Model": MockModelResultTPP(
                model=MagicMock(),
                metrics=MockMetricsTPP(auc_roc=0.88, test_auc=None),
            ),
        }

        summary = _build_summary(results, tmp_path / "v1")

        assert summary["best_auc"] == 0.88
