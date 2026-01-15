"""Tests MLflow Internal Functions - ISO 42001.

Document ID: ALICE-TEST-MLFLOW-INT
Version: 1.0.0
Tests: 6

Classes:
- TestLogModelResults: Tests logging résultats (2 tests)
- TestLogModelParams: Tests logging paramètres (2 tests)
- TestLogModelMetrics: Tests logging métriques (2 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 42001:2023 - AI Management (Traçabilité)
- ISO/IEC 5055:2021 - Code Quality (<80 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

import sys
from unittest.mock import MagicMock, patch

from scripts.ml_types import ModelMetrics, TrainingResult


class TestLogModelResults:
    """Tests pour _log_model_results."""

    def test_logs_each_model(self):
        """Log chaque modèle avec nested run."""
        from scripts.training.mlflow_tracking import _log_model_results

        mock_metrics = ModelMetrics(
            auc_roc=0.85,
            accuracy=0.80,
            precision=0.78,
            recall=0.82,
            f1_score=0.80,
            log_loss=0.45,
        )
        result = TrainingResult(model=MagicMock(), train_time=10.0, metrics=mock_metrics)
        results = {"CatBoost": result, "XGBoost": result}
        config = {"catboost": {"iterations": 100}}

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            _log_model_results(results, config)

        # Should create nested runs for each model
        assert mock_mlflow.start_run.call_count >= 2

    def test_skips_none_models(self):
        """Ne log pas les modèles None."""
        from scripts.training.mlflow_tracking import _log_model_results

        result_none = TrainingResult(model=None, train_time=0.0, metrics=MagicMock())
        results = {"Failed": result_none}

        mock_mlflow = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            _log_model_results(results, {})

        # Should not create any nested runs
        mock_mlflow.start_run.assert_not_called()


class TestLogModelParams:
    """Tests pour _log_model_params."""

    def test_logs_valid_params(self):
        """Log les paramètres valides."""
        from scripts.training.mlflow_tracking import _log_model_params

        config = {
            "catboost": {
                "iterations": 1000,
                "learning_rate": 0.03,
                "depth": 6,
            }
        }

        mock_mlflow = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            _log_model_params(config, "CatBoost")

        assert mock_mlflow.log_param.call_count == 3

    def test_handles_missing_config(self):
        """Gère l'absence de config modèle."""
        from scripts.training.mlflow_tracking import _log_model_params

        config = {}

        mock_mlflow = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            _log_model_params(config, "XGBoost")

        mock_mlflow.log_param.assert_not_called()


class TestLogModelMetrics:
    """Tests pour _log_model_metrics."""

    def test_logs_numeric_metrics(self):
        """Log les métriques numériques."""
        from scripts.training.mlflow_tracking import _log_model_metrics

        mock_metrics = MagicMock()
        mock_metrics.to_dict.return_value = {
            "auc_roc": 0.85,
            "accuracy": 0.80,
            "f1_score": 0.78,
        }
        mock_result = MagicMock()
        mock_result.metrics = mock_metrics

        mock_mlflow = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            _log_model_metrics(mock_result)

        assert mock_mlflow.log_metric.call_count == 3

    def test_skips_non_numeric(self):
        """Ignore les métriques non-numériques."""
        from scripts.training.mlflow_tracking import _log_model_metrics

        mock_metrics = MagicMock()
        mock_metrics.to_dict.return_value = {
            "auc_roc": 0.85,
            "model_name": "CatBoost",  # String - should skip
        }
        mock_result = MagicMock()
        mock_result.metrics = mock_metrics

        mock_mlflow = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            _log_model_metrics(mock_result)

        # Only numeric metric logged
        mock_mlflow.log_metric.assert_called_once()
