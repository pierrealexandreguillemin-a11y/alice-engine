"""Tests Parallel Training - ISO 42001/23894.

Document ID: ALICE-TEST-PARALLEL
Version: 1.0.0
Tests: 5

Classes:
- TestGetModelConfig: Tests extraction config (2 tests)
- TestTrainSingleModel: Tests training unitaire (2 tests)
- TestTrainAllModels: Tests orchestration (1 test)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 42001:2023 - AI Management (Gouvernance)
- ISO/IEC 23894:2023 - AI Risk Management (Ressources)
- ISO/IEC 5055:2021 - Code Quality (<100 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from unittest.mock import MagicMock, patch

from scripts.ml_types import ModelMetrics, TrainingResult
from scripts.training.parallel import (
    _get_model_config,
    _train_single_model,
)

# Fixtures loaded via pytest_plugins in conftest.py


class TestGetModelConfig:
    """Tests pour _get_model_config."""

    def test_extracts_config(self):
        """Extrait la config d'un modèle."""
        config = {
            "catboost": {"iterations": 500, "depth": 6},
            "xgboost": {"n_estimators": 1000},
        }
        result = _get_model_config(config, "catboost")

        assert result["iterations"] == 500
        assert result["depth"] == 6

    def test_missing_config_returns_empty(self):
        """Retourne dict vide si config absente."""
        config = {"catboost": {"iterations": 100}}
        result = _get_model_config(config, "lightgbm")

        assert result == {}


class TestTrainSingleModel:
    """Tests pour _train_single_model."""

    def test_successful_training(self, mock_model_metrics):
        """Training réussi retourne résultat."""
        expected = TrainingResult(model=MagicMock(), train_time=5.0, metrics=mock_model_metrics)

        def mock_train_fn():
            """Fonction mock de training."""
            return expected

        result = _train_single_model("TestModel", mock_train_fn)

        assert result.train_time == 5.0
        assert result.metrics.auc_roc == 0.85

    def test_failed_training_returns_empty_metrics(self):
        """Training échoué retourne métriques vides."""

        def failing_train_fn():
            """Fonction mock qui échoue."""
            raise RuntimeError("Training failed")

        result = _train_single_model("FailModel", failing_train_fn)

        assert result.model is None
        assert result.metrics.auc_roc == 0.0
        assert result.metrics.log_loss == 1.0


class TestTrainAllModels:
    """Tests pour train_all_models_parallel."""

    def test_orchestration_calls_all_trainers(
        self,
        sample_X_train,  # noqa: N803
        sample_y_train,
        sample_X_valid,  # noqa: N803
        sample_y_valid,
    ):
        """Appelle les 3 trainers séquentiellement."""
        from scripts.training.parallel import train_all_models_parallel

        config = {
            "catboost": {"iterations": 10},
            "xgboost": {"n_estimators": 10},
            "lightgbm": {"n_estimators": 10},
        }

        with (
            patch("scripts.training.parallel.train_catboost") as mock_cat,
            patch("scripts.training.parallel.train_xgboost") as mock_xgb,
            patch("scripts.training.parallel.train_lightgbm") as mock_lgb,
        ):
            mock_result = TrainingResult(
                model=MagicMock(),
                train_time=1.0,
                metrics=ModelMetrics(
                    auc_roc=0.8,
                    accuracy=0.8,
                    precision=0.8,
                    recall=0.8,
                    f1_score=0.8,
                    log_loss=0.4,
                ),
            )
            mock_cat.return_value = mock_result
            mock_xgb.return_value = mock_result
            mock_lgb.return_value = mock_result

            results = train_all_models_parallel(
                sample_X_train,
                sample_y_train,
                sample_X_valid,
                sample_y_valid,
                cat_features=["cat_feature"],
                config=config,
            )

            assert "CatBoost" in results
            assert "XGBoost" in results
            assert "LightGBM" in results
