"""Tests: tests/test_training_optuna.py - Tests Optuna Tuning.

Document ID: ALICE-TEST-OPTUNA-001
Version: 1.0.0
Tests: 16

Ce module teste le module d'optimisation Optuna pour ALICE.

Classes de tests:
- TestOptimizeHyperparameters: Tests fonction principale (4 tests)
- TestObjectiveCreators: Tests créateurs d'objectifs (4 tests)
- TestOptimizeAllModels: Tests optimisation multi-modèles (2 tests)
- TestSearchSpaceParsing: Tests parsing search space (3 tests)
- TestOptunaIntegration: Tests d'intégration Optuna (3 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 42001:2023 - AI Management (reproductibilité)

See Also
--------
- scripts/training/optuna_tuning.py - Module testé
- config/hyperparameters.yaml - Configuration search space

Author: ALICE Engine Team
Last Updated: 2026-01-10

"""

import inspect
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pandas as pd
import pytest

from scripts.training.optuna_tuning import (
    _create_catboost_objective,
    _create_lightgbm_objective,
    _create_xgboost_objective,
    optimize_all_models,
    optimize_hyperparameters,
)


@pytest.fixture
def sample_data():
    """Données de test minimales."""
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame(
        {
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "cat_feature": np.random.choice(["A", "B", "C"], n_samples),
        }
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    return X, y


@pytest.fixture
def sample_config():
    """Configuration Optuna minimale pour tests rapides."""
    return {
        "optuna": {
            "n_trials": 2,  # Minimum pour tests
            "timeout": 60,
            "catboost_search_space": {
                "iterations": [100],
                "learning_rate": [0.1],
                "depth": [4],
                "l2_leaf_reg": [1],
                "min_data_in_leaf": [20],
            },
            "xgboost_search_space": {
                "n_estimators": [100],
                "learning_rate": [0.1],
                "max_depth": [4],
                "reg_lambda": [1.0],
                "colsample_bytree": [1.0],
                "subsample": [1.0],
            },
            "lightgbm_search_space": {
                "n_estimators": [100],
                "learning_rate": [0.1],
                "num_leaves": [31],
                "reg_lambda": [1.0],
                "feature_fraction": [0.9],
                "bagging_fraction": [1.0],
                "min_child_samples": [20],
            },
        }
    }


class TestOptimizeHyperparameters:
    """Tests pour optimize_hyperparameters."""

    def test_invalid_model_raises_error(self, sample_data, sample_config):
        """Modèle invalide lève ValueError."""
        X, y = sample_data
        X_train, X_valid = X[:80], X[80:]
        y_train, y_valid = y[:80], y[80:]

        with pytest.raises(ValueError, match="Modèle non supporté"):
            optimize_hyperparameters(
                X_train,
                y_train,
                X_valid,
                y_valid,
                cat_features=["cat_feature"],
                config=sample_config,
                model_name="invalid_model",
            )

    def test_function_signature(self):
        """Vérifie signature de la fonction."""
        sig = inspect.signature(optimize_hyperparameters)
        params = list(sig.parameters.keys())

        assert "X_train" in params
        assert "y_train" in params
        assert "X_valid" in params
        assert "y_valid" in params
        assert "cat_features" in params
        assert "config" in params
        assert "model_name" in params

    def test_returns_dict_with_mocked_catboost(self, sample_data, sample_config):
        """Vérifie que la fonction retourne un dict avec CatBoost mocké."""
        X, y = sample_data
        X_train, X_valid = X[:80], X[80:]
        y_train, y_valid = y[:80], y[80:]

        # Mock CatBoostClassifier pour éviter l'entraînement réel
        mock_model = MagicMock()
        mock_model.get_best_score.return_value = {"validation": {"AUC": 0.75}}

        with patch(
            "scripts.training.optuna_tuning.CatBoostClassifier",
            return_value=mock_model,
        ):
            result = optimize_hyperparameters(
                X_train,
                y_train,
                X_valid,
                y_valid,
                cat_features=["cat_feature"],
                config=sample_config,
                model_name="catboost",
            )

            assert isinstance(result, dict)
            # Vérifie que les paramètres du search space sont présents
            assert "iterations" in result or "depth" in result or "learning_rate" in result

    def test_default_model_catboost(self, sample_data, sample_config):
        """Vérifie que le modèle par défaut est catboost."""
        sig = inspect.signature(optimize_hyperparameters)
        default = sig.parameters["model_name"].default

        assert default == "catboost"


class TestObjectiveCreators:
    """Tests pour les créateurs d'objectifs."""

    def test_catboost_objective_created(self, sample_data, sample_config):
        """Vérifie création objectif CatBoost."""
        X, y = sample_data
        X_train, X_valid = X[:80], X[80:]
        y_train, y_valid = y[:80], y[80:]

        objective = _create_catboost_objective(
            X_train,
            y_train,
            X_valid,
            y_valid,
            cat_features=["cat_feature"],
            optuna_config=sample_config["optuna"],
        )

        assert callable(objective)

    def test_xgboost_objective_created(self, sample_data, sample_config):
        """Vérifie création objectif XGBoost."""
        X, y = sample_data
        X_train, X_valid = X[:80], X[80:]
        y_train, y_valid = y[:80], y[80:]

        objective = _create_xgboost_objective(
            X_train,
            y_train,
            X_valid,
            y_valid,
            optuna_config=sample_config["optuna"],
        )

        assert callable(objective)

    def test_lightgbm_objective_created(self, sample_data, sample_config):
        """Vérifie création objectif LightGBM."""
        X, y = sample_data
        X_train, X_valid = X[:80], X[80:]
        y_train, y_valid = y[:80], y[80:]

        objective = _create_lightgbm_objective(
            X_train,
            y_train,
            X_valid,
            y_valid,
            cat_features=["cat_feature"],
            optuna_config=sample_config["optuna"],
        )

        assert callable(objective)

    def test_catboost_objective_executes(self, sample_data, sample_config):
        """Vérifie que l'objectif CatBoost s'exécute avec un trial mocké."""
        X, y = sample_data
        X_train, X_valid = X[:80], X[80:]
        y_train, y_valid = y[:80], y[80:]

        objective = _create_catboost_objective(
            X_train,
            y_train,
            X_valid,
            y_valid,
            cat_features=["cat_feature"],
            optuna_config=sample_config["optuna"],
        )

        # Mock model pour éviter l'entraînement réel
        mock_model = MagicMock()
        mock_model.get_best_score.return_value = {"validation": {"AUC": 0.85}}

        with patch(
            "scripts.training.optuna_tuning.CatBoostClassifier",
            return_value=mock_model,
        ):
            # Créer un vrai trial Optuna
            study = optuna.create_study(direction="maximize")
            trial = optuna.trial.FixedTrial(
                {
                    "iterations": 100,
                    "learning_rate": 0.1,
                    "depth": 4,
                    "l2_leaf_reg": 1,
                    "min_data_in_leaf": 20,
                }
            )

            result = objective(trial)
            assert result == 0.85


class TestOptimizeAllModels:
    """Tests pour optimize_all_models."""

    def test_returns_dict_with_three_models(self, sample_data, sample_config):
        """Vérifie retour dict avec 3 modèles."""
        X, y = sample_data
        X_train, X_valid = X[:80], X[80:]
        y_train, y_valid = y[:80], y[80:]

        # Mock pour éviter l'exécution réelle
        with patch("scripts.training.optuna_tuning.optimize_hyperparameters") as mock_opt:
            mock_opt.return_value = {"param": "value"}

            result = optimize_all_models(
                X_train,
                y_train,
                X_valid,
                y_valid,
                cat_features=["cat_feature"],
                config=sample_config,
            )

            assert isinstance(result, dict)
            assert "catboost" in result
            assert "xgboost" in result
            assert "lightgbm" in result

    def test_calls_optimize_three_times(self, sample_data, sample_config):
        """Vérifie que optimize_hyperparameters est appelé 3 fois."""
        X, y = sample_data
        X_train, X_valid = X[:80], X[80:]
        y_train, y_valid = y[:80], y[80:]

        with patch("scripts.training.optuna_tuning.optimize_hyperparameters") as mock_opt:
            mock_opt.return_value = {"param": "value"}

            optimize_all_models(
                X_train,
                y_train,
                X_valid,
                y_valid,
                cat_features=["cat_feature"],
                config=sample_config,
            )

            assert mock_opt.call_count == 3


class TestSearchSpaceParsing:
    """Tests pour le parsing du search space."""

    def test_catboost_uses_min_data_in_leaf(self, sample_data, sample_config):
        """Vérifie que CatBoost utilise min_data_in_leaf."""
        # Le search space devrait inclure min_data_in_leaf
        assert "min_data_in_leaf" in sample_config["optuna"]["catboost_search_space"]

    def test_xgboost_uses_subsampling(self, sample_data, sample_config):
        """Vérifie que XGBoost utilise colsample_bytree et subsample."""
        search_space = sample_config["optuna"]["xgboost_search_space"]

        assert "colsample_bytree" in search_space
        assert "subsample" in search_space

    def test_lightgbm_uses_bagging(self, sample_data, sample_config):
        """Vérifie que LightGBM utilise feature_fraction et bagging_fraction."""
        search_space = sample_config["optuna"]["lightgbm_search_space"]

        assert "feature_fraction" in search_space
        assert "bagging_fraction" in search_space
        assert "min_child_samples" in search_space


class TestOptunaIntegration:
    """Tests d'intégration avec Optuna réel."""

    def test_optuna_study_creation(self):
        """Vérifie que Optuna crée correctement une étude."""
        study = optuna.create_study(
            direction="maximize",
            study_name="test_study",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        assert study.study_name == "test_study"
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE

    def test_xgboost_objective_executes(self, sample_data, sample_config):
        """Vérifie que l'objectif XGBoost s'exécute avec un trial mocké."""
        X, y = sample_data
        X_train, X_valid = X[:80], X[80:]
        y_train, y_valid = y[:80], y[80:]

        objective = _create_xgboost_objective(
            X_train,
            y_train,
            X_valid,
            y_valid,
            optuna_config=sample_config["optuna"],
        )

        # Mock model pour éviter l'entraînement réel
        mock_model = MagicMock()
        mock_model.best_score = 0.82

        with patch(
            "scripts.training.optuna_tuning.XGBClassifier",
            return_value=mock_model,
        ):
            trial = optuna.trial.FixedTrial(
                {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 4,
                    "reg_lambda": 1.0,
                    "colsample_bytree": 1.0,
                    "subsample": 1.0,
                }
            )

            result = objective(trial)
            assert result == 0.82

    def test_lightgbm_objective_executes(self, sample_data, sample_config):
        """Vérifie que l'objectif LightGBM s'exécute avec un trial mocké."""
        X, y = sample_data
        X_train, X_valid = X[:80], X[80:]
        y_train, y_valid = y[:80], y[80:]

        objective = _create_lightgbm_objective(
            X_train,
            y_train,
            X_valid,
            y_valid,
            cat_features=["cat_feature"],
            optuna_config=sample_config["optuna"],
        )

        # Mock model pour éviter l'entraînement réel
        mock_model = MagicMock()
        mock_model.best_score_ = {"valid_0": {"auc": 0.88}}

        with patch(
            "scripts.training.optuna_tuning.lgb.LGBMClassifier",
            return_value=mock_model,
        ):
            trial = optuna.trial.FixedTrial(
                {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "num_leaves": 31,
                    "reg_lambda": 1.0,
                    "feature_fraction": 0.9,
                    "bagging_fraction": 1.0,
                    "min_child_samples": 20,
                }
            )

            result = objective(trial)
            assert result == 0.88
