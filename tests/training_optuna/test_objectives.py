"""Tests Objective Creators - ISO 29119.

Document ID: ALICE-TEST-TRAINING-OPTUNA-OBJECTIVES
Version: 1.0.0

Tests pour les créateurs d'objectifs Optuna.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from unittest.mock import MagicMock, patch

import optuna

from scripts.training.optuna_tuning import (
    _create_catboost_objective,
    _create_lightgbm_objective,
    _create_xgboost_objective,
)


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

        mock_model = MagicMock()
        mock_model.get_best_score.return_value = {"validation": {"AUC": 0.85}}

        with patch(
            "scripts.training.optuna_objectives.CatBoostClassifier",
            return_value=mock_model,
        ):
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
