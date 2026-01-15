"""Tests Integration Optuna - ISO 29119.

Document ID: ALICE-TEST-TRAINING-OPTUNA-INTEGRATION
Version: 1.0.0

Tests d'intégration avec Optuna réel.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from unittest.mock import MagicMock, patch

import optuna

from scripts.training.optuna_tuning import (
    _create_lightgbm_objective,
    _create_xgboost_objective,
)


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

        mock_model = MagicMock()
        mock_model.best_score = 0.82

        with patch(
            "scripts.training.optuna_objectives.XGBClassifier",
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

        mock_model = MagicMock()
        mock_model.best_score_ = {"valid_0": {"auc": 0.88}}

        with patch(
            "scripts.training.optuna_objectives.lgb.LGBMClassifier",
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
