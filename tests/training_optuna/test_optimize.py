"""Tests Optimize Hyperparameters - ISO 29119.

Document ID: ALICE-TEST-TRAINING-OPTUNA-OPTIMIZE
Version: 1.0.0

Tests pour optimize_hyperparameters.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest

from scripts.training.optuna_tuning import optimize_hyperparameters


class TestOptimizeHyperparameters:
    """Tests pour optimize_hyperparameters."""

    def test_invalid_model_raises_error(self, sample_data, sample_config):
        """Modèle invalide lève ValueError."""
        X, y = sample_data
        X_train, X_valid = X[:80], X[80:]
        y_train, y_valid = y[:80], y[80:]

        with pytest.raises(ValueError, match="non support"):
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

        mock_model = MagicMock()
        mock_model.get_best_score.return_value = {"validation": {"AUC": 0.75}}

        with patch(
            "scripts.training.optuna_objectives.CatBoostClassifier",
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
            assert "iterations" in result or "depth" in result or "learning_rate" in result

    def test_default_model_catboost(self, sample_data, sample_config):
        """Vérifie que le modèle par défaut est catboost."""
        sig = inspect.signature(optimize_hyperparameters)
        default = sig.parameters["model_name"].default

        assert default == "catboost"
