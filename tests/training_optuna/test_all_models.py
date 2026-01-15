"""Tests Optimize All Models - ISO 29119.

Document ID: ALICE-TEST-TRAINING-OPTUNA-ALL-MODELS
Version: 1.0.0

Tests pour optimize_all_models.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from unittest.mock import patch

from scripts.training.optuna_tuning import optimize_all_models


class TestOptimizeAllModels:
    """Tests pour optimize_all_models."""

    def test_returns_dict_with_three_models(self, sample_data, sample_config):
        """Vérifie retour dict avec 3 modèles."""
        X, y = sample_data
        X_train, X_valid = X[:80], X[80:]
        y_train, y_valid = y[:80], y[80:]

        with patch("scripts.training.optuna_core.optimize_hyperparameters") as mock_opt:
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

        with patch("scripts.training.optuna_core.optimize_hyperparameters") as mock_opt:
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
