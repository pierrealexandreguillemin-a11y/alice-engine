"""Tests Trainers - ISO 42001/29119.

Document ID: ALICE-TEST-TRAINERS
Version: 1.0.0
Tests: 6

Classes:
- TestTrainCatboost: Tests CatBoost training (2 tests)
- TestTrainXgboost: Tests XGBoost training (2 tests)
- TestTrainLightgbm: Tests LightGBM training (2 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 42001:2023 - AI Management (Traçabilité)
- ISO/IEC 5055:2021 - Code Quality (<80 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

import numpy as np
import pandas as pd
import pytest

from scripts.training.trainers import train_catboost, train_lightgbm, train_xgboost

# Fixtures loaded via pytest_plugins in conftest.py


@pytest.fixture
def mini_train_data():
    """Données minimales pour tests rapides."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "f1": np.random.randn(50),
            "f2": np.random.randn(50),
        }
    )
    y = pd.Series(np.random.randint(0, 2, 50))
    return X, y


@pytest.fixture
def mini_valid_data():
    """Données validation minimales."""
    np.random.seed(43)
    X = pd.DataFrame(
        {
            "f1": np.random.randn(20),
            "f2": np.random.randn(20),
        }
    )
    y = pd.Series(np.random.randint(0, 2, 20))
    return X, y


class TestTrainCatboost:
    """Tests pour train_catboost."""

    def test_returns_training_result(self, mini_train_data, mini_valid_data):
        """Retourne TrainingResult valide."""
        X_train, y_train = mini_train_data
        X_valid, y_valid = mini_valid_data

        result = train_catboost(
            X_train,
            y_train,
            X_valid,
            y_valid,
            cat_features=[],
            params={"iterations": 5, "verbose": 0},
        )

        assert result.model is not None
        assert result.train_time > 0
        assert 0 <= result.metrics.auc_roc <= 1

    def test_metrics_computed(self, mini_train_data, mini_valid_data):
        """Calcule toutes les métriques."""
        X_train, y_train = mini_train_data
        X_valid, y_valid = mini_valid_data

        result = train_catboost(
            X_train,
            y_train,
            X_valid,
            y_valid,
            cat_features=[],
            params={"iterations": 5, "verbose": 0},
        )

        assert result.metrics.accuracy >= 0
        assert result.metrics.precision >= 0
        assert result.metrics.recall >= 0


class TestTrainXgboost:
    """Tests pour train_xgboost."""

    def test_returns_training_result(self, mini_train_data, mini_valid_data):
        """Retourne TrainingResult valide."""
        X_train, y_train = mini_train_data
        X_valid, y_valid = mini_valid_data

        result = train_xgboost(
            X_train,
            y_train,
            X_valid,
            y_valid,
            params={"n_estimators": 5, "verbosity": 0},
        )

        assert result.model is not None
        assert result.train_time > 0

    def test_metrics_computed(self, mini_train_data, mini_valid_data):
        """Calcule toutes les métriques."""
        X_train, y_train = mini_train_data
        X_valid, y_valid = mini_valid_data

        result = train_xgboost(
            X_train,
            y_train,
            X_valid,
            y_valid,
            params={"n_estimators": 5, "verbosity": 0},
        )

        assert 0 <= result.metrics.auc_roc <= 1


class TestTrainLightgbm:
    """Tests pour train_lightgbm."""

    def test_returns_training_result(self, mini_train_data, mini_valid_data):
        """Retourne TrainingResult valide."""
        X_train, y_train = mini_train_data
        X_valid, y_valid = mini_valid_data

        result = train_lightgbm(
            X_train,
            y_train,
            X_valid,
            y_valid,
            cat_features=[],
            params={"n_estimators": 5, "verbose": -1},
        )

        assert result.model is not None
        assert result.train_time > 0

    def test_metrics_computed(self, mini_train_data, mini_valid_data):
        """Calcule toutes les métriques."""
        X_train, y_train = mini_train_data
        X_valid, y_valid = mini_valid_data

        result = train_lightgbm(
            X_train,
            y_train,
            X_valid,
            y_valid,
            cat_features=[],
            params={"n_estimators": 5, "verbose": -1},
        )

        assert result.metrics.f1_score >= 0
