"""Fixtures Training Optuna - ISO 29119.

Document ID: ALICE-TEST-TRAINING-OPTUNA-CONFTEST
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import numpy as np
import pandas as pd
import pytest


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
            "n_trials": 2,
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
