"""Fixtures ML Tests - ISO 29119.

Document ID: ALICE-TEST-ML-FIX
Version: 1.0.0

Shared fixtures for ML module tests (explainability, trainers, parallel).

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<80 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

import numpy as np
import pandas as pd
import pytest

from scripts.ml_types import ModelMetrics, TrainingResult


@pytest.fixture
def sample_X_train() -> pd.DataFrame:  # noqa: N802
    """DataFrame features pour training."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature_a": np.random.randn(100),
            "feature_b": np.random.randn(100),
            "cat_feature": np.random.choice(["A", "B", "C"], 100),
        }
    )


@pytest.fixture
def sample_y_train() -> pd.Series:
    """Series labels pour training."""
    np.random.seed(42)
    return pd.Series(np.random.randint(0, 2, 100))


@pytest.fixture
def sample_X_valid() -> pd.DataFrame:  # noqa: N802
    """DataFrame features pour validation."""
    np.random.seed(43)
    return pd.DataFrame(
        {
            "feature_a": np.random.randn(50),
            "feature_b": np.random.randn(50),
            "cat_feature": np.random.choice(["A", "B", "C"], 50),
        }
    )


@pytest.fixture
def sample_y_valid() -> pd.Series:
    """Series labels pour validation."""
    np.random.seed(43)
    return pd.Series(np.random.randint(0, 2, 50))


@pytest.fixture
def mock_model_metrics() -> ModelMetrics:
    """ModelMetrics mock pour tests."""
    return ModelMetrics(
        auc_roc=0.85,
        accuracy=0.80,
        precision=0.78,
        recall=0.82,
        f1_score=0.80,
        log_loss=0.45,
    )


@pytest.fixture
def mock_training_result(mock_model_metrics: ModelMetrics) -> TrainingResult:
    """TrainingResult mock pour tests."""
    return TrainingResult(
        model=None,
        train_time=10.5,
        metrics=mock_model_metrics,
    )


@pytest.fixture
def feature_names() -> list[str]:
    """Noms des features pour tests SHAP."""
    return ["feature_a", "feature_b", "cat_feature"]
