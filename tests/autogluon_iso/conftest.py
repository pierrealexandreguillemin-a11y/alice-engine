"""Fixtures AutoGluon ISO - ISO 29119.

Document ID: ALICE-TEST-AUTOGLUON-ISO-CONFTEST
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from scripts.autogluon.config import AutoGluonConfig


@pytest.fixture
def mock_training_result() -> MagicMock:
    """Resultat d'entrainement mock."""
    result = MagicMock()
    result.model_path = Path("models/test_model")
    result.data_hash = "abc123def456" * 5  # 64 chars
    result.best_model = "CatBoost"
    result.config = AutoGluonConfig(presets="best_quality")
    result.metrics = {
        "score_val": 0.85,
        "pred_time_val": 0.1,
        "fit_time": 10.0,
        "num_models": 5,
    }
    result.predictor = MagicMock()
    result.predictor.feature_importance.return_value = pd.DataFrame(
        {
            "importance": [0.3, 0.5, 0.2],
        },
        index=["feature_1", "feature_2", "feature_3"],
    )
    result.predictor.path = "models/test"
    result.predictor.label = "target"
    return result


@pytest.fixture
def test_data() -> pd.DataFrame:
    """Donnees de test."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "feature_1": np.random.randn(n),
            "feature_2": np.random.randn(n),
            "ligue_code": np.random.choice(["IDF", "ARA", "BRE"], n),
            "target": np.random.randint(0, 2, n),
        }
    )


@pytest.fixture
def mock_predictor() -> MagicMock:
    """Predictor mock."""
    predictor = MagicMock()
    predictor.path = "models/test"
    predictor.label = "target"
    return predictor
