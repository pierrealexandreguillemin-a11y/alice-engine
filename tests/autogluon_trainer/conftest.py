"""Fixtures AutoGluon Trainer - ISO 29119.

Document ID: ALICE-TEST-AUTOGLUON-TRAINER-CONFTEST
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_train_data() -> pd.DataFrame:
    """Donnees d'entrainement synthetiques."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.choice(["A", "B", "C"], n_samples),
            "target": np.random.randint(0, 2, n_samples),
        }
    )


@pytest.fixture
def sample_test_data() -> pd.DataFrame:
    """Donnees de test synthetiques."""
    np.random.seed(123)
    n_samples = 50
    return pd.DataFrame(
        {
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.choice(["A", "B", "C"], n_samples),
            "target": np.random.randint(0, 2, n_samples),
        }
    )


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Fichier de configuration temporaire."""
    config_content = """
autogluon:
  presets: "best_quality"
  time_limit: 60
  eval_metric: "roc_auc"
  num_bag_folds: 3
  num_stack_levels: 1
  tabpfn:
    enabled: true
    n_ensemble_configurations: 8
  models:
    include:
      - CatBoost
      - XGBoost
  random_seed: 42
  verbosity: 1
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content, encoding="utf-8")
    return config_path
