"""Fixtures Bias Tests - ISO 24027.

Document ID: ALICE-TEST-BIAS-FIX
Version: 1.0.0

Shared fixtures for bias monitoring tests.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<80 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

import numpy as np
import pytest

from scripts.monitoring.bias_types import BiasMonitorConfig


@pytest.fixture
def fair_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prédictions équitables entre groupes."""
    np.random.seed(42)
    n = 500
    y_true = np.random.randint(0, 2, n)
    y_pred = y_true.copy()
    y_pred[np.random.choice(n, 50, replace=False)] = (
        1 - y_pred[np.random.choice(n, 50, replace=False)]
    )
    protected = np.array(["A"] * 250 + ["B"] * 250)
    return y_true, y_pred, protected


@pytest.fixture
def biased_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prédictions biaisées (groupe A favorisé)."""
    np.random.seed(42)
    n = 500
    y_true = np.random.randint(0, 2, n)
    y_pred = np.zeros(n, dtype=int)
    # Group A: 80% positive predictions
    y_pred[:250] = np.random.choice([0, 1], 250, p=[0.2, 0.8])
    # Group B: 30% positive predictions
    y_pred[250:] = np.random.choice([0, 1], 250, p=[0.7, 0.3])
    protected = np.array(["A"] * 250 + ["B"] * 250)
    return y_true, y_pred, protected


@pytest.fixture
def default_config() -> BiasMonitorConfig:
    """Configuration par défaut."""
    return BiasMonitorConfig(
        protected_attributes=["gender", "age_group"],
        model_version="v1.0",
    )


@pytest.fixture
def strict_config() -> BiasMonitorConfig:
    """Configuration stricte."""
    return BiasMonitorConfig(
        protected_attributes=["gender"],
        demographic_parity_threshold=0.9,
        equalized_odds_threshold=0.05,
        model_version="v1.0",
    )


@pytest.fixture
def y_prob() -> np.ndarray:
    """Probabilités de prédiction."""
    np.random.seed(42)
    return np.random.uniform(0.3, 0.9, 500)
