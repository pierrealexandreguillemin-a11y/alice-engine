"""Fixtures McNemar - ISO 29119.

Document ID: ALICE-TEST-MCNEMAR-CONFTEST
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import numpy as np
import pytest


@pytest.fixture
def identical_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predictions identiques (pas de difference)."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = y_true.copy()
    return y_true, y_pred, y_pred


@pytest.fixture
def different_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predictions differentes (un modele meilleur)."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)

    # Modele A: 90% accuracy
    y_pred_a = y_true.copy()
    errors_a = np.random.choice(100, 10, replace=False)
    y_pred_a[errors_a] = 1 - y_pred_a[errors_a]

    # Modele B: 70% accuracy
    y_pred_b = y_true.copy()
    errors_b = np.random.choice(100, 30, replace=False)
    y_pred_b[errors_b] = 1 - y_pred_b[errors_b]

    return y_true, y_pred_a, y_pred_b


@pytest.fixture
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Donnees de classification."""
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y
