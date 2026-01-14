"""Fixtures partagées pour tests robustesse - ISO 24029.

Document ID: ALICE-TEST-ROBUST-CONF
Version: 1.0.0

Fixtures:
- simple_model: Modèle simple (signe features)
- robust_model: Modèle robuste (constant)
- fragile_model: Modèle fragile (sensible)
- sample_data: Données de test (100x5)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<80 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

import numpy as np
import pytest


@pytest.fixture
def simple_model():
    """Modèle simple qui retourne le signe des features."""

    def predict(x_input):
        x_input = np.asarray(x_input)
        if len(x_input.shape) == 1:
            return (x_input > 0).astype(int)
        return (x_input.mean(axis=1) > 0).astype(int)

    return predict


@pytest.fixture
def robust_model():
    """Modèle robuste (retourne toujours la même valeur)."""

    def predict(x_input):
        x_input = np.asarray(x_input)
        n = x_input.shape[0]
        return np.ones(n, dtype=int)

    return predict


@pytest.fixture
def fragile_model():
    """Modèle fragile (sensible au moindre changement)."""

    def predict(x_input):
        x_input = np.asarray(x_input)
        if len(x_input.shape) == 1:
            return (np.round(x_input * 100) % 2).astype(int)
        return (np.round(x_input.sum(axis=1) * 100) % 2).astype(int)

    return predict


@pytest.fixture
def sample_data():
    """Données de test standard."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X.mean(axis=1) > 0).astype(int)
    return X, y
