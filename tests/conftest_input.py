"""Fixtures Input Validation Tests - ISO 24029.

Document ID: ALICE-TEST-INPUT-FIX
Version: 1.0.0

Shared fixtures for input validation tests.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<80 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

import numpy as np
import pandas as pd
import pytest

from scripts.model_registry.input_types import FeatureBounds, InputBoundsConfig


@pytest.fixture
def training_data() -> pd.DataFrame:
    """Données d'entraînement pour créer les bornes."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "age": np.random.randint(18, 80, 1000),
            "income": np.random.normal(50000, 15000, 1000),
            "score": np.random.uniform(0, 100, 1000),
            "category": np.random.choice(["A", "B", "C"], 1000),
        }
    )


@pytest.fixture
def numeric_bounds() -> FeatureBounds:
    """Bornes pour feature numérique."""
    return FeatureBounds(
        feature_name="income",
        min_value=20000,
        max_value=100000,
        mean=50000,
        std=15000,
        p01=22000,
        p99=85000,
        n_samples=1000,
        is_categorical=False,
    )


@pytest.fixture
def categorical_bounds() -> FeatureBounds:
    """Bornes pour feature catégorielle."""
    return FeatureBounds(
        feature_name="category",
        min_value=0,
        max_value=0,
        mean=0,
        std=0,
        p01=0,
        p99=0,
        n_samples=1000,
        is_categorical=True,
        categories=["A", "B", "C"],
    )


@pytest.fixture
def bounds_config(
    numeric_bounds: FeatureBounds, categorical_bounds: FeatureBounds
) -> InputBoundsConfig:
    """Configuration de bornes complète."""
    return InputBoundsConfig(
        model_version="v1.0",
        created_at="2026-01-14T00:00:00",
        training_samples=1000,
        features={
            "income": numeric_bounds,
            "category": categorical_bounds,
        },
    )


@pytest.fixture
def valid_input() -> dict:
    """Input valide dans les bornes."""
    return {"income": 55000, "category": "A"}


@pytest.fixture
def ood_input() -> dict:
    """Input hors distribution."""
    return {"income": 200000, "category": "Z"}
