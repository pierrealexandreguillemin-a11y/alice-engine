"""Fixtures Drift Tests - ISO 23894.

Document ID: ALICE-TEST-DRIFT-FIX
Version: 1.0.0

Shared fixtures for drift monitoring tests.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<70 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def baseline_data() -> pd.DataFrame:
    """Données baseline pour les tests drift."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature_a": np.random.normal(0, 1, 1000),
            "feature_b": np.random.uniform(0, 10, 1000),
            "category": np.random.choice(["A", "B", "C"], 1000),
        }
    )


@pytest.fixture
def current_no_drift(baseline_data: pd.DataFrame) -> pd.DataFrame:
    """Données current sans drift."""
    np.random.seed(43)
    return pd.DataFrame(
        {
            "feature_a": np.random.normal(0, 1, 800),
            "feature_b": np.random.uniform(0, 10, 800),
            "category": np.random.choice(["A", "B", "C"], 800),
        }
    )


@pytest.fixture
def current_with_drift() -> pd.DataFrame:
    """Données current avec drift significatif."""
    np.random.seed(44)
    return pd.DataFrame(
        {
            "feature_a": np.random.normal(2, 2, 800),  # Mean shift + variance change
            "feature_b": np.random.uniform(5, 20, 800),  # Distribution shift
            "category": np.random.choice(["A", "D", "E"], 800),  # New categories
        }
    )


@pytest.fixture
def numeric_baseline() -> np.ndarray:
    """Array numpy baseline pour tests statistiques."""
    np.random.seed(42)
    return np.random.normal(0, 1, 500)


@pytest.fixture
def numeric_similar() -> np.ndarray:
    """Array numpy similaire au baseline."""
    np.random.seed(43)
    return np.random.normal(0, 1.05, 500)


@pytest.fixture
def numeric_shifted() -> np.ndarray:
    """Array numpy avec shift significatif."""
    np.random.seed(44)
    return np.random.normal(3, 2, 500)
