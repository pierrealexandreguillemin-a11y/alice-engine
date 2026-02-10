"""Fixtures Fairness Auto Report - ISO 29119.

Document ID: ALICE-TEST-FAIRNESS-AUTO-CONFTEST
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.fairness.auto_report.types import AttributeAnalysis


@pytest.fixture
def sample_y_true() -> np.ndarray:
    """Labels reels binaires."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, 500)


@pytest.fixture
def sample_y_pred() -> np.ndarray:
    """Predictions binaires."""
    rng = np.random.default_rng(43)
    return rng.integers(0, 2, 500)


@pytest.fixture
def sample_test_data() -> pd.DataFrame:
    """DataFrame test avec attributs proteges."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame(
        {
            "ligue_code": rng.choice(["IDF", "ARA", "OCC", "BRE"], n),
            "blanc_titre": rng.choice(["GM", "IM", "FM", "WGM", "WIM", ""], n),
        }
    )


@pytest.fixture
def mock_analysis_fair() -> AttributeAnalysis:
    """Analyse mock avec status fair."""
    return AttributeAnalysis(
        attribute_name="ligue_code",
        sample_count=500,
        group_count=4,
        demographic_parity_ratio=0.90,
        equalized_odds_tpr_diff=0.05,
        equalized_odds_fpr_diff=0.04,
        predictive_parity_diff=0.03,
        min_group_accuracy=0.82,
        status="fair",
    )


@pytest.fixture
def mock_analysis_critical() -> AttributeAnalysis:
    """Analyse mock avec status critical."""
    return AttributeAnalysis(
        attribute_name="blanc_titre",
        sample_count=500,
        group_count=6,
        demographic_parity_ratio=0.60,
        equalized_odds_tpr_diff=0.25,
        equalized_odds_fpr_diff=0.20,
        predictive_parity_diff=0.18,
        min_group_accuracy=0.55,
        status="critical",
    )
