"""Fixtures Protected Attributes - ISO 29119.

Document ID: ALICE-TEST-PROTECTED-ATTRS-CONFTEST
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

from scripts.fairness.protected.types import (
    ProtectedAttribute,
    ProtectionLevel,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """DataFrame echantillon avec features + attributs proteges."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "blanc_elo": rng.integers(1200, 2400, n),
            "noir_elo": rng.integers(1200, 2400, n),
            "diff_elo": rng.integers(-500, 500, n),
            "echiquier": rng.integers(1, 9, n),
            "ligue_code": rng.choice(["IDF", "ARA", "OCC", "BRE"], n),
            "blanc_titre": rng.choice(["GM", "IM", "FM", "WGM", "WIM", ""], n),
            "noir_titre": rng.choice(["GM", "IM", "FM", "WGM", "WIM", ""], n),
            "jour_semaine": rng.integers(0, 7, n),
        }
    )


@pytest.fixture
def ffe_protected_attributes() -> list[ProtectedAttribute]:
    """Configuration des attributs proteges FFE."""
    return [
        ProtectedAttribute(
            name="ligue_code",
            level=ProtectionLevel.PROXY_CHECK,
            reason="discrimination geographique regionale",
        ),
        ProtectedAttribute(
            name="blanc_titre",
            level=ProtectionLevel.PROXY_CHECK,
            reason="proxy genre via titres feminins (WGM, WIM, WFM)",
        ),
    ]
