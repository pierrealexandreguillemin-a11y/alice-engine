"""Fixtures Features ALI - ISO 29119.

Document ID: ALICE-TEST-FEATURES-ALI-CONFTEST
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_echiquiers() -> pd.DataFrame:
    """Fixture DataFrame échiquiers pour tests."""
    return pd.DataFrame(
        [
            # Joueur A: présent 7/9 rondes (régulier, titulaire)
            {"saison": 2025, "ronde": 1, "blanc_nom": "Joueur A", "noir_nom": "X", "echiquier": 1},
            {"saison": 2025, "ronde": 2, "blanc_nom": "Joueur A", "noir_nom": "Y", "echiquier": 1},
            {"saison": 2025, "ronde": 3, "blanc_nom": "Joueur A", "noir_nom": "Z", "echiquier": 1},
            {"saison": 2025, "ronde": 4, "blanc_nom": "Joueur A", "noir_nom": "W", "echiquier": 1},
            {"saison": 2025, "ronde": 5, "blanc_nom": "Joueur A", "noir_nom": "V", "echiquier": 1},
            {"saison": 2025, "ronde": 6, "blanc_nom": "Joueur A", "noir_nom": "U", "echiquier": 2},
            {"saison": 2025, "ronde": 7, "blanc_nom": "Joueur A", "noir_nom": "T", "echiquier": 1},
            # Joueur B: présent 4/9 rondes (rotation)
            {"saison": 2025, "ronde": 1, "blanc_nom": "X", "noir_nom": "Joueur B", "echiquier": 3},
            {"saison": 2025, "ronde": 3, "blanc_nom": "X", "noir_nom": "Joueur B", "echiquier": 4},
            {"saison": 2025, "ronde": 5, "blanc_nom": "X", "noir_nom": "Joueur B", "echiquier": 5},
            {"saison": 2025, "ronde": 7, "blanc_nom": "X", "noir_nom": "Joueur B", "echiquier": 6},
            # Joueur C: présent 2/9 rondes (rare, remplaçant)
            {"saison": 2025, "ronde": 8, "blanc_nom": "Joueur C", "noir_nom": "S", "echiquier": 8},
            {"saison": 2025, "ronde": 9, "blanc_nom": "Joueur C", "noir_nom": "R", "echiquier": 8},
            # Joueur D: polyvalent (plusieurs échiquiers)
            {"saison": 2025, "ronde": 1, "blanc_nom": "Joueur D", "noir_nom": "Q", "echiquier": 1},
            {"saison": 2025, "ronde": 2, "blanc_nom": "Joueur D", "noir_nom": "P", "echiquier": 4},
            {"saison": 2025, "ronde": 3, "blanc_nom": "Joueur D", "noir_nom": "O", "echiquier": 7},
            {"saison": 2025, "ronde": 4, "blanc_nom": "Joueur D", "noir_nom": "N", "echiquier": 2},
            {"saison": 2025, "ronde": 5, "blanc_nom": "Joueur D", "noir_nom": "M", "echiquier": 5},
            {"saison": 2025, "ronde": 6, "blanc_nom": "Joueur D", "noir_nom": "L", "echiquier": 8},
            {"saison": 2025, "ronde": 7, "blanc_nom": "Joueur D", "noir_nom": "K", "echiquier": 3},
            {"saison": 2025, "ronde": 8, "blanc_nom": "Joueur D", "noir_nom": "J", "echiquier": 6},
        ]
    )


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Fixture DataFrame vide."""
    return pd.DataFrame()
