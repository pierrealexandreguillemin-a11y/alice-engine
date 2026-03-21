"""Shared utilities for feature engineering — ISO 5055/5259.

Document ID: ALICE-FEAT-HELPERS
Version: 1.0.0

Single source of truth for:
- Forfait sentinel value (resultat_blanc=2.0)
- Result type classification (played vs non-played)
- W/D/L rate computation

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality (SRP, <300 lines)
- ISO/IEC 5259:2024 - Data Quality for ML (forfait exclusion)
"""

from __future__ import annotations

import pandas as pd

FORFAIT_RESULT = 2.0

PLAYED_RESULTS = {
    "victoire_blanc",
    "victoire_noir",
    "nulle",
    "victoire_blanc_ajournement",
    "victoire_noir_ajournement",
    "ajournement",
}

NON_PLAYED = {
    "non_joue",
    "forfait_blanc",
    "forfait_noir",
    "double_forfait",
}


def exclude_forfeits(df: pd.DataFrame) -> pd.DataFrame:
    """Remove forfait rows (resultat_blanc=2.0) from DataFrame."""
    return df[df["resultat_blanc"] != FORFAIT_RESULT].copy()


def filter_played_games(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only actually played games (no forfeits, no non-played)."""
    mask = df["type_resultat"].isin(PLAYED_RESULTS) & (df["resultat_blanc"] != FORFAIT_RESULT)
    return df[mask].copy()


def compute_wdl_rates(results: pd.Series) -> dict[str, float]:
    """Compute win/draw/loss rates from a series of resultat values (0, 0.5, 1)."""
    n = len(results)
    if n == 0:
        return {"win_rate": 0.0, "draw_rate": 0.0, "expected_score": 0.0}
    wins = (results == 1.0).sum()
    draws = (results == 0.5).sum()
    return {
        "win_rate": float(wins / n),
        "draw_rate": float(draws / n),
        "expected_score": float((wins + 0.5 * draws) / n),
    }
