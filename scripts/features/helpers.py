"""Shared utilities for feature engineering — ISO 5055/5259.

Document ID: ALICE-FEAT-HELPERS
Version: 2.0.0

Single source of truth for:
- Result type classification (played vs non-played via type_resultat)
- W/D/L rate computation (resultat_blanc: 0=loss, 0.5=draw, 1.0/2.0=win)
- FFE scoring: 2.0 = victoire jeunes (J02 §4.1: 2pts éch. non-U10), NOT forfait

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality (SRP, <300 lines)
- ISO/IEC 5259:2024 - Data Quality for ML (type_resultat-based filtering)

Fix 2026-03-25: resultat_blanc=2.0 was wrongly treated as forfait.
See docs/postmortem/2026-03-25-resultat-blanc-2.0-bug.md
"""

from __future__ import annotations

import pandas as pd

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

# Win values in resultat_blanc (1.0 = adulte/U10, 2.0 = jeunes non-U10)
WIN_VALUES = {1.0, 2.0}


def filter_played_games(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only actually played games (no forfeits, no non-played).

    Uses type_resultat column (NOT resultat_blanc) to identify non-played games.
    resultat_blanc=2.0 is a real win (jeunes FFE), NOT a forfeit.
    """
    if "type_resultat" not in df.columns:
        return df.copy()
    return df[df["type_resultat"].isin(PLAYED_RESULTS)].copy()


def compute_wdl_rates(results: pd.Series) -> dict[str, float]:
    """Compute win/draw/loss rates from resultat_blanc values (0, 0.5, 1.0, 2.0).

    Both 1.0 and 2.0 count as wins (2.0 = jeunes FFE J02 §4.1).
    """
    n = len(results)
    if n == 0:
        return {"win_rate": 0.0, "draw_rate": 0.0, "expected_score": 0.0}
    wins = results.isin(WIN_VALUES).sum()
    draws = (results == 0.5).sum()
    return {
        "win_rate": float(wins / n),
        "draw_rate": float(draws / n),
        "expected_score": float((wins + 0.5 * draws) / n),
    }
