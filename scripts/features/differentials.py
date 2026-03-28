"""Differential and interaction features for matchup-level prediction.

Transforms individual features (blanc/noir, dom/ext) into relative features
as recommended by multisport ML literature (PMC11265715, Hubacek 2019).

Stateless, vectorized, usable in batch (FE pipeline) and online (inference).
Training-serving skew prevention: same function called in both contexts.

Document ID: ALICE-DIFFERENTIALS
Version: 1.0.0
ISO: 5055 (SRP, <300 lines), 5259 (no leakage), 42001 (traceable)

References
----------
- PMC11265715: NBA XGBoost, "features subtracted from each other"
- Hubacek 2019: Soccer Prediction Challenge winner
- Hopsworks FTI: training-serving skew prevention
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_diff(df: pd.DataFrame, col_a: str, col_b: str, out_name: str) -> pd.DataFrame:
    """Compute col_a - col_b if both exist, skip otherwise."""
    if col_a in df.columns and col_b in df.columns:
        df[out_name] = df[col_a] - df[col_b]
    return df


def _player_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """8 player matchup differentials (blanc - noir)."""
    pairs = [
        ("expected_score_recent_blanc", "expected_score_recent_noir", "diff_form"),
        ("win_rate_recent_blanc", "win_rate_recent_noir", "diff_win_rate_recent"),
        ("draw_rate_blanc", "draw_rate_noir", "diff_draw_rate"),
        ("draw_rate_recent_blanc", "draw_rate_recent_noir", "diff_draw_rate_recent"),
        ("win_rate_normal_blanc", "win_rate_normal_noir", "diff_win_rate_normal"),
        ("clutch_win_blanc", "clutch_win_noir", "diff_clutch"),
        ("momentum_blanc", "momentum_noir", "diff_momentum"),
        ("derniere_presence_blanc", "derniere_presence_noir", "diff_derniere_presence"),
    ]
    for col_a, col_b, out in pairs:
        _safe_diff(df, col_a, col_b, out)
    return df


def _team_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """6 team matchup differentials (dom - ext)."""
    pairs = [
        ("position_dom", "position_ext", "diff_position"),
        ("points_cumules_dom", "points_cumules_ext", "diff_points_cumules"),
        ("profondeur_effectif_dom", "profondeur_effectif_ext", "diff_profondeur"),
        ("noyau_stable_dom", "noyau_stable_ext", "diff_stabilite"),
        ("win_rate_home_dom", "win_rate_home_ext", "diff_win_rate_home"),
        ("draw_rate_home_dom", "draw_rate_home_ext", "diff_draw_rate_home"),
    ]
    for col_a, col_b, out in pairs:
        _safe_diff(df, col_a, col_b, out)
    return df
