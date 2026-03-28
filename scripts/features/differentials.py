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

import numpy as np
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


def _board_match_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """6 board x match interactions (player context in team context)."""
    # form_in_danger: diff_form * zone_danger
    if "diff_form" in df.columns and "zone_enjeu_dom" in df.columns:
        is_danger = (df["zone_enjeu_dom"] == "danger").astype(int)
        df["form_in_danger"] = df["diff_form"] * is_danger

    # color_match: player has preferred color on this board (FFE convention)
    if "echiquier" in df.columns and "est_domicile_blanc" in df.columns:
        is_odd = df["echiquier"] % 2 == 1
        est_dom = df["est_domicile_blanc"] == 1
        blanc_plays_white = est_dom == is_odd
        pref = df.get("couleur_preferee_blanc", pd.Series("neutre", index=df.index))
        df["color_match"] = (
            ((pref == "blanc") & blanc_plays_white) | ((pref == "noir") & ~blanc_plays_white)
        ).astype(int)

    # decalage_important: strategic placement in key match
    if "decalage_position_blanc" in df.columns and "match_important" in df.columns:
        df["decalage_important"] = df["decalage_position_blanc"] * df["match_important"]

    # marge100_decale: deliberate captain strategy
    if "club_utilise_marge_100_dom" in df.columns and "decalage_position_blanc" in df.columns:
        df["marge100_decale"] = (
            df["club_utilise_marge_100_dom"] * df["decalage_position_blanc"].abs()
        )

    # flex_decale: flexible player moved vs specialist moved
    if "flexibilite_echiquier_blanc" in df.columns and "decalage_position_blanc" in df.columns:
        df["flex_decale"] = df["flexibilite_echiquier_blanc"] * df["decalage_position_blanc"].abs()

    # promu_vs_strong: reinforcement facing strong opponent
    if "joueur_promu_blanc" in df.columns and "diff_elo" in df.columns:
        df["promu_vs_strong"] = df["joueur_promu_blanc"] * np.clip(-df["diff_elo"], 0, 800) / 400

    return df
