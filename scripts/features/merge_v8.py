"""V8 merge helpers — draw rates, club level, player team context.

Document ID: ALICE-FEAT-MERGE-V8
Version: 1.0.0

Merge functions for V8 features that could not fit in merge_helpers.py
(ISO 5055 <300 lines constraint).

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality (SRP, <300 lines)
- ISO/IEC 5259:2024 - Data Quality for ML
"""

from __future__ import annotations

import pandas as pd

from scripts.features.merge_helpers import merge_player_features


def merge_draw_rate_player(
    result: pd.DataFrame,
    draw_rate_player: pd.DataFrame,
) -> pd.DataFrame:
    """Merge per-player draw rates for blanc and noir.

    Produces: draw_rate_blanc, draw_rate_noir.
    """
    return merge_player_features(result, draw_rate_player, ["draw_rate"])


def merge_draw_rate_equipe(
    result: pd.DataFrame,
    draw_rate_equipe: pd.DataFrame,
) -> pd.DataFrame:
    """Merge per-team draw rates for equipe_dom and equipe_ext.

    Produces: draw_rate_equipe_dom, draw_rate_equipe_ext.
    """
    if draw_rate_equipe.empty or "equipe" not in draw_rate_equipe.columns:
        return result

    for suffix, equipe_col in [("dom", "equipe_dom"), ("ext", "equipe_ext")]:
        if equipe_col not in result.columns:
            continue
        rename_map = {"draw_rate": f"draw_rate_equipe_{suffix}"}
        merge_df = draw_rate_equipe[["equipe", "draw_rate"]].rename(
            columns={"equipe": equipe_col} | rename_map
        )
        result = result.merge(
            merge_df[[equipe_col, f"draw_rate_equipe_{suffix}"]],
            on=equipe_col,
            how="left",
        )

    return result


def merge_club_level(
    result: pd.DataFrame,
    club_level: pd.DataFrame,
) -> pd.DataFrame:
    """Merge club hierarchy features for equipe_dom and equipe_ext.

    Produces per suffix (dom/ext):
    - team_rank_in_club_dom/ext
    - club_nb_teams_dom/ext
    - reinforcement_rate_dom/ext
    - stabilite_effectif_dom/ext
    - elo_moyen_evolution_dom/ext
    """
    if club_level.empty or "equipe" not in club_level.columns:
        return result

    feat_cols = [c for c in club_level.columns if c not in ("equipe", "saison")]

    for suffix, equipe_col in [("dom", "equipe_dom"), ("ext", "equipe_ext")]:
        if equipe_col not in result.columns:
            continue

        rename_map = {c: f"{c}_{suffix}" for c in feat_cols}
        merge_df = club_level.rename(columns={"equipe": equipe_col} | rename_map)
        merge_cols = [equipe_col, "saison"] + list(rename_map.values())
        merge_cols = [c for c in merge_cols if c in merge_df.columns]
        sub = merge_df[merge_cols].drop_duplicates(subset=[equipe_col, "saison"])
        result = result.merge(sub, on=[equipe_col, "saison"], how="left")

    return result


def merge_player_team_context(
    result: pd.DataFrame,
    player_team_ctx: pd.DataFrame,
) -> pd.DataFrame:
    """Merge player team context (promu, relegue, elo gap) per blanc/noir.

    Produces:
    - joueur_promu_blanc/noir
    - joueur_relegue_blanc/noir
    - player_team_elo_gap_blanc/noir
    """
    if player_team_ctx.empty:
        return result

    feat_cols = ["joueur_promu", "joueur_relegue", "player_team_elo_gap"]
    available = [c for c in feat_cols if c in player_team_ctx.columns]
    if not available:
        return result

    for color, equipe_col in [("blanc", "equipe_dom"), ("noir", "equipe_ext")]:
        nom_col = f"{color}_nom"
        if nom_col not in result.columns or equipe_col not in result.columns:
            continue

        rename_map = {c: f"{c}_{color}" for c in available}
        sub = player_team_ctx.rename(columns=rename_map)

        merge_keys_right = ["joueur_nom", "equipe", "saison", "ronde"]
        merge_keys_left = [nom_col, equipe_col, "saison", "ronde"]

        # Only use keys that exist in both
        valid_pairs = [
            (lk, rk)
            for lk, rk in zip(merge_keys_left, merge_keys_right, strict=False)
            if lk in result.columns and rk in sub.columns
        ]
        if not valid_pairs:
            continue

        left_keys = [p[0] for p in valid_pairs]
        right_keys = [p[1] for p in valid_pairs]
        value_cols = list(rename_map.values())
        sub_cols = right_keys + [c for c in value_cols if c in sub.columns]
        sub_dedup = sub[sub_cols].drop_duplicates(subset=right_keys, keep="first")

        result = result.merge(
            sub_dedup,
            left_on=left_keys,
            right_on=right_keys,
            how="left",
        )
        # Drop merge key artifacts
        for rk in right_keys:
            if rk in result.columns and rk not in left_keys:
                result = result.drop(columns=[rk], errors="ignore")
        # Clean up joueur_nom and equipe if they snuck in
        if "joueur_nom" in result.columns:
            result = result.drop(columns=["joueur_nom"], errors="ignore")
        if "equipe" in result.columns and "equipe" not in left_keys:
            result = result.drop(columns=["equipe"], errors="ignore")

    return result
