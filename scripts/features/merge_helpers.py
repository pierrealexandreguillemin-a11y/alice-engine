"""Helpers de merge pour feature engineering - ISO 5055.

Ce module contient les fonctions de merge extraites de compute_features_for_split
pour réduire la complexité cyclomatique.

Conformité ISO/IEC 5055 (Code Quality).
"""

from __future__ import annotations

import pandas as pd


def merge_player_features(
    result: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    prefix: str = "",
) -> pd.DataFrame:
    """Merge features joueur pour blanc et noir.

    Args:
    ----
        result: DataFrame cible
        feature_df: DataFrame source avec joueur_nom
        feature_cols: Colonnes à merger
        prefix: Préfixe optionnel pour les colonnes

    Returns:
    -------
        DataFrame avec features mergées
    """
    if feature_df.empty:
        return result

    for color in ["blanc", "noir"]:
        nom_col = f"{color}_nom"
        if nom_col not in result.columns:
            continue
        cols_exist = [c for c in feature_cols if c in feature_df.columns]
        if not cols_exist:
            continue
        col_mapping = {c: f"{prefix}{c}_{color}" for c in cols_exist}
        merge_df = feature_df.rename(columns=col_mapping)
        result = result.merge(
            merge_df[["joueur_nom"] + list(col_mapping.values())],
            left_on=nom_col,
            right_on="joueur_nom",
            how="left",
        )
        if "joueur_nom" in result.columns:
            result = result.drop(columns=["joueur_nom"])

    return result


def merge_club_reliability(
    result: pd.DataFrame,
    club_reliability: pd.DataFrame,
) -> pd.DataFrame:
    """Merge features de fiabilité club.

    Args:
    ----
        result: DataFrame cible
        club_reliability: DataFrame fiabilité clubs

    Returns:
    -------
        DataFrame avec features mergées
    """
    if club_reliability.empty:
        return result

    club_cols = ["taux_forfait", "taux_non_joue", "fiabilite_score"]

    for suffix, col in [("dom", "equipe_dom"), ("ext", "equipe_ext")]:
        if col not in result.columns:
            continue
        merge_df = club_reliability.rename(columns={c: f"{c}_{suffix}" for c in club_cols})
        result = result.merge(
            merge_df[["equipe"] + [f"{c}_{suffix}" for c in club_cols]],
            left_on=col,
            right_on="equipe",
            how="left",
        )
        if "equipe" in result.columns:
            result = result.drop(columns=["equipe"])

    return result


def merge_team_enjeu(
    result: pd.DataFrame,
    team_enjeu: pd.DataFrame,
) -> pd.DataFrame:
    """Merge features d'enjeu équipe.

    Args:
    ----
        result: DataFrame cible
        team_enjeu: DataFrame enjeu équipes

    Returns:
    -------
        DataFrame avec features mergées
    """
    if team_enjeu.empty or "ronde" not in team_enjeu.columns:
        return result

    for suffix, col in [("dom", "equipe_dom"), ("ext", "equipe_ext")]:
        result = _merge_single_team_enjeu(result, team_enjeu, suffix, col)

    return result


def _merge_single_team_enjeu(
    result: pd.DataFrame,
    team_enjeu: pd.DataFrame,
    suffix: str,
    col: str,
) -> pd.DataFrame:
    """Merge enjeu pour une equipe (dom ou ext)."""
    cols_exist = _get_enjeu_cols(result, team_enjeu, col)
    if not cols_exist:
        return result

    merge_df = team_enjeu.rename(columns={c: f"{c}_{suffix}" for c in cols_exist})
    return _execute_enjeu_merge(result, merge_df, suffix, col, cols_exist)


def _get_enjeu_cols(result: pd.DataFrame, team_enjeu: pd.DataFrame, col: str) -> list[str]:
    """Recupere les colonnes d'enjeu disponibles."""
    if col not in result.columns or "saison" not in result.columns:
        return []
    merge_cols = [
        "zone_enjeu",
        "niveau_hierarchique",
        "position",
        "ecart_premier",
        "ecart_dernier",
        "points_cumules",
        "nb_equipes",
    ]
    return [c for c in merge_cols if c in team_enjeu.columns]


def _execute_enjeu_merge(
    result: pd.DataFrame,
    merge_df: pd.DataFrame,
    suffix: str,
    col: str,
    cols_exist: list[str],
) -> pd.DataFrame:
    """Execute le merge d'enjeu avec contexte competition complet."""
    # Use full competition context to avoid cartesian product
    # (a club in 2 competitions has 2 legitimate enjeu rows)
    ctx_cols = ["competition", "division", "groupe"]
    merge_keys = ["equipe", "saison"]
    left_keys = [col, "saison"]
    for c in ["ronde"] + ctx_cols:
        if c in result.columns and c in merge_df.columns:
            merge_keys.append(c)
            left_keys.append(c)

    value_cols = [f"{c}_{suffix}" for c in cols_exist]
    sub = merge_df[merge_keys + value_cols].drop_duplicates(subset=merge_keys)

    result = result.merge(sub, left_on=left_keys, right_on=merge_keys, how="left")
    if "equipe" in result.columns:
        result = result.drop(columns=["equipe"])
    return result


def _get_merge_keys(result: pd.DataFrame, merge_df: pd.DataFrame) -> list[str]:
    """Determine les cles de merge."""
    merge_keys = ["equipe", "saison"]
    if "ronde" in result.columns and "ronde" in merge_df.columns:
        merge_keys.append("ronde")
    return merge_keys


def merge_noyau_features(
    result: pd.DataFrame,
    noyau_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge les features noyau par (joueur, equipe, saison, ronde).

    Produit les colonnes:
    - est_dans_noyau_blanc / est_dans_noyau_noir
    - pct_noyau_equipe_dom / pct_noyau_equipe_ext

    Args:
    ----
        result: DataFrame cible avec blanc_nom, noir_nom, equipe_dom,
                equipe_ext, saison, ronde
        noyau_df: DataFrame depuis extract_noyau_features()

    Returns:
    -------
        DataFrame avec features noyau mergees
    """
    if noyau_df.empty:
        return result

    required = {"joueur_nom", "equipe", "saison", "ronde", "est_dans_noyau", "pct_noyau_match"}
    if not required.issubset(noyau_df.columns):
        return result

    for color, equipe_col in [("blanc", "equipe_dom"), ("noir", "equipe_ext")]:
        nom_col = f"{color}_nom"
        if nom_col not in result.columns or equipe_col not in result.columns:
            continue

        rename_map = {
            "est_dans_noyau": f"est_dans_noyau_{color}",
            "pct_noyau_match": f"pct_noyau_equipe_{'dom' if color == 'blanc' else 'ext'}",
        }
        merge_keys = ["joueur_nom", "equipe", "saison", "ronde"]
        sub = noyau_df.rename(columns=rename_map)[
            merge_keys + list(rename_map.values())
        ].drop_duplicates(subset=merge_keys, keep="first")

        result = result.merge(
            sub,
            left_on=[nom_col, equipe_col, "saison", "ronde"],
            right_on=["joueur_nom", "equipe", "saison", "ronde"],
            how="left",
        )
        result = result.drop(columns=["joueur_nom", "equipe"], errors="ignore")

    return result


def merge_h2h_features(
    result: pd.DataFrame,
    h2h: pd.DataFrame,
) -> pd.DataFrame:
    """Merge features head-to-head.

    Produit les colonnes:
    - h2h_win_rate: taux victoire du joueur blanc (perspective blanc)
    - h2h_draw_rate: taux nulles (symétrique)
    - h2h_nb_confrontations: nombre de confrontations
    - h2h_exists: True si H2H existe, False sinon (jamais NaN)

    Args:
    ----
        result: DataFrame cible
        h2h: DataFrame depuis calculate_head_to_head()

    Returns:
    -------
        DataFrame avec features H2H mergées
    """
    if "blanc_nom" not in result.columns or "noir_nom" not in result.columns:
        return result

    if h2h.empty:
        result["h2h_win_rate"] = float("nan")
        result["h2h_draw_rate"] = float("nan")
        result["h2h_nb_confrontations"] = float("nan")
        result["h2h_exists"] = False
        return result

    # Build canonical key (alphabetical order) for both sides
    b = result["blanc_nom"].astype(str)
    n = result["noir_nom"].astype(str)
    mask_b_lt_n = b < n
    result = result.copy()
    result["_h2h_key"] = list(
        zip(
            b.where(mask_b_lt_n, n),
            n.where(mask_b_lt_n, b),
            strict=False,
        )
    )

    h2h_work = h2h.copy()
    h2h_work["_h2h_key"] = list(zip(h2h_work["joueur_a"], h2h_work["joueur_b"], strict=False))

    h2h_merge = h2h_work[["_h2h_key", "nb_confrontations", "h2h_win_rate", "h2h_draw_rate"]].copy()
    result = result.merge(h2h_merge, on="_h2h_key", how="left")

    # Flip win_rate when blanc > noir alphabetically (stored from joueur_a perspective)
    raw_win = result["h2h_win_rate"].copy()
    result["h2h_win_rate"] = raw_win.where(
        mask_b_lt_n | raw_win.isna(),
        1.0 - raw_win - result["h2h_draw_rate"],
    )
    # draw_rate is symmetric — no flip needed

    result["h2h_nb_confrontations"] = result["nb_confrontations"]
    result["h2h_exists"] = result["h2h_win_rate"].notna()
    result = result.drop(columns=["_h2h_key", "nb_confrontations"], errors="ignore")

    return result
