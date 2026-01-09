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
        if col not in result.columns or "saison" not in result.columns:
            continue
        merge_cols = ["zone_enjeu", "niveau_hierarchique", "position"]
        cols_exist = [c for c in merge_cols if c in team_enjeu.columns]
        if not cols_exist:
            continue
        merge_df = team_enjeu.rename(columns={c: f"{c}_{suffix}" for c in cols_exist})
        merge_keys = ["equipe", "saison"]
        if "ronde" in result.columns and "ronde" in merge_df.columns:
            merge_keys.append("ronde")
        result = result.merge(
            merge_df[merge_keys + [f"{c}_{suffix}" for c in cols_exist]].drop_duplicates(),
            left_on=[col, "saison"] + (["ronde"] if "ronde" in result.columns else []),
            right_on=merge_keys,
            how="left",
        )
        if "equipe" in result.columns:
            result = result.drop(columns=["equipe"])

    return result


def merge_fatigue_features(
    result: pd.DataFrame,
    fatigue: pd.DataFrame,
) -> pd.DataFrame:
    """Merge features de fatigue.

    Args:
    ----
        result: DataFrame cible
        fatigue: DataFrame fatigue joueurs

    Returns:
    -------
        DataFrame avec features mergées
    """
    if fatigue.empty or "date" not in result.columns:
        return result

    result["_date_parsed"] = pd.to_datetime(result["date"], errors="coerce")
    fatigue["_date_parsed"] = pd.to_datetime(fatigue["date_match"], errors="coerce")

    for color in ["blanc", "noir"]:
        nom_col = f"{color}_nom"
        if nom_col not in result.columns:
            continue
        fatigue_cols = ["jours_repos", "fatigue_level"]
        merge_df = fatigue.rename(columns={c: f"{c}_{color}" for c in fatigue_cols})
        result = result.merge(
            merge_df[["joueur_nom", "_date_parsed"] + [f"{c}_{color}" for c in fatigue_cols]],
            left_on=[nom_col, "_date_parsed"],
            right_on=["joueur_nom", "_date_parsed"],
            how="left",
        )
        if "joueur_nom" in result.columns:
            result = result.drop(columns=["joueur_nom"])

    result = result.drop(columns=["_date_parsed"], errors="ignore")
    return result


def merge_h2h_features(
    result: pd.DataFrame,
    h2h: pd.DataFrame,
) -> pd.DataFrame:
    """Merge features head-to-head.

    Args:
    ----
        result: DataFrame cible
        h2h: DataFrame confrontations directes

    Returns:
    -------
        DataFrame avec features mergées
    """
    if h2h.empty:
        return result

    if "blanc_nom" not in result.columns or "noir_nom" not in result.columns:
        return result

    def get_h2h_key(row: pd.Series) -> tuple[str, str]:
        b, n = str(row["blanc_nom"]), str(row["noir_nom"])
        return (b, n) if b < n else (n, b)

    result["_h2h_key"] = result.apply(get_h2h_key, axis=1)
    h2h["_h2h_key"] = list(zip(h2h["joueur_a"], h2h["joueur_b"], strict=False))

    h2h_merge = h2h[["_h2h_key", "nb_confrontations", "avantage_a"]].copy()
    result = result.merge(h2h_merge, on="_h2h_key", how="left")

    def adjust_h2h(row: pd.Series) -> float:
        if pd.isna(row.get("avantage_a")):
            return float("nan")
        b, n = str(row["blanc_nom"]), str(row["noir_nom"])
        return row["avantage_a"] if b < n else -row["avantage_a"]

    result["h2h_avantage_blanc"] = result.apply(adjust_h2h, axis=1)
    result["h2h_nb_confrontations"] = result["nb_confrontations"]
    result = result.drop(columns=["_h2h_key", "nb_confrontations", "avantage_a"], errors="ignore")

    return result
