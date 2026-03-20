"""Noyau equipe - features reglementaires FFE (A02 Art. 3.7.f) - ISO 5055.

Features:
- est_dans_noyau: joueur a-t-il deja joue pour cette equipe cette saison
  avant cette ronde ?
- pct_noyau_match: % de la composition actuelle dans le noyau

Vectorized implementation — no Python loops over matches.

Document ID: ALICE-FEA-NOYAU-001
Version: 2.0.0
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def extract_noyau_features(df_history_played: pd.DataFrame) -> pd.DataFrame:
    """Calcule les features noyau par partie (vectorise).

    Returns
    -------
        DataFrame avec colonnes: joueur_nom, equipe, saison, ronde,
        est_dans_noyau, pct_noyau_match.
    """
    if df_history_played.empty:
        return pd.DataFrame()

    parts = []
    for color, equipe_col in [("blanc", "equipe_dom"), ("noir", "equipe_ext")]:
        part = _compute_noyau_vectorized(df_history_played, color, equipe_col)
        if not part.empty:
            parts.append(part)

    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    logger.info("  Noyau: %d entrees joueur/match calculees", len(result))
    return result


def _compute_noyau_vectorized(df: pd.DataFrame, color: str, equipe_col: str) -> pd.DataFrame:
    """Vectorized noyau computation for one color."""
    nom_col = f"{color}_nom"
    if nom_col not in df.columns:
        return pd.DataFrame()

    # Build (joueur, equipe, saison, ronde) tuples
    sub = df[[nom_col, equipe_col, "saison", "ronde"]].copy()
    sub.columns = ["joueur_nom", "equipe", "saison", "ronde"]
    sub = sub.dropna(subset=["joueur_nom"])
    sub = sub[sub["joueur_nom"].str.strip() != ""]

    # For each (joueur, equipe, saison): first ronde they appeared
    first_appearance = (
        sub.groupby(["joueur_nom", "equipe", "saison"])["ronde"]
        .min()
        .reset_index()
        .rename(columns={"ronde": "premiere_ronde"})
    )

    # Merge back: joueur is in noyau if current ronde > premiere_ronde
    merged = sub.merge(first_appearance, on=["joueur_nom", "equipe", "saison"], how="left")
    merged["est_dans_noyau"] = (merged["ronde"] > merged["premiere_ronde"]).astype(int)

    # pct_noyau per match excluding focal player: (group_sum - own) / (group_count - 1)
    grp = ["equipe", "saison", "ronde"]
    group_sum = merged.groupby(grp)["est_dans_noyau"].transform("sum")
    group_count = merged.groupby(grp)["est_dans_noyau"].transform("count")
    others_count = group_count - 1
    merged["pct_noyau_match"] = (
        (group_sum - merged["est_dans_noyau"]) / others_count.replace(0, 1)
    ).round(3)

    return merged[["joueur_nom", "equipe", "saison", "ronde", "est_dans_noyau", "pct_noyau_match"]]
