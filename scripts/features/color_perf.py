"""Performance par couleur (blanc/noir) - ISO 5055/5259.

Ce module calcule les scores par couleur pour chaque joueur.

Conformité:
- ISO 5055: Module <300 lignes, responsabilité unique
- ISO 5259: Données réelles, pas d'estimation
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_color_performance(
    df: pd.DataFrame,
    min_games: int = 10,
    min_per_color: int = 5,
) -> pd.DataFrame:
    """Calcule la performance par couleur (blanc/noir) pour chaque joueur.

    CORRIGÉ (ISO 5259): Plus de fillna(0.5) - données manquantes = NaN.

    Convention échecs interclubs:
    - Échiquiers impairs (1, 3, 5, 7) = Blancs pour équipe domicile
    - Échiquiers pairs (2, 4, 6, 8) = Noirs pour équipe domicile

    Args:
    ----
        df: DataFrame échiquiers (parties jouées uniquement)
        min_games: Minimum de parties total pour inclure (défaut: 10)
        min_per_color: Minimum de parties PAR couleur pour calculer préférence

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom: nom complet
        - score_blancs: score moyen avec blancs [0, 1] ou NaN si insuffisant
        - score_noirs: score moyen avec noirs [0, 1] ou NaN si insuffisant
        - nb_blancs: nombre de parties avec blancs
        - nb_noirs: nombre de parties avec noirs
        - avantage_blancs: score_blancs - score_noirs ou NaN
        - couleur_preferee: 'blanc', 'noir', 'neutre', ou 'donnees_insuffisantes'
        - data_quality: 'complet', 'partiel_blancs', 'partiel_noirs', 'insuffisant'

    ISO 5259: AUCUNE estimation - données manquantes marquées explicitement.
    """
    logger.info("Calcul performance par couleur (blanc/noir)...")

    if df.empty:
        return pd.DataFrame()

    # Filtrer parties jouées
    parties_jouees = df[
        ~df["type_resultat"].isin(["non_joue", "forfait_blanc", "forfait_noir", "double_forfait"])
    ].copy()

    # Stats par joueur jouant avec blancs
    blancs_stats = _get_blancs_stats(parties_jouees)

    # Stats par joueur jouant avec noirs
    noirs_stats = _get_noirs_stats(parties_jouees)

    if blancs_stats.empty and noirs_stats.empty:
        return pd.DataFrame()

    # Fusionner - PAS de fillna(0.5) !
    result = _merge_color_stats(blancs_stats, noirs_stats)

    # Filtrer joueurs avec assez de parties TOTAL
    result["nb_total"] = result["nb_blancs"] + result["nb_noirs"]
    result = result[result["nb_total"] >= min_games].copy()

    # Calculer qualité données
    result["data_quality"] = result.apply(lambda row: _get_data_quality(row, min_per_color), axis=1)

    # Calculer avantage couleur SEULEMENT si données complètes
    result["avantage_blancs"] = result.apply(_calc_avantage, axis=1)

    # Catégoriser préférence (seuil 5% = significatif)
    result["couleur_preferee"] = result.apply(_categorize_preference, axis=1)

    # Nettoyer colonnes
    result = result.drop(columns=["nb_total"])

    # Stats logging
    _log_color_stats(result)

    return result


def _get_blancs_stats(parties_jouees: pd.DataFrame) -> pd.DataFrame:
    """Calcule stats joueurs avec blancs."""
    if "blanc_nom" not in parties_jouees.columns or "resultat_blanc" not in parties_jouees.columns:
        return pd.DataFrame()

    return (
        parties_jouees.groupby("blanc_nom")
        .agg(
            score_blancs=("resultat_blanc", "mean"),
            nb_blancs=("resultat_blanc", "count"),
        )
        .reset_index()
        .rename(columns={"blanc_nom": "joueur_nom"})
    )


def _get_noirs_stats(parties_jouees: pd.DataFrame) -> pd.DataFrame:
    """Calcule stats joueurs avec noirs."""
    if "noir_nom" not in parties_jouees.columns or "resultat_noir" not in parties_jouees.columns:
        return pd.DataFrame()

    return (
        parties_jouees.groupby("noir_nom")
        .agg(
            score_noirs=("resultat_noir", "mean"),
            nb_noirs=("resultat_noir", "count"),
        )
        .reset_index()
        .rename(columns={"noir_nom": "joueur_nom"})
    )


def _merge_color_stats(
    blancs_stats: pd.DataFrame,
    noirs_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Fusionne stats blancs et noirs."""
    if not blancs_stats.empty and not noirs_stats.empty:
        result = blancs_stats.merge(noirs_stats, on="joueur_nom", how="outer")
    elif not blancs_stats.empty:
        result = blancs_stats.copy()
        result["score_noirs"] = np.nan
        result["nb_noirs"] = 0
    else:
        result = noirs_stats.copy()
        result["score_blancs"] = np.nan
        result["nb_blancs"] = 0

    # Remplir nb_* NaN par 0 (count, pas score)
    result["nb_blancs"] = result["nb_blancs"].fillna(0).astype(int)
    result["nb_noirs"] = result["nb_noirs"].fillna(0).astype(int)

    return result


def _get_data_quality(row: pd.Series, min_per_color: int) -> str:
    """Évalue la qualité des données pour ce joueur."""
    has_blancs = row["nb_blancs"] >= min_per_color
    has_noirs = row["nb_noirs"] >= min_per_color

    if has_blancs and has_noirs:
        return "complet"
    elif has_blancs:
        return "partiel_noirs"  # Manque données noirs
    elif has_noirs:
        return "partiel_blancs"  # Manque données blancs
    return "insuffisant"


def _calc_avantage(row: pd.Series) -> float:
    """Calcule avantage blancs seulement si données suffisantes."""
    if row["data_quality"] != "complet":
        return np.nan
    # Vérifier que les scores ne sont pas NaN
    if pd.isna(row["score_blancs"]) or pd.isna(row["score_noirs"]):
        return np.nan
    return row["score_blancs"] - row["score_noirs"]


def _categorize_preference(row: pd.Series) -> str:
    """Catégorise préférence couleur."""
    if row["data_quality"] != "complet":
        return "donnees_insuffisantes"

    avantage = row["avantage_blancs"]
    if pd.isna(avantage):
        return "donnees_insuffisantes"

    if avantage > 0.05:
        return "blanc"
    elif avantage < -0.05:
        return "noir"
    return "neutre"


def _log_color_stats(result: pd.DataFrame) -> None:
    """Log statistiques performance couleur."""
    if result.empty:
        logger.info("  Aucun joueur avec assez de parties")
        return

    complet = (result["data_quality"] == "complet").sum()
    partiel = result["data_quality"].astype(str).str.startswith("partiel").sum()
    insuffisant = (result["data_quality"] == "insuffisant").sum()

    logger.info(f"  {len(result)} joueurs avec stats couleur")
    logger.info(f"  Qualité: {complet} complet, {partiel} partiel, {insuffisant} insuffisant")

    if complet > 0:
        prefs = result[result["data_quality"] == "complet"]["couleur_preferee"]
        logger.info(
            f"  Préférences (données complètes): "
            f"{(prefs == 'blanc').sum()} blanc, "
            f"{(prefs == 'noir').sum()} noir, "
            f"{(prefs == 'neutre').sum()} neutre"
        )
