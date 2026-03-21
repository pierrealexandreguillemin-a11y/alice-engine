"""Performance par couleur (blanc/noir) — W/D/L decomposition, rolling 3 saisons.

Ce module calcule win_rate/draw_rate par couleur pour chaque joueur,
sur une fenêtre glissante des 3 dernières saisons (ISO 5259).

Document ID: ALICE-FEAT-COLOR
Version: 2.0.0

Conformité:
- ISO 5055: Module <300 lignes, responsabilité unique
- ISO 5259: Données réelles, fenêtre temporelle, pas d'estimation
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scripts.features.helpers import filter_played_games

logger = logging.getLogger(__name__)

ROLLING_SEASONS = 3
ADVANTAGE_THRESHOLD = 0.05


def calculate_color_performance(
    df: pd.DataFrame,
    min_games: int = 10,
    min_per_color: int = 5,
) -> pd.DataFrame:
    """Calcule win_rate/draw_rate par couleur (blanc/noir) pour chaque joueur.

    Utilise une fenêtre glissante des 3 dernières saisons si la colonne
    `saison` est présente. Sinon utilise toutes les données disponibles.

    Convention FFE interclubs (A02 Art. 3.6.a):
    - Échiquiers impairs (1, 3, 5, 7) = Blancs pour équipe domicile
    - Échiquiers pairs (2, 4, 6, 8) = Noirs pour équipe domicile

    Args:
    ----
        df: DataFrame échiquiers (toutes parties)
        min_games: Minimum de parties total pour inclure (défaut: 10)
        min_per_color: Minimum de parties PAR couleur pour calculer préférence

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom: nom complet
        - win_rate_white: taux de victoires avec blancs [0, 1] ou NaN
        - draw_rate_white: taux de nulles avec blancs [0, 1] ou NaN
        - win_rate_black: taux de victoires avec noirs [0, 1] ou NaN
        - draw_rate_black: taux de nulles avec noirs [0, 1] ou NaN
        - nb_blancs: nombre de parties avec blancs
        - nb_noirs: nombre de parties avec noirs
        - win_adv_white: win_rate_white - win_rate_black ou NaN
        - draw_adv_white: draw_rate_white - draw_rate_black ou NaN
        - couleur_preferee: 'blanc', 'noir', 'neutre', ou 'donnees_insuffisantes'
        - data_quality: 'complet', 'partiel_blancs', 'partiel_noirs', 'insuffisant'

    ISO 5259: AUCUNE estimation — données manquantes marquées explicitement.
    """
    logger.info("Calcul performance par couleur W/D/L (rolling %d saisons)...", ROLLING_SEASONS)

    if df.empty:
        return pd.DataFrame()

    parties = filter_played_games(df)
    parties = _apply_rolling_window(parties)

    blancs = _get_blancs_wdl(parties)
    noirs = _get_noirs_wdl(parties)

    if blancs.empty and noirs.empty:
        return pd.DataFrame()

    result = _merge_color_stats(blancs, noirs)

    result["nb_total"] = result["nb_blancs"] + result["nb_noirs"]
    result = result[result["nb_total"] >= min_games].copy()
    result = result.drop(columns=["nb_total"])

    result["data_quality"] = result.apply(lambda row: _get_data_quality(row, min_per_color), axis=1)
    result["win_adv_white"] = result.apply(_calc_win_adv, axis=1)
    result["draw_adv_white"] = result.apply(_calc_draw_adv, axis=1)
    result["couleur_preferee"] = result.apply(_categorize_preference, axis=1)

    _log_color_stats(result)
    return result


def _apply_rolling_window(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre aux 3 dernières saisons si colonne saison présente."""
    if "saison" not in df.columns or df.empty:
        return df
    max_saison = df["saison"].max()
    cutoff = max_saison - (ROLLING_SEASONS - 1)
    return df[df["saison"] >= cutoff].copy()


def _get_blancs_wdl(parties: pd.DataFrame) -> pd.DataFrame:
    """Calcule win_rate/draw_rate des joueurs avec blancs."""
    if "blanc_nom" not in parties.columns or "resultat_blanc" not in parties.columns:
        return pd.DataFrame()

    rows = []
    for joueur, grp in parties.groupby("blanc_nom"):
        n = len(grp)
        wins = (grp["resultat_blanc"] == 1.0).sum()
        draws = (grp["resultat_blanc"] == 0.5).sum()
        rows.append(
            {
                "joueur_nom": joueur,
                "win_rate_white": float(wins / n),
                "draw_rate_white": float(draws / n),
                "nb_blancs": n,
            }
        )
    return pd.DataFrame(rows)


def _get_noirs_wdl(parties: pd.DataFrame) -> pd.DataFrame:
    """Calcule win_rate/draw_rate des joueurs avec noirs."""
    if "noir_nom" not in parties.columns or "resultat_noir" not in parties.columns:
        return pd.DataFrame()

    rows = []
    for joueur, grp in parties.groupby("noir_nom"):
        n = len(grp)
        wins = (grp["resultat_noir"] == 1.0).sum()
        draws = (grp["resultat_noir"] == 0.5).sum()
        rows.append(
            {
                "joueur_nom": joueur,
                "win_rate_black": float(wins / n),
                "draw_rate_black": float(draws / n),
                "nb_noirs": n,
            }
        )
    return pd.DataFrame(rows)


def _merge_color_stats(
    blancs: pd.DataFrame,
    noirs: pd.DataFrame,
) -> pd.DataFrame:
    """Fusionne stats blancs et noirs avec outer join."""
    if not blancs.empty and not noirs.empty:
        result = blancs.merge(noirs, on="joueur_nom", how="outer")
    elif not blancs.empty:
        result = blancs.copy()
        result["win_rate_black"] = np.nan
        result["draw_rate_black"] = np.nan
        result["nb_noirs"] = 0
    else:
        result = noirs.copy()
        result["win_rate_white"] = np.nan
        result["draw_rate_white"] = np.nan
        result["nb_blancs"] = 0

    result["nb_blancs"] = result["nb_blancs"].fillna(0).astype(int)
    result["nb_noirs"] = result["nb_noirs"].fillna(0).astype(int)
    return result


def _get_data_quality(row: pd.Series, min_per_color: int) -> str:
    """Évalue la qualité des données pour un joueur."""
    has_blancs = row["nb_blancs"] >= min_per_color
    has_noirs = row["nb_noirs"] >= min_per_color

    if has_blancs and has_noirs:
        return "complet"
    elif has_blancs:
        return "partiel_noirs"
    elif has_noirs:
        return "partiel_blancs"
    return "insuffisant"


def _calc_win_adv(row: pd.Series) -> float:
    """Calcule win_rate_white - win_rate_black (NaN si données incomplètes)."""
    if row["data_quality"] != "complet":
        return np.nan
    if pd.isna(row["win_rate_white"]) or pd.isna(row["win_rate_black"]):
        return np.nan
    return row["win_rate_white"] - row["win_rate_black"]


def _calc_draw_adv(row: pd.Series) -> float:
    """Calcule draw_rate_white - draw_rate_black (NaN si données incomplètes)."""
    if row["data_quality"] != "complet":
        return np.nan
    if pd.isna(row["draw_rate_white"]) or pd.isna(row["draw_rate_black"]):
        return np.nan
    return row["draw_rate_white"] - row["draw_rate_black"]


def _categorize_preference(row: pd.Series) -> str:
    """Catégorise la préférence couleur basée sur win_rate."""
    if row["data_quality"] != "complet":
        return "donnees_insuffisantes"
    win_adv = row["win_adv_white"]
    if pd.isna(win_adv):
        return "donnees_insuffisantes"
    if win_adv > ADVANTAGE_THRESHOLD:
        return "blanc"
    if win_adv < -ADVANTAGE_THRESHOLD:
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

    logger.info("  %d joueurs avec stats couleur", len(result))
    logger.info("  Qualité: %d complet, %d partiel, %d insuffisant", complet, partiel, insuffisant)

    if complet > 0:
        prefs = result[result["data_quality"] == "complet"]["couleur_preferee"]
        logger.info(
            "  Préférences: %d blanc, %d noir, %d neutre",
            (prefs == "blanc").sum(),
            (prefs == "noir").sum(),
            (prefs == "neutre").sum(),
        )
