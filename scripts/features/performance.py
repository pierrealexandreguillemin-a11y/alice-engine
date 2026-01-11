"""Features de performance joueur - ISO 5055/5259.

Ce module extrait les features de performance: forme récente,
position échiquier, et performance par couleur.

Conformité:
- ISO 5055: Module <300 lignes, responsabilité unique
- ISO 5259: Données réelles, pas d'estimation (CORRIGÉ fillna bug)
"""

from __future__ import annotations

import logging

import pandas as pd

# Ré-export pour compatibilité
from scripts.features.color_perf import calculate_color_performance

logger = logging.getLogger(__name__)

# Export public
__all__ = [
    "calculate_recent_form",
    "calculate_board_position",
    "calculate_color_performance",
]


def calculate_recent_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Calcule la forme récente de chaque joueur (score sur N derniers matchs).

    Args:
    ----
        df: DataFrame échiquiers filtré (parties jouées uniquement)
        window: nombre de matchs pour calculer la forme

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom: nom complet
        - forme_recente: score moyen sur les N derniers matchs [0, 1]
        - nb_matchs_forme: nombre de matchs utilisés
        - forme_tendance: 'hausse', 'baisse', 'stable' (momentum)

    ISO 5259: Forme calculée depuis résultats réels uniquement.
    """
    logger.info(f"Calcul forme récente (window={window})...")

    if df.empty:
        return pd.DataFrame()

    parties_jouees = _filter_played_games(df)
    forme_data = _collect_form_data(parties_jouees, window)
    result = _aggregate_form_data(forme_data)

    logger.info(f"  {len(result)} joueurs avec forme récente")
    return result


def _filter_played_games(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre les parties effectivement jouees."""
    parties = df[
        ~df["type_resultat"].isin(["non_joue", "forfait_blanc", "forfait_noir", "double_forfait"])
    ].copy()
    if "date" in parties.columns:
        parties = parties.sort_values("date")
    return parties


def _collect_form_data(parties: pd.DataFrame, window: int) -> list[dict]:
    """Collecte les donnees de forme par joueur."""
    forme_data = []

    for couleur in ["blanc", "noir"]:
        nom_col = f"{couleur}_nom"
        resultat_col = f"resultat_{couleur}"

        if nom_col not in parties.columns or resultat_col not in parties.columns:
            continue

        for joueur, group in parties.groupby(nom_col):
            if len(group) >= window:
                entry = _compute_player_form(joueur, group, window, resultat_col)
                forme_data.append(entry)

    return forme_data


def _compute_player_form(joueur: str, group: pd.DataFrame, window: int, res_col: str) -> dict:
    """Calcule la forme d'un joueur."""
    last_n = group.tail(window)
    forme = last_n[res_col].mean()
    tendance = _compute_tendance(last_n, window, res_col)

    return {
        "joueur_nom": joueur,
        "forme_recente": forme,
        "nb_matchs_forme": len(last_n),
        "forme_tendance": tendance,
    }


def _compute_tendance(last_n: pd.DataFrame, window: int, res_col: str) -> str:
    """Calcule la tendance (hausse/baisse/stable)."""
    mid = window // 2
    first_half = last_n.head(mid)[res_col].mean()
    second_half = last_n.tail(mid)[res_col].mean()

    if second_half > first_half + 0.1:
        return "hausse"
    if second_half < first_half - 0.1:
        return "baisse"
    return "stable"


def _aggregate_form_data(forme_data: list[dict]) -> pd.DataFrame:
    """Agrege les donnees de forme."""
    result = pd.DataFrame(forme_data)
    if len(result) > 0:
        result = (
            result.groupby("joueur_nom")
            .agg(
                forme_recente=("forme_recente", "mean"),
                nb_matchs_forme=("nb_matchs_forme", "sum"),
                forme_tendance=("forme_tendance", "first"),
            )
            .reset_index()
        )
    return result


def calculate_board_position(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule la position moyenne sur l'échiquier pour chaque joueur.

    Un joueur habitué à jouer sur échiquier 1 vs échiquier 8
    n'a pas le même niveau.

    Args:
    ----
        df: DataFrame échiquiers

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom: nom complet
        - echiquier_moyen: position moyenne
        - echiquier_std: écart-type (variabilité)
        - echiquier_min: échiquier le plus fort joué
        - echiquier_max: échiquier le plus faible joué

    ISO 5259: Position calculée depuis historique réel.
    """
    logger.info("Calcul position échiquier moyenne...")

    if df.empty or "echiquier" not in df.columns:
        return pd.DataFrame()

    board_data = []

    for couleur in ["blanc", "noir"]:
        nom_col = f"{couleur}_nom"
        if nom_col not in df.columns:
            continue

        for joueur, group in df.groupby(nom_col):
            board_data.append(
                {
                    "joueur_nom": joueur,
                    "echiquier_moyen": group["echiquier"].mean(),
                    "echiquier_std": group["echiquier"].std(),
                    "echiquier_min": group["echiquier"].min(),
                    "echiquier_max": group["echiquier"].max(),
                }
            )

    result = pd.DataFrame(board_data)
    if len(result) > 0:
        result = (
            result.groupby("joueur_nom")
            .agg(
                echiquier_moyen=("echiquier_moyen", "mean"),
                echiquier_std=("echiquier_std", "mean"),
                echiquier_min=("echiquier_min", "min"),
                echiquier_max=("echiquier_max", "max"),
            )
            .reset_index()
        )

    logger.info(f"  {len(result)} joueurs avec stats échiquier")
    return result
