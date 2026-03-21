"""Feature forme recente joueur - ISO 5055/5259.

Ce module extrait la forme recente des joueurs:
- calculate_recent_form: Score sur N derniers matchs

Conformite:
- ISO 5055: Module <300 lignes, responsabilite unique
- ISO 5259: Donnees reelles, pas d'estimation
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_recent_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Calcule la forme recente de chaque joueur (score sur N derniers matchs).

    Args:
    ----
        df: DataFrame echiquiers filtre (parties jouees uniquement)
        window: nombre de matchs pour calculer la forme

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom: nom complet
        - forme_recente: score moyen sur les N derniers matchs [0, 1]
        - nb_matchs_forme: nombre de matchs utilises
        - forme_tendance: 'hausse', 'baisse', 'stable' (momentum)

    ISO 5259: Forme calculee depuis resultats reels uniquement.
    """
    logger.info(f"Calcul forme recente (window={window})...")

    if df.empty:
        return pd.DataFrame()

    # Filtrer parties jouees
    parties_jouees = df[
        ~df["type_resultat"].isin(["non_joue", "forfait_blanc", "forfait_noir", "double_forfait"])
    ].copy()

    if "date" in parties_jouees.columns:
        parties_jouees = parties_jouees.sort_values("date")

    forme_data = []

    for couleur in ["blanc", "noir"]:
        nom_col = f"{couleur}_nom"
        resultat_col = f"resultat_{couleur}"

        if nom_col not in parties_jouees.columns or resultat_col not in parties_jouees.columns:
            continue

        for joueur, group in parties_jouees.groupby(nom_col):
            if len(group) >= window:
                last_n = group.tail(window)
                forme = last_n[resultat_col].mean()

                # Calcul tendance (premiere moitie vs seconde moitie)
                tendance = _calculate_tendance(last_n, resultat_col, window)

                forme_data.append(
                    {
                        "joueur_nom": joueur,
                        "forme_recente": forme,
                        "nb_matchs_forme": len(last_n),
                        "forme_tendance": tendance,
                    }
                )

    result = pd.DataFrame(forme_data)
    if len(result) > 0:
        # Agreger si joueur joue blanc ET noir
        result = (
            result.groupby("joueur_nom")
            .agg(
                forme_recente=("forme_recente", "mean"),
                nb_matchs_forme=("nb_matchs_forme", "sum"),
                forme_tendance=("forme_tendance", "first"),  # Prendre la premiere
            )
            .reset_index()
        )

    logger.info(f"  {len(result)} joueurs avec forme recente")
    return result


def _calculate_tendance(
    last_n: pd.DataFrame,
    resultat_col: str,
    window: int,
) -> str:
    """Calcule la tendance (hausse/baisse/stable).

    Args:
    ----
        last_n: DataFrame des N derniers matchs
        resultat_col: Nom colonne resultat
        window: Taille fenetre

    Returns:
    -------
        'hausse', 'baisse', ou 'stable'
    """
    mid = window // 2
    first_half = last_n.head(mid)[resultat_col].mean()
    second_half = last_n.tail(mid)[resultat_col].mean()

    if second_half > first_half + 0.1:
        return "hausse"
    elif second_half < first_half - 0.1:
        return "baisse"
    return "stable"
