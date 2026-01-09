"""Performance domicile vs extérieur - ISO 5055/5259.

Ce module implémente le calcul de la performance domicile vs extérieur.
Feature: certains joueurs performent différemment selon le lieu.

Sources:
- Home Advantage Research (Taylor & Francis)
- AI Sports Predictions 2025

Conformité:
- ISO 5055: Module <300 lignes, responsabilité unique
- ISO 5259: Features calculées depuis données réelles
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_home_away_performance(df: pd.DataFrame, min_games: int = 5) -> pd.DataFrame:
    """Calcule la performance domicile vs extérieur par joueur.

    Feature: certains joueurs performent différemment selon le lieu.

    Args:
    ----
        df: DataFrame échiquiers
        min_games: Minimum de matchs par catégorie

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom: nom joueur
        - score_domicile: score moyen à domicile [0, 1]
        - score_exterieur: score moyen à l'extérieur [0, 1]
        - nb_domicile: matchs à domicile
        - nb_exterieur: matchs à l'extérieur
        - avantage_domicile: score_dom - score_ext
        - home_away_pref: 'domicile', 'exterieur', 'neutre'

    ISO 5259: Performance calculée depuis lieux réels.
    """
    logger.info("Calcul performance domicile vs extérieur...")

    if df.empty:
        return pd.DataFrame()

    parties = df[
        ~df["type_resultat"].isin(["non_joue", "forfait_blanc", "forfait_noir", "double_forfait"])
    ].copy()

    home_away: dict[str, dict[str, list[float]]] = {}

    # Blancs à domicile (échiquiers impairs pour équipe_dom)
    if "blanc_nom" in parties.columns and "resultat_blanc" in parties.columns:
        for _, row in parties.iterrows():
            joueur = str(row["blanc_nom"])
            resultat = row["resultat_blanc"]
            # Convention: blancs = domicile sur échiquiers impairs
            echiquier = row.get("echiquier", 1)
            is_home = echiquier % 2 == 1

            if joueur not in home_away:
                home_away[joueur] = {"home": [], "away": []}

            if is_home:
                home_away[joueur]["home"].append(resultat)
            else:
                home_away[joueur]["away"].append(resultat)

    # Noirs (inverse)
    if "noir_nom" in parties.columns and "resultat_noir" in parties.columns:
        for _, row in parties.iterrows():
            joueur = str(row["noir_nom"])
            resultat = row["resultat_noir"]
            echiquier = row.get("echiquier", 1)
            is_home = echiquier % 2 == 0  # Noirs = domicile sur échiquiers pairs

            if joueur not in home_away:
                home_away[joueur] = {"home": [], "away": []}

            if is_home:
                home_away[joueur]["home"].append(resultat)
            else:
                home_away[joueur]["away"].append(resultat)

    # Construire résultat
    result_data = []
    for joueur, stats in home_away.items():
        nb_home = len(stats["home"])
        nb_away = len(stats["away"])

        if nb_home >= min_games and nb_away >= min_games:
            score_home = np.mean(stats["home"])
            score_away = np.mean(stats["away"])
            avantage = score_home - score_away

            if avantage > 0.05:
                pref = "domicile"
            elif avantage < -0.05:
                pref = "exterieur"
            else:
                pref = "neutre"

            result_data.append(
                {
                    "joueur_nom": joueur,
                    "score_domicile": score_home,
                    "score_exterieur": score_away,
                    "nb_domicile": nb_home,
                    "nb_exterieur": nb_away,
                    "avantage_domicile": avantage,
                    "home_away_pref": pref,
                }
            )

    result = pd.DataFrame(result_data)
    logger.info(f"  {len(result)} joueurs avec stats domicile/extérieur")

    return result
