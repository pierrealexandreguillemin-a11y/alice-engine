"""Fatigue / jours de repos - ISO 5055/5259.

Ce module implémente le calcul des jours de repos depuis le dernier match.
Feature importante: la fatigue affecte la performance.
Moins de repos = performance potentiellement diminuée.

Sources:
- AI Sports Predictions 2025 (ainewshub.org)
- Sports Prediction PMC

Conformité:
- ISO 5055: Module <300 lignes, responsabilité unique
- ISO 5259: Features calculées depuis données réelles
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_fatigue_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les jours de repos depuis le dernier match par joueur.

    Feature importante: la fatigue affecte la performance.
    Moins de repos = performance potentiellement diminuée.

    Args:
    ----
        df: DataFrame échiquiers avec colonne 'date'

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom: nom joueur
        - date_match: date du match
        - jours_repos: jours depuis dernier match (None si premier)
        - fatigue_level: 'repose' (>7j), 'normal' (3-7j), 'fatigue' (<3j)

    ISO 5259: Fatigue calculée depuis calendrier réel.
    """
    logger.info("Calcul fatigue / jours de repos...")

    if df.empty or "date" not in df.columns:
        logger.warning("  Colonne date manquante")
        return pd.DataFrame()

    df_dated = df[df["date"].notna()].copy()
    df_dated["date"] = pd.to_datetime(df_dated["date"])

    fatigue_data = []

    for couleur in ["blanc", "noir"]:
        nom_col = f"{couleur}_nom"
        if nom_col not in df_dated.columns:
            continue

        for joueur, group in df_dated.groupby(nom_col):
            group_sorted = group.sort_values("date")
            dates = group_sorted["date"].tolist()

            for i, date in enumerate(dates):
                if i == 0:
                    jours_repos = None
                    fatigue = "inconnu"
                else:
                    delta = (date - dates[i - 1]).days
                    jours_repos = delta

                    if delta > 7:
                        fatigue = "repose"
                    elif delta >= 3:
                        fatigue = "normal"
                    else:
                        fatigue = "fatigue"

                fatigue_data.append(
                    {
                        "joueur_nom": joueur,
                        "date_match": date,
                        "jours_repos": jours_repos,
                        "fatigue_level": fatigue,
                    }
                )

    result = pd.DataFrame(fatigue_data)
    if not result.empty:
        # Stats
        known = result[result["fatigue_level"] != "inconnu"]
        if not known.empty:
            logger.info(f"  {len(result)} entrées fatigue")
            logger.info(
                f"  Distribution: "
                f"{(known['fatigue_level'] == 'repose').sum()} reposés, "
                f"{(known['fatigue_level'] == 'normal').sum()} normal, "
                f"{(known['fatigue_level'] == 'fatigue').sum()} fatigués"
            )

    return result
