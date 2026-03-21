"""Features de performance joueur - ISO 5055/5259.

Ce module extrait les features de performance: forme recente,
position echiquier, et performance par couleur.

Conformite:
- ISO 5055: Module <300 lignes, responsabilite unique
- ISO 5259: Donnees reelles, pas d'estimation

Note: calculate_recent_form est re-exporte depuis recent_form.py (SRP).
"""

from __future__ import annotations

import logging

import pandas as pd

# Re-exports SRP
from scripts.features.color_perf import calculate_color_performance
from scripts.features.recent_form import calculate_recent_form

logger = logging.getLogger(__name__)

# Export public
__all__ = [
    "calculate_recent_form",
    "calculate_board_position",
    "calculate_color_performance",
]


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
