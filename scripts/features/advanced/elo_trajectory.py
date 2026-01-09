"""Trajectoire Elo (progression/régression) - ISO 5055/5259.

Ce module implémente le calcul de la trajectoire Elo par joueur.
Feature importante: un joueur en progression peut surperformer son Elo.

Sources:
- EloMetrics IEEE 2025
- AI Sports Predictions 2025

Conformité:
- ISO 5055: Module <300 lignes, responsabilité unique
- ISO 5259: Features calculées depuis données réelles
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_elo_trajectory(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Calcule la trajectoire Elo (progression/régression) par joueur.

    Feature importante: un joueur en progression peut surperformer son Elo.

    Args:
    ----
        df: DataFrame échiquiers avec colonnes Elo et date
        window: Nombre de matchs pour calculer tendance

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom: nom joueur
        - elo_debut: Elo au début de la fenêtre
        - elo_fin: Elo en fin de fenêtre
        - elo_delta: variation absolue
        - elo_trajectory: 'progression' (>50), 'stable' (+-50), 'regression' (<-50)
        - momentum: score basé sur tendance récente

    ISO 5259: Trajectoire calculée depuis historique Elo réel.
    """
    logger.info(f"Calcul trajectoire Elo (window={window})...")

    if df.empty:
        return pd.DataFrame()

    # Besoin de dates pour ordonner
    if "date" not in df.columns:
        logger.warning("  Colonne date manquante pour trajectoire Elo")
        return pd.DataFrame()

    df_dated = df[df["date"].notna()].copy()
    df_dated["date"] = pd.to_datetime(df_dated["date"])

    trajectory_data = []

    for couleur in ["blanc", "noir"]:
        nom_col = f"{couleur}_nom"
        elo_col = f"{couleur}_elo"

        if nom_col not in df_dated.columns or elo_col not in df_dated.columns:
            continue

        for joueur, group in df_dated.groupby(nom_col):
            if len(group) < window:
                continue

            group_sorted = group.sort_values("date")
            elos = group_sorted[elo_col].dropna().tolist()

            if len(elos) < window:
                continue

            elo_debut = elos[0]
            elo_fin = elos[-1]
            delta = elo_fin - elo_debut

            if delta > 50:
                trajectory = "progression"
                momentum = min(1.0, delta / 200)  # Normalize
            elif delta < -50:
                trajectory = "regression"
                momentum = max(-1.0, delta / 200)
            else:
                trajectory = "stable"
                momentum = 0.0

            trajectory_data.append(
                {
                    "joueur_nom": joueur,
                    "elo_debut": elo_debut,
                    "elo_fin": elo_fin,
                    "elo_delta": delta,
                    "elo_trajectory": trajectory,
                    "momentum": momentum,
                    "nb_matchs": len(elos),
                }
            )

    # Dédupliquer (joueur peut jouer blanc ET noir)
    result = pd.DataFrame(trajectory_data)
    if not result.empty:
        result = (
            result.groupby("joueur_nom")
            .agg(
                elo_debut=("elo_debut", "first"),
                elo_fin=("elo_fin", "last"),
                elo_delta=("elo_delta", "mean"),
                elo_trajectory=("elo_trajectory", "first"),
                momentum=("momentum", "mean"),
                nb_matchs=("nb_matchs", "sum"),
            )
            .reset_index()
        )

    logger.info(f"  {len(result)} joueurs avec trajectoire Elo")

    return result
