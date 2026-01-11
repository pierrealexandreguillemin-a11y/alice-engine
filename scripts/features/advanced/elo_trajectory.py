"""Trajectoire Elo (progression/régression) - ISO 5055/5259.

Ce module implémente le calcul de la trajectoire Elo par joueur.
Feature importante: un joueur en progression peut surperformer son Elo actuel.

Classification trajectoire (SEUILS DOCUMENTÉS ISO 5259):
- "progression": delta > +50 pts - Joueur en amélioration significative
- "regression": delta < -50 pts - Joueur en déclin significatif
- "stable": -50 <= delta <= +50 - Performance constante

Justification seuil ±50 pts:
- Correspond à environ 1 catégorie FFE (50-100 pts par catégorie)
- Variation significative mesurable sur 6 matchs (window par défaut)
- Référence FIDE Handbook: K-factor implique ~12-20 pts par partie
- Statistiquement: ~1 écart-type de variation Elo sur une saison

Momentum (SEUIL DOCUMENTÉ):
- Normalisé sur [-1, 1] avec diviseur 200 pts
- 200 pts = progression/régression extrême sur 6 matchs
- Permet comparaison entre joueurs

Sources:
- EloMetrics IEEE 2025 (https://ieeexplore.ieee.org/document/10879733/)
- AI Sports Predictions 2025 (ainewshub.org)
- FIDE Handbook - Elo Rating System (K-factor rules)

Conformité:
- ISO 5055: Module <300 lignes, responsabilité unique
- ISO 5259: Features calculées depuis données réelles, seuils documentés
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

    df_dated = _prepare_elo_df(df)
    if df_dated.empty:
        return pd.DataFrame()

    trajectory_data = _collect_trajectory_data(df_dated, window)
    result = _aggregate_trajectory_data(trajectory_data)

    logger.info(f"  {len(result)} joueurs avec trajectoire Elo")
    return result


def _prepare_elo_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare le DataFrame pour calcul trajectoire."""
    if df.empty:
        return pd.DataFrame()

    if "date" not in df.columns:
        logger.warning("  Colonne date manquante pour trajectoire Elo")
        return pd.DataFrame()

    df_dated = df[df["date"].notna()].copy()
    df_dated["date"] = pd.to_datetime(df_dated["date"])
    return df_dated


def _collect_trajectory_data(df_dated: pd.DataFrame, window: int) -> list[dict]:
    """Collecte les donnees de trajectoire par joueur."""
    trajectory_data = []

    for couleur in ["blanc", "noir"]:
        nom_col, elo_col = f"{couleur}_nom", f"{couleur}_elo"

        if nom_col not in df_dated.columns or elo_col not in df_dated.columns:
            continue

        for joueur, group in df_dated.groupby(nom_col):
            entry = _compute_player_trajectory(joueur, group, window, elo_col)
            if entry:
                trajectory_data.append(entry)

    return trajectory_data


def _compute_player_trajectory(
    joueur: str, group: pd.DataFrame, window: int, elo_col: str
) -> dict | None:
    """Calcule la trajectoire d'un joueur."""
    if len(group) < window:
        return None

    elos = group.sort_values("date")[elo_col].dropna().tolist()
    if len(elos) < window:
        return None

    elo_debut, elo_fin = elos[0], elos[-1]
    delta = elo_fin - elo_debut
    trajectory, momentum = _classify_trajectory(delta)

    return {
        "joueur_nom": joueur,
        "elo_debut": elo_debut,
        "elo_fin": elo_fin,
        "elo_delta": delta,
        "elo_trajectory": trajectory,
        "momentum": momentum,
        "nb_matchs": len(elos),
    }


def _classify_trajectory(delta: float) -> tuple[str, float]:
    """Classifie la trajectoire et calcule le momentum."""
    if delta > 50:
        return "progression", min(1.0, delta / 200)
    if delta < -50:
        return "regression", max(-1.0, delta / 200)
    return "stable", 0.0


def _aggregate_trajectory_data(trajectory_data: list[dict]) -> pd.DataFrame:
    """Agrege les donnees de trajectoire."""
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
    return result
