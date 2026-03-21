"""Head-to-head (H2H) historique - ISO 5055/5259.

Ce module implémente le calcul de l'historique H2H entre paires de joueurs.
Feature importante pour prédiction: certains joueurs ont des résultats
atypiques contre des adversaires spécifiques.

Sources:
- AI Sports Predictions 2025 (ainewshub.org)
- EloMetrics IEEE 2025

Conformité:
- ISO 5055: Module <300 lignes, responsabilité unique
- ISO 5259: Features calculées depuis données réelles (forfaits exclus)
"""

from __future__ import annotations

import logging

import pandas as pd

from scripts.features.helpers import filter_played_games

logger = logging.getLogger(__name__)


def calculate_head_to_head(df: pd.DataFrame, min_games: int = 3) -> pd.DataFrame:
    """Calcule l'historique H2H entre paires de joueurs.

    Feature importante pour prédiction: certains joueurs ont
    des résultats atypiques contre des adversaires spécifiques.
    Forfaits exclus via filter_played_games() (ISO 5259).

    Args:
    ----
        df: DataFrame échiquiers avec parties jouées
        min_games: Minimum de confrontations pour inclure

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_a: premier joueur (ordre alphabétique)
        - joueur_b: second joueur
        - nb_confrontations: nombre total de parties
        - h2h_win_rate: taux de victoires joueur_a [0, 1]
        - h2h_draw_rate: taux de nulles [0, 1] (symétrique)
        - h2h_exists: True (H2H data disponible)

    ISO 5259: H2H calculé depuis confrontations réelles uniquement.
    """
    logger.info("Calcul historique H2H entre joueurs...")

    if df.empty:
        return pd.DataFrame()

    # Exclude forfeits and non-played (ISO 5259)
    parties = filter_played_games(df)

    if "blanc_nom" not in parties.columns or "noir_nom" not in parties.columns:
        return pd.DataFrame()

    # Vectorized: normalize pair order (alphabetical) for unique keys
    b = parties["blanc_nom"].astype(str)
    n = parties["noir_nom"].astype(str)
    # Filter out self-confrontations (data quality guard)
    valid = b != n
    b, n = b[valid], n[valid]
    parties = parties[valid]
    mask = b < n

    joueur_a = b.where(mask, n)
    joueur_b = n.where(mask, b)
    # result from joueur_a perspective: win=1.0, draw=0.5, loss=0.0
    score_a = parties["resultat_blanc"].where(mask, parties["resultat_noir"])

    h2h_df = pd.DataFrame(
        {
            "joueur_a": joueur_a.values,
            "joueur_b": joueur_b.values,
            "score_a": score_a.values,
        }
    )

    result = (
        h2h_df.groupby(["joueur_a", "joueur_b"])
        .agg(
            nb_confrontations=("score_a", "count"),
            wins_a=("score_a", lambda x: (x == 1.0).sum()),
            draws=("score_a", lambda x: (x == 0.5).sum()),
        )
        .reset_index()
    )

    result = result[result["nb_confrontations"] >= min_games].copy()
    result["h2h_win_rate"] = (result["wins_a"] / result["nb_confrontations"]).round(4)
    result["h2h_draw_rate"] = (result["draws"] / result["nb_confrontations"]).round(4)
    result["h2h_exists"] = True
    result = result.drop(columns=["wins_a", "draws"])

    logger.info("  %d paires H2H avec >= %d confrontations", len(result), min_games)

    return result
