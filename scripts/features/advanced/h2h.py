"""Head-to-head (H2H) historique - ISO 5055/5259.

Ce module implémente le calcul de l'historique H2H entre paires de joueurs.
Feature importante pour prédiction: certains joueurs ont des résultats
atypiques contre des adversaires spécifiques.

Sources:
- AI Sports Predictions 2025 (ainewshub.org)
- EloMetrics IEEE 2025

Conformité:
- ISO 5055: Module <300 lignes, responsabilité unique
- ISO 5259: Features calculées depuis données réelles
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_head_to_head(df: pd.DataFrame, min_games: int = 3) -> pd.DataFrame:
    """Calcule l'historique H2H entre paires de joueurs.

    Feature importante pour prédiction: certains joueurs ont
    des résultats atypiques contre des adversaires spécifiques.

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
        - score_a: score moyen joueur A contre B [0, 1]
        - score_b: score moyen joueur B contre A [0, 1]
        - avantage_a: score_a - score_b

    ISO 5259: H2H calculé depuis confrontations réelles.
    """
    logger.info("Calcul historique H2H entre joueurs...")

    if df.empty:
        return pd.DataFrame()

    # Filtrer parties jouées
    parties = df[
        ~df["type_resultat"].isin(["non_joue", "forfait_blanc", "forfait_noir", "double_forfait"])
    ].copy()

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
    score_a = parties["resultat_blanc"].where(mask, parties["resultat_noir"])
    score_b = parties["resultat_noir"].where(mask, parties["resultat_blanc"])

    h2h_df = pd.DataFrame(
        {
            "joueur_a": joueur_a.values,
            "joueur_b": joueur_b.values,
            "score_a": score_a.values,
            "score_b": score_b.values,
        }
    )

    result = (
        h2h_df.groupby(["joueur_a", "joueur_b"])
        .agg(
            nb_confrontations=("score_a", "count"),
            score_a=("score_a", "mean"),
            score_b=("score_b", "mean"),
        )
        .reset_index()
    )

    result = result[result["nb_confrontations"] >= min_games].copy()
    result["avantage_a"] = result["score_a"] - result["score_b"]

    logger.info("  %d paires H2H avec >= %d confrontations", len(result), min_games)

    return result
