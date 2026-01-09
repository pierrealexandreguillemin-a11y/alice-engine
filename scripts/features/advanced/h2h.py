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

    # Collecter confrontations
    h2h_data: dict[tuple[str, str], dict[str, float]] = {}

    for _, row in parties.iterrows():
        blanc = str(row["blanc_nom"])
        noir = str(row["noir_nom"])
        res_blanc = row.get("resultat_blanc", 0.5)
        res_noir = row.get("resultat_noir", 0.5)

        # Ordonner alphabétiquement pour clé unique
        if blanc < noir:
            key = (blanc, noir)
            score_a, score_b = res_blanc, res_noir
        else:
            key = (noir, blanc)
            score_a, score_b = res_noir, res_blanc

        if key not in h2h_data:
            h2h_data[key] = {"games": 0, "score_a": 0.0, "score_b": 0.0}

        h2h_data[key]["games"] += 1
        h2h_data[key]["score_a"] += score_a
        h2h_data[key]["score_b"] += score_b

    # Construire résultat
    result_data = []
    for (joueur_a, joueur_b), stats in h2h_data.items():
        if stats["games"] >= min_games:
            avg_a = stats["score_a"] / stats["games"]
            avg_b = stats["score_b"] / stats["games"]
            result_data.append(
                {
                    "joueur_a": joueur_a,
                    "joueur_b": joueur_b,
                    "nb_confrontations": int(stats["games"]),
                    "score_a": avg_a,
                    "score_b": avg_b,
                    "avantage_a": avg_a - avg_b,
                }
            )

    result = pd.DataFrame(result_data)
    logger.info(f"  {len(result)} paires H2H avec >= {min_games} confrontations")

    return result
