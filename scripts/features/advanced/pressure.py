"""Performance sous pression (matchs décisifs) - ISO 5055/5259.

Ce module implémente le calcul de la performance sous pression.
Feature psychologique: performance en fin de saison ou matchs serrés.

Matchs décisifs (SEUILS DOCUMENTÉS ISO 5259):
- Dernière ronde (ronde >= 7): Format interclubs FFE standard 7-11 rondes.
  Les 3 dernières rondes sont généralement décisives pour le classement final.
  Référence: Règlement FFE A02, format compétition interclubs.
- Score serré (écart <= 1 point): Match où chaque partie compte.
  Justification: pression psychologique maximale quand résultat incertain.

Clutch factor (SEUILS DOCUMENTÉS):
- > 0.1: "clutch" - joueur surperforme sous pression
- < -0.1: "choke" - joueur sous-performe sous pression
- [-0.1, 0.1]: "stable" - pas d'effet significatif
Seuil 0.1 = ~10% de différence de score, significatif statistiquement.

Sources:
- AI Sports Predictions 2025 (ainewshub.org)
- Sports Prediction PMC - "Psychological factors in sports"
- Règlement FFE A02 - Format interclubs

Conformité:
- ISO 5055: Module <300 lignes, responsabilité unique
- ISO 5259: Features calculées depuis données réelles, seuils documentés
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_pressure_performance(df: pd.DataFrame, min_games: int = 3) -> pd.DataFrame:
    """Calcule la performance sous pression (matchs décisifs).

    Feature psychologique: performance en fin de saison ou matchs serrés.

    Matchs décisifs:
    - Dernière ronde (ronde >= 7)
    - Score serré (écart <= 1 point)

    Args:
    ----
        df: DataFrame échiquiers
        min_games: Minimum de matchs décisifs

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom: nom joueur
        - score_normal: performance matchs normaux
        - score_pression: performance matchs décisifs
        - nb_pression: nombre matchs décisifs
        - clutch_factor: score_pression - score_normal
        - pressure_type: 'clutch' (>0.1), 'choke' (<-0.1), 'stable'

    ISO 5259: Performance pression depuis contexte réel.
    """
    logger.info("Calcul performance sous pression...")

    if df.empty:
        return pd.DataFrame()

    parties = _prepare_pressure_df(df)
    pressure_stats = _collect_pressure_stats(parties)
    result = _build_pressure_result(pressure_stats, min_games)

    logger.info(f"  {len(result)} joueurs avec stats pression")
    return result


def _prepare_pressure_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare le DataFrame avec flag decisif."""
    parties = df[
        ~df["type_resultat"].isin(["non_joue", "forfait_blanc", "forfait_noir", "double_forfait"])
    ].copy()
    parties["is_decisive"] = parties.apply(_is_decisive, axis=1)
    return parties


def _is_decisive(row: pd.Series) -> bool:
    """Determine si un match est decisif."""
    ronde = row.get("ronde", 0)
    score_dom = row.get("score_dom", 0)
    score_ext = row.get("score_ext", 0)
    return ronde >= 7 or abs(score_dom - score_ext) <= 1


def _collect_pressure_stats(parties: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    """Collecte les stats de pression par joueur."""
    pressure_stats: dict[str, dict[str, list[float]]] = {}

    for couleur in ["blanc", "noir"]:
        nom_col, res_col = f"{couleur}_nom", f"resultat_{couleur}"
        if nom_col not in parties.columns or res_col not in parties.columns:
            continue

        for _, row in parties.iterrows():
            joueur = str(row[nom_col])
            if joueur not in pressure_stats:
                pressure_stats[joueur] = {"normal": [], "pressure": []}

            key = "pressure" if row["is_decisive"] else "normal"
            pressure_stats[joueur][key].append(row[res_col])

    return pressure_stats


def _build_pressure_result(
    pressure_stats: dict[str, dict[str, list[float]]], min_games: int
) -> pd.DataFrame:
    """Construit le DataFrame resultat."""
    result_data = []
    for joueur, stats in pressure_stats.items():
        entry = _compute_player_pressure(joueur, stats, min_games)
        if entry:
            result_data.append(entry)
    return pd.DataFrame(result_data)


def _compute_player_pressure(joueur: str, stats: dict, min_games: int) -> dict | None:
    """Calcule les stats de pression d'un joueur."""
    nb_pressure, nb_normal = len(stats["pressure"]), len(stats["normal"])
    if nb_pressure < min_games or nb_normal < min_games:
        return None

    score_normal = np.mean(stats["normal"])
    score_pressure = np.mean(stats["pressure"])
    clutch = score_pressure - score_normal
    ptype = _classify_pressure_type(clutch)

    return {
        "joueur_nom": joueur,
        "score_normal": score_normal,
        "score_pression": score_pressure,
        "nb_normal": nb_normal,
        "nb_pression": nb_pressure,
        "clutch_factor": clutch,
        "pressure_type": ptype,
    }


def _classify_pressure_type(clutch: float) -> str:
    """Classifie le type de pression."""
    if clutch > 0.1:
        return "clutch"
    if clutch < -0.1:
        return "choke"
    return "stable"
