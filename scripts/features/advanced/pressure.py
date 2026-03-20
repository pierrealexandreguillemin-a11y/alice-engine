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

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_pressure_performance(df: pd.DataFrame, min_games: int = 3) -> pd.DataFrame:
    """Calcule la performance sous pression (vectorise).

    Matchs decisifs: ronde >= 7 ou score serre (ecart <= 1 pt).
    ISO 5259: Performance pression depuis contexte reel.
    """
    logger.info("Calcul performance sous pression...")

    if df.empty:
        return pd.DataFrame()

    parties = df[
        ~df["type_resultat"].isin(["non_joue", "forfait_blanc", "forfait_noir", "double_forfait"])
    ].copy()

    # Vectorized is_decisive (no .apply)
    ronde = parties["ronde"] if "ronde" in parties.columns else pd.Series(0, index=parties.index)
    has_scores = "score_dom" in parties.columns and "score_ext" in parties.columns
    score_gap_flag = (
        (parties["score_dom"] - parties["score_ext"]).abs() <= 1
        if has_scores
        else pd.Series(False, index=parties.index)
    )
    parties["is_decisive"] = (ronde >= 7) | score_gap_flag

    # Build long-form (joueur, resultat, is_decisive) — no iterrows
    parts: list[pd.DataFrame] = []
    for couleur in ["blanc", "noir"]:
        nom_col, res_col = f"{couleur}_nom", f"resultat_{couleur}"
        if nom_col not in parties.columns or res_col not in parties.columns:
            continue
        sub = parties[[nom_col, res_col, "is_decisive"]].dropna(subset=[nom_col]).copy()
        sub.columns = ["joueur_nom", "resultat", "is_decisive"]
        parts.append(sub)

    if not parts:
        return pd.DataFrame()

    all_data = pd.concat(parts, ignore_index=True)
    all_data["joueur_nom"] = all_data["joueur_nom"].astype(str)

    # Group by joueur and is_decisive, compute mean + count
    stats = (
        all_data.groupby(["joueur_nom", "is_decisive"])["resultat"]
        .agg(["mean", "count"])
        .reset_index()
    )
    stats.columns = ["joueur_nom", "is_decisive", "score", "count"]

    normal = stats[~stats["is_decisive"]][["joueur_nom", "score", "count"]].rename(
        columns={"score": "score_normal", "count": "nb_normal"}
    )
    pressure = stats[stats["is_decisive"]][["joueur_nom", "score", "count"]].rename(
        columns={"score": "score_pression", "count": "nb_pression"}
    )

    result = normal.merge(pressure, on="joueur_nom", how="inner")
    result = result[(result["nb_normal"] >= min_games) & (result["nb_pression"] >= min_games)]
    result["clutch_factor"] = result["score_pression"] - result["score_normal"]
    result["pressure_type"] = result["clutch_factor"].apply(_classify_pressure_type)

    logger.info("  %d joueurs avec stats pression", len(result))
    return result


def _classify_pressure_type(clutch: float) -> str:
    """Classifie le type de pression."""
    if clutch > 0.1:
        return "clutch"
    if clutch < -0.1:
        return "choke"
    return "stable"
