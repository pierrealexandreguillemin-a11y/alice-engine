"""Performance sous pression (matchs décisifs) - ISO 5055/5259.

Ce module implémente le calcul de la performance sous pression.
Feature psychologique: performance en fin de saison ou zones d'enjeu réelles.

Matchs décisifs (SEUILS DOCUMENTÉS ISO 5259):
- Zone d'enjeu active (montee/danger): Pression réelle du classement.
  Pas de leakage circulaire — score_dom/ext est le résultat du MATCH,
  pas une mesure de l'enjeu avant le match.
  Référence: calculer_zone_enjeu() dans ffe_rules_features.py.
- Fallback ronde >= 7: Si zone_enjeu non disponible dans le DataFrame.
  Format interclubs FFE standard 7-11 rondes (Règlement FFE A02).

W/D/L decomposition (ISO 5259 — décomposition explicite):
- win_rate: proportion de victoires (résultat = 1.0)
- draw_rate: proportion de nulles (résultat = 0.5)
- clutch_win = win_rate_pression - win_rate_normal
- clutch_draw = draw_rate_pression - draw_rate_normal

pressure_type (SEUILS DOCUMENTÉS):
- "clutch": clutch_win > 0.1 — surperformance en victoires sous pression
- "choke": clutch_win < -0.1 — sous-performance en victoires sous pression
- "stable": clutch_win in [-0.1, 0.1] — pas d'effet significatif

Sources:
- AI Sports Predictions 2025 (ainewshub.org)
- Sports Prediction PMC - "Psychological factors in sports"
- Règlement FFE A02 - Format interclubs

Conformité:
- ISO 5055: Module <300 lignes, responsabilité unique
- ISO 5259: Features calculées depuis données réelles, seuils documentés
- ISO 5259: Pas de leakage circulaire (zone_enjeu != score résultat match)
"""

from __future__ import annotations

import logging

import pandas as pd

from scripts.features.helpers import filter_played_games

logger = logging.getLogger(__name__)

_ZONE_ENJEU_DECISIVE = {"montee", "danger"}


def calculate_pressure_performance(df: pd.DataFrame, min_games: int = 3) -> pd.DataFrame:
    """Calcule la performance sous pression (W/D/L décomposés).

    Matchs decisifs: zone_enjeu IN (montee, danger) OU ronde >= 7 (fallback).
    ISO 5259: Zone enjeu depuis classement réel — pas de leakage score_dom/ext.

    Args:
    ----
        df: DataFrame échiquiers (type_resultat, resultat_blanc/noir, ronde,
            optionnel: zone_enjeu_dom, zone_enjeu_ext)
        min_games: Nombre minimum de parties par contexte (normal/pression)

    Returns:
    -------
        DataFrame avec joueur_nom, win_rate_normal, draw_rate_normal,
        win_rate_pression, draw_rate_pression, clutch_win, clutch_draw,
        pressure_type
    """
    logger.info("Calcul performance sous pression (zone_enjeu)...")

    if df.empty:
        return pd.DataFrame()

    parties = filter_played_games(df)

    if parties.empty:
        return pd.DataFrame()

    parties = parties.copy()
    parties["is_decisive"] = _compute_is_decisive(parties)

    all_data = _build_long_form(parties)
    if all_data.empty:
        return pd.DataFrame()

    result = _aggregate_wdl(all_data, min_games)
    logger.info("  %d joueurs avec stats pression", len(result))
    return result


def _compute_is_decisive(parties: pd.DataFrame) -> pd.Series:
    """Détermine si chaque partie est décisive (vectorisé, sans score_dom/ext).

    Priorité: zone_enjeu_dom / zone_enjeu_ext si présentes.
    Fallback: ronde >= 7 uniquement.

    ISO 5259: Pas de leakage — n'utilise PAS score_dom/score_ext.
    """
    ronde = parties["ronde"] if "ronde" in parties.columns else pd.Series(0, index=parties.index)
    ronde_flag = ronde >= 7

    has_zone = "zone_enjeu_dom" in parties.columns or "zone_enjeu_ext" in parties.columns
    if not has_zone:
        return ronde_flag

    zone_dom_flag = (
        parties["zone_enjeu_dom"].isin(_ZONE_ENJEU_DECISIVE)
        if "zone_enjeu_dom" in parties.columns
        else pd.Series(False, index=parties.index)
    )
    zone_ext_flag = (
        parties["zone_enjeu_ext"].isin(_ZONE_ENJEU_DECISIVE)
        if "zone_enjeu_ext" in parties.columns
        else pd.Series(False, index=parties.index)
    )
    return zone_dom_flag | zone_ext_flag


def _build_long_form(parties: pd.DataFrame) -> pd.DataFrame:
    """Construit vue longue (joueur, resultat, is_decisive) — sans iterrows."""
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
    return all_data


def _aggregate_wdl(all_data: pd.DataFrame, min_games: int) -> pd.DataFrame:
    """Agrège W/D/L par joueur x contexte (normal / pression) — vectorisé."""
    all_data = all_data.copy()
    all_data["is_win"] = (all_data["resultat"] == 1.0).astype(float)
    all_data["is_draw"] = (all_data["resultat"] == 0.5).astype(float)

    grp = all_data.groupby(["joueur_nom", "is_decisive"])
    agg = (
        grp[["is_win", "is_draw", "resultat"]]
        .agg(
            win_rate=("is_win", "mean"),
            draw_rate=("is_draw", "mean"),
            nb=("resultat", "count"),
        )
        .reset_index()
    )

    normal_df = _extract_context(agg, is_decisive=False, suffix="normal")
    pressure_df = _extract_context(agg, is_decisive=True, suffix="pression")

    result = normal_df.merge(pressure_df, on="joueur_nom", how="inner")
    result = result[(result["nb_normal"] >= min_games) & (result["nb_pression"] >= min_games)]

    result["clutch_win"] = result["win_rate_pression"] - result["win_rate_normal"]
    result["clutch_draw"] = result["draw_rate_pression"] - result["draw_rate_normal"]
    result["pressure_type"] = result["clutch_win"].map(_classify_pressure_type)

    return result[
        [
            "joueur_nom",
            "win_rate_normal",
            "draw_rate_normal",
            "win_rate_pression",
            "draw_rate_pression",
            "clutch_win",
            "clutch_draw",
            "pressure_type",
        ]
    ]


def _extract_context(agg: pd.DataFrame, *, is_decisive: bool, suffix: str) -> pd.DataFrame:
    """Extrait et renomme les colonnes pour un contexte (normal/pression)."""
    ctx = agg[agg["is_decisive"] == is_decisive][
        ["joueur_nom", "win_rate", "draw_rate", "nb"]
    ].copy()
    return ctx.rename(
        columns={
            "win_rate": f"win_rate_{suffix}",
            "draw_rate": f"draw_rate_{suffix}",
            "nb": f"nb_{suffix}",
        }
    )


def _classify_pressure_type(clutch_win: float) -> str:
    """Classifie le type de pression depuis clutch_win (delta win_rate)."""
    if clutch_win > 0.1:
        return "clutch"
    if clutch_win < -0.1:
        return "choke"
    return "stable"
