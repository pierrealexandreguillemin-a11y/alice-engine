"""Feature forme recente joueur — W/D/L decomposition + stratification.

Document ID: ALICE-FEAT-RECENT-FORM
Version: 2.0.0

Ce module extrait la forme recente des joueurs avec decomposition W/D/L
et stratification par type de competition.

Conformite:
- ISO 5055: Module <300 lignes, responsabilite unique (SRP)
- ISO 5259: Donnees reelles, pas d'estimation
- ISO 42001: Traceabilite features ML
"""

from __future__ import annotations

import logging

import pandas as pd

from scripts.features.helpers import compute_wdl_rates, filter_played_games

logger = logging.getLogger(__name__)

_STRATIFY_MIN_GAMES = 3  # Seuil minimum de parties pour stratification


def calculate_recent_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Calcule la forme recente de chaque joueur avec decomposition W/D/L.

    Args:
    ----
        df: DataFrame echiquiers (parties jouees + non-jouees)
        window: nombre de matchs pour la fenetre de forme

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom: nom complet
        - win_rate_recent: taux de victoires [0, 1]
        - draw_rate_recent: taux de nulles [0, 1]
        - expected_score_recent: score estime (win + 0.5*draw)
        - win_trend: 'hausse'/'baisse'/'stable'
        - draw_trend: 'hausse'/'baisse'/'stable'
        - nb_matchs_forme: nombre de matchs dans la fenetre

    ISO 5259: Forme calculee depuis resultats reels uniquement.
    ISO 42001: Decomposition W/D/L — pas de confusion draws/losses.
    """
    logger.info(f"Calcul forme recente W/D/L (window={window})...")

    if df.empty:
        return pd.DataFrame()

    parties = filter_played_games(df)

    if "date" in parties.columns:
        parties = parties.sort_values("date")

    forme_data = _collect_form_data(parties, window)
    result = _aggregate_form_data(forme_data)

    logger.info(f"  {len(result)} joueurs avec forme recente")
    return result


def _collect_form_data(parties: pd.DataFrame, window: int) -> list[dict]:
    """Collecte les donnees de forme par joueur et couleur.

    Utilise la stratification par type_competition si disponible.
    """
    forme_data = []

    has_type_comp = "type_competition" in parties.columns

    for couleur in ["blanc", "noir"]:
        nom_col = f"{couleur}_nom"
        resultat_col = f"resultat_{couleur}"

        if nom_col not in parties.columns or resultat_col not in parties.columns:
            continue

        if has_type_comp:
            entries = _collect_stratified(parties, nom_col, resultat_col, window)
        else:
            entries = _collect_flat(parties, nom_col, resultat_col, window)

        forme_data.extend(entries)

    return forme_data


def _collect_stratified(
    parties: pd.DataFrame,
    nom_col: str,
    resultat_col: str,
    window: int,
) -> list[dict]:
    """Collecte la forme par (joueur, type_competition).

    Pour chaque (joueur, type_competition), prend les N derniers matchs
    au meme niveau. Si < _STRATIFY_MIN_GAMES matchs au meme niveau,
    bascule sur tous les matchs du joueur.
    """
    entries: list[dict] = []

    for joueur, player_group in parties.groupby(nom_col):
        has_any_stratified = False
        for type_comp, comp_group in player_group.groupby("type_competition"):
            if len(comp_group) >= _STRATIFY_MIN_GAMES:
                last_n = comp_group.tail(window)
                entry = _build_form_entry(joueur, last_n, resultat_col, type_comp)
                entries.append(entry)
                has_any_stratified = True
            # Skip under-represented competitions — don't fall back per-comp

        # Fallback: if NO competition had enough games, use all games
        if not has_any_stratified and len(player_group) >= _STRATIFY_MIN_GAMES:
            last_n = player_group.tail(window)
            entry = _build_form_entry(joueur, last_n, resultat_col, type_comp=None)
            entries.append(entry)

    return entries


def _collect_flat(
    parties: pd.DataFrame,
    nom_col: str,
    resultat_col: str,
    window: int,
) -> list[dict]:
    """Collecte la forme sans stratification (type_competition absent)."""
    entries: list[dict] = []

    for joueur, group in parties.groupby(nom_col):
        if len(group) < _STRATIFY_MIN_GAMES:
            continue
        last_n = group.tail(window)
        entry = _build_form_entry(joueur, last_n, resultat_col, type_comp=None)
        entries.append(entry)

    return entries


def _build_form_entry(
    joueur: str,
    last_n: pd.DataFrame,
    resultat_col: str,
    type_comp: str | None,
) -> dict:
    """Construit une entree de forme pour un joueur sur une fenetre."""
    results = last_n[resultat_col]
    wdl = compute_wdl_rates(results)
    win_trend, draw_trend = _compute_trends(last_n, resultat_col)

    return {
        "joueur_nom": joueur,
        "type_competition": type_comp,
        "win_rate_recent": wdl["win_rate"],
        "draw_rate_recent": wdl["draw_rate"],
        "expected_score_recent": wdl["expected_score"],
        "win_trend": win_trend,
        "draw_trend": draw_trend,
        "nb_matchs_forme": len(last_n),
    }


def _compute_trends(last_n: pd.DataFrame, resultat_col: str) -> tuple[str, str]:
    """Calcule les tendances win et draw (hausse/baisse/stable).

    Compare premiere moitie vs seconde moitie de la fenetre.
    """
    mid = len(last_n) // 2
    if mid == 0:
        return "stable", "stable"

    first = last_n.head(mid)[resultat_col]
    second = last_n.tail(mid)[resultat_col]

    win_trend = _trend_label(
        (first == 1.0).mean(),
        (second == 1.0).mean(),
    )
    draw_trend = _trend_label(
        (first == 0.5).mean(),
        (second == 0.5).mean(),
    )
    return win_trend, draw_trend


def _trend_label(first_rate: float, second_rate: float) -> str:
    """Retourne 'hausse', 'baisse', ou 'stable' selon le delta."""
    delta = second_rate - first_rate
    if delta > 0.1:
        return "hausse"
    if delta < -0.1:
        return "baisse"
    return "stable"


def _aggregate_form_data(forme_data: list[dict]) -> pd.DataFrame:
    """Agrege les donnees de forme — UNE ligne par joueur_nom.

    La stratification par type_competition sert a choisir quels matchs utiliser,
    mais le resultat final est agrege par joueur (pas par joueur+competition).
    On prend le type_competition avec le plus de matchs comme forme principale,
    puis on moyenne blanc+noir.
    """
    if not forme_data:
        return pd.DataFrame()

    df = pd.DataFrame(forme_data)

    # Step 1: pour chaque (joueur, type_competition), agreger blanc+noir
    group_cols = ["joueur_nom", "type_competition"]
    per_comp = (
        df.groupby(group_cols, dropna=False)
        .agg(
            win_rate_recent=("win_rate_recent", "mean"),
            draw_rate_recent=("draw_rate_recent", "mean"),
            expected_score_recent=("expected_score_recent", "mean"),
            win_trend=("win_trend", "first"),
            draw_trend=("draw_trend", "first"),
            nb_matchs_forme=("nb_matchs_forme", "sum"),
        )
        .reset_index()
    )

    # Step 2: pour chaque joueur, garder le type_competition avec le plus de matchs
    idx = per_comp.groupby("joueur_nom")["nb_matchs_forme"].idxmax()
    result = per_comp.loc[idx].drop(columns=["type_competition"]).reset_index(drop=True)

    return result
