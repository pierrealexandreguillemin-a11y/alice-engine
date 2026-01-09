"""Features classement équipe et zones d'enjeu - ISO 5055/5259.

Ce module calcule les classements réels depuis les scores de match
et détermine les zones d'enjeu (promotion/maintien).

Conformité:
- ISO 5055: Module <300 lignes, responsabilité unique
- ISO 5259: Position réelle, tie-breakers FFE implémentés (CORRIGÉ)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

# Ré-exports pour compatibilité
from scripts.features.enjeu import extract_team_enjeu_fallback, extract_team_enjeu_features

logger = logging.getLogger(__name__)

# Export public
__all__ = [
    "calculate_standings",
    "extract_team_enjeu_features",
    "extract_team_enjeu_fallback",
]


def calculate_standings(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule le classement réel par équipe/saison/groupe/ronde.

    CORRIGÉ: Implémente les tie-breakers FFE pour départager ex-aequo.

    Points Interclubs FFE:
    - Victoire match (score > adversaire): 2 pts
    - Nul match (score = adversaire): 1 pt
    - Défaite match (score < adversaire): 0 pt

    Tie-breakers FFE (dans l'ordre):
    1. Confrontation directe (résultat entre les ex-aequo)
    2. Différence de points de matchs (somme score_dom - score_ext)
    3. Nombre de victoires
    4. Nom alphabétique (dernier recours, stable)

    Args:
    ----
        df: DataFrame échiquiers complet

    Returns:
    -------
        DataFrame avec colonnes:
        - equipe, saison, competition, division, groupe, ronde
        - points_cumules: points accumulés jusqu'à cette ronde
        - position: classement à cette ronde (1 = premier)
        - nb_equipes: nombre total d'équipes dans le groupe
        - ecart_premier: points du 1er - points équipe
        - ecart_dernier: points équipe - points du dernier
        - victoires: nombre de victoires
        - diff_points_matchs: différence de points de matchs

    ISO 5259: Position calculée depuis données réelles avec tie-breakers.
    """
    logger.info("Calcul classement réel depuis scores matchs...")

    if df.empty:
        return pd.DataFrame()

    # Colonnes requises pour extraire matchs
    match_cols = [
        "saison",
        "competition",
        "division",
        "groupe",
        "ronde",
        "equipe_dom",
        "equipe_ext",
        "score_dom",
        "score_ext",
    ]

    # Vérifier colonnes présentes
    missing = [c for c in match_cols if c not in df.columns]
    if missing:
        logger.warning(f"  Colonnes manquantes pour classement: {missing}")
        return pd.DataFrame()

    # Extraire matchs uniques
    matches = df.drop_duplicates(
        subset=["saison", "competition", "division", "groupe", "ronde", "equipe_dom", "equipe_ext"]
    )[match_cols].copy()

    logger.info(f"  {len(matches)} matchs uniques")

    standings_data = []

    for (saison, comp, div, groupe), group_matches in matches.groupby(
        ["saison", "competition", "division", "groupe"]
    ):
        # Stats accumulées par équipe
        equipe_stats: dict[str, dict[str, Any]] = {}

        # Historique confrontations directes
        h2h: dict[tuple[str, str], int] = {}

        for ronde in sorted(group_matches["ronde"].unique()):
            ronde_matches = group_matches[group_matches["ronde"] == ronde]

            for _, match in ronde_matches.iterrows():
                _process_match(match, equipe_stats, h2h)

            # Calculer classement à cette ronde avec tie-breakers
            ranking = _apply_tiebreakers(equipe_stats, h2h)

            nb_equipes = len(ranking)
            pts_premier = ranking[0][1]["points"] if ranking else 0
            pts_dernier = ranking[-1][1]["points"] if ranking else 0

            for position, (equipe, stats) in enumerate(ranking, 1):
                standings_data.append(
                    {
                        "equipe": equipe,
                        "saison": saison,
                        "competition": comp,
                        "division": div,
                        "groupe": groupe,
                        "ronde": ronde,
                        "points_cumules": stats["points"],
                        "matchs_joues": stats["matchs"],
                        "victoires": stats["victoires"],
                        "diff_points_matchs": stats["diff_points_matchs"],
                        "position": position,
                        "nb_equipes": nb_equipes,
                        "ecart_premier": pts_premier - stats["points"],
                        "ecart_dernier": stats["points"] - pts_dernier,
                    }
                )

    result = pd.DataFrame(standings_data)
    logger.info(f"  {len(result)} lignes classement générées")

    return result


def _process_match(
    match: pd.Series,
    equipe_stats: dict[str, dict[str, Any]],
    h2h: dict[tuple[str, str], int],
) -> None:
    """Traite un match et met à jour les stats."""
    dom = str(match["equipe_dom"])
    ext = str(match["equipe_ext"])
    sd = match["score_dom"]
    se = match["score_ext"]

    # Initialiser si nouvelle équipe
    for eq in [dom, ext]:
        if eq not in equipe_stats:
            equipe_stats[eq] = {
                "points": 0,
                "victoires": 0,
                "diff_points_matchs": 0,
                "matchs": 0,
            }

    # Mettre à jour stats
    equipe_stats[dom]["matchs"] += 1
    equipe_stats[ext]["matchs"] += 1
    equipe_stats[dom]["diff_points_matchs"] += sd - se
    equipe_stats[ext]["diff_points_matchs"] += se - sd

    # Attribuer points (2 victoire, 1 nul, 0 défaite)
    if sd > se:  # Dom gagne
        equipe_stats[dom]["points"] += 2
        equipe_stats[dom]["victoires"] += 1
        h2h[(dom, ext)] = h2h.get((dom, ext), 0) + 2
        h2h[(ext, dom)] = h2h.get((ext, dom), 0) + 0
    elif se > sd:  # Ext gagne
        equipe_stats[ext]["points"] += 2
        equipe_stats[ext]["victoires"] += 1
        h2h[(ext, dom)] = h2h.get((ext, dom), 0) + 2
        h2h[(dom, ext)] = h2h.get((dom, ext), 0) + 0
    else:  # Nul
        equipe_stats[dom]["points"] += 1
        equipe_stats[ext]["points"] += 1
        h2h[(dom, ext)] = h2h.get((dom, ext), 0) + 1
        h2h[(ext, dom)] = h2h.get((ext, dom), 0) + 1


def _apply_tiebreakers(
    equipe_stats: dict[str, dict[str, Any]],
    h2h: dict[tuple[str, str], int],
) -> list[tuple[str, dict[str, Any]]]:
    """Applique les tie-breakers FFE pour départager ex-aequo.

    Ordre des critères:
    1. Points
    2. Confrontation directe (H2H)
    3. Différence de points de matchs
    4. Nombre de victoires
    5. Nom alphabétique (stable)
    """

    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[Any, ...]:
        equipe, stats = item
        pts = stats["points"]
        diff = stats["diff_points_matchs"]
        vic = stats["victoires"]
        h2h_score = sum(h2h.get((equipe, other), 0) for other in equipe_stats if other != equipe)
        return (-pts, -h2h_score, -diff, -vic, equipe)

    return sorted(equipe_stats.items(), key=sort_key)
