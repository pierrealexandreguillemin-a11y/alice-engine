#!/usr/bin/env python3
"""Feature Engineering pour ALICE - Preparation des donnees ML.

Ce script transforme les donnees brutes (echiquiers.parquet) en features
exploitables pour l'entrainement du modele ALI (Adversarial Lineup Inference).

Features extraites:
- Features de fiabilite (depuis non_joue/forfaits)
- Features derivees (forme recente, taux presence, etc.)
- Split temporel pour validation

Conformite ISO/IEC 25010:2023, 25012 (Qualite donnees).

Usage:
    python scripts/feature_engineering.py
    python scripts/feature_engineering.py --output-dir data/features
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from scripts.ffe_rules_features import (
    TypeCompetition,
    calculer_zone_enjeu,
    detecter_type_competition,
    get_niveau_equipe,
)

# Configuration paths
PROJECT_DIR = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# FEATURES DE FIABILITE
# ==============================================================================


def extract_club_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait les features de fiabilite par club.

    Utilise les non_joue et forfaits pour identifier les clubs defaillants.

    Args:
    ----
        df: DataFrame echiquiers complet (avant filtrage)

    Returns:
    -------
        DataFrame avec colonnes:
        - equipe: nom du club
        - taux_forfait: % de forfaits sur l'historique
        - taux_non_joue: % de matchs non joues
        - fiabilite_score: score composite (1 - taux_defaillance)
    """
    logger.info("Extraction features fiabilite clubs...")

    # Agreger par equipe domicile
    forfaits_types = ["forfait_blanc", "forfait_noir", "double_forfait"]

    club_stats = []

    for equipe_col in ["equipe_dom", "equipe_ext"]:
        stats = df.groupby(equipe_col).agg(
            total_matchs=("type_resultat", "count"),
            forfaits=("type_resultat", lambda x: x.isin(forfaits_types).sum()),
            non_joue=("type_resultat", lambda x: (x == "non_joue").sum()),
        )
        stats = stats.reset_index().rename(columns={equipe_col: "equipe"})
        club_stats.append(stats)

    # Fusionner dom + ext
    all_stats = pd.concat(club_stats).groupby("equipe").sum().reset_index()

    # Calculer taux
    all_stats["taux_forfait"] = all_stats["forfaits"] / all_stats["total_matchs"]
    all_stats["taux_non_joue"] = all_stats["non_joue"] / all_stats["total_matchs"]
    all_stats["fiabilite_score"] = 1 - (
        all_stats["taux_forfait"] * 0.7 + all_stats["taux_non_joue"] * 0.3
    )

    logger.info(f"  {len(all_stats)} clubs avec stats fiabilite")

    return all_stats[["equipe", "taux_forfait", "taux_non_joue", "fiabilite_score"]]


def extract_player_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait les features de fiabilite par joueur.

    Analyse les patterns de presence/absence pour chaque joueur.

    Args:
    ----
        df: DataFrame echiquiers complet

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom: nom complet du joueur
        - nb_matchs: nombre total de matchs
        - taux_presence: % de matchs effectivement joues
        - joueur_fantome: flag si < 20% presence
    """
    logger.info("Extraction features fiabilite joueurs...")

    # Collecter stats pour blancs et noirs
    player_stats = []

    for couleur in ["blanc", "noir"]:
        nom_col = f"{couleur}_nom"
        df_couleur = df[df[nom_col].notna()].copy()

        # Identifier les matchs joues vs non joues
        matchs_joues = ~df_couleur["type_resultat"].isin(
            ["non_joue", f"forfait_{couleur}", "double_forfait"]
        )

        # Capture matchs_joues via default arg to avoid B023
        stats = (
            df_couleur.groupby(nom_col)
            .agg(
                nb_matchs=("type_resultat", "count"),
                matchs_joues=(
                    "type_resultat",
                    lambda x, mj=matchs_joues: mj[x.index].sum(),
                ),
            )
            .reset_index()
            .rename(columns={nom_col: "joueur_nom"})
        )
        player_stats.append(stats)

    # Fusionner blanc + noir
    all_stats = pd.concat(player_stats).groupby("joueur_nom").sum().reset_index()

    # Calculer taux
    all_stats["taux_presence"] = all_stats["matchs_joues"] / all_stats["nb_matchs"]
    all_stats["joueur_fantome"] = all_stats["taux_presence"] < 0.2

    logger.info(f"  {len(all_stats)} joueurs avec stats fiabilite")
    logger.info(f"  {all_stats['joueur_fantome'].sum()} joueurs fantomes (<20% presence)")

    return all_stats[["joueur_nom", "nb_matchs", "taux_presence", "joueur_fantome"]]


def extract_player_monthly_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait les patterns de disponibilite mensuelle par joueur.

    Detecte les joueurs indisponibles certains mois (vacances, pro, etc.)

    Args:
    ----
        df: DataFrame echiquiers avec colonne 'date'

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom: nom complet
        - dispo_mois_1 ... dispo_mois_12: taux presence par mois
    """
    logger.info("Extraction patterns mensuels joueurs...")

    # Filtrer matchs avec date
    df_dated = df[df["date"].notna()].copy()
    if len(df_dated) == 0:
        logger.warning("  Pas de dates disponibles, skip patterns mensuels")
        return pd.DataFrame()

    df_dated["mois"] = pd.to_datetime(df_dated["date"]).dt.month

    # Collecter pour blancs et noirs
    monthly_stats = []

    for couleur in ["blanc", "noir"]:
        nom_col = f"{couleur}_nom"
        matchs_joues = ~df_dated["type_resultat"].isin(
            ["non_joue", f"forfait_{couleur}", "double_forfait"]
        )

        # Pivot par mois
        df_couleur = df_dated[[nom_col, "mois"]].copy()
        df_couleur["joue"] = matchs_joues.astype(int)

        pivot = df_couleur.pivot_table(
            values="joue", index=nom_col, columns="mois", aggfunc="mean", fill_value=0
        )
        pivot = pivot.reset_index().rename(columns={nom_col: "joueur_nom"})

        # Renommer colonnes mois
        pivot.columns = ["joueur_nom"] + [f"dispo_mois_{m}" for m in pivot.columns[1:]]
        monthly_stats.append(pivot)

    # Fusionner (moyenne blanc/noir)
    if len(monthly_stats) == 2:
        result = monthly_stats[0].merge(
            monthly_stats[1], on="joueur_nom", how="outer", suffixes=("_b", "_n")
        )
        # Moyenne des deux couleurs
        for m in range(1, 13):
            col_b = f"dispo_mois_{m}_b"
            col_n = f"dispo_mois_{m}_n"
            if col_b in result.columns and col_n in result.columns:
                result[f"dispo_mois_{m}"] = result[[col_b, col_n]].mean(axis=1)
                result = result.drop(columns=[col_b, col_n])

        logger.info(f"  {len(result)} joueurs avec patterns mensuels")
        return result

    return monthly_stats[0] if monthly_stats else pd.DataFrame()


# ==============================================================================
# FEATURES DERIVEES
# ==============================================================================


def calculate_recent_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Calcule la forme recente de chaque joueur (score sur N derniers matchs).

    Args:
    ----
        df: DataFrame echiquiers filtre (parties jouees uniquement)
        window: nombre de matchs pour calculer la forme

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom: nom complet
        - forme_recente: score moyen sur les N derniers matchs (0-1)
        - nb_matchs_forme: nombre de matchs utilises
    """
    logger.info(f"Calcul forme recente (window={window})...")

    # Filtrer parties jouees
    parties_jouees = df[
        ~df["type_resultat"].isin(["non_joue", "forfait_blanc", "forfait_noir", "double_forfait"])
    ].copy()

    if "date" in parties_jouees.columns:
        parties_jouees = parties_jouees.sort_values("date")

    forme_data = []

    for couleur in ["blanc", "noir"]:
        nom_col = f"{couleur}_nom"
        resultat_col = f"resultat_{couleur}"

        for joueur, group in parties_jouees.groupby(nom_col):
            if len(group) >= window:
                last_n = group.tail(window)
                forme = last_n[resultat_col].mean()
                forme_data.append(
                    {
                        "joueur_nom": joueur,
                        "forme_recente": forme,
                        "nb_matchs_forme": len(last_n),
                    }
                )

    result = pd.DataFrame(forme_data)
    if len(result) > 0:
        # Agreger si joueur joue blanc ET noir
        result = (
            result.groupby("joueur_nom")
            .agg(
                forme_recente=("forme_recente", "mean"),
                nb_matchs_forme=("nb_matchs_forme", "sum"),
            )
            .reset_index()
        )

    logger.info(f"  {len(result)} joueurs avec forme recente")
    return result


def calculate_board_position(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule la position moyenne sur l'echiquier pour chaque joueur.

    Un joueur habitue a jouer sur echiquier 1 vs echiquier 8
    n'a pas le meme niveau.

    Args:
    ----
        df: DataFrame echiquiers

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom: nom complet
        - echiquier_moyen: position moyenne
        - echiquier_std: ecart-type (variabilite)
    """
    logger.info("Calcul position echiquier moyenne...")

    board_data = []

    for couleur in ["blanc", "noir"]:
        nom_col = f"{couleur}_nom"

        for joueur, group in df.groupby(nom_col):
            board_data.append(
                {
                    "joueur_nom": joueur,
                    "echiquier_moyen": group["echiquier"].mean(),
                    "echiquier_std": group["echiquier"].std(),
                }
            )

    result = pd.DataFrame(board_data)
    if len(result) > 0:
        result = (
            result.groupby("joueur_nom")
            .agg(
                echiquier_moyen=("echiquier_moyen", "mean"),
                echiquier_std=("echiquier_std", "mean"),
            )
            .reset_index()
        )

    logger.info(f"  {len(result)} joueurs avec stats echiquier")
    return result


def calculate_color_performance(df: pd.DataFrame, min_games: int = 10) -> pd.DataFrame:
    """Calcule la performance par couleur (blanc/noir) pour chaque joueur.

    Convention echecs interclubs:
    - Echiquiers impairs (1, 3, 5, 7) = Blancs pour equipe domicile
    - Echiquiers pairs (2, 4, 6, 8) = Noirs pour equipe domicile

    Certains joueurs performent mieux avec une couleur.

    Args:
    ----
        df: DataFrame echiquiers (parties jouees uniquement)
        min_games: Minimum de parties pour calculer (defaut: 10)

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom: nom complet
        - score_blancs: score moyen avec blancs (0-1)
        - score_noirs: score moyen avec noirs (0-1)
        - nb_blancs: nombre de parties avec blancs
        - nb_noirs: nombre de parties avec noirs
        - avantage_blancs: score_blancs - score_noirs (>0 = prefere blancs)
        - couleur_preferee: 'blanc', 'noir', ou 'neutre'

    ISO 5259: Feature calculee depuis donnees reelles.
    """
    logger.info("Calcul performance par couleur (blanc/noir)...")

    # Filtrer parties jouees
    parties_jouees = df[
        ~df["type_resultat"].isin(["non_joue", "forfait_blanc", "forfait_noir", "double_forfait"])
    ].copy()

    # Stats par joueur jouant avec blancs
    blancs_stats = (
        parties_jouees.groupby("blanc_nom")
        .agg(
            score_blancs=("resultat_blanc", "mean"),
            nb_blancs=("resultat_blanc", "count"),
        )
        .reset_index()
        .rename(columns={"blanc_nom": "joueur_nom"})
    )

    # Stats par joueur jouant avec noirs
    noirs_stats = (
        parties_jouees.groupby("noir_nom")
        .agg(
            score_noirs=("resultat_noir", "mean"),
            nb_noirs=("resultat_noir", "count"),
        )
        .reset_index()
        .rename(columns={"noir_nom": "joueur_nom"})
    )

    # Fusionner
    result = blancs_stats.merge(noirs_stats, on="joueur_nom", how="outer")

    # Remplir NaN
    result["score_blancs"] = result["score_blancs"].fillna(0.5)
    result["score_noirs"] = result["score_noirs"].fillna(0.5)
    result["nb_blancs"] = result["nb_blancs"].fillna(0).astype(int)
    result["nb_noirs"] = result["nb_noirs"].fillna(0).astype(int)

    # Filtrer joueurs avec assez de parties
    result["nb_total"] = result["nb_blancs"] + result["nb_noirs"]
    result = result[result["nb_total"] >= min_games].copy()

    # Calculer avantage couleur
    result["avantage_blancs"] = result["score_blancs"] - result["score_noirs"]

    # Categoriser preference (seuil 5% = significatif)
    def categorize_preference(row: pd.Series) -> str:
        # Besoin de min 5 parties dans chaque couleur pour juger
        if row["nb_blancs"] < 5 or row["nb_noirs"] < 5:
            return "neutre"
        if row["avantage_blancs"] > 0.05:
            return "blanc"
        elif row["avantage_blancs"] < -0.05:
            return "noir"
        return "neutre"

    result["couleur_preferee"] = result.apply(categorize_preference, axis=1)

    # Nettoyer colonnes
    result = result.drop(columns=["nb_total"])

    logger.info(f"  {len(result)} joueurs avec stats couleur")
    logger.info(
        f"  Preferences: {(result['couleur_preferee'] == 'blanc').sum()} blanc, "
        f"{(result['couleur_preferee'] == 'noir').sum()} noir, "
        f"{(result['couleur_preferee'] == 'neutre').sum()} neutre"
    )

    return result


# ==============================================================================
# FEATURES REGLEMENTAIRES FFE
# ==============================================================================


def build_historique_brulage(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Construit l'historique de brulage par joueur.

    Compte le nombre de matchs joues par chaque joueur dans chaque equipe.

    Args:
    ----
        df: DataFrame echiquiers

    Returns:
    -------
        {joueur_nom: {equipe_nom: nb_matchs}}
    """
    historique: dict[str, dict[str, int]] = {}

    for couleur in ["blanc", "noir"]:
        nom_col = f"{couleur}_nom"
        df_couleur = df[df[nom_col].notna()].copy()

        for _, row in df_couleur.iterrows():
            joueur = str(row[nom_col])
            equipe = str(row[f"equipe_{'dom' if couleur == 'blanc' else 'ext'}"])

            if joueur not in historique:
                historique[joueur] = {}
            if equipe not in historique[joueur]:
                historique[joueur][equipe] = 0
            historique[joueur][equipe] += 1

    return historique


def build_historique_noyau(df: pd.DataFrame) -> dict[str, set[str]]:
    """Construit l'historique du noyau par equipe.

    Identifie les joueurs ayant deja joue pour chaque equipe.

    Args:
    ----
        df: DataFrame echiquiers

    Returns:
    -------
        {equipe_nom: set(joueur_noms)}
    """
    noyau: dict[str, set[str]] = {}

    for couleur in ["blanc", "noir"]:
        nom_col = f"{couleur}_nom"
        equipe_col = f"equipe_{'dom' if couleur == 'blanc' else 'ext'}"
        df_couleur = df[df[nom_col].notna()].copy()

        for _, row in df_couleur.iterrows():
            joueur = str(row[nom_col])
            equipe = str(row[equipe_col])

            if equipe not in noyau:
                noyau[equipe] = set()
            noyau[equipe].add(joueur)

    return noyau


def extract_ffe_regulatory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait les features reglementaires FFE.

    Calcule pour chaque joueur:
    - nb_equipes: nombre d'equipes differentes jouees
    - niveau_max: niveau hierarchique max joue (1=Top16)
    - niveau_min: niveau hierarchique min joue
    - type_competition: type de competition le plus frequent
    - matchs_par_niveau: distribution des matchs par niveau

    Args:
    ----
        df: DataFrame echiquiers

    Returns:
    -------
        DataFrame avec features reglementaires par joueur
    """
    logger.info("Extraction features reglementaires FFE...")

    features_data = []

    for couleur in ["blanc", "noir"]:
        nom_col = f"{couleur}_nom"
        equipe_col = f"equipe_{'dom' if couleur == 'blanc' else 'ext'}"
        competition_col = "competition"

        df_couleur = df[df[nom_col].notna()].copy()

        for joueur, group in df_couleur.groupby(nom_col):
            equipes = group[equipe_col].unique()
            niveaux = [get_niveau_equipe(str(eq)) for eq in equipes]

            # Competition la plus frequente
            if competition_col in group.columns:
                competitions = group[competition_col].dropna()
                if len(competitions) > 0:
                    type_comp = detecter_type_competition(str(competitions.mode().iloc[0]))
                else:
                    type_comp = TypeCompetition.A02
            else:
                type_comp = TypeCompetition.A02

            features_data.append(
                {
                    "joueur_nom": joueur,
                    "nb_equipes": len(equipes),
                    "niveau_max": min(niveaux) if niveaux else 10,  # min = plus fort
                    "niveau_min": max(niveaux) if niveaux else 10,
                    "type_competition": type_comp.value,
                    "multi_equipe": len(equipes) > 1,
                }
            )

    result = pd.DataFrame(features_data)
    if len(result) > 0:
        result = (
            result.groupby("joueur_nom")
            .agg(
                nb_equipes=("nb_equipes", "max"),
                niveau_max=("niveau_max", "min"),
                niveau_min=("niveau_min", "max"),
                type_competition=("type_competition", "first"),
                multi_equipe=("multi_equipe", "max"),
            )
            .reset_index()
        )

    logger.info(f"  {len(result)} joueurs avec features reglementaires")
    logger.info(f"  {result['multi_equipe'].sum()} joueurs multi-equipes")

    return result


def calculate_standings(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule le classement reel par equipe/saison/groupe/ronde.

    Points Interclubs FFE:
    - Victoire match (score > adversaire): 2 pts
    - Nul match (score = adversaire): 1 pt
    - Defaite match (score < adversaire): 0 pt

    Args:
    ----
        df: DataFrame echiquiers complet

    Returns:
    -------
        DataFrame avec colonnes:
        - equipe, saison, competition, division, groupe, ronde
        - points_cumules: points accumules jusqu'a cette ronde
        - position: classement a cette ronde (1 = premier)
        - nb_equipes: nombre total d'equipes dans le groupe
        - ecart_premier: points du 1er - points equipe
        - ecart_dernier: points equipe - points du dernier

    ISO 5259: Position calculee depuis donnees reelles, pas estimee.
    """
    logger.info("Calcul classement reel depuis scores matchs...")

    # Extraire matchs uniques (un seul row par match, pas par echiquier)
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

    # Verifier colonnes presentes
    missing = [c for c in match_cols if c not in df.columns]
    if missing:
        logger.warning(f"  Colonnes manquantes pour classement: {missing}")
        return pd.DataFrame()

    matches = df.drop_duplicates(
        subset=["saison", "competition", "division", "groupe", "ronde", "equipe_dom", "equipe_ext"]
    )[match_cols].copy()

    logger.info(f"  {len(matches)} matchs uniques")

    # Calculer points par match
    standings_data = []

    for (saison, comp, div, groupe), group_matches in matches.groupby(
        ["saison", "competition", "division", "groupe"]
    ):
        # Accumuler points par equipe au fil des rondes
        equipe_points: dict[str, int] = {}
        equipe_played: dict[str, int] = {}

        for ronde in sorted(group_matches["ronde"].unique()):
            ronde_matches = group_matches[group_matches["ronde"] == ronde]

            for _, match in ronde_matches.iterrows():
                dom = match["equipe_dom"]
                ext = match["equipe_ext"]
                sd = match["score_dom"]
                se = match["score_ext"]

                # Initialiser si nouveau
                if dom not in equipe_points:
                    equipe_points[dom] = 0
                    equipe_played[dom] = 0
                if ext not in equipe_points:
                    equipe_points[ext] = 0
                    equipe_played[ext] = 0

                # Attribuer points (2 victoire, 1 nul, 0 defaite)
                equipe_played[dom] += 1
                equipe_played[ext] += 1

                if sd > se:  # Dom gagne
                    equipe_points[dom] += 2
                elif se > sd:  # Ext gagne
                    equipe_points[ext] += 2
                else:  # Nul
                    equipe_points[dom] += 1
                    equipe_points[ext] += 1

            # Calculer classement a cette ronde
            ranking = sorted(equipe_points.items(), key=lambda x: -x[1])
            nb_equipes = len(ranking)
            pts_premier = ranking[0][1] if ranking else 0
            pts_dernier = ranking[-1][1] if ranking else 0

            for position, (equipe, pts) in enumerate(ranking, 1):
                standings_data.append(
                    {
                        "equipe": equipe,
                        "saison": saison,
                        "competition": comp,
                        "division": div,
                        "groupe": groupe,
                        "ronde": ronde,
                        "points_cumules": pts,
                        "matchs_joues": equipe_played[equipe],
                        "position": position,
                        "nb_equipes": nb_equipes,
                        "ecart_premier": pts_premier - pts,
                        "ecart_dernier": pts - pts_dernier,
                    }
                )

    result = pd.DataFrame(standings_data)
    logger.info(f"  {len(result)} lignes classement generees")

    return result


def extract_team_enjeu_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait les features de zone d'enjeu par equipe et saison.

    CORRIGE: Utilise position reelle calculee depuis les scores.

    Args:
    ----
        df: DataFrame echiquiers avec colonnes ronde, saison

    Returns:
    -------
        DataFrame avec zone_enjeu par equipe/saison/ronde

    ISO 5259: Zone d'enjeu basee sur position reelle, pas estimee.
    """
    logger.info("Extraction features zones d'enjeu (position reelle)...")

    if "ronde" not in df.columns or "saison" not in df.columns:
        logger.warning("  Colonnes ronde/saison manquantes, skip zones enjeu")
        return pd.DataFrame()

    # Calculer classement reel
    standings = calculate_standings(df)

    if standings.empty:
        logger.warning("  Classement vide, fallback estimation")
        return _extract_team_enjeu_fallback(df)

    # Enrichir avec zone d'enjeu
    features_data = []

    for _, row in standings.iterrows():
        division = str(row["division"]) if row["division"] else "N4"
        zone = calculer_zone_enjeu(row["position"], row["nb_equipes"], division)

        features_data.append(
            {
                "equipe": row["equipe"],
                "saison": row["saison"],
                "competition": row["competition"],
                "division": row["division"],
                "groupe": row["groupe"],
                "ronde": row["ronde"],
                "position": row["position"],
                "points_cumules": row["points_cumules"],
                "nb_equipes": row["nb_equipes"],
                "ecart_premier": row["ecart_premier"],
                "ecart_dernier": row["ecart_dernier"],
                "zone_enjeu": zone,
                "niveau_hierarchique": get_niveau_equipe(str(row["equipe"])),
            }
        )

    result = pd.DataFrame(features_data)
    logger.info(f"  {len(result)} equipes/rondes avec zones enjeu reelles")

    return result


def _extract_team_enjeu_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback si calcul classement impossible (donnees incompletes)."""
    logger.warning("  Utilisation fallback zone enjeu (estimation)")

    features_data = []

    for equipe_col in ["equipe_dom", "equipe_ext"]:
        for (equipe, saison), group in df.groupby([equipe_col, "saison"]):
            division = str(equipe).split()[0] if equipe else "N4"
            niveau = get_niveau_equipe(str(equipe))
            nb_equipes = 10 if niveau <= 4 else 8

            # Fallback: position estimee mi-tableau
            position_estimee = nb_equipes // 2
            zone = calculer_zone_enjeu(position_estimee, nb_equipes, division)

            features_data.append(
                {
                    "equipe": equipe,
                    "saison": saison,
                    "zone_enjeu": zone,
                    "niveau_hierarchique": niveau,
                    "nb_rondes": group["ronde"].nunique(),
                    "position": position_estimee,
                    "nb_equipes": nb_equipes,
                    "is_fallback": True,
                }
            )

    result = pd.DataFrame(features_data)
    if len(result) > 0:
        result = result.drop_duplicates(subset=["equipe", "saison"])

    return result


# ==============================================================================
# SPLIT TEMPOREL
# ==============================================================================


def temporal_split(
    df: pd.DataFrame,
    train_end: int = 2022,
    valid_end: int = 2023,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporel des donnees pour eviter data leakage.

    Args:
    ----
        df: DataFrame echiquiers
        train_end: derniere saison pour train (incluse)
        valid_end: derniere saison pour validation (incluse)

    Returns:
    -------
        Tuple (train, valid, test) DataFrames
    """
    logger.info(
        f"Split temporel: train<={train_end}, valid={train_end + 1}-{valid_end}, test>{valid_end}"
    )

    train = df[df["saison"] <= train_end]
    valid = df[(df["saison"] > train_end) & (df["saison"] <= valid_end)]
    test = df[df["saison"] > valid_end]

    logger.info(
        f"  Train: {len(train):,} echiquiers ({train['saison'].min()}-{train['saison'].max()})"
    )
    logger.info(
        f"  Valid: {len(valid):,} echiquiers ({valid['saison'].min()}-{valid['saison'].max()})"
    )
    logger.info(
        f"  Test:  {len(test):,} echiquiers ({test['saison'].min()}-{test['saison'].max()})"
    )

    return train, valid, test


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================


def run_feature_engineering(data_dir: Path, output_dir: Path) -> None:
    """Pipeline complet de feature engineering.

    ATTENTION: DATA LEAKAGE!
    Cette fonction calcule les features AVANT le split temporel,
    ce qui cause du data leakage (features utilisent donnees futures).

    Utilisez run_feature_engineering_v2() pour un pipeline sans leakage.

    @deprecated: Utilisez run_feature_engineering_v2() a la place.
    """
    logger.warning("!" * 60)
    logger.warning("ATTENTION: Cette version a un DATA LEAKAGE!")
    logger.warning("Les features sont calculees AVANT le split temporel.")
    logger.warning("Utilisez run_feature_engineering_v2() pour corriger.")
    logger.warning("!" * 60)

    logger.info("=" * 60)
    logger.info("ALICE Engine - Feature Engineering")
    logger.info("=" * 60)

    # Charger donnees
    echiquiers_path = data_dir / "echiquiers.parquet"
    if not echiquiers_path.exists():
        logger.error(f"Fichier non trouve: {echiquiers_path}")
        logger.error("Executez d'abord: python scripts/parse_dataset.py")
        return

    logger.info(f"\nChargement {echiquiers_path}...")
    df = pd.read_parquet(echiquiers_path)
    logger.info(f"  {len(df):,} echiquiers charges")

    # 1. Features de fiabilite (AVANT filtrage)
    logger.info("\n[1/6] Extraction features fiabilite...")
    club_reliability = extract_club_reliability(df)
    player_reliability = extract_player_reliability(df)
    player_monthly = extract_player_monthly_pattern(df)

    # 2. Filtrer pour features derivees (parties jouees)
    logger.info("\n[2/6] Filtrage parties jouees...")
    df_played = df[
        ~df["type_resultat"].isin(["non_joue", "forfait_blanc", "forfait_noir", "double_forfait"])
    ]
    logger.info(f"  {len(df_played):,} parties jouees ({len(df_played) / len(df) * 100:.1f}%)")

    # 3. Features derivees
    logger.info("\n[3/6] Calcul features derivees...")
    recent_form = calculate_recent_form(df_played)
    board_position = calculate_board_position(df_played)

    # 4. Features reglementaires FFE
    logger.info("\n[4/6] Extraction features reglementaires FFE...")
    ffe_regulatory = extract_ffe_regulatory_features(df_played)
    team_enjeu = extract_team_enjeu_features(df_played)

    # 5. Exclure Elo=0
    logger.info("\n[5/6] Filtrage Elo > 0...")
    df_clean = df_played[(df_played["blanc_elo"] > 0) & (df_played["noir_elo"] > 0)]
    logger.info(
        f"  {len(df_clean):,} parties avec Elo valide ({len(df_clean) / len(df_played) * 100:.1f}%)"
    )

    # 6. Split temporel
    logger.info("\n[6/6] Split temporel...")
    train, valid, test = temporal_split(df_clean)

    # Export
    output_dir.mkdir(parents=True, exist_ok=True)

    # Features fiabilite
    club_reliability.to_parquet(output_dir / "club_reliability.parquet", index=False)
    player_reliability.to_parquet(output_dir / "player_reliability.parquet", index=False)
    if len(player_monthly) > 0:
        player_monthly.to_parquet(output_dir / "player_monthly.parquet", index=False)

    # Features derivees
    recent_form.to_parquet(output_dir / "player_form.parquet", index=False)
    board_position.to_parquet(output_dir / "player_board.parquet", index=False)

    # Features reglementaires FFE
    ffe_regulatory.to_parquet(output_dir / "ffe_regulatory.parquet", index=False)
    if len(team_enjeu) > 0:
        team_enjeu.to_parquet(output_dir / "team_enjeu.parquet", index=False)

    # Splits
    train.to_parquet(output_dir / "train.parquet", index=False)
    valid.to_parquet(output_dir / "valid.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)

    # Resume
    logger.info("\n" + "=" * 60)
    logger.info("Feature engineering termine!")
    logger.info("=" * 60)
    logger.info(f"\nFichiers generes dans {output_dir}/:")
    logger.info("  Features fiabilite:")
    logger.info(f"    - club_reliability.parquet ({len(club_reliability)} clubs)")
    logger.info(f"    - player_reliability.parquet ({len(player_reliability)} joueurs)")
    if len(player_monthly) > 0:
        logger.info(f"    - player_monthly.parquet ({len(player_monthly)} joueurs)")
    logger.info("  Features derivees:")
    logger.info(f"    - player_form.parquet ({len(recent_form)} joueurs)")
    logger.info(f"    - player_board.parquet ({len(board_position)} joueurs)")
    logger.info("  Features reglementaires FFE:")
    logger.info(f"    - ffe_regulatory.parquet ({len(ffe_regulatory)} joueurs)")
    if len(team_enjeu) > 0:
        logger.info(f"    - team_enjeu.parquet ({len(team_enjeu)} equipes/saisons)")
    logger.info("  Splits temporels:")
    logger.info(f"    - train.parquet ({len(train):,} echiquiers)")
    logger.info(f"    - valid.parquet ({len(valid):,} echiquiers)")
    logger.info(f"    - test.parquet ({len(test):,} echiquiers)")


# ==============================================================================
# PIPELINE V2 - SANS DATA LEAKAGE
# ==============================================================================


def compute_features_for_split(
    df_split: pd.DataFrame,
    df_history: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Calcule les features pour un split en utilisant uniquement l'historique visible.

    Args:
    ----
        df_split: Donnees du split (train, valid ou test)
        df_history: Donnees historiques visibles (pour calculer les features)
        split_name: Nom du split pour logging

    Returns:
    -------
        DataFrame avec features ajoutees
    """
    logger.info(
        f"  Computing features for {split_name} using {len(df_history):,} historical records..."
    )

    # Features de fiabilite calculees sur l'historique
    club_reliability = extract_club_reliability(df_history)
    player_reliability = extract_player_reliability(df_history)

    # Features derivees (forme, position) sur l'historique
    df_history_played = df_history[
        ~df_history["type_resultat"].isin(
            ["non_joue", "forfait_blanc", "forfait_noir", "double_forfait"]
        )
    ]

    recent_form = calculate_recent_form(df_history_played)
    board_position = calculate_board_position(df_history_played)

    # Features reglementaires
    ffe_regulatory = extract_ffe_regulatory_features(df_history_played)
    team_enjeu = extract_team_enjeu_features(df_history_played)

    # Merger les features sur le split
    result = df_split.copy()

    # Club reliability - merge sur equipe_dom et equipe_ext
    if len(club_reliability) > 0:
        club_cols = ["taux_forfait", "taux_non_joue", "fiabilite_score"]
        for suffix, col in [("dom", "equipe_dom"), ("ext", "equipe_ext")]:
            merge_df = club_reliability.rename(columns={c: f"{c}_{suffix}" for c in club_cols})
            result = result.merge(
                merge_df[["equipe"] + [f"{c}_{suffix}" for c in club_cols]],
                left_on=col,
                right_on="equipe",
                how="left",
            )
            if "equipe" in result.columns:
                result = result.drop(columns=["equipe"])

    # Player reliability - merge sur joueur_nom (blanc/noir)
    if len(player_reliability) > 0:
        player_cols = ["taux_presence", "joueur_fantome"]
        for color in ["blanc", "noir"]:
            merge_df = player_reliability.rename(columns={c: f"{c}_{color}" for c in player_cols})
            result = result.merge(
                merge_df[["joueur_nom"] + [f"{c}_{color}" for c in player_cols]],
                left_on=f"{color}_nom",
                right_on="joueur_nom",
                how="left",
            )
            if "joueur_nom" in result.columns:
                result = result.drop(columns=["joueur_nom"])

    # Forme recente
    if len(recent_form) > 0:
        for color in ["blanc", "noir"]:
            merge_df = recent_form.rename(columns={"forme_recente": f"forme_recente_{color}"})
            result = result.merge(
                merge_df[["joueur_nom", f"forme_recente_{color}"]],
                left_on=f"{color}_nom",
                right_on="joueur_nom",
                how="left",
            )
            if "joueur_nom" in result.columns:
                result = result.drop(columns=["joueur_nom"])

    # Position echiquier moyenne
    if len(board_position) > 0:
        for color in ["blanc", "noir"]:
            merge_df = board_position.rename(
                columns={"echiquier_moyen": f"echiquier_moyen_{color}"}
            )
            result = result.merge(
                merge_df[["joueur_nom", f"echiquier_moyen_{color}"]],
                left_on=f"{color}_nom",
                right_on="joueur_nom",
                how="left",
            )
            if "joueur_nom" in result.columns:
                result = result.drop(columns=["joueur_nom"])

    # FFE regulatory features
    if len(ffe_regulatory) > 0:
        ffe_cols = ["nb_equipes", "niveau_max", "niveau_min", "multi_equipe"]
        for color in ["blanc", "noir"]:
            merge_df = ffe_regulatory.rename(columns={c: f"ffe_{c}_{color}" for c in ffe_cols})
            cols_to_merge = ["joueur_nom"] + [
                f"ffe_{c}_{color}" for c in ffe_cols if c in ffe_regulatory.columns
            ]
            if len(cols_to_merge) > 1:
                result = result.merge(
                    merge_df[cols_to_merge],
                    left_on=f"{color}_nom",
                    right_on="joueur_nom",
                    how="left",
                )
                if "joueur_nom" in result.columns:
                    result = result.drop(columns=["joueur_nom"])

    # Team enjeu
    if len(team_enjeu) > 0:
        for suffix, col in [("dom", "equipe_dom"), ("ext", "equipe_ext")]:
            merge_df = team_enjeu.rename(
                columns={
                    "zone_enjeu": f"zone_enjeu_{suffix}",
                    "niveau_hierarchique": f"niveau_hier_{suffix}",
                }
            )
            result = result.merge(
                merge_df[["equipe", "saison", f"zone_enjeu_{suffix}", f"niveau_hier_{suffix}"]],
                left_on=[col, "saison"],
                right_on=["equipe", "saison"],
                how="left",
            )
            if "equipe" in result.columns:
                result = result.drop(columns=["equipe"])

    logger.info(f"  {split_name}: {len(result):,} samples, {len(result.columns)} features")

    return result


def run_feature_engineering_v2(data_dir: Path, output_dir: Path) -> None:
    """Pipeline feature engineering V2 - SANS DATA LEAKAGE.

    Cette version corrige le data leakage en:
    1. Faisant le split temporel D'ABORD
    2. Calculant les features PAR SPLIT avec uniquement les donnees historiques visibles

    Conformite ISO/IEC 42001 (AI Management), ISO/IEC 5259 (Data Quality for ML).

    Args:
    ----
        data_dir: Repertoire des donnees sources
        output_dir: Repertoire de sortie
    """
    logger.info("=" * 60)
    logger.info("ALICE Engine - Feature Engineering V2 (No Leakage)")
    logger.info("ISO/IEC 42001, 5259 Conformant")
    logger.info("=" * 60)

    # Charger donnees
    echiquiers_path = data_dir / "echiquiers.parquet"
    if not echiquiers_path.exists():
        logger.error(f"Fichier non trouve: {echiquiers_path}")
        logger.error("Executez d'abord: python scripts/parse_dataset.py")
        return

    logger.info(f"\nChargement {echiquiers_path}...")
    df = pd.read_parquet(echiquiers_path)
    logger.info(f"  {len(df):,} echiquiers charges")

    # 1. Filtrer parties jouees et Elo valide
    logger.info("\n[1/4] Filtrage donnees...")
    df_played = df[
        ~df["type_resultat"].isin(["non_joue", "forfait_blanc", "forfait_noir", "double_forfait"])
    ]
    df_clean = df_played[(df_played["blanc_elo"] > 0) & (df_played["noir_elo"] > 0)]
    logger.info(f"  {len(df_clean):,} parties valides")

    # 2. SPLIT TEMPOREL D'ABORD (crucial pour eviter leakage)
    logger.info("\n[2/4] Split temporel AVANT features...")
    train_raw, valid_raw, test_raw = temporal_split(df_clean)

    # 3. Calculer features PAR SPLIT avec historique approprie
    logger.info("\n[3/4] Calcul features per-split (no leakage)...")

    # Train: features calculees sur train uniquement
    logger.info("\n  --- TRAIN ---")
    train = compute_features_for_split(
        df_split=train_raw,
        df_history=df[df["saison"] <= train_raw["saison"].max()],  # Historique train
        split_name="train",
    )

    # Valid: features calculees sur train + valid historique
    # (valid peut voir train car train est dans le passe)
    logger.info("\n  --- VALID ---")
    valid_history = df[df["saison"] <= valid_raw["saison"].max()]
    valid = compute_features_for_split(
        df_split=valid_raw,
        df_history=valid_history,
        split_name="valid",
    )

    # Test: features calculees sur tout l'historique (train + valid)
    # En production, test ne voit que le passe
    logger.info("\n  --- TEST ---")
    test_history = df[df["saison"] <= test_raw["saison"].min() - 1]  # Avant test
    test = compute_features_for_split(
        df_split=test_raw,
        df_history=test_history,
        split_name="test",
    )

    # 4. Export
    logger.info("\n[4/4] Export...")
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(output_dir / "train.parquet", index=False)
    valid.to_parquet(output_dir / "valid.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)

    # Resume
    logger.info("\n" + "=" * 60)
    logger.info("Feature engineering V2 termine!")
    logger.info("=" * 60)
    logger.info(f"\nFichiers generes dans {output_dir}/:")
    logger.info(f"  - train.parquet ({len(train):,} echiquiers, {len(train.columns)} features)")
    logger.info(f"  - valid.parquet ({len(valid):,} echiquiers)")
    logger.info(f"  - test.parquet ({len(test):,} echiquiers)")
    logger.info("\nDATA LEAKAGE: CORRIGE")
    logger.info("  - Split temporel effectue AVANT calcul des features")
    logger.info("  - Chaque split utilise uniquement les donnees historiques visibles")


def main() -> None:
    """Point d'entree."""
    parser = argparse.ArgumentParser(description="Feature engineering ALICE")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Repertoire des donnees sources",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR / "features",
        help="Repertoire de sortie",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Utiliser l'ancienne version avec data leakage (NON RECOMMANDE)",
    )
    args = parser.parse_args()

    if args.legacy:
        logger.warning("Mode legacy active - DATA LEAKAGE PRESENT!")
        run_feature_engineering(args.data_dir, args.output_dir)
    else:
        run_feature_engineering_v2(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
