"""Features de fiabilité club/joueur - ISO 5055/25012.

Ce module extrait les features de fiabilité basées sur les forfaits
et les patterns de présence.

Conformité:
- ISO 5055: Module <300 lignes, responsabilité unique
- ISO 25012: Qualité données (fiabilité, complétude)
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def extract_club_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait les features de fiabilité par club.

    Utilise les non_joue et forfaits pour identifier les clubs défaillants.

    Args:
    ----
        df: DataFrame échiquiers complet (avant filtrage)

    Returns:
    -------
        DataFrame avec colonnes:
        - equipe: nom du club
        - taux_forfait: % de forfaits sur l'historique
        - taux_non_joue: % de matchs non joués
        - fiabilite_score: score composite (1 - taux_défaillance)

    ISO 25012: Fiabilité mesurée depuis données réelles.
    """
    logger.info("Extraction features fiabilité clubs...")

    if df.empty or "type_resultat" not in df.columns:
        logger.warning("  DataFrame vide ou colonnes manquantes")
        return pd.DataFrame()

    forfaits_types = ["forfait_blanc", "forfait_noir", "double_forfait"]

    club_stats = []

    for equipe_col in ["equipe_dom", "equipe_ext"]:
        if equipe_col not in df.columns:
            continue

        stats = df.groupby(equipe_col).agg(
            total_matchs=("type_resultat", "count"),
            forfaits=("type_resultat", lambda x: x.isin(forfaits_types).sum()),
            non_joue=("type_resultat", lambda x: (x == "non_joue").sum()),
        )
        stats = stats.reset_index().rename(columns={equipe_col: "equipe"})
        club_stats.append(stats)

    if not club_stats:
        return pd.DataFrame()

    # Fusionner dom + ext
    all_stats = pd.concat(club_stats).groupby("equipe").sum().reset_index()

    # Calculer taux (éviter division par zéro)
    all_stats["taux_forfait"] = all_stats["forfaits"] / all_stats["total_matchs"].replace(0, 1)
    all_stats["taux_non_joue"] = all_stats["non_joue"] / all_stats["total_matchs"].replace(0, 1)
    all_stats["fiabilite_score"] = 1 - (
        all_stats["taux_forfait"] * 0.7 + all_stats["taux_non_joue"] * 0.3
    )

    logger.info(f"  {len(all_stats)} clubs avec stats fiabilité")

    return all_stats[["equipe", "taux_forfait", "taux_non_joue", "fiabilite_score"]]


def extract_player_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait les features de fiabilité par joueur.

    Analyse les patterns de présence/absence pour chaque joueur.

    Args:
    ----
        df: DataFrame échiquiers complet

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom: nom complet du joueur
        - nb_matchs: nombre total de matchs
        - taux_presence: % de matchs effectivement joués
        - joueur_fantome: flag si < 20% présence

    ISO 25012: Complétude mesurée par taux de présence.
    """
    logger.info("Extraction features fiabilité joueurs...")

    if df.empty or "type_resultat" not in df.columns:
        logger.warning("  DataFrame vide ou colonnes manquantes")
        return pd.DataFrame()

    player_stats = []

    for couleur in ["blanc", "noir"]:
        nom_col = f"{couleur}_nom"
        if nom_col not in df.columns:
            continue

        df_couleur = df[df[nom_col].notna()].copy()

        # Identifier les matchs joués vs non joués
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

    if not player_stats:
        return pd.DataFrame()

    # Fusionner blanc + noir
    all_stats = pd.concat(player_stats).groupby("joueur_nom").sum().reset_index()

    # Calculer taux (éviter division par zéro)
    all_stats["taux_presence"] = all_stats["matchs_joues"] / all_stats["nb_matchs"].replace(0, 1)
    all_stats["joueur_fantome"] = all_stats["taux_presence"] < 0.2

    logger.info(f"  {len(all_stats)} joueurs avec stats fiabilité")
    logger.info(f"  {all_stats['joueur_fantome'].sum()} joueurs fantômes (<20% présence)")

    return all_stats[["joueur_nom", "nb_matchs", "taux_presence", "joueur_fantome"]]


def extract_player_monthly_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait les patterns de disponibilité mensuelle par joueur.

    Détecte les joueurs indisponibles certains mois (vacances, pro, etc.)

    Args:
    ----
        df: DataFrame échiquiers avec colonne 'date'

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom: nom complet
        - dispo_mois_1 ... dispo_mois_12: taux présence par mois

    ISO 5259: Feature temporelle pour prédiction disponibilité.
    """
    logger.info("Extraction patterns mensuels joueurs...")

    if df.empty or "date" not in df.columns:
        logger.warning("  Colonnes date manquantes, skip patterns mensuels")
        return pd.DataFrame()

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
        if nom_col not in df_dated.columns:
            continue

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
