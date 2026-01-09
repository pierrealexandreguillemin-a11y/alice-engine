"""Features FFE (Federation Francaise des Echecs) - ISO 5055.

Ce module contient les features reglementaires specifiques a la FFE:
- Historique de brulage
- Historique du noyau
- Features reglementaires joueur (multi-equipe, niveaux)

Conformite ISO/IEC:
- 5055: Code maintainable, responsabilite unique
- 5259: Qualite donnees ML
"""

from __future__ import annotations

import logging

import pandas as pd

from scripts.ffe_rules_features import (
    TypeCompetition,
    detecter_type_competition,
    get_niveau_equipe,
)

logger = logging.getLogger(__name__)


def build_historique_brulage(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Construit l'historique de brulage par joueur.

    Args:
    ----
        df: DataFrame avec colonnes blanc_nom, noir_nom, equipe_dom, equipe_ext

    Returns:
    -------
        Dict[joueur_nom, Dict[equipe, nb_parties]]
    """
    historique: dict[str, dict[str, int]] = {}

    for couleur in ["blanc", "noir"]:
        nom_col = f"{couleur}_nom"
        if nom_col not in df.columns:
            continue

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

    Args:
    ----
        df: DataFrame avec colonnes blanc_nom, noir_nom, equipe_dom, equipe_ext

    Returns:
    -------
        Dict[equipe, Set[joueurs]]
    """
    noyau: dict[str, set[str]] = {}

    for couleur in ["blanc", "noir"]:
        nom_col = f"{couleur}_nom"
        equipe_col = f"equipe_{'dom' if couleur == 'blanc' else 'ext'}"

        if nom_col not in df.columns or equipe_col not in df.columns:
            continue

        df_couleur = df[df[nom_col].notna()].copy()

        for _, row in df_couleur.iterrows():
            joueur = str(row[nom_col])
            equipe = str(row[equipe_col])

            if equipe not in noyau:
                noyau[equipe] = set()
            noyau[equipe].add(joueur)

    return noyau


def _extract_player_ffe_features(
    group: pd.DataFrame, equipe_col: str, competition_col: str
) -> dict[str, object]:
    """Extrait les features FFE pour un joueur.

    Args:
    ----
        group: DataFrame groupe pour un joueur
        equipe_col: Nom de la colonne equipe
        competition_col: Nom de la colonne competition

    Returns:
    -------
        Dict avec nb_equipes, niveau_max, niveau_min, type_competition, multi_equipe
    """
    equipes = group[equipe_col].unique() if equipe_col in group.columns else []
    niveaux = [get_niveau_equipe(str(eq)) for eq in equipes]

    type_comp = TypeCompetition.A02
    if competition_col in group.columns:
        competitions = group[competition_col].dropna()
        if len(competitions) > 0:
            type_comp = detecter_type_competition(str(competitions.mode().iloc[0]))

    return {
        "nb_equipes": len(equipes),
        "niveau_max": min(niveaux) if niveaux else 10,
        "niveau_min": max(niveaux) if niveaux else 10,
        "type_competition": type_comp.value,
        "multi_equipe": len(equipes) > 1,
    }


def extract_ffe_regulatory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait les features reglementaires FFE.

    Analyse les donnees pour extraire les features liees aux regles FFE:
    - Nombre d'equipes par joueur
    - Niveau min/max des equipes
    - Type de competition
    - Indicateur multi-equipe

    Args:
    ----
        df: DataFrame des parties jouees

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom
        - nb_equipes
        - niveau_max
        - niveau_min
        - type_competition
        - multi_equipe
    """
    logger.info("Extraction features reglementaires FFE...")

    if df.empty:
        return pd.DataFrame()

    features_data = []

    for couleur in ["blanc", "noir"]:
        nom_col = f"{couleur}_nom"
        equipe_col = f"equipe_{'dom' if couleur == 'blanc' else 'ext'}"

        if nom_col not in df.columns:
            continue

        df_couleur = df[df[nom_col].notna()].copy()

        for joueur, group in df_couleur.groupby(nom_col):
            feat = _extract_player_ffe_features(group, equipe_col, "competition")
            feat["joueur_nom"] = joueur
            features_data.append(feat)

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
    return result
