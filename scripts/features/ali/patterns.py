"""Patterns de sélection (ALI) - ISO 5055/5259.

Ce module calcule les patterns de sélection des joueurs
pour prédire leur présence future.

Features (DOCUMENTÉES ISO 5259):
- role_type: Classification du rôle dans l'équipe
  - 'titulaire': Joue > 70% des rondes, position stable
  - 'rotation': Joue 30-70% des rondes, rotation régulière
  - 'remplacant': Joue < 30% des rondes, utilisé ponctuellement
  - 'polyvalent': Joue sur plusieurs échiquiers différents (std > 2)
- echiquier_prefere: Échiquier modal (le plus fréquent)
- flexibilite_echiquier: Écart-type des positions jouées
  - 0: Toujours même échiquier
  - > 2: Très flexible, peut jouer partout

Justification seuils:
- 70%/30%: Même seuils que regularité (cohérence)
- std > 2: Un joueur jouant éch. 1-5 régulièrement = polyvalent

Conformité:
- ISO 5055: Module <300 lignes, responsabilité unique
- ISO 5259: Features depuis données réelles, seuils documentés
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_selection_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les patterns de sélection par joueur.

    Args:
    ----
        df: DataFrame échiquiers avec colonnes joueur, echiquier, ronde

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom: nom joueur
        - saison: saison concernée
        - role_type: 'titulaire', 'remplacant', 'polyvalent'
        - echiquier_prefere: int (échiquier modal)
        - flexibilite_echiquier: float (std positions)
        - nb_echiquiers_differents: int

    ISO 5259: Patterns calculés depuis compositions réelles.
    """
    logger.info("Calcul patterns sélection joueur...")

    if df.empty or "echiquier" not in df.columns:
        return pd.DataFrame()

    pattern_data = []

    for saison in df["saison"].unique():
        df_saison = df[df["saison"] == saison]
        nb_rondes_total = df_saison["ronde"].nunique()
        if nb_rondes_total == 0:
            continue

        for couleur in ["blanc", "noir"]:
            _process_color_patterns(df_saison, couleur, saison, nb_rondes_total, pattern_data)

    return _deduplicate_patterns(pattern_data)


def _process_color_patterns(
    df_saison: pd.DataFrame,
    couleur: str,
    saison: int,
    nb_rondes_total: int,
    pattern_data: list[dict],
) -> None:
    """Traite les patterns pour une couleur donnee."""
    nom_col = f"{couleur}_nom"
    if nom_col not in df_saison.columns:
        return

    for joueur, group in df_saison.groupby(nom_col):
        pattern = _analyze_player_pattern(joueur, group, saison, nb_rondes_total)
        pattern_data.append(pattern)


def _analyze_player_pattern(
    joueur: str,
    group: pd.DataFrame,
    saison: int,
    nb_rondes_total: int,
) -> dict:
    """Analyse le pattern d'un joueur."""
    echiquiers = group["echiquier"].tolist()
    nb_rondes_jouees = group["ronde"].nunique()
    nb_echiquiers_diff = len(set(echiquiers))

    echiquier_prefere = pd.Series(echiquiers).value_counts().index[0]
    flexibilite = float(np.std(echiquiers)) if len(echiquiers) > 1 else 0.0
    taux_presence = nb_rondes_jouees / nb_rondes_total

    role = _classify_role(flexibilite, nb_echiquiers_diff, taux_presence)

    return {
        "joueur_nom": joueur,
        "saison": saison,
        "role_type": role,
        "echiquier_prefere": int(echiquier_prefere),
        "flexibilite_echiquier": round(flexibilite, 2),
        "nb_echiquiers_differents": nb_echiquiers_diff,
        "taux_presence": round(taux_presence, 3),
    }


def _classify_role(flexibilite: float, nb_echiquiers_diff: int, taux_presence: float) -> str:
    """Classifie le role du joueur."""
    if flexibilite > 2 and nb_echiquiers_diff >= 3:
        return "polyvalent"
    if taux_presence > 0.7:
        return "titulaire"
    if taux_presence < 0.3:
        return "remplacant"
    return "rotation"


def _deduplicate_patterns(pattern_data: list[dict]) -> pd.DataFrame:
    """Deduplique les patterns par joueur/saison."""
    result = pd.DataFrame(pattern_data)
    if result.empty:
        return result

    result = (
        result.groupby(["joueur_nom", "saison"])
        .agg(
            role_type=("role_type", "first"),
            echiquier_prefere=("echiquier_prefere", lambda x: int(pd.Series(x).mode().iloc[0])),
            flexibilite_echiquier=("flexibilite_echiquier", "max"),
            nb_echiquiers_differents=("nb_echiquiers_differents", "max"),
            taux_presence=("taux_presence", "max"),
        )
        .reset_index()
    )

    logger.info(f"  {len(result)} joueurs avec patterns sélection")
    return result
