"""Présence joueur (ALI) - ISO 5055/5259.

Ce module calcule les features de présence pour ALI
(Adversarial Lineup Inference).

Features (DOCUMENTÉES ISO 5259):
- taux_presence_saison: % rondes jouées cette saison [0, 1]
- derniere_presence: Nombre de rondes depuis dernière apparition
- regularite: Classification basée sur taux de présence
  - 'regulier': > 70% des rondes (titulaire stable)
  - 'occasionnel': 30-70% des rondes (rotation)
  - 'rare': < 30% des rondes (remplaçant, blessé, indisponible)

Justification seuils:
- 70%: Joueur présent 6+ rondes sur 9 = titulaire
- 30%: Joueur présent 2-3 rondes sur 9 = remplaçant occasionnel
- Référence: Analyse statistique compositions interclubs FFE

Conformité:
- ISO 5055: Module <300 lignes, responsabilité unique
- ISO 5259: Features depuis données réelles, seuils documentés
- ISO 42001: Prédictions AI traçables
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_presence_features(
    df: pd.DataFrame,
    saison: int | None = None,
) -> pd.DataFrame:
    """Calcule les features de présence par joueur.

    Args:
    ----
        df: DataFrame échiquiers avec colonnes joueur, ronde, saison
        saison: Saison à analyser (None = toutes)

    Returns:
    -------
        DataFrame avec colonnes:
        - joueur_nom: nom joueur
        - saison: saison concernée
        - taux_presence_saison: float [0, 1]
        - derniere_presence: int (nb rondes depuis dernier match)
        - nb_rondes_jouees: int
        - nb_rondes_total: int
        - regularite: 'regulier', 'occasionnel', 'rare'

    ISO 5259: Présence calculée depuis historique réel.
    """
    logger.info("Calcul features présence joueur...")

    df_filtered = _filter_by_saison(df, saison)
    if df_filtered.empty:
        return pd.DataFrame()

    presence_data = _collect_presence_data(df_filtered)
    result = _aggregate_presence_data(presence_data)

    logger.info(f"  {len(result)} joueurs avec features présence")
    return result


def _filter_by_saison(df: pd.DataFrame, saison: int | None) -> pd.DataFrame:
    """Filtre le DataFrame par saison."""
    if df.empty:
        return pd.DataFrame()
    if saison is not None:
        return df[df["saison"] == saison].copy()
    return df


def _collect_presence_data(df: pd.DataFrame) -> list[dict]:
    """Collecte les donnees de presence par joueur/saison."""
    presence_data = []

    for s in df["saison"].unique():
        df_saison = df[df["saison"] == s]
        nb_rondes_total = df_saison["ronde"].nunique()
        if nb_rondes_total == 0:
            continue

        ronde_max = df_saison["ronde"].max()
        _collect_saison_presence(df_saison, s, nb_rondes_total, ronde_max, presence_data)

    return presence_data


def _collect_saison_presence(
    df_saison: pd.DataFrame, s: int, nb_rondes_total: int, ronde_max: int, presence_data: list
) -> None:
    """Collecte presence pour une saison."""
    for couleur in ["blanc", "noir"]:
        nom_col = f"{couleur}_nom"
        if nom_col not in df_saison.columns:
            continue

        for joueur, group in df_saison.groupby(nom_col):
            entry = _compute_player_presence(joueur, group, s, nb_rondes_total, ronde_max)
            presence_data.append(entry)


def _compute_player_presence(
    joueur: str, group: pd.DataFrame, saison: int, nb_rondes_total: int, ronde_max: int
) -> dict:
    """Calcule les stats de presence d'un joueur."""
    rondes_jouees = group["ronde"].unique()
    nb_rondes_jouees = len(rondes_jouees)
    taux = nb_rondes_jouees / nb_rondes_total
    regularite = _classify_regularite(taux)

    return {
        "joueur_nom": joueur,
        "saison": saison,
        "taux_presence_saison": round(taux, 3),
        "derniere_presence": ronde_max - max(rondes_jouees),
        "nb_rondes_jouees": nb_rondes_jouees,
        "nb_rondes_total": nb_rondes_total,
        "regularite": regularite,
    }


def _classify_regularite(taux: float) -> str:
    """Classifie la regularite du joueur."""
    if taux > 0.7:
        return "regulier"
    if taux >= 0.3:
        return "occasionnel"
    return "rare"


def _aggregate_presence_data(presence_data: list[dict]) -> pd.DataFrame:
    """Agrege les donnees de presence."""
    result = pd.DataFrame(presence_data)
    if not result.empty:
        result = (
            result.groupby(["joueur_nom", "saison"])
            .agg(
                taux_presence_saison=("taux_presence_saison", "max"),
                derniere_presence=("derniere_presence", "min"),
                nb_rondes_jouees=("nb_rondes_jouees", "max"),
                nb_rondes_total=("nb_rondes_total", "first"),
                regularite=("regularite", "first"),
            )
            .reset_index()
        )
    return result
