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

    if df.empty:
        return pd.DataFrame()

    # Filtrer saison si spécifiée
    if saison is not None:
        df = df[df["saison"] == saison].copy()

    if df.empty:
        return pd.DataFrame()

    presence_data = []

    # Collecter présence par joueur/saison
    for s in df["saison"].unique():
        df_saison = df[df["saison"] == s]
        nb_rondes_total = df_saison["ronde"].nunique()

        if nb_rondes_total == 0:
            continue

        ronde_max = df_saison["ronde"].max()

        for couleur in ["blanc", "noir"]:
            nom_col = f"{couleur}_nom"
            if nom_col not in df_saison.columns:
                continue

            for joueur, group in df_saison.groupby(nom_col):
                rondes_jouees = group["ronde"].unique()
                nb_rondes_jouees = len(rondes_jouees)
                derniere_ronde = max(rondes_jouees)

                # Calculs
                taux = nb_rondes_jouees / nb_rondes_total
                derniere_presence = ronde_max - derniere_ronde

                # Classification régularité (seuils documentés)
                if taux > 0.7:
                    regularite = "regulier"
                elif taux >= 0.3:
                    regularite = "occasionnel"
                else:
                    regularite = "rare"

                presence_data.append(
                    {
                        "joueur_nom": joueur,
                        "saison": s,
                        "taux_presence_saison": round(taux, 3),
                        "derniere_presence": derniere_presence,
                        "nb_rondes_jouees": nb_rondes_jouees,
                        "nb_rondes_total": nb_rondes_total,
                        "regularite": regularite,
                    }
                )

    # Dédupliquer (joueur peut jouer blanc ET noir)
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

    logger.info(f"  {len(result)} joueurs avec features présence")
    return result
