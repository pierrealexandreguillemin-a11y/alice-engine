"""Patterns de sélection (ALI) - ISO 5055/5259.

Ce module calcule les patterns de sélection des joueurs
pour prédire leur présence future.

Features (DOCUMENTÉES ISO 5259):
- role_type: Classification du rôle dans l'équipe
  - 'titulaire': Joue > 70% des rondes, position stable
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
            nom_col = f"{couleur}_nom"
            if nom_col not in df_saison.columns:
                continue

            for joueur, group in df_saison.groupby(nom_col):
                echiquiers = group["echiquier"].tolist()
                rondes = group["ronde"].unique()

                nb_rondes_jouees = len(rondes)
                nb_echiquiers_diff = len(set(echiquiers))

                # Échiquier préféré (mode)
                echiquier_counts = pd.Series(echiquiers).value_counts()
                echiquier_prefere = echiquier_counts.index[0]

                # Flexibilité
                if len(echiquiers) > 1:
                    flexibilite = float(np.std(echiquiers))
                else:
                    flexibilite = 0.0

                # Classification rôle
                taux_presence = nb_rondes_jouees / nb_rondes_total

                if flexibilite > 2 and nb_echiquiers_diff >= 3:
                    role = "polyvalent"
                elif taux_presence > 0.7:
                    role = "titulaire"
                elif taux_presence < 0.3:
                    role = "remplacant"
                else:
                    role = "titulaire"  # 30-70% = titulaire avec rotation

                pattern_data.append(
                    {
                        "joueur_nom": joueur,
                        "saison": saison,
                        "role_type": role,
                        "echiquier_prefere": int(echiquier_prefere),
                        "flexibilite_echiquier": round(flexibilite, 2),
                        "nb_echiquiers_differents": nb_echiquiers_diff,
                        "taux_presence": round(taux_presence, 3),
                    }
                )

    # Dédupliquer
    result = pd.DataFrame(pattern_data)
    if not result.empty:
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
