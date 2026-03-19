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
    """Calcule les patterns de selection par joueur (vectorise).

    ISO 5259: Patterns calcules depuis compositions reelles.
    """
    logger.info("Calcul patterns sélection joueur...")

    if df.empty or "echiquier" not in df.columns:
        return pd.DataFrame()

    # Build long-form (joueur, saison, ronde, echiquier)
    parts: list[pd.DataFrame] = []
    for couleur in ["blanc", "noir"]:
        nom_col = f"{couleur}_nom"
        if nom_col not in df.columns:
            continue
        sub = df[[nom_col, "saison", "ronde", "echiquier"]].dropna(subset=[nom_col]).copy()
        sub = sub.rename(columns={nom_col: "joueur_nom"})
        parts.append(sub)

    if not parts:
        return pd.DataFrame()

    all_data = pd.concat(parts, ignore_index=True)

    # Deduplicate per (joueur, saison, ronde) BEFORE aggregation
    # A player appearing as blanc AND noir in the same round should count once
    all_data = all_data.drop_duplicates(subset=["joueur_nom", "saison", "ronde"])

    # Total rondes per saison
    rondes_per_saison = all_data.groupby("saison")["ronde"].nunique().reset_index()
    rondes_per_saison.columns = ["saison", "nb_rondes_total"]

    # Per joueur×saison: stats vectorisees (one row per joueur×saison after dedup)
    stats = (
        all_data.groupby(["joueur_nom", "saison"])
        .agg(
            nb_rondes_jouees=("ronde", "nunique"),
            flexibilite_echiquier=("echiquier", "std"),
            nb_echiquiers_differents=("echiquier", "nunique"),
        )
        .reset_index()
    )

    # echiquier_prefere: most frequent via idxmax (fully vectorized)
    counts = all_data.groupby(["joueur_nom", "saison", "echiquier"]).size().reset_index(name="cnt")
    idx = counts.groupby(["joueur_nom", "saison"])["cnt"].idxmax()
    mode_df = counts.loc[idx, ["joueur_nom", "saison", "echiquier"]].rename(
        columns={"echiquier": "echiquier_prefere"}
    )

    stats = stats.merge(mode_df, on=["joueur_nom", "saison"], how="left")
    stats = stats.merge(rondes_per_saison, on="saison", how="left")

    stats["taux_presence"] = (stats["nb_rondes_jouees"] / stats["nb_rondes_total"]).round(3)
    stats["flexibilite_echiquier"] = stats["flexibilite_echiquier"].fillna(0.0).round(2)
    stats["echiquier_prefere"] = stats["echiquier_prefere"].fillna(1).astype(int)

    # Classify role (vectorized via np.select)
    stats["role_type"] = np.where(
        (stats["flexibilite_echiquier"] > 2) & (stats["nb_echiquiers_differents"] >= 3),
        "polyvalent",
        np.where(
            stats["taux_presence"] > 0.7,
            "titulaire",
            np.where(stats["taux_presence"] < 0.3, "remplacant", "rotation"),
        ),
    )

    # No second dedup needed: drop_duplicates before groupby ensures
    # one row per (joueur_nom, saison) after the first aggregation.
    logger.info("  %d joueurs avec patterns sélection", len(stats))
    return stats
