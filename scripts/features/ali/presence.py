"""Presence joueur (ALI) - ISO 5055/5259.

Features:
- taux_presence_saison: % rondes jouees cette saison [0, 1]
- derniere_presence: Nombre de rondes depuis derniere apparition
- regularite: 'regulier' (>70%), 'occasionnel' (30-70%), 'rare' (<30%)

Vectorized implementation v2 — no Python loops over players.

Document ID: ALICE-FEA-PRESENCE-001
Version: 2.0.0
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_presence_features(
    df: pd.DataFrame,
    saison: int | None = None,
) -> pd.DataFrame:
    """Calcule les features de presence par joueur (vectorise).

    Returns
    -------
        DataFrame avec colonnes: joueur_nom, saison,
        taux_presence_saison, derniere_presence, regularite
    """
    logger.info("Calcul features presence joueur...")

    if df.empty:
        return pd.DataFrame()

    df_filtered = df[df["saison"] == saison].copy() if saison else df.copy()
    if df_filtered.empty:
        return pd.DataFrame()

    parts = []
    for color in ("blanc", "noir"):
        nom_col = f"{color}_nom"
        if nom_col not in df_filtered.columns:
            continue
        part = _compute_presence_vectorized(df_filtered, nom_col)
        if not part.empty:
            parts.append(part)

    if not parts:
        return pd.DataFrame()

    # Deduplicate: same player may appear as blanc and noir
    result = pd.concat(parts, ignore_index=True)
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

    logger.info("  %d joueurs avec features presence", len(result))
    return result


def _compute_presence_vectorized(df: pd.DataFrame, nom_col: str) -> pd.DataFrame:
    """Vectorized presence for one color."""
    sub = df[[nom_col, "saison", "ronde"]].dropna(subset=[nom_col]).copy()
    sub.columns = ["joueur_nom", "saison", "ronde"]

    # Rondes total per saison
    rondes_total = sub.groupby("saison")["ronde"].nunique().reset_index()
    rondes_total.columns = ["saison", "nb_rondes_total"]

    ronde_max = sub.groupby("saison")["ronde"].max().reset_index()
    ronde_max.columns = ["saison", "ronde_max"]

    # Per joueur×saison: nb rondes jouees + derniere ronde
    stats = (
        sub.groupby(["joueur_nom", "saison"])
        .agg(
            nb_rondes_jouees=("ronde", "nunique"),
            derniere_ronde=("ronde", "max"),
        )
        .reset_index()
    )

    stats = stats.merge(rondes_total, on="saison", how="left")
    stats = stats.merge(ronde_max, on="saison", how="left")

    stats["taux_presence_saison"] = (stats["nb_rondes_jouees"] / stats["ronde_max"]).round(3)
    stats["derniere_presence"] = stats["ronde_max"] - stats["derniere_ronde"]
    stats["regularite"] = stats["taux_presence_saison"].apply(_classify_regularite)

    return stats[
        [
            "joueur_nom",
            "saison",
            "taux_presence_saison",
            "derniere_presence",
            "nb_rondes_jouees",
            "nb_rondes_total",
            "regularite",
        ]
    ]


def _classify_regularite(taux: float) -> str:
    """Classifie la regularite du joueur."""
    if taux > 0.7:
        return "regulier"
    if taux >= 0.3:
        return "occasionnel"
    return "rare"
