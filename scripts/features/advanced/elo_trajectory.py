"""Trajectoire Elo (progression/régression) - ISO 5055/5259.

Ce module implémente le calcul de la trajectoire Elo par joueur.
Feature importante: un joueur en progression peut surperformer son Elo actuel.

Classification trajectoire (SEUILS DOCUMENTÉS ISO 5259):
- "progression": delta > +50 pts - Joueur en amélioration significative
- "regression": delta < -50 pts - Joueur en déclin significatif
- "stable": -50 <= delta <= +50 - Performance constante

Justification seuil ±50 pts:
- Correspond à environ 1 catégorie FFE (50-100 pts par catégorie)
- Variation significative mesurable sur 6 matchs (window par défaut)
- Référence FIDE Handbook: K-factor implique ~12-20 pts par partie
- Statistiquement: ~1 écart-type de variation Elo sur une saison

Momentum (SEUIL DOCUMENTÉ):
- Normalisé sur [-1, 1] avec diviseur 200 pts
- 200 pts = progression/régression extrême sur 6 matchs
- Permet comparaison entre joueurs

Sources:
- EloMetrics IEEE 2025 (https://ieeexplore.ieee.org/document/10879733/)
- AI Sports Predictions 2025 (ainewshub.org)
- FIDE Handbook - Elo Rating System (K-factor rules)

Conformité:
- ISO 5055: Module <300 lignes, responsabilité unique
- ISO 5259: Features calculées depuis données réelles, seuils documentés
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_elo_trajectory(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Calcule la trajectoire Elo par joueur (vectorise).

    ISO 5259: Trajectoire calculee depuis historique Elo reel.
    """
    logger.info("Calcul trajectoire Elo (window=%d)...", window)

    if "date" not in df.columns:
        logger.warning("  Colonne date manquante pour trajectoire Elo")
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    df_dated = df[df["date"].notna()].copy()
    df_dated["date"] = pd.to_datetime(df_dated["date"])

    if df_dated.empty:
        return pd.DataFrame()

    # Build long-form (joueur, elo, date) from both colors
    parts: list[pd.DataFrame] = []
    for couleur in ["blanc", "noir"]:
        nom_col, elo_col = f"{couleur}_nom", f"{couleur}_elo"
        if nom_col not in df_dated.columns or elo_col not in df_dated.columns:
            continue
        sub = df_dated[[nom_col, elo_col, "date"]].dropna(subset=[nom_col, elo_col]).copy()
        sub.columns = ["joueur_nom", "elo", "date"]
        parts.append(sub)

    if not parts:
        return pd.DataFrame()

    all_data = pd.concat(parts, ignore_index=True).sort_values("date")

    # Keep only the last `window` games per player for trajectory computation
    all_data = all_data.groupby("joueur_nom").tail(window)

    # Per joueur: first elo, last elo, count (within the window)
    stats = (
        all_data.groupby("joueur_nom")
        .agg(
            elo_debut=("elo", "first"),
            elo_fin=("elo", "last"),
            nb_matchs=("elo", "count"),
        )
        .reset_index()
    )

    # Filter by minimum window
    stats = stats[stats["nb_matchs"] >= window].copy()
    stats["elo_delta"] = stats["elo_fin"] - stats["elo_debut"]

    # Vectorized classification
    stats["elo_trajectory"] = pd.cut(
        stats["elo_delta"],
        bins=[-float("inf"), -50, 50, float("inf")],
        labels=["regression", "stable", "progression"],
    ).astype(str)

    stats["momentum"] = stats["elo_delta"].clip(-200, 200) / 200
    # Zero out momentum for stable players
    stats.loc[stats["elo_trajectory"] == "stable", "momentum"] = 0.0

    # No dedup needed: all_data already merged both colors before groupby,
    # so groupby("joueur_nom") already produces one row per player.
    logger.info("  %d joueurs avec trajectoire Elo", len(stats))
    return stats
