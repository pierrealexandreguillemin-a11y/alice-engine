"""Pipeline extended features — ALI, CE, temporelles (ISO 5055).

Branche les modules ALI (presence, patterns) et CE (scenarios) dans
le pipeline de feature engineering, plus les features temporelles
et contextuelles manquantes.

Conformite ISO/IEC:
- 5055: Code maintenable (<300 lignes, SRP)
- 5259: Qualite donnees ML (features documentees)
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def extract_ali_features(df_history_played: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Extrait les features ALI (presence + patterns) depuis l'historique."""
    from scripts.features.ali.patterns import calculate_selection_patterns
    from scripts.features.ali.presence import calculate_presence_features

    features = {}

    # Presence: taux_presence_saison, derniere_presence, regularite
    presence = calculate_presence_features(df_history_played)
    if not presence.empty:
        # Agreger par joueur (derniere saison = plus pertinente)
        presence = presence.sort_values("saison").groupby("joueur_nom").last().reset_index()
        features["ali_presence"] = presence

    # Patterns: role_type, echiquier_prefere, flexibilite_echiquier
    patterns = calculate_selection_patterns(df_history_played)
    if not patterns.empty:
        patterns = patterns.sort_values("saison").groupby("joueur_nom").last().reset_index()
        features["ali_patterns"] = patterns

    return features


def extract_temporal_features(df_split: pd.DataFrame) -> None:
    """Ajoute les features temporelles directement sur le split (in-place).

    Features ajoutees:
    - phase_saison: debut (R1-R3), milieu (R4-R6), fin (R7+)
    - ronde_normalisee: ronde / max_ronde du groupe
    """
    if "ronde" not in df_split.columns:
        return

    df_split["phase_saison"] = pd.cut(
        df_split["ronde"],
        bins=[0, 3, 6, 99],
        labels=["debut", "milieu", "fin"],
    ).astype(str)

    # Ronde normalisee par groupe
    max_ronde = df_split.groupby(["saison", "competition", "division", "groupe"])[
        "ronde"
    ].transform("max")
    df_split["ronde_normalisee"] = (df_split["ronde"] / max_ronde).clip(0, 1)

    logger.info("  Features temporelles ajoutees: phase_saison, ronde_normalisee")


def extract_adversaire_niveau(
    df_split: pd.DataFrame,
    standings: pd.DataFrame,
) -> None:
    """Ajoute la position actuelle de l'adversaire au classement (in-place)."""
    if standings.empty or "equipe_ext" not in df_split.columns:
        return

    # Position de l'equipe ext dans le classement
    if "equipe" in standings.columns and "position" in standings.columns:
        latest = standings.sort_values("ronde").groupby("equipe").last().reset_index()
        mapping = dict(zip(latest["equipe"], latest["position"], strict=False))
        df_split["adversaire_niveau_dom"] = df_split["equipe_ext"].map(mapping)
        df_split["adversaire_niveau_ext"] = df_split["equipe_dom"].map(mapping)
        logger.info("  Feature adversaire_niveau ajoutee")


def merge_ali_features(
    result: pd.DataFrame,
    features: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge les features ALI sur le DataFrame cible."""
    from scripts.features.merge_helpers import merge_player_features

    if "ali_presence" in features:
        result = merge_player_features(
            result,
            features["ali_presence"],
            ["taux_presence_saison", "derniere_presence", "regularite"],
        )

    if "ali_patterns" in features:
        result = merge_player_features(
            result,
            features["ali_patterns"],
            ["role_type", "echiquier_prefere", "flexibilite_echiquier"],
        )

    return result
