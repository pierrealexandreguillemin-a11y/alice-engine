"""Pipeline helpers pour feature engineering - ISO 5055.

Ce module contient les fonctions d'orchestration du pipeline de features:
- _extract_all_features: Extraction de toutes les features depuis l'historique
- _merge_all_features: Merge de toutes les features sur le DataFrame cible

Ces fonctions sont extraites de feature_engineering.py pour respecter
la limite de 300 lignes (ISO 5055).

Conformite ISO/IEC:
- 5055: Code maintainable (<300 lignes, SRP)
- 5259: Qualite donnees ML
"""

from __future__ import annotations

import pandas as pd

from scripts.features.advanced import (
    calculate_elo_trajectory,
    calculate_head_to_head,
    calculate_pressure_performance,
)
from scripts.features.ffe_features import extract_ffe_regulatory_features
from scripts.features.merge_helpers import (
    merge_club_reliability,
    merge_h2h_features,
    merge_player_features,
    merge_team_enjeu,
)
from scripts.features.performance import (
    calculate_board_position,
    calculate_color_performance,
    calculate_recent_form,
)
from scripts.features.reliability import (
    extract_club_reliability,
    extract_player_reliability,
)
from scripts.features.standings import calculate_standings, extract_team_enjeu_features


def extract_all_features(
    df_history: pd.DataFrame,
    df_history_played: pd.DataFrame,
    include_advanced: bool,
) -> dict[str, pd.DataFrame]:
    """Extrait toutes les features de l'historique.

    Args:
    ----
        df_history: DataFrame historique complet
        df_history_played: DataFrame parties jouees uniquement (sans forfaits)
        include_advanced: Inclure features avancees (H2H, fatigue, etc.)

    Returns:
    -------
        Dict[nom_feature, DataFrame] avec toutes les features calculees
    """
    # Calcul classement pour zones enjeu (ISO 5259 - position reelle)
    standings = calculate_standings(df_history_played)

    features = {
        "club_reliability": extract_club_reliability(df_history),
        "player_reliability": extract_player_reliability(df_history),
        "recent_form": calculate_recent_form(df_history_played),
        "board_position": calculate_board_position(df_history_played),
        "color_perf": calculate_color_performance(df_history_played),
        "ffe_regulatory": extract_ffe_regulatory_features(df_history_played),
        "team_enjeu": extract_team_enjeu_features(df_history_played, standings),
    }

    if include_advanced:
        features.update(
            {
                "h2h": calculate_head_to_head(df_history_played),
                "pressure": calculate_pressure_performance(df_history_played),
                "trajectory": calculate_elo_trajectory(df_history_played),
            }
        )

    return features


def merge_all_features(
    result: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    include_advanced: bool,
) -> pd.DataFrame:
    """Merge toutes les features sur le DataFrame cible.

    Args:
    ----
        result: DataFrame cible (copie du split)
        features: Dict des features extraites par extract_all_features
        include_advanced: Inclure features avancees

    Returns:
    -------
        DataFrame avec toutes les features mergees
    """
    # Club reliability
    result = merge_club_reliability(result, features["club_reliability"])

    # Player-based features
    result = merge_player_features(
        result, features["player_reliability"], ["taux_presence", "joueur_fantome"]
    )
    result = merge_player_features(
        result, features["recent_form"], ["forme_recente", "forme_tendance"]
    )
    result = merge_player_features(
        result, features["board_position"], ["echiquier_moyen", "echiquier_std"]
    )
    result = merge_player_features(
        result,
        features["color_perf"],
        ["score_blancs", "score_noirs", "couleur_preferee", "data_quality"],
    )

    # FFE regulatory features
    result = merge_player_features(
        result,
        features["ffe_regulatory"],
        ["nb_equipes", "niveau_max", "niveau_min", "multi_equipe"],
        prefix="ffe_",
    )

    # Team enjeu
    result = merge_team_enjeu(result, features["team_enjeu"])

    # Advanced features
    if include_advanced:
        result = merge_player_features(
            result,
            features.get("trajectory", pd.DataFrame()),
            ["elo_trajectory", "momentum"],
        )
        result = merge_player_features(
            result,
            features.get("pressure", pd.DataFrame()),
            ["clutch_factor", "pressure_type"],
        )
        result = merge_h2h_features(result, features.get("h2h", pd.DataFrame()))

    return result
