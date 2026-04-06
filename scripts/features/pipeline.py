"""Pipeline helpers pour feature engineering - ISO 5055.

Ce module contient les fonctions d'orchestration du pipeline de features:
- extract_all_features: Extraction de toutes les features depuis l'historique
- merge_all_features: Merge de toutes les features sur le DataFrame cible

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
from scripts.features.club_behavior import extract_club_behavior
from scripts.features.club_level import extract_club_level_features, extract_player_team_context
from scripts.features.composition import extract_composition_strategy
from scripts.features.draw_priors import compute_equipe_draw_rates, compute_player_draw_rates
from scripts.features.ffe_features import extract_ffe_regulatory_features
from scripts.features.merge_helpers import (
    merge_club_reliability,
    merge_h2h_features,
    merge_noyau_features,
    merge_player_features,
    merge_team_enjeu,
)
from scripts.features.merge_v8 import (
    merge_club_level,
    merge_draw_rate_equipe,
    merge_draw_rate_player,
    merge_player_team_context,
)
from scripts.features.noyau import extract_noyau_features
from scripts.features.performance import (
    calculate_board_position,
    calculate_color_performance,
    calculate_recent_form,
)
from scripts.features.pipeline_extended import (
    extract_ali_features,
    merge_ali_features,
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
        "board_position": calculate_board_position(
            df_history_played,
            max_saison=int(df_history_played["saison"].max())
            if not df_history_played.empty and "saison" in df_history_played.columns
            else None,
        ),
        "color_perf": calculate_color_performance(df_history_played),
        "ffe_regulatory": extract_ffe_regulatory_features(df_history_played),
        "team_enjeu": extract_team_enjeu_features(df_history_played, standings),
        "club_behavior": extract_club_behavior(df_history),
        "noyau": extract_noyau_features(df_history_played),
        # V8 features
        "draw_rate_player": compute_player_draw_rates(df_history_played),
        "draw_rate_equipe": compute_equipe_draw_rates(df_history_played),
        "club_level": extract_club_level_features(df_history),
        "player_team_context": extract_player_team_context(df_history),
    }

    # ALI features (presence + patterns + absence)
    ali_features = extract_ali_features(df_history_played)
    features.update(ali_features)

    # Composition strategy (A02 Art. 3.6.e) — always computed
    compo_raw = extract_composition_strategy(df_history_played)
    compo_agg = _aggregate_composition(compo_raw)
    features["composition"] = compo_agg

    if include_advanced:
        features.update(
            {
                "h2h": calculate_head_to_head(df_history_played),
                "pressure": calculate_pressure_performance(df_history_played),
                "trajectory": calculate_elo_trajectory(df_history_played),
            }
        )

    return features


def _aggregate_composition(compo_raw: pd.DataFrame) -> pd.DataFrame:
    """Agrège les features composition par joueur (moyenne historique)."""
    if compo_raw.empty:
        return pd.DataFrame(
            columns=["joueur_nom", "decalage_position", "joueur_decale_haut", "joueur_decale_bas"]
        )
    return (
        compo_raw.groupby("nom")
        .agg(
            decalage_position=("decalage_position", "mean"),
            joueur_decale_haut=("joueur_decale_haut", "mean"),
            joueur_decale_bas=("joueur_decale_bas", "mean"),
        )
        .reset_index()
        .rename(columns={"nom": "joueur_nom"})
    )


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
        result,
        features["recent_form"],
        [
            "win_rate_recent",
            "draw_rate_recent",
            "expected_score_recent",
            "win_trend",
            "draw_trend",
        ],
    )
    result = merge_player_features(
        result, features["board_position"], ["echiquier_moyen", "echiquier_std"]
    )
    result = merge_player_features(
        result,
        features["color_perf"],
        [
            "win_rate_white",
            "draw_rate_white",
            "win_rate_black",
            "draw_rate_black",
            "win_adv_white",
            "draw_adv_white",
            "couleur_preferee",
            "data_quality",
        ],
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

    # Club behavior (merge by equipe_dom AND equipe_ext)
    result = _merge_club_behavior(result, features.get("club_behavior", pd.DataFrame()))

    # Noyau features (joueur x equipe x ronde)
    result = merge_noyau_features(result, features.get("noyau", pd.DataFrame()))

    # ALI features (presence + patterns + absence per player)
    result = merge_ali_features(result, features)

    # Composition strategy (always merged)
    result = merge_player_features(
        result,
        features.get("composition", pd.DataFrame()),
        ["decalage_position", "joueur_decale_haut", "joueur_decale_bas"],
    )

    # V8 features
    result = merge_draw_rate_player(result, features.get("draw_rate_player", pd.DataFrame()))
    result = merge_draw_rate_equipe(result, features.get("draw_rate_equipe", pd.DataFrame()))
    result = merge_club_level(result, features.get("club_level", pd.DataFrame()))
    result = merge_player_team_context(result, features.get("player_team_context", pd.DataFrame()))

    # Advanced features
    if include_advanced:
        result = _merge_advanced_features(result, features)

    return result


def _merge_club_behavior(
    result: pd.DataFrame,
    club_beh: pd.DataFrame,
) -> pd.DataFrame:
    """Merge club behavior pour equipe_dom ET equipe_ext.

    Colonnes dom: nb_joueurs_utilises_dom, rotation_effectif_dom, etc.
    Colonnes ext: nb_joueurs_utilises_ext, rotation_effectif_ext, etc.
    """
    if club_beh.empty or "equipe" not in club_beh.columns:
        return result

    beh_cols = [c for c in club_beh.columns if c not in ("equipe", "saison")]

    for suffix, equipe_col in [("dom", "equipe_dom"), ("ext", "equipe_ext")]:
        if equipe_col not in result.columns:
            continue
        rename_map = {c: f"{c}_{suffix}" for c in beh_cols}
        merge_df = club_beh.rename(columns={"equipe": equipe_col} | rename_map)
        result = result.merge(
            merge_df[[equipe_col, "saison"] + list(rename_map.values())],
            on=[equipe_col, "saison"],
            how="left",
        )

    return result


def _merge_advanced_features(
    result: pd.DataFrame,
    features: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge les features avancees (H2H, pressure, trajectory, composition)."""
    result = merge_player_features(
        result,
        features.get("trajectory", pd.DataFrame()),
        ["elo_trajectory", "momentum"],
    )
    result = merge_player_features(
        result,
        features.get("pressure", pd.DataFrame()),
        [
            "win_rate_normal",
            "draw_rate_normal",
            "win_rate_pression",
            "draw_rate_pression",
            "clutch_win",
            "clutch_draw",
            "pressure_type",
        ],
    )
    result = merge_h2h_features(result, features.get("h2h", pd.DataFrame()))
    return result
