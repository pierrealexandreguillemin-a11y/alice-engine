"""Features module - ISO 5055 compliant modular structure.

Ce module regroupe les fonctions d'extraction de features pour ALICE Engine.
Chaque sous-module a une responsabilite unique (SRP - ISO 25010).

Modules:
- reliability: Features de fiabilite club/joueur
- performance: Features de performance (forme, couleur, position)
- standings: Classement equipe et zones d'enjeu
- advanced: Features avancees (H2H, fatigue, Elo trajectory)
- ffe_features: Features reglementaires FFE
- pipeline: Orchestration extraction/merge
"""

from scripts.features.advanced import (
    calculate_elo_trajectory,
    calculate_head_to_head,
    calculate_pressure_performance,
)
from scripts.features.ffe_features import (
    build_historique_brulage,
    build_historique_noyau,
    extract_ffe_regulatory_features,
)
from scripts.features.performance import (
    calculate_board_position,
    calculate_color_performance,
    calculate_recent_form,
)
from scripts.features.pipeline import (
    extract_all_features,
    merge_all_features,
)
from scripts.features.reliability import (
    extract_club_reliability,
    extract_player_monthly_pattern,
    extract_player_reliability,
)
from scripts.features.standings import (
    calculate_standings,
    extract_team_enjeu_features,
)

__all__ = [
    # Reliability
    "extract_club_reliability",
    "extract_player_reliability",
    "extract_player_monthly_pattern",
    # Performance
    "calculate_recent_form",
    "calculate_board_position",
    "calculate_color_performance",
    # Standings
    "calculate_standings",
    "extract_team_enjeu_features",
    # Advanced
    "calculate_head_to_head",
    "calculate_pressure_performance",
    "calculate_elo_trajectory",
    # FFE Features
    "build_historique_brulage",
    "build_historique_noyau",
    "extract_ffe_regulatory_features",
    # Pipeline
    "extract_all_features",
    "merge_all_features",
]
