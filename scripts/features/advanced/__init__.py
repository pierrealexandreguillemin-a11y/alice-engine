"""Features avancées ML - ISO 5055/5259.

Ce package implémente les features avancées identifiées par recherche web:
- Head-to-head (H2H) historique
- Fatigue / jours de repos
- Performance domicile vs extérieur
- Performance sous pression (matchs décisifs)
- Trajectoire Elo (progression/régression)

Sources:
- AI Sports Predictions 2025 (ainewshub.org)
- EloMetrics IEEE 2025
- Home Advantage Research (Taylor & Francis)
- Sports Prediction PMC

Conformité:
- ISO 5055: Modules <300 lignes, responsabilité unique
- ISO 5259: Features calculées depuis données réelles
"""

from scripts.features.advanced.elo_trajectory import calculate_elo_trajectory
from scripts.features.advanced.fatigue import calculate_fatigue_rest_days
from scripts.features.advanced.h2h import calculate_head_to_head
from scripts.features.advanced.home_away import calculate_home_away_performance
from scripts.features.advanced.pressure import calculate_pressure_performance

__all__ = [
    "calculate_head_to_head",
    "calculate_fatigue_rest_days",
    "calculate_home_away_performance",
    "calculate_pressure_performance",
    "calculate_elo_trajectory",
]
