"""Features avancées ML - ISO 5055/5259.

Ce package implémente les features avancées identifiées par recherche web:
- Head-to-head (H2H) historique
- Performance sous pression (matchs décisifs)
- Trajectoire Elo (progression/régression)

NOTE: Features supprimées (non pertinentes pour échecs):
- fatigue.py: Fatigue physique non applicable (échecs = cognitif)
- home_away.py: Confusion conceptuelle domicile/couleur (voir color_perf.py)

Sources:
- AI Sports Predictions 2025 (ainewshub.org)
- EloMetrics IEEE 2025
- Sports Prediction PMC

Conformité:
- ISO 5055: Modules <300 lignes, responsabilité unique
- ISO 5259: Features calculées depuis données réelles
"""

from scripts.features.advanced.elo_trajectory import calculate_elo_trajectory
from scripts.features.advanced.h2h import calculate_head_to_head
from scripts.features.advanced.pressure import calculate_pressure_performance

__all__ = [
    "calculate_head_to_head",
    "calculate_pressure_performance",
    "calculate_elo_trajectory",
]
