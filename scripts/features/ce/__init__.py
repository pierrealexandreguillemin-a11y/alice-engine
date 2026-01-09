"""CE (Composition Engine) Features - ISO 5055/5259.

Ce package implémente les features pour l'optimisation de composition.
Utilisé par le module CE pour déterminer la meilleure composition.

Modules:
- scenarios: Scénarios de classement (course titre, danger, confort)
- urgency: Urgence mathématique (montée possible, maintien assuré)

Conformité:
- ISO 5055: Modules <300 lignes, responsabilité unique
- ISO 5259: Features calculées depuis données réelles
- ISO 42001: Décisions AI traçables
"""

from scripts.features.ce.scenarios import calculate_scenario_features
from scripts.features.ce.urgency import calculate_urgency_features

__all__ = [
    "calculate_scenario_features",
    "calculate_urgency_features",
]
