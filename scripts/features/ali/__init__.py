"""ALI (Adversarial Lineup Inference) Features - ISO 5055/5259.

Ce package implémente les features pour la prédiction de présence joueur.
Utilisé par le module ALI pour prédire la composition adverse probable.

Modules:
- presence: Taux de présence, dernière apparition, régularité
- patterns: Rôle (titulaire/remplacant), échiquier préféré, flexibilité

Conformité:
- ISO 5055: Modules <300 lignes, responsabilité unique
- ISO 5259: Features calculées depuis données réelles
- ISO 42001: Prédictions AI traçables
"""

from scripts.features.ali.patterns import calculate_selection_patterns
from scripts.features.ali.presence import calculate_presence_features

__all__ = [
    "calculate_presence_features",
    "calculate_selection_patterns",
]
