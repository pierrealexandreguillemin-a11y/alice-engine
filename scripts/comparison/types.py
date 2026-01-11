"""Types de comparaison statistique - ISO 24029.

Ce module contient les types de base pour la comparaison de modeles.

ISO Compliance:
- ISO/IEC 24029:2021 - Statistical validation
- ISO/IEC 5055:2021 - Code Quality (<50 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from scripts.comparison.mcnemar_test import McNemarResult


@dataclass
class ModelComparison:
    """Resultat de comparaison entre deux modeles.

    Attributes
    ----------
        model_a_name: Nom du modele A
        model_b_name: Nom du modele B
        mcnemar_result: Resultat du test McNemar
        metrics_a: Metriques du modele A
        metrics_b: Metriques du modele B
        winner: Modele gagnant
        practical_significance: True si difference pratiquement significative
        recommendation: Recommandation basee sur l'analyse
    """

    model_a_name: str
    model_b_name: str
    mcnemar_result: McNemarResult
    metrics_a: dict[str, float]
    metrics_b: dict[str, float]
    winner: str
    practical_significance: bool
    recommendation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
