"""Types de robustesse - ISO 24029.

Ce module contient les types de base pour les tests de robustesse:
- RobustnessLevel: Enum des niveaux de robustesse
- RobustnessMetrics: Dataclass des métriques par test
- RobustnessReport: Dataclass du rapport complet

ISO Compliance:
- ISO/IEC 24029-1:2021 - Neural Network Robustness Assessment
- ISO/IEC 5055:2021 - Code Quality (<100 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RobustnessLevel(Enum):
    """Niveau de robustesse selon ISO 24029."""

    ROBUST = "robust"
    ACCEPTABLE = "acceptable"
    WARNING = "warning"
    FRAGILE = "fragile"


@dataclass
class RobustnessMetrics:
    """Métriques de robustesse pour un test.

    Attributes
    ----------
        test_name: Nom du test
        original_score: Score sur données originales
        perturbed_score: Score après perturbation
        degradation: Dégradation relative (%)
        stability_ratio: % de prédictions inchangées
        level: Niveau de robustesse
        details: Détails additionnels
    """

    test_name: str
    original_score: float
    perturbed_score: float
    degradation: float
    stability_ratio: float
    level: RobustnessLevel = RobustnessLevel.ACCEPTABLE
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RobustnessReport:
    """Rapport complet de robustesse ISO 24029.

    Attributes
    ----------
        timestamp: Date/heure du rapport
        model_name: Nom du modèle testé
        total_tests: Nombre de tests effectués
        metrics: Liste des métriques par test
        overall_level: Niveau de robustesse global
        recommendations: Recommandations d'action
    """

    timestamp: str
    model_name: str
    total_tests: int
    original_accuracy: float
    metrics: list[RobustnessMetrics] = field(default_factory=list)
    overall_level: RobustnessLevel = RobustnessLevel.ACCEPTABLE
    recommendations: list[str] = field(default_factory=list)
    iso_compliance: dict[str, Any] = field(default_factory=dict)
