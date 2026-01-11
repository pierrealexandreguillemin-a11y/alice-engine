"""Types de biais - ISO 24027.

Ce module contient les types de base pour la detection de biais:
- BiasLevel: Enum des niveaux de biais
- BiasMetrics: Dataclass des metriques par groupe
- BiasReport: Dataclass du rapport complet

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems
- ISO/IEC 5055:2021 - Code Quality (<100 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BiasLevel(Enum):
    """Niveau de biais selon ISO 24027."""

    ACCEPTABLE = "acceptable"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class BiasMetrics:
    """Metriques de biais pour un groupe.

    Attributes
    ----------
        group_name: Nom du groupe (ex: "N1", "GM", "1800-2000")
        group_size: Nombre d'echantillons dans le groupe
        positive_rate: Taux de prediction positive
        true_positive_rate: TPR (recall) du groupe
        spd: Statistical Parity Difference vs reference
        eod: Equal Opportunity Difference vs reference
        dir: Disparate Impact Ratio vs reference
        level: Niveau de biais (acceptable/warning/critical)
    """

    group_name: str
    group_size: int
    positive_rate: float
    true_positive_rate: float
    spd: float = 0.0
    eod: float = 0.0
    dir: float = 1.0
    level: BiasLevel = BiasLevel.ACCEPTABLE


@dataclass
class BiasReport:
    """Rapport complet de biais ISO 24027.

    Attributes
    ----------
        timestamp: Date/heure du rapport
        model_name: Nom du modele analyse
        total_samples: Nombre total d'echantillons
        metrics_by_group: Metriques par groupe
        overall_level: Niveau de biais global
        recommendations: Recommandations d'action
    """

    timestamp: str
    model_name: str
    total_samples: int
    feature_analyzed: str
    reference_group: str
    metrics_by_group: list[BiasMetrics] = field(default_factory=list)
    overall_level: BiasLevel = BiasLevel.ACCEPTABLE
    recommendations: list[str] = field(default_factory=list)
    iso_compliance: dict[str, Any] = field(default_factory=dict)
