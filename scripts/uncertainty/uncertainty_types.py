"""Types pour uncertainty quantification - ISO 24029.

Structures de données pour quantification d'incertitude ML.

ISO Compliance:
- ISO/IEC 24029:2021 - Neural Network Robustness
- ISO/IEC 5055:2021 - Code Quality (SRP)

Author: ALICE Engine Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class UncertaintyMethod(Enum):
    """Méthodes de quantification d'incertitude."""

    CONFORMAL = "conformal"  # Conformal prediction
    BOOTSTRAP = "bootstrap"  # Bootstrap intervals
    ENSEMBLE = "ensemble"  # Ensemble disagreement


@dataclass
class UncertaintyConfig:
    """Configuration pour quantification d'incertitude.

    Attributes:
        method: Méthode de quantification
        alpha: Niveau de significativité (1 - confidence)
        n_bootstrap: Nombre d'itérations bootstrap
    """

    method: UncertaintyMethod = UncertaintyMethod.CONFORMAL
    alpha: float = 0.10  # 90% confidence
    n_bootstrap: int = 100


@dataclass
class PredictionInterval:
    """Intervalle de prédiction pour une observation.

    Attributes:
        point_estimate: Probabilité ponctuelle
        lower: Borne inférieure de l'intervalle
        upper: Borne supérieure de l'intervalle
        confidence: Niveau de confiance (1 - alpha)
        in_prediction_set: Classes dans l'ensemble de prédiction (conformal)
    """

    point_estimate: float
    lower: float
    upper: float
    confidence: float
    in_prediction_set: list[int] = field(default_factory=list)

    def interval_width(self) -> float:
        """Retourne la largeur de l'intervalle."""
        return self.upper - self.lower


@dataclass
class UncertaintyMetrics:
    """Métriques globales d'incertitude.

    Attributes:
        mean_interval_width: Largeur moyenne des intervalles
        coverage: Couverture empirique (% vrais dans intervalle)
        efficiency: Efficacité (1 - taille moyenne ensemble)
        empty_set_rate: Taux d'ensembles vides
        singleton_rate: Taux d'ensembles singleton
    """

    mean_interval_width: float
    coverage: float
    efficiency: float
    empty_set_rate: float = 0.0
    singleton_rate: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convertit en dictionnaire."""
        return {
            "mean_interval_width": self.mean_interval_width,
            "coverage": self.coverage,
            "efficiency": self.efficiency,
            "empty_set_rate": self.empty_set_rate,
            "singleton_rate": self.singleton_rate,
        }


@dataclass
class UncertaintyResult:
    """Résultat de quantification d'incertitude.

    Attributes:
        intervals: Intervalles de prédiction par observation
        metrics: Métriques globales
        method: Méthode utilisée
        calibration_scores: Scores de non-conformité (conformal)
    """

    intervals: list[PredictionInterval]
    metrics: UncertaintyMetrics
    method: UncertaintyMethod
    calibration_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convertit en dictionnaire."""
        return {
            "method": self.method.value,
            "metrics": self.metrics.to_dict(),
            "n_predictions": len(self.intervals),
        }
