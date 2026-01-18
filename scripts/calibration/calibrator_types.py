"""Types pour calibration - ISO 24029.

Structures de données pour calibration des probabilités ML.

ISO Compliance:
- ISO/IEC 24029:2021 - Neural Network Robustness
- ISO/IEC 5055:2021 - Code Quality (<100 lignes, SRP)

Author: ALICE Engine Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CalibrationMethod(Enum):
    """Méthodes de calibration (ISO 24029)."""

    PLATT = "platt"  # Platt scaling (sigmoid)
    ISOTONIC = "isotonic"  # Isotonic regression
    BETA = "beta"  # Beta calibration (future)


@dataclass
class CalibrationConfig:
    """Configuration de calibration.

    Attributes:
        method: Méthode de calibration (platt, isotonic)
        cv: Nombre de folds pour cross-validation (0 = prefit)
        n_bins: Nombre de bins pour calibration curve
    """

    method: CalibrationMethod = CalibrationMethod.ISOTONIC
    cv: int = 5
    n_bins: int = 10


@dataclass
class CalibrationMetrics:
    """Métriques de calibration.

    Attributes:
        brier_before: Brier score avant calibration
        brier_after: Brier score après calibration
        ece_before: Expected Calibration Error avant
        ece_after: Expected Calibration Error après
        improvement_pct: Amélioration en pourcentage
    """

    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float
    improvement_pct: float

    def to_dict(self) -> dict[str, float]:
        """Convertit en dictionnaire."""
        return {
            "brier_before": self.brier_before,
            "brier_after": self.brier_after,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "improvement_pct": self.improvement_pct,
        }


@dataclass
class CalibrationResult:
    """Résultat de calibration.

    Attributes:
        calibrator: Modèle calibré (sklearn CalibratedClassifierCV)
        method: Méthode utilisée
        metrics: Métriques de calibration
        calibration_curve: Données pour courbe de calibration
    """

    calibrator: Any  # CalibratedClassifierCV
    method: CalibrationMethod
    metrics: CalibrationMetrics
    calibration_curve: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire (sans calibrator)."""
        return {
            "method": self.method.value,
            "metrics": self.metrics.to_dict(),
            "calibration_curve": self.calibration_curve,
        }
