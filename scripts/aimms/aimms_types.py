"""Types pour AIMMS - ISO/IEC 42001:2023.

Structures de données pour AI Management System lifecycle.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System
- ISO/IEC 5055:2021 - Code Quality (SRP)

Author: ALICE Engine Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class LifecyclePhase(Enum):
    """Phases du cycle de vie AI (ISO 42001 Clause 8.2)."""

    DEVELOPMENT = "development"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    RETIREMENT = "retirement"


@dataclass
class AIMSConfig:
    """Configuration AIMMS post-processing.

    Attributes:
        enable_calibration: Activer calibration probabilités
        enable_uncertainty: Activer quantification incertitude
        enable_alerting: Configurer alerting drift
        calibration_cv: CV folds (0=prefit)
        uncertainty_alpha: Niveau alpha conformal (0.10=90% coverage)
        min_calibration_samples: Minimum échantillons calibration
    """

    enable_calibration: bool = True
    enable_uncertainty: bool = True
    enable_alerting: bool = True
    calibration_cv: int = 0  # prefit by default
    uncertainty_alpha: float = 0.10
    min_calibration_samples: int = 30


@dataclass
class CalibrationSummary:
    """Résumé calibration (ISO 24029)."""

    method: str
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float
    improvement_pct: float

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "method": self.method,
            "brier_before": self.brier_before,
            "brier_after": self.brier_after,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "improvement_pct": self.improvement_pct,
        }


@dataclass
class UncertaintySummary:
    """Résumé uncertainty (ISO 24029)."""

    method: str
    coverage: float
    mean_interval_width: float
    singleton_rate: float
    empty_set_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "method": self.method,
            "coverage": self.coverage,
            "mean_interval_width": self.mean_interval_width,
            "singleton_rate": self.singleton_rate,
            "empty_set_rate": self.empty_set_rate,
        }


@dataclass
class AlertingSummary:
    """Résumé configuration alerting (ISO 23894)."""

    enabled: bool
    slack_configured: bool
    min_severity: str
    cooldown_minutes: int

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "enabled": self.enabled,
            "slack_configured": self.slack_configured,
            "min_severity": self.min_severity,
            "cooldown_minutes": self.cooldown_minutes,
        }


@dataclass
class AIMSResult:
    """Résultat complet AIMMS post-processing.

    Attributes:
        phase: Phase lifecycle actuelle
        timestamp: Horodatage ISO 8601
        model_version: Version du modèle
        calibration: Résumé calibration (si activé)
        uncertainty: Résumé uncertainty (si activé)
        alerting: Résumé alerting (si activé)
        recommendations: Recommandations ISO 42001 Clause 10.2
    """

    phase: LifecyclePhase
    timestamp: str
    model_version: str
    calibration: CalibrationSummary | None = None
    uncertainty: UncertaintySummary | None = None
    alerting: AlertingSummary | None = None
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour export JSON."""
        return {
            "iso_standard": "ISO/IEC 42001:2023",
            "phase": self.phase.value,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "uncertainty": self.uncertainty.to_dict() if self.uncertainty else None,
            "alerting": self.alerting.to_dict() if self.alerting else None,
            "recommendations": self.recommendations,
        }

    @staticmethod
    def create(model_version: str, phase: LifecyclePhase = LifecyclePhase.VALIDATION) -> AIMSResult:
        """Factory method pour créer un résultat vide."""
        return AIMSResult(
            phase=phase,
            timestamp=datetime.now().isoformat(),
            model_version=model_version,
        )
