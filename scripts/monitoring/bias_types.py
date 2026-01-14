"""Types Bias Tracker - ISO 24027.

Enums et dataclasses pour le monitoring de biais.

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias Detection in AI
- ISO/IEC 5055:2021 - Code Quality (<150 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Constants (EEOC 80% rule)
DEMOGRAPHIC_PARITY_THRESHOLD = 0.8
DISPARATE_IMPACT_THRESHOLD = 0.8
EQUALIZED_ODDS_THRESHOLD = 0.1
CALIBRATION_THRESHOLD = 0.1


class FairnessStatus(str, Enum):
    """Statut de fairness selon ISO 24027."""

    FAIR = "fair"
    CAUTION = "caution"
    BIASED = "biased"
    CRITICAL = "critical"


class BiasAlertLevel(str, Enum):
    """Niveaux d'alerte pour le biais."""

    NONE = "none"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class BiasMetrics:
    """Métriques de biais pour un groupe protégé."""

    group_name: str
    group_value: str
    n_samples: int
    positive_rate: float
    true_positive_rate: float | None = None
    false_positive_rate: float | None = None
    calibration: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "group": f"{self.group_name}={self.group_value}",
            "n_samples": self.n_samples,
            "positive_rate": self.positive_rate,
            "tpr": self.true_positive_rate,
            "fpr": self.false_positive_rate,
            "calibration": self.calibration,
        }


@dataclass
class BiasAlert:
    """Alerte de biais détectée."""

    metric_name: str
    level: BiasAlertLevel
    message: str
    value: float
    threshold: float
    affected_groups: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "metric": self.metric_name,
            "level": self.level.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "affected_groups": self.affected_groups,
        }


@dataclass
class BiasMonitorResult:
    """Résultat complet du monitoring de biais."""

    timestamp: str
    protected_attribute: str
    n_total_samples: int
    group_metrics: list[BiasMetrics] = field(default_factory=list)
    demographic_parity: float = 0.0
    disparate_impact: float = 0.0
    equalized_odds_tpr: float | None = None
    equalized_odds_fpr: float | None = None
    overall_status: FairnessStatus = FairnessStatus.FAIR
    alerts: list[BiasAlert] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "timestamp": self.timestamp,
            "protected_attribute": self.protected_attribute,
            "n_samples": self.n_total_samples,
            "metrics": {
                "demographic_parity": self.demographic_parity,
                "disparate_impact": self.disparate_impact,
                "equalized_odds": {
                    "tpr_diff": self.equalized_odds_tpr,
                    "fpr_diff": self.equalized_odds_fpr,
                },
            },
            "groups": [g.to_dict() for g in self.group_metrics],
            "status": self.overall_status.value,
            "alerts": [a.to_dict() for a in self.alerts],
            "recommendations": self.recommendations,
        }


@dataclass
class BiasMonitorConfig:
    """Configuration pour le monitoring de biais."""

    protected_attributes: list[str] = field(default_factory=list)
    demographic_parity_threshold: float = DEMOGRAPHIC_PARITY_THRESHOLD
    disparate_impact_threshold: float = DISPARATE_IMPACT_THRESHOLD
    equalized_odds_threshold: float = EQUALIZED_ODDS_THRESHOLD
    calibration_threshold: float = CALIBRATION_THRESHOLD
    min_group_size: int = 30
    model_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "protected_attributes": self.protected_attributes,
            "thresholds": {
                "demographic_parity": self.demographic_parity_threshold,
                "disparate_impact": self.disparate_impact_threshold,
                "equalized_odds": self.equalized_odds_threshold,
                "calibration": self.calibration_threshold,
            },
            "min_group_size": self.min_group_size,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BiasMonitorConfig:
        """Reconstruit depuis un dictionnaire."""
        t = data.get("thresholds", {})
        return cls(
            protected_attributes=data.get("protected_attributes", []),
            demographic_parity_threshold=t.get("demographic_parity", DEMOGRAPHIC_PARITY_THRESHOLD),
            disparate_impact_threshold=t.get("disparate_impact", DISPARATE_IMPACT_THRESHOLD),
            equalized_odds_threshold=t.get("equalized_odds", EQUALIZED_ODDS_THRESHOLD),
            calibration_threshold=t.get("calibration", CALIBRATION_THRESHOLD),
            min_group_size=data.get("min_group_size", 30),
            model_version=data.get("model_version", ""),
        )
