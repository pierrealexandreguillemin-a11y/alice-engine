"""Types Drift Monitor - ISO 23894.

Enums et dataclasses pour le drift monitoring.

ISO Compliance:
- ISO/IEC 23894:2023 - AI Risk Management
- ISO/IEC 5055:2021 - Code Quality (<80 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Constants
PSI_THRESHOLD_OK = 0.1
PSI_THRESHOLD_WARNING = 0.2
PSI_THRESHOLD_CRITICAL = 0.25

KS_PVALUE_OK = 0.05
KS_PVALUE_WARNING = 0.01
KS_PVALUE_CRITICAL = 0.001


class DriftSeverity(str, Enum):
    """Niveaux de sévérité du drift."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftType(str, Enum):
    """Types de drift détectables."""

    COVARIATE = "covariate"
    CONCEPT = "concept"
    PRIOR = "prior"


@dataclass
class FeatureDriftResult:
    """Résultat de drift pour une feature."""

    feature_name: str
    psi_score: float
    ks_statistic: float | None = None
    ks_pvalue: float | None = None
    chi2_statistic: float | None = None
    chi2_pvalue: float | None = None
    js_divergence: float | None = None
    severity: DriftSeverity = DriftSeverity.NONE
    is_categorical: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "feature": self.feature_name,
            "psi": self.psi_score,
            "ks_statistic": self.ks_statistic,
            "ks_pvalue": self.ks_pvalue,
            "chi2_statistic": self.chi2_statistic,
            "chi2_pvalue": self.chi2_pvalue,
            "js_divergence": self.js_divergence,
            "severity": self.severity.value,
            "is_categorical": self.is_categorical,
        }


@dataclass
class DriftMonitorResult:
    """Résultat complet du monitoring de drift."""

    timestamp: str
    model_version: str
    baseline_samples: int
    current_samples: int
    feature_results: list[FeatureDriftResult] = field(default_factory=list)
    overall_severity: DriftSeverity = DriftSeverity.NONE
    drift_detected: bool = False
    drifted_features: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation."""
        return {
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "samples": {
                "baseline": self.baseline_samples,
                "current": self.current_samples,
            },
            "features": [f.to_dict() for f in self.feature_results],
            "summary": {
                "overall_severity": self.overall_severity.value,
                "drift_detected": self.drift_detected,
                "drifted_features": self.drifted_features,
                "drifted_count": len(self.drifted_features),
                "total_features": len(self.feature_results),
            },
            "recommendations": self.recommendations,
        }
