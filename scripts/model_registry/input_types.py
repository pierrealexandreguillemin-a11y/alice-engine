"""Types Input Validator - ISO 24029.

Enums et dataclasses pour la validation OOD.

ISO Compliance:
- ISO/IEC 24029:2021 - Neural Network Robustness
- ISO/IEC 5055:2021 - Code Quality (<150 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

DEFAULT_STD_TOLERANCE = 4.0
OOD_REJECTION_THRESHOLD = 0.3


class OODSeverity(str, Enum):
    """Niveaux de sévérité pour les inputs OOD."""

    VALID = "valid"
    WARNING = "warning"
    OUT_OF_BOUNDS = "out_of_bounds"
    EXTREME = "extreme"


class OODAction(str, Enum):
    """Actions possibles pour les inputs OOD."""

    ACCEPT = "accept"
    WARN = "warn"
    REJECT = "reject"
    CLAMP = "clamp"


@dataclass
class FeatureBounds:
    """Bornes d'une feature basées sur le training set."""

    feature_name: str
    min_value: float
    max_value: float
    mean: float
    std: float
    p01: float
    p99: float
    n_samples: int
    is_categorical: bool = False
    categories: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "feature": self.feature_name,
            "bounds": {
                "min": self.min_value,
                "max": self.max_value,
                "mean": self.mean,
                "std": self.std,
                "p01": self.p01,
                "p99": self.p99,
            },
            "n_samples": self.n_samples,
            "is_categorical": self.is_categorical,
            "categories": self.categories,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureBounds:
        """Reconstruit depuis un dictionnaire."""
        b = data.get("bounds", {})
        return cls(
            feature_name=data["feature"],
            min_value=b.get("min", 0),
            max_value=b.get("max", 0),
            mean=b.get("mean", 0),
            std=b.get("std", 1),
            p01=b.get("p01", 0),
            p99=b.get("p99", 0),
            n_samples=data.get("n_samples", 0),
            is_categorical=data.get("is_categorical", False),
            categories=data.get("categories"),
        )

    def check_value(
        self, value: float | str, std_tolerance: float = DEFAULT_STD_TOLERANCE
    ) -> tuple[OODSeverity, str]:
        """Vérifie si une valeur est dans les bornes."""
        if self.is_categorical:
            is_unknown = self.categories and str(value) not in self.categories
            return (
                (OODSeverity.OUT_OF_BOUNDS, f"Unknown category: {value}")
                if is_unknown
                else (OODSeverity.VALID, "")
            )

        if not isinstance(value, int | float) or np.isnan(value):
            return OODSeverity.OUT_OF_BOUNDS, f"Invalid value: {value}"

        if value < self.min_value or value > self.max_value:
            deviation = abs(value - self.mean) / max(self.std, 1e-8)
            severity = (
                OODSeverity.EXTREME if deviation > std_tolerance * 2 else OODSeverity.OUT_OF_BOUNDS
            )
            msg = (
                f"Value {value} is extreme"
                if severity == OODSeverity.EXTREME
                else f"Value {value} outside bounds"
            )
            return severity, msg

        if value < self.p01 or value > self.p99:
            return OODSeverity.WARNING, f"Value {value} outside p01-p99"

        return OODSeverity.VALID, ""


@dataclass
class FeatureValidationResult:
    """Résultat de validation pour une feature."""

    feature_name: str
    value: Any
    severity: OODSeverity
    message: str
    clamped_value: Any | None = None


@dataclass
class InputValidationResult:
    """Résultat complet de validation d'un input."""

    is_valid: bool
    action: OODAction
    feature_results: list[FeatureValidationResult] = field(default_factory=list)
    ood_features: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    ood_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "is_valid": self.is_valid,
            "action": self.action.value,
            "ood_ratio": self.ood_ratio,
            "ood_features": self.ood_features,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class InputBoundsConfig:
    """Configuration des bornes pour validation."""

    features: dict[str, FeatureBounds] = field(default_factory=dict)
    model_version: str = ""
    created_at: str = ""
    training_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "model_version": self.model_version,
            "created_at": self.created_at,
            "training_samples": self.training_samples,
            "features": {k: v.to_dict() for k, v in self.features.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InputBoundsConfig:
        """Reconstruit depuis un dictionnaire."""
        features = {
            fname: FeatureBounds.from_dict(fdata)
            for fname, fdata in data.get("features", {}).items()
        }
        return cls(
            model_version=data.get("model_version", ""),
            created_at=data.get("created_at", ""),
            training_samples=data.get("training_samples", 0),
            features=features,
        )
