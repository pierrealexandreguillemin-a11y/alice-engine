"""Input Validator - ISO 24029 (Neural Network Robustness).

Module de validation OOD pour les inputs de prédiction.

ISO Compliance:
- ISO/IEC 24029:2021 - Neural Network Robustness
- ISO/IEC 5055:2021 - Code Quality (<150 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.model_registry.input_types import (
    DEFAULT_STD_TOLERANCE,
    OOD_REJECTION_THRESHOLD,
    FeatureBounds,
    FeatureValidationResult,
    InputBoundsConfig,
    InputValidationResult,
    OODAction,
    OODSeverity,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def compute_feature_bounds(
    data: pd.DataFrame, feature_name: str, is_categorical: bool = False
) -> FeatureBounds:
    """Calcule les bornes d'une feature depuis le training set."""
    col = data[feature_name]
    n_samples = len(col.dropna())

    if is_categorical:
        categories = [str(c) for c in col.dropna().unique().tolist()]
        return FeatureBounds(
            feature_name=feature_name,
            min_value=0,
            max_value=0,
            mean=0,
            std=0,
            p01=0,
            p99=0,
            n_samples=n_samples,
            is_categorical=True,
            categories=categories,
        )

    col_clean = col.dropna()
    return FeatureBounds(
        feature_name=feature_name,
        min_value=float(col_clean.min()),
        max_value=float(col_clean.max()),
        mean=float(col_clean.mean()),
        std=float(col_clean.std()),
        p01=float(np.percentile(col_clean, 1)),
        p99=float(np.percentile(col_clean, 99)),
        n_samples=n_samples,
        is_categorical=False,
    )


def create_bounds_config(
    training_data: pd.DataFrame,
    model_version: str,
    categorical_features: list[str] | None = None,
    features_to_validate: list[str] | None = None,
) -> InputBoundsConfig:
    """Crée une configuration de bornes depuis le training set."""
    categorical_features = categorical_features or []
    features = features_to_validate or list(training_data.columns)

    config = InputBoundsConfig(
        model_version=model_version,
        created_at=datetime.now().isoformat(),
        training_samples=len(training_data),
    )

    for feature in features:
        if feature in training_data.columns:
            config.features[feature] = compute_feature_bounds(
                training_data, feature, feature in categorical_features
            )

    logger.info(f"Created bounds config for {len(config.features)} features")
    return config


def _record_severity(
    result: InputValidationResult, feature_name: str, severity: OODSeverity, message: str
) -> None:
    """Enregistre la sévérité d'une feature dans le résultat de validation.

    Met à jour les listes warnings, ood_features et errors selon la sévérité.

    Args:
    ----
        result: Résultat de validation à mettre à jour (mutation in-place).
        feature_name: Nom de la feature validée.
        severity: Niveau de sévérité détecté.
        message: Message descriptif de l'anomalie.
    """
    if severity == OODSeverity.WARNING:
        result.warnings.append(f"[{feature_name}] {message}")
    elif severity in (OODSeverity.OUT_OF_BOUNDS, OODSeverity.EXTREME):
        result.ood_features.append(feature_name)
        result.errors.append(f"[{feature_name}] {message}")


def _determine_action(result: InputValidationResult, rejection_threshold: float) -> None:
    """Détermine l'action finale basée sur le ratio OOD.

    Met à jour is_valid et action selon le ratio de features OOD.

    Args:
    ----
        result: Résultat de validation à mettre à jour (mutation in-place).
        rejection_threshold: Seuil de ratio OOD au-delà duquel rejeter.

    Note:
    ----
        - REJECT si ood_ratio >= rejection_threshold
        - WARN si features OOD ou warnings présents
        - ACCEPT sinon (inchangé)
    """
    if result.ood_ratio >= rejection_threshold:
        result.is_valid = False
        result.action = OODAction.REJECT
    elif result.ood_features or result.warnings:
        result.action = OODAction.WARN


def validate_input(
    input_data: dict[str, Any] | pd.DataFrame,
    bounds_config: InputBoundsConfig,
    std_tolerance: float = DEFAULT_STD_TOLERANCE,
    rejection_threshold: float = OOD_REJECTION_THRESHOLD,
) -> InputValidationResult:
    """Valide un input contre les bornes du training set (ISO 24029)."""
    input_dict = input_data.to_dict() if hasattr(input_data, "to_dict") else dict(input_data)
    result = InputValidationResult(is_valid=True, action=OODAction.ACCEPT)

    for feature_name, bounds in bounds_config.features.items():
        if feature_name not in input_dict:
            continue
        value = input_dict[feature_name]
        severity, message = bounds.check_value(value, std_tolerance)
        result.feature_results.append(
            FeatureValidationResult(
                feature_name=feature_name, value=value, severity=severity, message=message
            )
        )
        _record_severity(result, feature_name, severity, message)

    n_validated = len([f for f in bounds_config.features if f in input_dict])
    result.ood_ratio = len(result.ood_features) / n_validated if n_validated > 0 else 0.0
    _determine_action(result, rejection_threshold)
    return result


def validate_batch(
    inputs: pd.DataFrame,
    bounds_config: InputBoundsConfig,
    std_tolerance: float = DEFAULT_STD_TOLERANCE,
    rejection_threshold: float = OOD_REJECTION_THRESHOLD,
) -> list[InputValidationResult]:
    """Valide un batch d'inputs."""
    return [
        validate_input(inputs.iloc[idx], bounds_config, std_tolerance, rejection_threshold)
        for idx in range(len(inputs))
    ]


def save_bounds_config(config: InputBoundsConfig, path: Path) -> None:
    """Sauvegarde la configuration des bornes en JSON."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f"Bounds config saved to {path}")


def load_bounds_config(path: Path) -> InputBoundsConfig | None:
    """Charge la configuration des bornes depuis JSON."""
    if not path.exists():
        logger.warning(f"Bounds config not found: {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        return InputBoundsConfig.from_dict(json.load(f))


__all__ = [
    "DEFAULT_STD_TOLERANCE",
    "OOD_REJECTION_THRESHOLD",
    "OODSeverity",
    "OODAction",
    "FeatureBounds",
    "FeatureValidationResult",
    "InputValidationResult",
    "InputBoundsConfig",
    "compute_feature_bounds",
    "create_bounds_config",
    "validate_input",
    "validate_batch",
    "save_bounds_config",
    "load_bounds_config",
]
