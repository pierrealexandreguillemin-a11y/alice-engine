"""Module: parallel.py - Entraînement séquentiel des modèles ML.

Ce module entraîne les modèles un par un pour économiser la RAM.
Chaque modèle est entraîné puis libéré avant le suivant via gc.collect().

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (Gouvernance training)
- ISO/IEC 23894:2023 - AI Risk Management (Gestion ressources mémoire)
- ISO/IEC 5055 - Code Quality (0 CWE critiques)

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from scripts.ml_types import ModelMetrics, ModelResults, TrainingResult
from scripts.training.trainers import train_catboost, train_lightgbm, train_xgboost

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def train_all_models_parallel(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
    config: dict[str, object],
) -> ModelResults:
    """Entraine CatBoost, XGBoost et LightGBM séquentiellement (économie RAM)."""
    results: ModelResults = {}

    logger.info("\n" + "=" * 60)
    logger.info("SEQUENTIAL TRAINING - 3 MODELS (RAM optimized)")
    logger.info("=" * 60)

    # Définir les tâches d'entraînement
    tasks: list[tuple[str, Callable[[], TrainingResult]]] = [
        (
            "CatBoost",
            lambda: train_catboost(
                X_train,
                y_train,
                X_valid,
                y_valid,
                cat_features,
                _get_model_config(config, "catboost"),
            ),
        ),
        (
            "XGBoost",
            lambda: train_xgboost(
                X_train,
                y_train,
                X_valid,
                y_valid,
                _get_model_config(config, "xgboost"),
            ),
        ),
        (
            "LightGBM",
            lambda: train_lightgbm(
                X_train,
                y_train,
                X_valid,
                y_valid,
                cat_features,
                _get_model_config(config, "lightgbm"),
            ),
        ),
    ]

    # Entraîner chaque modèle séquentiellement
    for name, train_fn in tasks:
        logger.info(f"\n[{name}] Starting training...")
        result = _train_single_model(name, train_fn)
        results[name] = result
        # Libérer la mémoire entre chaque modèle
        gc.collect()

    return results


def _get_model_config(config: dict[str, object], model_name: str) -> dict[str, object]:
    """Extrait et valide la config d'un modèle."""
    model_config = config.get(model_name, {})
    if not isinstance(model_config, dict):
        return {}
    return model_config


def _train_single_model(name: str, train_fn: Callable[[], TrainingResult]) -> TrainingResult:
    """Entraîne un seul modèle avec gestion d'erreur."""
    try:
        result = train_fn()
        logger.info(f"[{name}] Completed - AUC: {result.metrics.auc_roc:.4f}")
        return result
    except Exception as e:
        logger.exception(f"[{name}] FAILED: {e}")
        return TrainingResult(
            model=None,
            train_time=0.0,
            metrics=ModelMetrics(
                auc_roc=0.0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                log_loss=1.0,
            ),
        )
