"""Entraînement parallèle des modèles - ISO 5055.

Ce module contient les fonctions d'exécution parallèle.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    """Entraine CatBoost, XGBoost et LightGBM en parallele."""
    results: ModelResults = {}

    logger.info("\n" + "=" * 60)
    logger.info("PARALLEL TRAINING - 3 MODELS")
    logger.info("=" * 60)

    catboost_config = _get_model_config(config, "catboost")
    xgboost_config = _get_model_config(config, "xgboost")
    lightgbm_config = _get_model_config(config, "lightgbm")

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = _submit_training_tasks(
            executor,
            X_train,
            y_train,
            X_valid,
            y_valid,
            cat_features,
            catboost_config,
            xgboost_config,
            lightgbm_config,
        )

        for future in as_completed(futures):
            name = futures[future]
            result = _handle_training_result(future, name)
            results[name] = result

    return results


def _get_model_config(config: dict[str, object], model_name: str) -> dict[str, object]:
    """Extrait et valide la config d'un modèle."""
    model_config = config.get(model_name, {})
    if not isinstance(model_config, dict):
        return {}
    return model_config


def _submit_training_tasks(
    executor: ThreadPoolExecutor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
    catboost_config: dict[str, object],
    xgboost_config: dict[str, object],
    lightgbm_config: dict[str, object],
) -> dict[object, str]:
    """Soumet les tâches d'entraînement à l'executor."""
    return {
        executor.submit(
            train_catboost,
            X_train,
            y_train,
            X_valid,
            y_valid,
            cat_features,
            catboost_config,  # type: ignore[arg-type]
        ): "CatBoost",
        executor.submit(
            train_xgboost,
            X_train,
            y_train,
            X_valid,
            y_valid,
            xgboost_config,  # type: ignore[arg-type]
        ): "XGBoost",
        executor.submit(
            train_lightgbm,
            X_train,
            y_train,
            X_valid,
            y_valid,
            cat_features,
            lightgbm_config,  # type: ignore[arg-type]
        ): "LightGBM",
    }


def _handle_training_result(future: object, name: str) -> TrainingResult:
    """Gère le résultat d'une tâche d'entraînement."""
    try:
        result = future.result()  # type: ignore[union-attr]
        logger.info(f"[{name}] Completed successfully")
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
