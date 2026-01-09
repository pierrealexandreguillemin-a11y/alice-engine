"""Fonctions d'entraînement des modèles - ISO 5055.

Ce module contient les fonctions d'entraînement pour CatBoost, XGBoost et LightGBM.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from scripts.ml_types import (
    CatBoostConfig,
    CatBoostModel,
    LightGBMConfig,
    LightGBMModel,
    TrainingResult,
    XGBoostConfig,
    XGBoostModel,
)
from scripts.training.metrics import compute_all_metrics

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
    params: CatBoostConfig,
) -> TrainingResult:
    """Entraine CatBoost avec les hyperparametres specifies."""
    from catboost import CatBoostClassifier

    logger.info("[CatBoost] Starting training...")

    model: CatBoostModel = CatBoostClassifier(
        iterations=params.get("iterations", 1000),
        learning_rate=params.get("learning_rate", 0.03),
        depth=params.get("depth", 6),
        l2_leaf_reg=params.get("l2_leaf_reg", 3),
        cat_features=cat_features,
        early_stopping_rounds=params.get("early_stopping_rounds", 50),
        eval_metric="AUC",
        random_seed=params.get("random_seed", 42),
        verbose=params.get("verbose", 100),
        thread_count=params.get("thread_count", -1),
    )

    start = time.time()
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    train_time = time.time() - start

    # Metriques sur validation
    y_proba: NDArray[np.float64] = model.predict_proba(X_valid)[:, 1]
    y_pred: NDArray[np.int64] = (y_proba >= 0.5).astype(np.int64)
    metrics = compute_all_metrics(y_valid.values, y_pred, y_proba)
    metrics.train_time_s = train_time

    logger.info(f"[CatBoost] Done in {train_time:.1f}s | AUC: {metrics.auc_roc:.4f}")

    return TrainingResult(model=model, train_time=train_time, metrics=metrics)


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: XGBoostConfig,
) -> TrainingResult:
    """Entraine XGBoost avec les hyperparametres specifies."""
    from xgboost import XGBClassifier

    logger.info("[XGBoost] Starting training...")

    model: XGBoostModel = XGBClassifier(
        n_estimators=params.get("n_estimators", 1000),
        learning_rate=params.get("learning_rate", 0.03),
        max_depth=params.get("max_depth", 6),
        reg_lambda=params.get("reg_lambda", 1.0),
        reg_alpha=params.get("reg_alpha", 0.0),
        tree_method=params.get("tree_method", "hist"),
        early_stopping_rounds=params.get("early_stopping_rounds", 50),
        eval_metric="auc",
        random_state=params.get("random_state", 42),
        verbosity=params.get("verbosity", 1),
        n_jobs=params.get("n_jobs", -1),
    )

    start = time.time()
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)
    train_time = time.time() - start

    # Metriques
    y_proba: NDArray[np.float64] = model.predict_proba(X_valid)[:, 1]
    y_pred: NDArray[np.int64] = (y_proba >= 0.5).astype(np.int64)
    metrics = compute_all_metrics(y_valid.values, y_pred, y_proba)
    metrics.train_time_s = train_time

    logger.info(f"[XGBoost] Done in {train_time:.1f}s | AUC: {metrics.auc_roc:.4f}")

    return TrainingResult(model=model, train_time=train_time, metrics=metrics)


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
    params: LightGBMConfig,
) -> TrainingResult:
    """Entraine LightGBM avec les hyperparametres specifies."""
    import lightgbm as lgb

    logger.info("[LightGBM] Starting training...")

    # Indices des features categorielles
    cat_indices = [X_train.columns.get_loc(c) for c in cat_features if c in X_train.columns]

    model: LightGBMModel = lgb.LGBMClassifier(
        n_estimators=params.get("n_estimators", 1000),
        learning_rate=params.get("learning_rate", 0.03),
        num_leaves=params.get("num_leaves", 63),
        max_depth=params.get("max_depth", -1),
        reg_lambda=params.get("reg_lambda", 1.0),
        reg_alpha=params.get("reg_alpha", 0.0),
        categorical_feature=cat_indices,
        random_state=params.get("random_state", 42),
        verbose=params.get("verbose", -1),
        n_jobs=params.get("n_jobs", -1),
    )

    start = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=params.get("early_stopping_rounds", 50)),
            lgb.log_evaluation(period=100),
        ],
    )
    train_time = time.time() - start

    # Metriques
    y_proba: NDArray[np.float64] = model.predict_proba(X_valid)[:, 1]
    y_pred: NDArray[np.int64] = (y_proba >= 0.5).astype(np.int64)
    metrics = compute_all_metrics(y_valid.values, y_pred, y_proba)
    metrics.train_time_s = train_time

    logger.info(f"[LightGBM] Done in {train_time:.1f}s | AUC: {metrics.auc_roc:.4f}")

    return TrainingResult(model=model, train_time=train_time, metrics=metrics)
