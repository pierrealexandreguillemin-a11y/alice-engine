"""Model trainers for CatBoost, XGBoost, and LightGBM.

This module contains the training functions for each boosting model.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from scripts.evaluation.constants import (
    DEFAULT_DEPTH,
    DEFAULT_EARLY_STOPPING,
    DEFAULT_ITERATIONS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_RANDOM_SEED,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
) -> tuple[Any, float]:
    """Entraine CatBoost et retourne le modele + temps."""
    from catboost import CatBoostClassifier

    logger.info("  Training CatBoost...")

    model = CatBoostClassifier(
        iterations=DEFAULT_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE,
        depth=DEFAULT_DEPTH,
        cat_features=cat_features,
        early_stopping_rounds=DEFAULT_EARLY_STOPPING,
        eval_metric="AUC",
        verbose=100,
        random_seed=DEFAULT_RANDOM_SEED,
    )

    start = time.time()
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=100)
    train_time = time.time() - start

    return model, train_time


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[Any, float]:
    """Entraine XGBoost et retourne le modele + temps."""
    from xgboost import XGBClassifier

    logger.info("  Training XGBoost...")

    model = XGBClassifier(
        n_estimators=DEFAULT_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE,
        max_depth=DEFAULT_DEPTH,
        tree_method="hist",
        early_stopping_rounds=DEFAULT_EARLY_STOPPING,
        eval_metric="auc",
        verbosity=1,
        random_state=DEFAULT_RANDOM_SEED,
    )

    start = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=100,
    )
    train_time = time.time() - start

    return model, train_time


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
) -> tuple[Any, float]:
    """Entraine LightGBM et retourne le modele + temps."""
    import lightgbm as lgb

    logger.info("  Training LightGBM...")

    # Convertir cat_features en indices
    cat_indices = [X_train.columns.get_loc(c) for c in cat_features if c in X_train.columns]

    model = lgb.LGBMClassifier(
        n_estimators=DEFAULT_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE,
        max_depth=DEFAULT_DEPTH,
        categorical_feature=cat_indices,
        early_stopping_rounds=DEFAULT_EARLY_STOPPING,
        verbose=100,
        random_state=DEFAULT_RANDOM_SEED,
    )

    start = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
    )
    train_time = time.time() - start

    return model, train_time
