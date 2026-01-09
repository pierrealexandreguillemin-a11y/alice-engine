"""OOF (Out-of-Fold) predictions pour Ensemble - ISO 5055.

Ce module contient les fonctions de calcul des predictions OOF.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.ensemble.model_factory import (
    create_catboost_model,
    create_lightgbm_model,
    create_xgboost_model,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def compute_oof_for_model(
    model_name: str,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    X_test: NDArray[np.float64],
    kfold: StratifiedKFold,
    cat_indices: list[int],
    config: dict[str, object],
) -> tuple[str, NDArray[np.float64], NDArray[np.float64], float]:
    """Calcule les OOF predictions pour un modele avec K-fold CV.

    Args:
    ----
        model_name: Nom du modele
        X_train: Features train (numpy array)
        y_train: Labels train
        X_test: Features test
        kfold: StratifiedKFold splitter
        cat_indices: Indices des features categorielles
        config: Configuration hyperparametres

    Returns:
    -------
        (model_name, oof_preds, test_preds, auc_score)
    """
    n_train = len(X_train)
    n_test = len(X_test)
    n_folds = kfold.n_splits

    oof_preds: NDArray[np.float64] = np.zeros(n_train, dtype=np.float64)
    test_preds_folds: NDArray[np.float64] = np.zeros((n_test, n_folds), dtype=np.float64)

    logger.info(f"  [{model_name}] Starting {n_folds}-fold CV...")
    start_time = time.time()

    model_params = config.get(model_name.lower(), {})
    if not isinstance(model_params, dict):
        model_params = {}

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Create and train model
        model = _train_model_fold(model_name, model_params, cat_indices, X_tr, y_tr, X_val, y_val)

        # OOF predictions
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        # Test predictions (averaged over folds)
        test_preds_folds[:, fold_idx] = model.predict_proba(X_test)[:, 1]

    # Average test predictions
    test_preds: NDArray[np.float64] = test_preds_folds.mean(axis=1)

    # OOF AUC
    oof_auc = float(roc_auc_score(y_train, oof_preds))
    elapsed = time.time() - start_time

    logger.info(f"  [{model_name}] OOF AUC: {oof_auc:.4f} | Time: {elapsed:.1f}s")

    return model_name, oof_preds, test_preds, oof_auc


def _train_model_fold(
    model_name: str,
    model_params: dict[str, object],
    cat_indices: list[int],
    X_tr: NDArray[np.float64],
    y_tr: NDArray[np.int64],
    X_val: NDArray[np.float64],
    y_val: NDArray[np.int64],
) -> object:
    """Train a model for a single fold."""
    import lightgbm as lgb

    if model_name == "CatBoost":
        model = create_catboost_model(model_params, cat_indices)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    elif model_name == "XGBoost":
        model = create_xgboost_model(model_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    elif model_name == "LightGBM":
        model = create_lightgbm_model(model_params, cat_indices)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50)],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model
