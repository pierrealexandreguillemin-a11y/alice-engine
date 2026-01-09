"""Module: stacking.py - Stacking Ensemble avec OOF predictions et meta-learner.

Ce module crée un ensemble stacking (Level 0 + Level 1).
Mode séquentiel par défaut pour économiser la RAM (gc.collect après chaque modèle).

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (Métriques qualité, Explicabilité)
- ISO/IEC 25059:2023 - AI Quality Model (Benchmarks ensemble)
- ISO/IEC 23894:2023 - AI Risk Management (Gestion ressources mémoire)
- ISO/IEC 5055 - Code Quality (0 CWE critiques)

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

from __future__ import annotations

import gc
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np

from scripts.ensemble.oof import compute_oof_for_model
from scripts.ensemble.types import StackingMetrics, StackingResult
from scripts.ensemble.voting import MODEL_NAMES, compute_soft_voting

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray
    from sklearn.model_selection import StratifiedKFold

    from scripts.ml_types import MLClassifier

logger = logging.getLogger(__name__)


def create_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cat_features: list[str],
    config: dict[str, object],
    n_folds: int = 5,
    *,
    parallel: bool = False,  # Sequential by default (RAM optimized)
) -> StackingResult:
    """Cree un ensemble stacking avec out-of-fold predictions."""
    from sklearn.metrics import roc_auc_score  # Lazy import
    from sklearn.model_selection import StratifiedKFold  # Lazy import

    logger.info("\n" + "=" * 60)
    logger.info(f"STACKING ENSEMBLE - {n_folds}-Fold CV")
    logger.info("=" * 60)

    # Convert to numpy
    X_train_np: NDArray[np.float64] = X_train.values.astype(np.float64)
    y_train_np: NDArray[np.int64] = y_train.values.astype(np.int64)
    X_test_np: NDArray[np.float64] = X_test.values.astype(np.float64)
    y_test_np: NDArray[np.int64] = y_test.values.astype(np.int64)

    cat_indices = [X_train.columns.get_loc(c) for c in cat_features if c in X_train.columns]
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    logger.info(f"\n[Level 0] Computing OOF predictions for {len(MODEL_NAMES)} models...")

    # Compute OOF predictions
    oof_matrix, test_matrix, model_aucs = _compute_all_oof(
        X_train_np, y_train_np, X_test_np, kfold, cat_indices, config, parallel
    )

    # Soft voting
    soft_voting_proba = compute_soft_voting(test_matrix)
    soft_voting_auc = float(roc_auc_score(y_test_np, soft_voting_proba))
    logger.info(f"\n[Soft Voting] Test AUC: {soft_voting_auc:.4f}")

    # Meta-learner
    meta_model, stacking_train_auc, stacking_test_auc, stacking_test_proba = _train_meta_learner(
        oof_matrix, y_train_np, test_matrix, y_test_np, config
    )

    # Model weights from meta-learner
    model_weights = _extract_model_weights(meta_model)

    # Metrics
    metrics = _compute_metrics(
        model_aucs, stacking_train_auc, stacking_test_auc, soft_voting_auc, y_test_np, test_matrix
    )

    # Decide stacking vs best single
    use_stacking = stacking_test_auc >= metrics.best_single_auc

    return StackingResult(
        meta_model=meta_model,
        oof_predictions=oof_matrix,
        test_predictions=test_matrix,
        stacking_test_proba=stacking_test_proba,
        soft_voting_test_proba=soft_voting_proba,
        model_weights=model_weights,
        model_names=list(MODEL_NAMES),
        metrics=metrics,
        use_stacking=use_stacking,
        n_folds=n_folds,
    )


def _compute_all_oof(
    X_train_np: NDArray[np.float64],
    y_train_np: NDArray[np.int64],
    X_test_np: NDArray[np.float64],
    kfold: StratifiedKFold,
    cat_indices: list[int],
    config: dict[str, object],
    parallel: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float]]:
    """Compute OOF predictions for all models."""
    n_train = len(X_train_np)
    n_test = len(X_test_np)
    n_models = len(MODEL_NAMES)

    oof_matrix = np.zeros((n_train, n_models), dtype=np.float64)
    test_matrix = np.zeros((n_test, n_models), dtype=np.float64)
    model_aucs: dict[str, float] = {}

    if parallel:
        _compute_oof_parallel(
            X_train_np,
            y_train_np,
            X_test_np,
            kfold,
            cat_indices,
            config,
            oof_matrix,
            test_matrix,
            model_aucs,
        )
    else:
        _compute_oof_sequential(
            X_train_np,
            y_train_np,
            X_test_np,
            kfold,
            cat_indices,
            config,
            oof_matrix,
            test_matrix,
            model_aucs,
        )

    return oof_matrix, test_matrix, model_aucs


def _compute_oof_parallel(
    X_train_np: NDArray[np.float64],
    y_train_np: NDArray[np.int64],
    X_test_np: NDArray[np.float64],
    kfold: StratifiedKFold,
    cat_indices: list[int],
    config: dict[str, object],
    oof_matrix: NDArray[np.float64],
    test_matrix: NDArray[np.float64],
    model_aucs: dict[str, float],
) -> None:
    """Compute OOF in parallel."""
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                compute_oof_for_model,
                name,
                X_train_np,
                y_train_np,
                X_test_np,
                kfold,
                cat_indices,
                config,
            ): idx
            for idx, name in enumerate(MODEL_NAMES)
        }
        for future in as_completed(futures):
            idx = futures[future]
            name, oof_preds, test_preds, oof_auc = future.result()
            oof_matrix[:, idx] = oof_preds
            test_matrix[:, idx] = test_preds
            model_aucs[name] = oof_auc


def _compute_oof_sequential(
    X_train_np: NDArray[np.float64],
    y_train_np: NDArray[np.int64],
    X_test_np: NDArray[np.float64],
    kfold: StratifiedKFold,
    cat_indices: list[int],
    config: dict[str, object],
    oof_matrix: NDArray[np.float64],
    test_matrix: NDArray[np.float64],
    model_aucs: dict[str, float],
) -> None:
    """Compute OOF sequentially (RAM optimized)."""
    for idx, name in enumerate(MODEL_NAMES):
        logger.info(f"  [{name}] Computing OOF predictions...")
        _, oof_preds, test_preds, oof_auc = compute_oof_for_model(
            name,
            X_train_np,
            y_train_np,
            X_test_np,
            kfold,
            cat_indices,
            config,
        )
        oof_matrix[:, idx] = oof_preds
        test_matrix[:, idx] = test_preds
        model_aucs[name] = oof_auc
        logger.info(f"  [{name}] OOF AUC: {oof_auc:.4f}")
        gc.collect()  # Free memory after each model


def _train_meta_learner(
    oof_matrix: NDArray[np.float64],
    y_train_np: NDArray[np.int64],
    test_matrix: NDArray[np.float64],
    y_test_np: NDArray[np.int64],
    config: dict[str, object],
) -> tuple[MLClassifier, float, float, NDArray[np.float64]]:
    """Train meta-learner on OOF predictions."""
    from sklearn.metrics import roc_auc_score  # Lazy import

    logger.info("\n[Level 1] Training meta-learner...")

    meta_model = _create_meta_model(config)
    meta_model.fit(oof_matrix, y_train_np)

    # Train AUC
    train_proba = meta_model.predict_proba(oof_matrix)[:, 1]
    stacking_train_auc = float(roc_auc_score(y_train_np, train_proba))

    # Test AUC
    test_proba = meta_model.predict_proba(test_matrix)[:, 1]
    stacking_test_auc = float(roc_auc_score(y_test_np, test_proba))

    logger.info(f"  Stacking Train AUC: {stacking_train_auc:.4f}")
    logger.info(f"  Stacking Test AUC:  {stacking_test_auc:.4f}")

    return meta_model, stacking_train_auc, stacking_test_auc, test_proba


def _create_meta_model(config: dict[str, object]) -> MLClassifier:
    """Create meta-learner model from config."""
    from sklearn.linear_model import LogisticRegression, RidgeClassifier  # Lazy import

    stacking_config = config.get("stacking", {})
    if not isinstance(stacking_config, dict):
        stacking_config = {}

    meta_learner_type = stacking_config.get("meta_learner", "logistic_regression")

    if meta_learner_type == "logistic_regression":
        lr_params = stacking_config.get("logistic_regression", {})
        if not isinstance(lr_params, dict):
            lr_params = {}
        return LogisticRegression(
            C=lr_params.get("C", 1.0),
            max_iter=lr_params.get("max_iter", 1000),
            random_state=lr_params.get("random_state", 42),
        )

    ridge_params = stacking_config.get("ridge", {})
    if not isinstance(ridge_params, dict):
        ridge_params = {}
    return RidgeClassifier(
        alpha=ridge_params.get("alpha", 1.0),
        random_state=ridge_params.get("random_state", 42),
    )


def _extract_model_weights(meta_model: MLClassifier) -> dict[str, float]:
    """Extract model weights from meta-learner."""
    if hasattr(meta_model, "coef_"):
        coefs = meta_model.coef_[0]
        weights = np.abs(coefs) / np.abs(coefs).sum()
        return {name: float(w) for name, w in zip(MODEL_NAMES, weights, strict=False)}
    return {name: 1.0 / len(MODEL_NAMES) for name in MODEL_NAMES}


def _compute_metrics(
    model_aucs: dict[str, float],
    stacking_train_auc: float,
    stacking_test_auc: float,
    soft_voting_auc: float,
    y_test_np: NDArray[np.int64],
    test_matrix: NDArray[np.float64],
) -> StackingMetrics:
    """Compute final metrics."""
    from sklearn.metrics import roc_auc_score  # Lazy import

    # Single model test AUCs
    single_models: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(MODEL_NAMES):
        test_auc = float(roc_auc_score(y_test_np, test_matrix[:, idx]))
        single_models[name] = {"oof_auc": model_aucs[name], "test_auc": test_auc}

    # Best single model
    best_name = max(single_models, key=lambda x: single_models[x]["test_auc"])
    best_auc = single_models[best_name]["test_auc"]

    return StackingMetrics(
        single_models=single_models,
        stacking_train_auc=stacking_train_auc,
        stacking_test_auc=stacking_test_auc,
        soft_voting_test_auc=soft_voting_auc,
        gain_vs_best_single=stacking_test_auc - best_auc,
        gain_vs_soft_voting=stacking_test_auc - soft_voting_auc,
        best_single_name=best_name,
        best_single_auc=best_auc,
    )
