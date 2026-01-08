#!/usr/bin/env python3
"""Stacking Ensemble pour ALICE.

Ce script cree un ensemble stacking avec out-of-fold predictions
combinant CatBoost, XGBoost et LightGBM.

Architecture:
    Level 0: 3 base models avec K-fold CV (OOF predictions)
    Level 1: Meta-learner (LogisticRegression) sur OOF

Includes:
    - Stacking (learned weights via meta-learner)
    - Soft Voting (simple average as baseline)

Conformite:
- ISO/IEC 42001 (AI Management)
- ISO/IEC 5259 (Data Quality for ML)
- ISO/IEC 5055 (Code Quality) - 0 Any

Usage:
    python -m scripts.ensemble_stacking
    python -m scripts.ensemble_stacking --n-folds 10
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

if TYPE_CHECKING:
    from numpy.typing import NDArray

from scripts.ml_types import MLClassifier
from scripts.model_registry import (
    compute_data_lineage,
    compute_file_checksum,
    get_environment_info,
)

# Configuration paths
PROJECT_DIR = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "features"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "hyperparameters.yaml"
DEFAULT_MODELS_DIR = PROJECT_DIR / "models"

# Model names
MODEL_NAMES: tuple[str, ...] = ("CatBoost", "XGBoost", "LightGBM")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# DATACLASSES
# ==============================================================================


@dataclass
class StackingMetrics:
    """Metriques du stacking."""

    single_models: dict[str, dict[str, float]]
    stacking_train_auc: float
    stacking_test_auc: float
    soft_voting_test_auc: float
    gain_vs_best_single: float
    gain_vs_soft_voting: float
    best_single_name: str
    best_single_auc: float


@dataclass
class StackingResult:
    """Resultat complet du stacking."""

    meta_model: MLClassifier
    oof_predictions: NDArray[np.float64]
    test_predictions: NDArray[np.float64]
    stacking_test_proba: NDArray[np.float64]
    soft_voting_test_proba: NDArray[np.float64]
    model_weights: dict[str, float]
    model_names: list[str]
    metrics: StackingMetrics
    use_stacking: bool
    n_folds: int


# ==============================================================================
# CONFIGURATION
# ==============================================================================


def load_hyperparameters(config_path: Path) -> dict[str, object]:
    """Charge les hyperparametres depuis un fichier YAML."""
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        result: dict[str, object] = yaml.safe_load(f)
        return result


# ==============================================================================
# FEATURES
# ==============================================================================

NUMERIC_FEATURES: list[str] = [
    "blanc_elo",
    "noir_elo",
    "diff_elo",
    "echiquier",
    "niveau",
    "ronde",
]

CATEGORICAL_FEATURES: list[str] = [
    "type_competition",
    "division",
    "ligue_code",
    "blanc_titre",
    "noir_titre",
    "jour_semaine",
]


def prepare_features(
    df: pd.DataFrame,
    label_encoders: dict[str, LabelEncoder] | None = None,
    *,
    fit_encoders: bool = False,
) -> tuple[pd.DataFrame, pd.Series, dict[str, LabelEncoder]]:
    """Prepare les features pour l'entrainement."""
    df = df.copy()
    y = (df["resultat_blanc"] == 1.0).astype(int)
    X_numeric = df[NUMERIC_FEATURES].fillna(0)

    if label_encoders is None:
        label_encoders = {}

    X_cat_encoded = pd.DataFrame()
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue

        df[col] = df[col].fillna("UNKNOWN").astype(str)

        if fit_encoders:
            le = LabelEncoder()
            X_cat_encoded[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le is None:
                continue
            known_classes = set(le.classes_)
            df[col] = df[col].apply(lambda x, kc=known_classes: x if x in kc else "UNKNOWN")
            if "UNKNOWN" not in known_classes:
                le.classes_ = np.append(le.classes_, "UNKNOWN")
            X_cat_encoded[col] = le.transform(df[col])

    X = pd.concat(
        [X_numeric.reset_index(drop=True), X_cat_encoded.reset_index(drop=True)],
        axis=1,
    )
    return X, y, label_encoders


# ==============================================================================
# MODEL CREATION
# ==============================================================================


def create_catboost_model(
    params: dict[str, object],
    cat_indices: list[int],
) -> MLClassifier:
    """Cree une instance CatBoost."""
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=params.get("iterations", 1000),
        learning_rate=params.get("learning_rate", 0.03),
        depth=params.get("depth", 6),
        l2_leaf_reg=params.get("l2_leaf_reg", 3),
        cat_features=cat_indices,
        early_stopping_rounds=params.get("early_stopping_rounds", 50),
        eval_metric="AUC",
        random_seed=params.get("random_seed", 42),
        verbose=0,
        thread_count=-1,
    )


def create_xgboost_model(params: dict[str, object]) -> MLClassifier:
    """Cree une instance XGBoost."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=params.get("n_estimators", 1000),
        learning_rate=params.get("learning_rate", 0.03),
        max_depth=params.get("max_depth", 6),
        reg_lambda=params.get("reg_lambda", 1.0),
        tree_method="hist",
        early_stopping_rounds=params.get("early_stopping_rounds", 50),
        eval_metric="auc",
        random_state=params.get("random_state", 42),
        verbosity=0,
        n_jobs=-1,
    )


def create_lightgbm_model(
    params: dict[str, object],
    cat_indices: list[int],
) -> MLClassifier:
    """Cree une instance LightGBM."""
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=params.get("n_estimators", 1000),
        learning_rate=params.get("learning_rate", 0.03),
        num_leaves=params.get("num_leaves", 63),
        reg_lambda=params.get("reg_lambda", 1.0),
        categorical_feature=cat_indices,
        random_state=params.get("random_state", 42),
        verbose=-1,
        n_jobs=-1,
    )


# ==============================================================================
# OOF PREDICTIONS
# ==============================================================================


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
    import lightgbm as lgb

    n_train = len(X_train)
    n_test = len(X_test)
    n_folds = kfold.n_splits

    oof_preds: NDArray[np.float64] = np.zeros(n_train, dtype=np.float64)
    test_preds_folds: NDArray[np.float64] = np.zeros((n_test, n_folds), dtype=np.float64)

    logger.info(f"  [{model_name}] Starting {n_folds}-fold CV...")
    start_time = time.time()

    # Get model params
    model_params = config.get(model_name.lower(), {})
    if not isinstance(model_params, dict):
        model_params = {}

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Create and train model
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
            msg = f"Unknown model: {model_name}"
            raise ValueError(msg)

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


# ==============================================================================
# SOFT VOTING
# ==============================================================================


def compute_soft_voting(
    test_matrix: NDArray[np.float64],
    weights: dict[str, float] | None = None,
) -> NDArray[np.float64]:
    """Calcule le soft voting (moyenne ponderee des probabilites).

    Args:
    ----
        test_matrix: Matrice (n_samples, n_models) des probabilites
        weights: Poids par modele (None = moyenne simple)

    Returns:
    -------
        Probabilites moyennees
    """
    if weights is None:
        # Simple average
        return test_matrix.mean(axis=1)

    # Weighted average
    weight_array = np.array([weights.get(name, 1.0) for name in MODEL_NAMES])
    weight_array = weight_array / weight_array.sum()  # Normalize
    return np.average(test_matrix, weights=weight_array, axis=1)


# ==============================================================================
# STACKING
# ==============================================================================


def create_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cat_features: list[str],
    config: dict[str, object],
    n_folds: int = 5,
    *,
    parallel: bool = True,
) -> StackingResult:
    """Cree un ensemble stacking avec out-of-fold predictions.

    Args:
    ----
        X_train: Features d'entrainement
        y_train: Labels d'entrainement
        X_test: Features de test
        y_test: Labels de test
        cat_features: Liste des features categorielles
        config: Configuration des hyperparametres
        n_folds: Nombre de folds pour CV
        parallel: Executer les modeles en parallele

    Returns:
    -------
        StackingResult avec toutes les metriques
    """
    logger.info("\n" + "=" * 60)
    logger.info(f"STACKING ENSEMBLE - {n_folds}-Fold CV")
    logger.info("=" * 60)

    # Convert to numpy
    X_train_np: NDArray[np.float64] = X_train.values.astype(np.float64)
    y_train_np: NDArray[np.int64] = y_train.values.astype(np.int64)
    X_test_np: NDArray[np.float64] = X_test.values.astype(np.float64)
    y_test_np: NDArray[np.int64] = y_test.values.astype(np.int64)

    # Categorical indices
    cat_indices = [X_train.columns.get_loc(c) for c in cat_features if c in X_train.columns]

    # KFold
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # OOF matrices
    n_train = len(X_train)
    n_test = len(X_test)
    n_models = len(MODEL_NAMES)

    oof_matrix: NDArray[np.float64] = np.zeros((n_train, n_models), dtype=np.float64)
    test_matrix: NDArray[np.float64] = np.zeros((n_test, n_models), dtype=np.float64)
    model_aucs: dict[str, float] = {}

    logger.info(f"\n[Level 0] Computing OOF predictions for {n_models} models...")

    if parallel:
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
    else:
        for idx, name in enumerate(MODEL_NAMES):
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

    # ==========================================================================
    # SOFT VOTING (baseline)
    # ==========================================================================
    logger.info("\n[Soft Voting] Computing simple average...")
    soft_voting_proba = compute_soft_voting(test_matrix)
    soft_voting_auc = float(roc_auc_score(y_test_np, soft_voting_proba))
    logger.info(f"  Soft Voting Test AUC: {soft_voting_auc:.4f}")

    # ==========================================================================
    # STACKING (Level 1 meta-learner)
    # ==========================================================================
    logger.info("\n[Level 1] Training meta-learner...")

    stacking_config = config.get("stacking", {})
    if not isinstance(stacking_config, dict):
        stacking_config = {}

    meta_learner_type = stacking_config.get("meta_learner", "logistic_regression")

    meta_model: MLClassifier
    if meta_learner_type == "logistic_regression":
        lr_params = stacking_config.get("logistic_regression", {})
        if not isinstance(lr_params, dict):
            lr_params = {}
        meta_model = LogisticRegression(
            C=lr_params.get("C", 1.0),
            max_iter=lr_params.get("max_iter", 1000),
            random_state=lr_params.get("random_state", 42),
        )
    else:
        ridge_params = stacking_config.get("ridge", {})
        if not isinstance(ridge_params, dict):
            ridge_params = {}
        meta_model = RidgeClassifier(alpha=ridge_params.get("alpha", 1.0))

    meta_model.fit(oof_matrix, y_train_np)

    # Stacking predictions
    if hasattr(meta_model, "predict_proba"):
        stacking_train_proba: NDArray[np.float64] = meta_model.predict_proba(oof_matrix)[:, 1]
        stacking_test_proba: NDArray[np.float64] = meta_model.predict_proba(test_matrix)[:, 1]
    else:
        # RidgeClassifier
        stacking_train_proba = 1 / (1 + np.exp(-meta_model.decision_function(oof_matrix)))
        stacking_test_proba = 1 / (1 + np.exp(-meta_model.decision_function(test_matrix)))

    stacking_train_auc = float(roc_auc_score(y_train_np, stacking_train_proba))
    stacking_test_auc = float(roc_auc_score(y_test_np, stacking_test_proba))

    # Model weights from meta-learner
    if hasattr(meta_model, "coef_"):
        raw_weights: NDArray[np.float64] = meta_model.coef_[0]
        weights_sum = float(np.sum(np.abs(raw_weights)))
        normalized_weights = {
            name: float(np.abs(raw_weights[idx]) / weights_sum)
            for idx, name in enumerate(MODEL_NAMES)
        }
    else:
        normalized_weights = dict.fromkeys(MODEL_NAMES, 1.0 / n_models)

    logger.info(f"\n  Stacking Train AUC: {stacking_train_auc:.4f}")
    logger.info(f"  Stacking Test AUC:  {stacking_test_auc:.4f}")
    logger.info("\n  Model weights:")
    for name, weight in normalized_weights.items():
        logger.info(f"    {name}: {weight:.3f}")

    # ==========================================================================
    # COMPARISON
    # ==========================================================================
    # Single model test AUCs
    single_test_aucs: dict[str, float] = {}
    for idx, name in enumerate(MODEL_NAMES):
        single_test_aucs[name] = float(roc_auc_score(y_test_np, test_matrix[:, idx]))

    best_single_name = max(single_test_aucs, key=single_test_aucs.get)  # type: ignore[arg-type]
    best_single_auc = single_test_aucs[best_single_name]

    gain_vs_best = stacking_test_auc - best_single_auc
    gain_vs_soft_voting = stacking_test_auc - soft_voting_auc

    logger.info(f"\n  Best single model (test): {best_single_name} ({best_single_auc:.4f})")
    logger.info(f"  Soft Voting (test): {soft_voting_auc:.4f}")
    logger.info(f"  Stacking gain vs best single: {gain_vs_best:+.4f}")
    logger.info(f"  Stacking gain vs soft voting: {gain_vs_soft_voting:+.4f}")

    # Decision
    selection_config = stacking_config.get("selection", {})
    if not isinstance(selection_config, dict):
        selection_config = {}
    min_gain = selection_config.get("min_gain_vs_best_single", 0.01)
    if not isinstance(min_gain, float):
        min_gain = 0.01

    use_stacking = gain_vs_best >= min_gain

    logger.info(f"\n  Decision: {'STACKING' if use_stacking else 'BEST SINGLE'}")

    # Build metrics
    single_metrics = {
        name: {"oof_auc": model_aucs[name], "test_auc": single_test_aucs[name]}
        for name in MODEL_NAMES
    }

    metrics = StackingMetrics(
        single_models=single_metrics,
        stacking_train_auc=stacking_train_auc,
        stacking_test_auc=stacking_test_auc,
        soft_voting_test_auc=soft_voting_auc,
        gain_vs_best_single=gain_vs_best,
        gain_vs_soft_voting=gain_vs_soft_voting,
        best_single_name=best_single_name,
        best_single_auc=best_single_auc,
    )

    return StackingResult(
        meta_model=meta_model,
        oof_predictions=oof_matrix,
        test_predictions=test_matrix,
        stacking_test_proba=stacking_test_proba,
        soft_voting_test_proba=soft_voting_proba,
        model_weights=normalized_weights,
        model_names=list(MODEL_NAMES),
        metrics=metrics,
        use_stacking=use_stacking,
        n_folds=n_folds,
    )


# ==============================================================================
# SAVE
# ==============================================================================


def save_stacking_ensemble(
    result: StackingResult,
    models_dir: Path,
    config: dict[str, object],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_dir: Path,
) -> Path:
    """Sauvegarde l'ensemble stacking avec conformité ISO."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = models_dir / f"stacking_v{timestamp}"
    version_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving stacking ensemble to: {version_dir}")

    # Environment info (ISO 42001 - Reproductibilité)
    env_info = get_environment_info()
    if env_info.git_commit:
        logger.info(f"  Git: {env_info.git_commit[:8]} ({env_info.git_branch})")

    # Meta-model avec checksum (ISO 27001 - Intégrité)
    meta_path = version_dir / "stacking_meta.joblib"
    joblib.dump(result.meta_model, meta_path)
    meta_checksum = compute_file_checksum(meta_path)
    logger.info(f"  Saved {meta_path.name} (SHA256: {meta_checksum[:12]}...)")

    # OOF predictions avec checksum
    oof_path = version_dir / "oof_predictions.npy"
    np.save(oof_path, result.oof_predictions)
    oof_checksum = compute_file_checksum(oof_path)
    logger.info(f"  Saved {oof_path.name} (SHA256: {oof_checksum[:12]}...)")

    # Test predictions
    test_path = version_dir / "test_predictions.npy"
    np.save(test_path, result.test_predictions)
    test_checksum = compute_file_checksum(test_path)
    logger.info(f"  Saved {test_path.name} (SHA256: {test_checksum[:12]}...)")

    # Data lineage (ISO 5259 - Traçabilité)
    train_full = pd.concat([train_df, valid_df], ignore_index=True)
    data_lineage = compute_data_lineage(
        data_dir / "train.parquet",
        data_dir / "valid.parquet",
        data_dir / "test.parquet",
        train_full,
        valid_df,
        test_df,
    )

    # Metadata complète
    metadata: dict[str, object] = {
        "version": f"stacking_v{timestamp}",
        "created_at": datetime.now().isoformat(),
        "environment": env_info.to_dict(),
        "data_lineage": data_lineage.to_dict(),
        "artifacts": {
            "meta_model": {
                "path": str(meta_path),
                "checksum": meta_checksum,
                "size_bytes": meta_path.stat().st_size,
            },
            "oof_predictions": {
                "path": str(oof_path),
                "checksum": oof_checksum,
                "size_bytes": oof_path.stat().st_size,
            },
            "test_predictions": {
                "path": str(test_path),
                "checksum": test_checksum,
                "size_bytes": test_path.stat().st_size,
            },
        },
        "n_folds": result.n_folds,
        "model_weights": result.model_weights,
        "metrics": {
            "single_models": result.metrics.single_models,
            "stacking_train_auc": result.metrics.stacking_train_auc,
            "stacking_test_auc": result.metrics.stacking_test_auc,
            "soft_voting_test_auc": result.metrics.soft_voting_test_auc,
            "gain_vs_best_single": result.metrics.gain_vs_best_single,
            "gain_vs_soft_voting": result.metrics.gain_vs_soft_voting,
            "best_single": {
                "name": result.metrics.best_single_name,
                "auc": result.metrics.best_single_auc,
            },
        },
        "use_stacking": result.use_stacking,
        "config": {"stacking": config.get("stacking", {})},
        "conformance": {
            "iso_42001": "AI Management System - Model Card",
            "iso_5259": "Data Quality - Data Lineage",
            "iso_27001": "Information Security - Checksums",
            "method": "K-Fold OOF Stacking + Soft Voting",
        },
    }

    metadata_path = version_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("  Saved metadata.json (ISO 42001 Model Card)")

    return version_dir


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================


def run_stacking(
    data_dir: Path,
    config_path: Path,
    models_dir: Path,
    n_folds: int = 5,
    *,
    parallel: bool = True,
) -> StackingResult:
    """Pipeline complet de stacking ensemble."""
    logger.info("=" * 60)
    logger.info("ALICE Engine - Stacking Ensemble")
    logger.info("ISO/IEC 42001, 5055 Conformant - 0 Any")
    logger.info("=" * 60)

    # Config
    logger.info("\n[1/4] Loading configuration...")
    config = load_hyperparameters(config_path)

    # Data
    logger.info("\n[2/4] Loading data...")
    train = pd.read_parquet(data_dir / "train.parquet")
    valid = pd.read_parquet(data_dir / "valid.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")

    # Combine train + valid for stacking
    train_full = pd.concat([train, valid], ignore_index=True)

    logger.info(f"  Train (combined): {len(train_full):,} samples")
    logger.info(f"  Test: {len(test):,} samples")

    # Features
    logger.info("\n[3/4] Preparing features...")
    X_train, y_train, encoders = prepare_features(train_full, fit_encoders=True)
    X_test, y_test, _ = prepare_features(test, label_encoders=encoders)

    cat_features = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    # Stacking
    result = create_stacking_ensemble(
        X_train,
        y_train,
        X_test,
        y_test,
        cat_features,
        config,
        n_folds=n_folds,
        parallel=parallel,
    )

    # Save (ISO 42001/5259/27001)
    logger.info("\n[4/4] Saving ensemble with production compliance...")
    version_dir = save_stacking_ensemble(
        result, models_dir, config, train_full, valid, test, data_dir
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("STACKING COMPLETE")
    logger.info("=" * 60)

    logger.info("\nSingle models (test):")
    for name in MODEL_NAMES:
        auc = result.metrics.single_models[name]["test_auc"]
        logger.info(f"  {name:10s} | AUC: {auc:.4f}")

    logger.info(f"\nSoft Voting: AUC {result.metrics.soft_voting_test_auc:.4f}")
    logger.info(f"Stacking:    AUC {result.metrics.stacking_test_auc:.4f}")
    logger.info(f"Gain vs best single: {result.metrics.gain_vs_best_single:+.4f}")
    logger.info(f"Gain vs soft voting: {result.metrics.gain_vs_soft_voting:+.4f}")
    logger.info(f"\nSelected: {'STACKING' if result.use_stacking else 'BEST SINGLE'}")
    logger.info(f"Saved to: {version_dir}")

    return result


def main() -> None:
    """Point d'entree."""
    parser = argparse.ArgumentParser(description="ALICE - Stacking Ensemble (ISO 42001)")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Repertoire des donnees features",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Fichier de configuration YAML",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Repertoire de sauvegarde",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Nombre de folds pour CV (default: 5)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Executer les modeles sequentiellement (debug)",
    )
    args = parser.parse_args()

    run_stacking(
        data_dir=args.data_dir,
        config_path=args.config,
        models_dir=args.models_dir,
        n_folds=args.n_folds,
        parallel=not args.sequential,
    )


if __name__ == "__main__":
    main()
