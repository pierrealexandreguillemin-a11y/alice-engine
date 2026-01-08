#!/usr/bin/env python3
"""
Stacking Ensemble pour ALICE.

Ce script cree un ensemble stacking avec out-of-fold predictions
combinant CatBoost, XGBoost et LightGBM.

Architecture:
    Level 0: 3 base models avec K-fold CV (OOF predictions)
    Level 1: Meta-learner (LogisticRegression) sur OOF

Conformite:
- ISO/IEC 42001 (AI Management)
- ISO/IEC 5259 (Data Quality for ML)
- MLOps Best Practices

Usage:
    python scripts/ensemble_stacking.py
    python scripts/ensemble_stacking.py --n-folds 10
    python scripts/ensemble_stacking.py --from-trained models/current/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Configuration paths
PROJECT_DIR = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "features"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "hyperparameters.yaml"
DEFAULT_MODELS_DIR = PROJECT_DIR / "models"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================


def load_hyperparameters(config_path: Path) -> dict[str, Any]:
    """Charge les hyperparametres depuis un fichier YAML."""
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f)


# ==============================================================================
# FEATURES
# ==============================================================================

NUMERIC_FEATURES = [
    "blanc_elo",
    "noir_elo",
    "diff_elo",
    "echiquier",
    "niveau",
    "ronde",
]

CATEGORICAL_FEATURES = [
    "type_competition",
    "division",
    "ligue_code",
    "blanc_titre",
    "noir_titre",
    "jour_semaine",
]


def prepare_features(
    df: pd.DataFrame,
    label_encoders: dict | None = None,
    fit_encoders: bool = False,
) -> tuple[pd.DataFrame, pd.Series, dict]:
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

    X = pd.concat([X_numeric.reset_index(drop=True), X_cat_encoded.reset_index(drop=True)], axis=1)
    return X, y, label_encoders


# ==============================================================================
# BASE MODEL FACTORIES
# ==============================================================================


def create_catboost(params: dict[str, Any], cat_features: list[int]) -> Any:
    """Cree une instance CatBoost."""
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=params.get("iterations", 1000),
        learning_rate=params.get("learning_rate", 0.03),
        depth=params.get("depth", 6),
        l2_leaf_reg=params.get("l2_leaf_reg", 3),
        cat_features=cat_features,
        early_stopping_rounds=params.get("early_stopping_rounds", 50),
        eval_metric="AUC",
        random_seed=params.get("random_seed", 42),
        verbose=0,  # Silent for CV
        thread_count=-1,
    )


def create_xgboost(params: dict[str, Any]) -> Any:
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


def create_lightgbm(params: dict[str, Any], cat_indices: list[int]) -> Any:
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
# STACKING IMPLEMENTATION
# ==============================================================================


def compute_oof_predictions_for_model(
    model_name: str,
    model_factory: callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    kfold: StratifiedKFold,
    cat_features: list[str],
    cat_indices: list[int],
    config: dict[str, Any],
) -> tuple[str, np.ndarray, np.ndarray, float]:
    """
    Calcule les OOF predictions pour un modele avec K-fold CV.

    Args:
        model_name: Nom du modele
        model_factory: Fonction pour creer le modele
        X_train: Features train (numpy array)
        y_train: Labels train
        X_test: Features test
        kfold: StratifiedKFold splitter
        cat_features: Liste des noms de features categorielles
        cat_indices: Indices des features categorielles
        config: Configuration hyperparametres

    Returns:
        (model_name, oof_preds, test_preds, auc_score)
    """
    import lightgbm as lgb

    n_train = len(X_train)
    n_test = len(X_test)
    n_folds = kfold.n_splits

    oof_preds = np.zeros(n_train)
    test_preds_folds = np.zeros((n_test, n_folds))

    logger.info(f"  [{model_name}] Starting {n_folds}-fold CV...")
    start_time = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Creer modele selon le type
        model_params = config.get(model_name.lower(), {})

        if model_name == "CatBoost":
            model = create_catboost(model_params, cat_indices)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)

        elif model_name == "XGBoost":
            model = create_xgboost(model_params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        elif model_name == "LightGBM":
            model = create_lightgbm(model_params, cat_indices)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                ],
            )

        # OOF predictions sur validation fold
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        # Test predictions (seront moyennees)
        test_preds_folds[:, fold_idx] = model.predict_proba(X_test)[:, 1]

    # Moyenne des predictions test sur les folds
    test_preds = test_preds_folds.mean(axis=1)

    # Calculer AUC OOF
    oof_auc = roc_auc_score(y_train, oof_preds)
    elapsed = time.time() - start_time

    logger.info(f"  [{model_name}] OOF AUC: {oof_auc:.4f} | Time: {elapsed:.1f}s")

    return model_name, oof_preds, test_preds, oof_auc


def create_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cat_features: list[str],
    config: dict[str, Any],
    n_folds: int = 5,
    parallel: bool = True,
) -> dict[str, Any]:
    """
    Cree un ensemble stacking avec out-of-fold predictions.

    Args:
        X_train: Features d'entrainement
        y_train: Labels d'entrainement
        X_test: Features de test
        y_test: Labels de test
        cat_features: Liste des features categorielles
        config: Configuration des hyperparametres
        n_folds: Nombre de folds pour CV
        parallel: Executer les modeles en parallele

    Returns:
        {
            "meta_model": meta-learner entraine,
            "oof_predictions": OOF predictions train,
            "test_predictions": predictions test,
            "model_weights": poids appris par le meta-learner,
            "metrics": metriques de performance
        }
    """
    logger.info("\n" + "=" * 60)
    logger.info(f"STACKING ENSEMBLE - {n_folds}-Fold CV")
    logger.info("=" * 60)

    # Convertir en numpy pour la vitesse
    X_train_np = X_train.values
    y_train_np = y_train.values
    X_test_np = X_test.values

    # Indices des features categorielles
    cat_indices = [X_train.columns.get_loc(c) for c in cat_features if c in X_train.columns]

    # KFold stratifie
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Calculer OOF predictions pour chaque modele
    n_train = len(X_train)
    n_test = len(X_test)
    n_models = 3

    oof_matrix = np.zeros((n_train, n_models))
    test_matrix = np.zeros((n_test, n_models))
    model_names = ["CatBoost", "XGBoost", "LightGBM"]
    model_aucs = {}

    logger.info(f"\n[Level 0] Computing OOF predictions for {n_models} models...")

    if parallel:
        # Entrainement parallele
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for idx, name in enumerate(model_names):
                future = executor.submit(
                    compute_oof_predictions_for_model,
                    name,
                    None,  # model_factory not used directly
                    X_train_np,
                    y_train_np,
                    X_test_np,
                    kfold,
                    cat_features,
                    cat_indices,
                    config,
                )
                futures[future] = idx

            for future in as_completed(futures):
                idx = futures[future]
                name, oof_preds, test_preds, oof_auc = future.result()
                oof_matrix[:, idx] = oof_preds
                test_matrix[:, idx] = test_preds
                model_aucs[name] = oof_auc
    else:
        # Entrainement sequentiel
        for idx, name in enumerate(model_names):
            _, oof_preds, test_preds, oof_auc = compute_oof_predictions_for_model(
                name,
                None,
                X_train_np,
                y_train_np,
                X_test_np,
                kfold,
                cat_features,
                cat_indices,
                config,
            )
            oof_matrix[:, idx] = oof_preds
            test_matrix[:, idx] = test_preds
            model_aucs[name] = oof_auc

    # Level 1: Meta-learner
    logger.info("\n[Level 1] Training meta-learner...")

    stacking_config = config.get("stacking", {})
    meta_learner_type = stacking_config.get("meta_learner", "logistic_regression")

    if meta_learner_type == "logistic_regression":
        lr_params = stacking_config.get("logistic_regression", {})
        meta_model = LogisticRegression(
            C=lr_params.get("C", 1.0),
            max_iter=lr_params.get("max_iter", 1000),
            random_state=lr_params.get("random_state", 42),
        )
    else:  # ridge
        ridge_params = stacking_config.get("ridge", {})
        meta_model = RidgeClassifier(
            alpha=ridge_params.get("alpha", 1.0),
        )

    meta_model.fit(oof_matrix, y_train_np)

    # Predictions stacking
    if hasattr(meta_model, "predict_proba"):
        stacking_train_proba = meta_model.predict_proba(oof_matrix)[:, 1]
        stacking_test_proba = meta_model.predict_proba(test_matrix)[:, 1]
    else:
        # RidgeClassifier n'a pas predict_proba
        stacking_train_proba = meta_model.decision_function(oof_matrix)
        stacking_train_proba = 1 / (1 + np.exp(-stacking_train_proba))  # Sigmoid
        stacking_test_proba = meta_model.decision_function(test_matrix)
        stacking_test_proba = 1 / (1 + np.exp(-stacking_test_proba))

    # Metriques stacking
    stacking_train_auc = roc_auc_score(y_train_np, stacking_train_proba)
    stacking_test_auc = roc_auc_score(y_test.values, stacking_test_proba)

    # Poids du meta-learner
    if hasattr(meta_model, "coef_"):
        raw_weights = meta_model.coef_[0]
        # Normaliser pour interpretation
        weights_sum = np.sum(np.abs(raw_weights))
        normalized_weights = {
            name: float(np.abs(raw_weights[idx]) / weights_sum)
            for idx, name in enumerate(model_names)
        }
    else:
        normalized_weights = {name: 1.0 / n_models for name in model_names}

    logger.info(f"\n  Stacking Train AUC: {stacking_train_auc:.4f}")
    logger.info(f"  Stacking Test AUC:  {stacking_test_auc:.4f}")
    logger.info("\n  Model weights:")
    for name, weight in normalized_weights.items():
        logger.info(f"    {name}: {weight:.3f}")

    # Comparer avec best single model
    best_single_name = max(model_aucs, key=model_aucs.get)
    _ = model_aucs[best_single_name]  # OOF AUC, utilise pour log uniquement

    # AUC sur test pour chaque modele single
    single_test_aucs = {}
    for idx, name in enumerate(model_names):
        single_test_aucs[name] = roc_auc_score(y_test.values, test_matrix[:, idx])

    best_single_test_name = max(single_test_aucs, key=single_test_aucs.get)
    best_single_test_auc = single_test_aucs[best_single_test_name]

    gain_vs_best = stacking_test_auc - best_single_test_auc

    logger.info(
        f"\n  Best single model (test): {best_single_test_name} ({best_single_test_auc:.4f})"
    )
    logger.info(f"  Stacking gain: {gain_vs_best:+.4f} ({gain_vs_best * 100:+.2f}%)")

    # Decision: utiliser stacking ou single?
    min_gain = stacking_config.get("selection", {}).get("min_gain_vs_best_single", 0.01)
    use_stacking = gain_vs_best >= min_gain

    logger.info(f"\n  Decision: {'STACKING' if use_stacking else 'SINGLE MODEL'}")
    if use_stacking:
        logger.info(f"    Reason: gain ({gain_vs_best:.4f}) >= threshold ({min_gain})")
    else:
        logger.info(f"    Reason: gain ({gain_vs_best:.4f}) < threshold ({min_gain})")

    return {
        "meta_model": meta_model,
        "oof_predictions": oof_matrix,
        "test_predictions": test_matrix,
        "stacking_test_proba": stacking_test_proba,
        "model_weights": normalized_weights,
        "model_names": model_names,
        "metrics": {
            "single_models": {
                name: {"oof_auc": model_aucs[name], "test_auc": single_test_aucs[name]}
                for name in model_names
            },
            "stacking": {
                "train_auc": stacking_train_auc,
                "test_auc": stacking_test_auc,
                "gain_vs_best_single": gain_vs_best,
            },
            "best_single": {
                "name": best_single_test_name,
                "test_auc": best_single_test_auc,
            },
        },
        "use_stacking": use_stacking,
        "n_folds": n_folds,
    }


# ==============================================================================
# SAUVEGARDE
# ==============================================================================


def save_stacking_ensemble(
    stacking_result: dict[str, Any],
    models_dir: Path,
    config: dict[str, Any],
) -> Path:
    """
    Sauvegarde l'ensemble stacking.

    Args:
        stacking_result: Resultat de create_stacking_ensemble
        models_dir: Repertoire de destination
        config: Configuration utilisee

    Returns:
        Chemin du repertoire cree
    """
    # Creer repertoire versionne
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = models_dir / f"stacking_v{timestamp}"
    version_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving stacking ensemble to: {version_dir}")

    # Sauvegarder meta-model
    meta_path = version_dir / "stacking_meta.joblib"
    joblib.dump(stacking_result["meta_model"], meta_path)
    logger.info(f"  Saved {meta_path.name}")

    # Sauvegarder OOF predictions (utiles pour debug/analyse)
    oof_path = version_dir / "oof_predictions.npy"
    np.save(oof_path, stacking_result["oof_predictions"])
    logger.info(f"  Saved {oof_path.name}")

    # Creer metadata
    metadata = {
        "version": f"stacking_v{timestamp}",
        "created_at": datetime.now().isoformat(),
        "n_folds": stacking_result["n_folds"],
        "model_weights": stacking_result["model_weights"],
        "metrics": stacking_result["metrics"],
        "use_stacking": stacking_result["use_stacking"],
        "config": {
            "stacking": config.get("stacking", {}),
        },
        "conformance": {
            "iso_42001": "AI Management System",
            "method": "K-Fold OOF Stacking",
        },
    }

    metadata_path = version_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Saved {metadata_path.name}")

    return version_dir


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================


def run_stacking(
    data_dir: Path,
    config_path: Path,
    models_dir: Path,
    n_folds: int = 5,
    parallel: bool = True,
) -> dict:
    """
    Pipeline complet de stacking ensemble.

    Args:
        data_dir: Repertoire des donnees features
        config_path: Chemin du fichier de configuration
        models_dir: Repertoire de sauvegarde
        n_folds: Nombre de folds pour CV
        parallel: Executer les modeles en parallele

    Returns:
        Resultat du stacking
    """
    logger.info("=" * 60)
    logger.info("ALICE Engine - Stacking Ensemble")
    logger.info("ISO/IEC 42001 Conformant")
    logger.info("=" * 60)

    # Charger config
    logger.info("\n[1/4] Loading configuration...")
    config = load_hyperparameters(config_path)

    # Charger donnees
    logger.info("\n[2/4] Loading data...")
    train = pd.read_parquet(data_dir / "train.parquet")
    valid = pd.read_parquet(data_dir / "valid.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")

    # Pour le stacking, on combine train + valid pour avoir plus de donnees
    train_full = pd.concat([train, valid], ignore_index=True)

    logger.info(f"  Train (combined): {len(train_full):,} samples")
    logger.info(f"  Test: {len(test):,} samples")

    # Preparer features
    logger.info("\n[3/4] Preparing features...")
    X_train, y_train, encoders = prepare_features(train_full, fit_encoders=True)
    X_test, y_test, _ = prepare_features(test, label_encoders=encoders)

    cat_features = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    # Creer stacking ensemble
    stacking_result = create_stacking_ensemble(
        X_train,
        y_train,
        X_test,
        y_test,
        cat_features,
        config,
        n_folds=n_folds,
        parallel=parallel,
    )

    # Sauvegarder
    logger.info("\n[4/4] Saving ensemble...")
    version_dir = save_stacking_ensemble(stacking_result, models_dir, config)

    # Resume final
    logger.info("\n" + "=" * 60)
    logger.info("STACKING COMPLETE")
    logger.info("=" * 60)

    metrics = stacking_result["metrics"]
    logger.info("\nSingle models (test):")
    for name, m in metrics["single_models"].items():
        logger.info(f"  {name:10s} | AUC: {m['test_auc']:.4f}")

    logger.info(f"\nStacking: AUC {metrics['stacking']['test_auc']:.4f}")
    logger.info(f"Gain vs best single: {metrics['stacking']['gain_vs_best_single']:+.4f}")
    logger.info(f"\nSelected: {'STACKING' if stacking_result['use_stacking'] else 'SINGLE'}")
    logger.info(f"Saved to: {version_dir}")

    return stacking_result


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
