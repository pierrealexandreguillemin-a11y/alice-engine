#!/usr/bin/env python3
"""Entrainement parallele des modeles ML pour ALICE.

Ce script entraine CatBoost, XGBoost et LightGBM en parallele
avec ThreadPoolExecutor, tracking MLflow et sauvegarde des modeles.

Conformite:
- ISO/IEC 42001 (AI Management)
- ISO/IEC 5259 (Data Quality for ML)
- ISO/IEC 5055 (Code Quality) - 0 Any

Usage:
    python -m scripts.train_models_parallel
    python -m scripts.train_models_parallel --config config/hyperparameters.yaml
    python -m scripts.train_models_parallel --no-mlflow
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

from scripts.ml_types import (
    CatBoostConfig,
    CatBoostModel,
    LightGBMConfig,
    LightGBMModel,
    ModelMetrics,
    ModelResults,
    TrainingResult,
    XGBoostConfig,
    XGBoostModel,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

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


def load_hyperparameters(config_path: Path) -> dict[str, object]:
    """Charge les hyperparametres depuis un fichier YAML."""
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}, using defaults")
        return get_default_hyperparameters()

    with config_path.open() as f:
        result: dict[str, object] = yaml.safe_load(f)
        return result


def get_default_hyperparameters() -> dict[str, object]:
    """Retourne les hyperparametres par defaut."""
    return {
        "global": {
            "random_seed": 42,
            "early_stopping_rounds": 50,
            "verbose": 100,
        },
        "catboost": {
            "iterations": 1000,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 3,
            "random_seed": 42,
            "verbose": 100,
            "early_stopping_rounds": 50,
        },
        "xgboost": {
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "max_depth": 6,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "random_state": 42,
            "verbosity": 1,
            "early_stopping_rounds": 50,
        },
        "lightgbm": {
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbose": -1,
            "early_stopping_rounds": 50,
        },
    }


def _get_config_value(config: dict[str, object], key: str, default: object) -> object:
    """Extrait une valeur de config avec default."""
    return config.get(key, default)


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
    """Prepare les features pour l'entrainement.

    Args:
        df: DataFrame brut
        label_encoders: encodeurs existants (pour valid/test)
        fit_encoders: True pour train, False pour valid/test

    Returns:
        X, y, label_encoders
    """
    df = df.copy()

    # Target: resultat blanc (0=defaite, 0.5=nulle, 1=victoire)
    y = (df["resultat_blanc"] == 1.0).astype(int)

    # Features numeriques
    X_numeric = df[NUMERIC_FEATURES].fillna(0)

    # Features categorielles - encodage
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

    # Combiner
    X = pd.concat(
        [X_numeric.reset_index(drop=True), X_cat_encoded.reset_index(drop=True)],
        axis=1,
    )

    return X, y, label_encoders


# ==============================================================================
# METRIQUES (ISO 25010)
# ==============================================================================


def compute_all_metrics(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    y_proba: NDArray[np.float64],
) -> ModelMetrics:
    """Calcule toutes les metriques conformes ISO 25010."""
    cm = confusion_matrix(y_true, y_pred)

    return ModelMetrics(
        auc_roc=float(roc_auc_score(y_true, y_proba)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1_score=float(f1_score(y_true, y_pred, zero_division=0)),
        log_loss=float(log_loss(y_true, y_proba)),
        true_negatives=int(cm[0, 0]),
        false_positives=int(cm[0, 1]),
        false_negatives=int(cm[1, 0]),
        true_positives=int(cm[1, 1]),
    )


# ==============================================================================
# ENTRAINEMENT DES MODELES
# ==============================================================================


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


# ==============================================================================
# ENTRAINEMENT PARALLELE
# ==============================================================================


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

    catboost_config = config.get("catboost", {})
    xgboost_config = config.get("xgboost", {})
    lightgbm_config = config.get("lightgbm", {})

    # Type narrowing
    if not isinstance(catboost_config, dict):
        catboost_config = {}
    if not isinstance(xgboost_config, dict):
        xgboost_config = {}
    if not isinstance(lightgbm_config, dict):
        lightgbm_config = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
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

        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                results[name] = result
                logger.info(f"[{name}] Completed successfully")
            except Exception as e:
                logger.exception(f"[{name}] FAILED: {e}")
                results[name] = TrainingResult(
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

    return results


# ==============================================================================
# SAUVEGARDE DES MODELES
# ==============================================================================


def save_models(
    results: ModelResults,
    models_dir: Path,
    label_encoders: dict[str, LabelEncoder],
    config: dict[str, object],
) -> Path:
    """Sauvegarde les modeles et metadata (Model Card)."""
    # Creer repertoire versionne
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = models_dir / f"v{timestamp}"
    version_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving models to: {version_dir}")

    # Sauvegarder chaque modele
    for name, result in results.items():
        if result.model is None:
            continue

        if name == "CatBoost":
            model_path = version_dir / "catboost.cbm"
            model = result.model
            if hasattr(model, "save_model"):
                model.save_model(str(model_path))  # type: ignore[union-attr]
            logger.info(f"  Saved {model_path.name}")

        elif name == "XGBoost":
            model_path = version_dir / "xgboost.ubj"
            model = result.model
            if hasattr(model, "save_model"):
                model.save_model(str(model_path))  # type: ignore[union-attr]
            logger.info(f"  Saved {model_path.name}")

        elif name == "LightGBM":
            model_path = version_dir / "lightgbm.txt"
            model = result.model
            if hasattr(model, "booster_"):
                model.booster_.save_model(str(model_path))  # type: ignore[union-attr]
            logger.info(f"  Saved {model_path.name}")

    # Sauvegarder label encoders
    encoders_path = version_dir / "label_encoders.joblib"
    joblib.dump(label_encoders, encoders_path)
    logger.info(f"  Saved {encoders_path.name}")

    # Creer Model Card (metadata.json)
    model_card = create_model_card(results, config, timestamp)
    metadata_path = version_dir / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(model_card, f, indent=2)
    logger.info(f"  Saved {metadata_path.name}")

    # Mettre a jour symlink "current"
    current_link = models_dir / "current"
    if current_link.exists() or current_link.is_symlink():
        current_link.unlink()

    # Sur Windows, utiliser junction au lieu de symlink
    import platform
    import subprocess

    if platform.system() == "Windows":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(current_link), str(version_dir)],
            capture_output=True,
            check=False,
        )
    else:
        current_link.symlink_to(version_dir.name)

    logger.info(f"  Updated 'current' -> {version_dir.name}")

    return version_dir


def create_model_card(
    results: ModelResults,
    config: dict[str, object],
    timestamp: str,
) -> dict[str, object]:
    """Cree une Model Card (ISO 42001) avec metadata et metriques."""
    import catboost
    import lightgbm
    import sklearn
    import xgboost

    # Trouver le meilleur modele
    best_name: str | None = None
    best_auc = 0.0
    metrics_all: dict[str, dict[str, float]] = {}

    for name, result in results.items():
        if result.model is None:
            continue
        auc = result.metrics.auc_roc
        metrics_all[name.lower()] = {
            "auc": auc,
            "accuracy": result.metrics.accuracy,
            "f1": result.metrics.f1_score,
            "precision": result.metrics.precision,
            "recall": result.metrics.recall,
            "log_loss": result.metrics.log_loss,
            "train_time_s": result.metrics.train_time_s,
        }
        if auc > best_auc:
            best_auc = auc
            best_name = name

    return {
        "version": f"v{timestamp}",
        "created_at": datetime.now().isoformat(),
        "framework": {
            "catboost": catboost.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
            "sklearn": sklearn.__version__,
        },
        "dataset": {
            "features_numeric": NUMERIC_FEATURES,
            "features_categorical": CATEGORICAL_FEATURES,
            "target": "resultat_blanc",
        },
        "metrics": metrics_all,
        "best_single_model": {
            "name": best_name,
            "auc": best_auc,
        },
        "hyperparameters": {
            "catboost": config.get("catboost", {}),
            "xgboost": config.get("xgboost", {}),
            "lightgbm": config.get("lightgbm", {}),
        },
        "training": {
            "parallel": True,
            "max_workers": 3,
        },
        "conformance": {
            "iso_42001": "AI Management System",
            "iso_5259": "Data Quality for ML",
            "iso_29119": "Software Testing",
        },
    }


# ==============================================================================
# MLFLOW TRACKING
# ==============================================================================


def setup_mlflow(config: dict[str, object]) -> bool:
    """Configure MLflow si disponible."""
    try:
        import mlflow

        mlflow_config = config.get("mlflow", {})
        if not isinstance(mlflow_config, dict):
            mlflow_config = {}

        experiment_name = mlflow_config.get("experiment_name", "alice-ml-training")
        tracking_uri = mlflow_config.get("tracking_uri", "./mlruns")

        if not isinstance(experiment_name, str):
            experiment_name = "alice-ml-training"
        if not isinstance(tracking_uri, str):
            tracking_uri = "./mlruns"

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        logger.info(f"MLflow tracking: {tracking_uri} / {experiment_name}")
        return True
    except ImportError:
        logger.warning("MLflow not installed, skipping experiment tracking")
        return False


def log_to_mlflow(
    results: ModelResults,
    config: dict[str, object],
) -> None:
    """Log les resultats dans MLflow."""
    try:
        import mlflow

        with mlflow.start_run(run_name="parallel_training"):
            # Log config globale
            mlflow.log_param("parallel_workers", 3)
            global_config = config.get("global", {})
            if isinstance(global_config, dict):
                mlflow.log_param("random_seed", global_config.get("random_seed", 42))

            for name, result in results.items():
                if result.model is None:
                    continue

                with mlflow.start_run(run_name=name, nested=True):
                    # Params
                    model_config = config.get(name.lower(), {})
                    if isinstance(model_config, dict):
                        for key, value in model_config.items():
                            if isinstance(value, int | float | str | bool):
                                mlflow.log_param(key, value)

                    # Metrics
                    metrics_dict = result.metrics.to_dict()
                    for metric_name, metric_value in metrics_dict.items():
                        if isinstance(metric_value, int | float):
                            mlflow.log_metric(metric_name, metric_value)

            logger.info("Results logged to MLflow")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================


def run_training(
    data_dir: Path,
    config_path: Path,
    models_dir: Path,
    *,
    use_mlflow: bool = True,
) -> dict[str, object]:
    """Pipeline complet d'entrainement parallele."""
    logger.info("=" * 60)
    logger.info("ALICE Engine - Parallel ML Training")
    logger.info("ISO/IEC 42001, 5259, 29119 Conformant")
    logger.info("=" * 60)

    # Charger config
    logger.info("\n[1/5] Loading configuration...")
    config = load_hyperparameters(config_path)
    logger.info(f"  Config: {config_path}")

    # Setup MLflow
    if use_mlflow:
        setup_mlflow(config)

    # Charger donnees
    logger.info("\n[2/5] Loading data...")
    train = pd.read_parquet(data_dir / "train.parquet")
    valid = pd.read_parquet(data_dir / "valid.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")

    logger.info(f"  Train: {len(train):,} samples")
    logger.info(f"  Valid: {len(valid):,} samples")
    logger.info(f"  Test:  {len(test):,} samples")

    # Preparer features
    logger.info("\n[3/5] Preparing features...")
    X_train, y_train, encoders = prepare_features(train, fit_encoders=True)
    X_valid, y_valid, _ = prepare_features(valid, label_encoders=encoders)
    X_test, y_test, _ = prepare_features(test, label_encoders=encoders)

    logger.info(f"  Features: {X_train.shape[1]}")
    logger.info(f"  Target balance: {y_train.mean():.2%} positive class")

    cat_features = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    # Entrainement parallele
    logger.info("\n[4/5] Training models in parallel...")
    results = train_all_models_parallel(X_train, y_train, X_valid, y_valid, cat_features, config)

    # Evaluation sur test set
    logger.info("\n[4.5/5] Evaluating on test set...")
    for name, result in results.items():
        if result.model is None:
            continue
        y_proba: NDArray[np.float64] = result.model.predict_proba(X_test)[:, 1]
        y_pred: NDArray[np.int64] = (y_proba >= 0.5).astype(np.int64)
        test_metrics = compute_all_metrics(y_test.values, y_pred, y_proba)
        result.metrics.test_auc = test_metrics.auc_roc
        result.metrics.test_accuracy = test_metrics.accuracy
        result.metrics.test_f1 = test_metrics.f1_score
        logger.info(f"  [{name}] Test AUC: {test_metrics.auc_roc:.4f}")

    # Sauvegarder modeles
    logger.info("\n[5/5] Saving models...")
    version_dir = save_models(results, models_dir, encoders, config)

    # MLflow logging
    if use_mlflow:
        log_to_mlflow(results, config)

    # Resume
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    best_name: str | None = None
    best_auc = 0.0
    for name, result in results.items():
        if result.model is None:
            continue
        auc = result.metrics.test_auc or result.metrics.auc_roc
        logger.info(
            f"  {name:10s} | Test AUC: {auc:.4f} | " f"Train: {result.metrics.train_time_s:.1f}s"
        )
        if auc > best_auc:
            best_auc = auc
            best_name = name

    logger.info(f"\nBest single model: {best_name} (AUC: {best_auc:.4f})")
    logger.info(f"Models saved to: {version_dir}")

    return {
        "results": {name: result.metrics.to_dict() for name, result in results.items()},
        "best_model": best_name,
        "best_auc": best_auc,
        "version_dir": str(version_dir),
    }


def main() -> None:
    """Point d'entree."""
    parser = argparse.ArgumentParser(description="ALICE - Parallel ML Training (ISO 42001)")
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
        help="Repertoire de sauvegarde des modeles",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Desactiver le tracking MLflow",
    )
    args = parser.parse_args()

    run_training(
        data_dir=args.data_dir,
        config_path=args.config,
        models_dir=args.models_dir,
        use_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
