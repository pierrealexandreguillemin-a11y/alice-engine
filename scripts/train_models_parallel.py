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
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scripts.ml_types import ModelResults
from scripts.model_registry import save_production_models
from scripts.training import (
    CATEGORICAL_FEATURES,
    compute_all_metrics,
    load_hyperparameters,
    log_to_mlflow,
    prepare_features,
    setup_mlflow,
    train_all_models_parallel,
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


def _log_header() -> None:
    """Log training header."""
    logger.info("=" * 60)
    logger.info("ALICE Engine - Parallel ML Training")
    logger.info("ISO/IEC 42001, 5259, 29119 Conformant")
    logger.info("=" * 60)


def _prepare_all_features(
    train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict, list]:
    """Prepare features for all datasets."""
    logger.info("\n[3/5] Preparing features...")
    X_train, y_train, encoders = prepare_features(train, fit_encoders=True)
    X_valid, y_valid, _ = prepare_features(valid, label_encoders=encoders)
    X_test, y_test, _ = prepare_features(test, label_encoders=encoders)
    logger.info(f"  Features: {X_train.shape[1]}")
    logger.info(f"  Target balance: {y_train.mean():.2%} positive class")
    cat_features = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    return X_train, y_train, X_valid, y_valid, X_test, y_test, encoders, cat_features


def run_training(
    data_dir: Path,
    config_path: Path,
    models_dir: Path,
    *,
    use_mlflow: bool = True,
) -> dict[str, object]:
    """Pipeline complet d'entrainement parallele."""
    _log_header()

    logger.info("\n[1/5] Loading configuration...")
    config = load_hyperparameters(config_path)
    logger.info(f"  Config: {config_path}")
    if use_mlflow:
        setup_mlflow(config)

    logger.info("\n[2/5] Loading data...")
    train, valid, test = _load_datasets(data_dir)
    X_train, y_train, X_valid, y_valid, X_test, y_test, encoders, cat_features = (
        _prepare_all_features(train, valid, test)
    )

    logger.info("\n[4/5] Training models in parallel...")
    results = train_all_models_parallel(X_train, y_train, X_valid, y_valid, cat_features, config)

    logger.info("\n[4.5/5] Evaluating on test set...")
    _evaluate_on_test(results, X_test, y_test)

    logger.info("\n[5/5] Saving models with production compliance...")
    version_dir = _save_models(
        results, config, models_dir, train, valid, test, data_dir, X_train, encoders
    )
    if use_mlflow:
        log_to_mlflow(results, config)

    return _build_summary(results, version_dir)


def _load_datasets(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge les datasets train/valid/test."""
    train = pd.read_parquet(data_dir / "train.parquet")
    valid = pd.read_parquet(data_dir / "valid.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")

    logger.info(f"  Train: {len(train):,} samples")
    logger.info(f"  Valid: {len(valid):,} samples")
    logger.info(f"  Test:  {len(test):,} samples")

    return train, valid, test


def _evaluate_on_test(
    results: ModelResults,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Évalue les modèles sur le test set."""
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


def _save_models(
    results: ModelResults,
    config: dict[str, object],
    models_dir: Path,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    data_dir: Path,
    X_train: pd.DataFrame,
    encoders: dict[str, object],
) -> Path:
    """Sauvegarde les modèles avec conformité ISO."""
    models_dict = {name: result.model for name, result in results.items() if result.model}
    metrics_dict = {
        name: {
            "auc_roc": result.metrics.auc_roc,
            "accuracy": result.metrics.accuracy,
            "precision": result.metrics.precision,
            "recall": result.metrics.recall,
            "f1_score": result.metrics.f1_score,
            "log_loss": result.metrics.log_loss,
            "train_time_s": result.metrics.train_time_s,
            "test_auc": result.metrics.test_auc,
        }
        for name, result in results.items()
        if result.model
    }
    return save_production_models(
        models=models_dict,
        metrics=metrics_dict,
        models_dir=models_dir,
        train_df=train,
        valid_df=valid,
        test_df=test,
        train_path=data_dir / "train.parquet",
        valid_path=data_dir / "valid.parquet",
        test_path=data_dir / "test.parquet",
        feature_names=list(X_train.columns),
        hyperparameters=config,
        label_encoders=encoders,
    )


def _build_summary(results: ModelResults, version_dir: Path) -> dict[str, object]:
    """Construit le résumé des résultats."""
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
            f"  {name:10s} | Test AUC: {auc:.4f} | Train: {result.metrics.train_time_s:.1f}s"
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
