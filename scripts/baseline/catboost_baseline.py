"""CatBoost Baseline - Entraînement isolé pour comparaison AutoGluon.

Ce script entraîne CatBoost de manière INDÉPENDANTE d'AutoGluon,
RÉUTILISANT les modules existants (ISO 5055 - pas de duplication).

ISO Compliance:
- ISO/IEC 24029:2021 - Independent implementation for comparison
- ISO/IEC 5055:2021 - Code Quality (no duplication)

Usage:
    python -m scripts.baseline.catboost_baseline

Author: ALICE Engine Team
Version: 1.1.0 (refactored - reuses existing modules)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

from scripts.baseline.types import BaselineMetrics
from scripts.training import (
    CATEGORICAL_FEATURES,
    compute_all_metrics,
    load_hyperparameters,
    prepare_features,
    train_catboost,
)
from scripts.training.config import get_config_value

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "features"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "models" / "baseline"
DEFAULT_CONFIG = PROJECT_DIR / "config" / "hyperparameters.yaml"


def train_catboost_baseline(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config_path: Path = DEFAULT_CONFIG,
) -> BaselineMetrics:
    """Entraîne CatBoost baseline isolé - RÉUTILISE scripts/training."""
    import pandas as pd

    logger.info("=" * 60)
    logger.info("CATBOOST BASELINE - Reusing scripts/training")
    logger.info("=" * 60)

    # Charger config existante
    config = load_hyperparameters(config_path)
    catboost_params = get_config_value(config, "catboost", {})

    # Charger données
    logger.info("\n[1/4] Loading data...")
    train = pd.read_parquet(data_dir / "train.parquet")
    valid = pd.read_parquet(data_dir / "valid.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    logger.info(f"  Train: {len(train):,} | Valid: {len(valid):,} | Test: {len(test):,}")

    # Réutiliser prepare_features existant
    logger.info("\n[2/4] Preparing features (reusing scripts/training)...")
    X_train, y_train, encoders = prepare_features(train, fit_encoders=True)
    X_valid, y_valid, _ = prepare_features(valid, label_encoders=encoders)
    X_test, y_test, _ = prepare_features(test, label_encoders=encoders)

    cat_features = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    # Réutiliser train_catboost existant
    logger.info("\n[3/4] Training (reusing scripts/training/trainers)...")
    result = train_catboost(X_train, y_train, X_valid, y_valid, cat_features, catboost_params)

    # Évaluer sur test
    logger.info("\n[4/4] Evaluating on test set...")
    y_proba = result.model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    test_metrics = compute_all_metrics(y_test.values, y_pred, y_proba)

    metrics = BaselineMetrics(
        model_name="CatBoost_Baseline",
        auc_roc=test_metrics.auc_roc,
        accuracy=test_metrics.accuracy,
        f1_score=test_metrics.f1_score,
        log_loss=test_metrics.log_loss,
        train_time_s=result.train_time,
        n_train_samples=len(train),
        n_test_samples=len(test),
    )

    # Sauvegarder
    output_dir.mkdir(parents=True, exist_ok=True)
    result.model.save_model(str(output_dir / "catboost_baseline.cbm"))
    joblib.dump(encoders, output_dir / "catboost_encoders.pkl")
    np.save(output_dir / "catboost_predictions.npy", y_pred)
    with open(output_dir / "catboost_metrics.json", "w") as f:
        json.dump(asdict(metrics), f, indent=2)

    logger.info(f"\n  AUC-ROC: {metrics.auc_roc:.4f} | Time: {metrics.train_time_s:.1f}s")
    return metrics


if __name__ == "__main__":
    train_catboost_baseline()
