"""LightGBM Baseline - Entraînement isolé pour comparaison AutoGluon.

RÉUTILISE les modules existants (ISO 5055 - pas de duplication).

Usage:
    python -m scripts.baseline.lightgbm_baseline
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import joblib

from scripts.baseline.types import BaselineMetrics
from scripts.training import (
    CATEGORICAL_FEATURES,
    compute_all_metrics,
    load_hyperparameters,
    prepare_features,
    train_lightgbm,
)
from scripts.training.config import get_config_value

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "features"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "models" / "baseline"
DEFAULT_CONFIG = PROJECT_DIR / "config" / "hyperparameters.yaml"


def train_lightgbm_baseline(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config_path: Path = DEFAULT_CONFIG,
) -> BaselineMetrics:
    """Entraîne LightGBM baseline isolé - RÉUTILISE scripts/training."""
    import pandas as pd

    logger.info("LIGHTGBM BASELINE - Reusing scripts/training")

    config = load_hyperparameters(config_path)
    lightgbm_params = get_config_value(config, "lightgbm", {})

    train = pd.read_parquet(data_dir / "train.parquet")
    valid = pd.read_parquet(data_dir / "valid.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")

    X_train, y_train, encoders = prepare_features(train, fit_encoders=True)
    X_valid, y_valid, _ = prepare_features(valid, label_encoders=encoders)
    X_test, y_test, _ = prepare_features(test, label_encoders=encoders)

    cat_features = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    result = train_lightgbm(X_train, y_train, X_valid, y_valid, cat_features, lightgbm_params)

    y_proba = result.model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    test_metrics = compute_all_metrics(y_test.values, y_pred, y_proba)

    metrics = BaselineMetrics(
        model_name="LightGBM_Baseline",
        auc_roc=test_metrics.auc_roc,
        accuracy=test_metrics.accuracy,
        f1_score=test_metrics.f1_score,
        log_loss=test_metrics.log_loss,
        train_time_s=result.train_time,
        n_train_samples=len(train),
        n_test_samples=len(test),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result.model.booster_.save_model(str(output_dir / "lightgbm_baseline.txt"))
    joblib.dump(encoders, output_dir / "lightgbm_encoders.pkl")
    with open(output_dir / "lightgbm_metrics.json", "w") as f:
        json.dump(asdict(metrics), f, indent=2)

    logger.info(f"  AUC-ROC: {metrics.auc_roc:.4f} | Time: {metrics.train_time_s:.1f}s")
    return metrics


if __name__ == "__main__":
    train_lightgbm_baseline()
