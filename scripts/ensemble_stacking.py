#!/usr/bin/env python3
"""Stacking Ensemble pour ALICE.

Ce script cree un ensemble stacking avec out-of-fold predictions
combinant CatBoost, XGBoost et LightGBM.

Architecture:
    Level 0: 3 base models avec K-fold CV (OOF predictions)
    Level 1: Meta-learner (LogisticRegression) sur OOF

Conformite:
- ISO/IEC 42001 (AI Management)
- ISO/IEC 5259 (Data Quality for ML)
- ISO/IEC 5055 (Code Quality) - Modular structure

Usage:
    python -m scripts.ensemble_stacking
    python -m scripts.ensemble_stacking --n-folds 10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from scripts.ensemble import (
    MODEL_NAMES,
    StackingResult,
    create_stacking_ensemble,
    save_stacking_ensemble,
)
from scripts.training import CATEGORICAL_FEATURES, prepare_features

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


def load_hyperparameters(config_path: Path) -> dict[str, object]:
    """Charge les hyperparametres depuis un fichier YAML."""
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        result: dict[str, object] = yaml.safe_load(f)
        return result


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
    logger.info("ISO/IEC 42001, 5055 Conformant")
    logger.info("=" * 60)

    config = load_hyperparameters(config_path)

    # Load data
    train = pd.read_parquet(data_dir / "train.parquet")
    valid = pd.read_parquet(data_dir / "valid.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")

    train_full = pd.concat([train, valid], ignore_index=True)
    logger.info(f"  Train (combined): {len(train_full):,} samples")
    logger.info(f"  Test: {len(test):,} samples")

    # Prepare features
    X_train, y_train, encoders = prepare_features(train_full, fit_encoders=True)
    X_test, y_test, _ = prepare_features(test, label_encoders=encoders)
    cat_features = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    # Create stacking ensemble
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

    # Save ensemble
    version_dir = save_stacking_ensemble(
        result, models_dir, config, train_full, valid, test, data_dir
    )

    # Log summary
    _log_summary(result, version_dir)
    return result


def _log_summary(result: StackingResult, version_dir: Path) -> None:
    """Log final summary."""
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
    logger.info(f"Selected: {'STACKING' if result.use_stacking else 'BEST SINGLE'}")
    logger.info(f"Saved to: {version_dir}")


def main() -> None:
    """Point d'entree."""
    parser = argparse.ArgumentParser(description="ALICE - Stacking Ensemble (ISO 42001)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--sequential", action="store_true")
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
