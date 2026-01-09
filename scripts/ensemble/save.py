"""Sauvegarde de l'ensemble stacking - ISO 5055.

Ce module contient les fonctions de sauvegarde avec conformité ISO.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd

from scripts.ensemble.types import StackingResult
from scripts.model_registry import (
    compute_data_lineage,
    compute_file_checksum,
    get_environment_info,
)

if TYPE_CHECKING:
    from scripts.model_registry import DataLineage, EnvironmentInfo

logger = logging.getLogger(__name__)


def save_stacking_ensemble(
    result: StackingResult,
    models_dir: Path,
    config: dict[str, object],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_dir: Path,
) -> Path:
    """Sauvegarde l'ensemble stacking avec conformite ISO."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = models_dir / f"stacking_v{timestamp}"
    version_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving stacking ensemble to: {version_dir}")

    env_info = get_environment_info()
    if env_info.git_commit:
        logger.info(f"  Git: {env_info.git_commit[:8]} ({env_info.git_branch})")

    # Save artifacts
    artifacts = _save_artifacts(result, version_dir)

    # Data lineage
    train_full = pd.concat([train_df, valid_df], ignore_index=True)
    data_lineage = compute_data_lineage(
        data_dir / "train.parquet",
        data_dir / "valid.parquet",
        data_dir / "test.parquet",
        train_full,
        valid_df,
        test_df,
    )

    # Metadata
    metadata = _build_metadata(result, timestamp, env_info, data_lineage, config, artifacts)

    metadata_path = version_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("  Saved metadata.json (ISO 42001 Model Card)")
    return version_dir


def _save_artifacts(
    result: StackingResult,
    version_dir: Path,
) -> dict[str, dict[str, object]]:
    """Save model artifacts and return checksums."""
    meta_path = version_dir / "stacking_meta.joblib"
    joblib.dump(result.meta_model, meta_path)
    meta_checksum = compute_file_checksum(meta_path)

    oof_path = version_dir / "oof_predictions.npy"
    np.save(oof_path, result.oof_predictions)
    oof_checksum = compute_file_checksum(oof_path)

    test_path = version_dir / "test_predictions.npy"
    np.save(test_path, result.test_predictions)
    test_checksum = compute_file_checksum(test_path)

    return {
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
    }


def _build_metadata(
    result: StackingResult,
    timestamp: str,
    env_info: EnvironmentInfo,
    data_lineage: DataLineage,
    config: dict[str, object],
    artifacts: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Build metadata dict."""
    return {
        "version": f"stacking_v{timestamp}",
        "created_at": datetime.now().isoformat(),
        "environment": env_info.to_dict(),
        "data_lineage": data_lineage.to_dict(),
        "artifacts": artifacts,
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
