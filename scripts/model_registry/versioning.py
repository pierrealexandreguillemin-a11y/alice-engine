"""Versionnement et API haut niveau - ISO 42001.

Ce module gère:
- Rollback de versions
- API de sauvegarde production
- Model Card création

Conformité ISO/IEC 42001 (AI Management), ISO/IEC 5055 (Code Quality).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess  # nosec B404 - subprocess for internal dev tools only
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import joblib

from scripts.model_registry.artifacts import (
    extract_feature_importance,
    save_model_artifact,
)
from scripts.model_registry.dataclasses import (
    DataLineage,
    EnvironmentInfo,
    ModelArtifact,
    ProductionModelCard,
)
from scripts.model_registry.utils import (
    compute_data_lineage,
    compute_file_checksum,
    get_environment_info,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


# ==============================================================================
# ROLLBACK MECHANISM
# ==============================================================================


def rollback_to_version(
    models_dir: Path,
    target_version: str,
) -> bool:
    """Rollback vers une version spécifique."""
    target_dir = models_dir / target_version
    if not target_dir.exists():
        logger.error(f"Version not found: {target_version}")
        return False

    current_link = models_dir / "current"
    if current_link.exists() or current_link.is_symlink():
        if current_link.is_symlink():
            current_link.unlink()
        elif current_link.is_dir():
            try:
                os.rmdir(current_link)
            except OSError:
                import shutil

                shutil.rmtree(current_link)

    if platform.system() == "Windows":
        subprocess.run(  # nosec B603, B607 - trusted mklink command for symlink
            ["cmd", "/c", "mklink", "/J", str(current_link), str(target_dir)],
            capture_output=True,
            check=False,
        )
    else:
        current_link.symlink_to(target_dir.name)

    logger.info(f"Rolled back to version: {target_version}")
    return True


def get_current_version(models_dir: Path) -> str | None:
    """Récupère la version courante."""
    current_link = models_dir / "current"
    if current_link.exists():
        if current_link.is_symlink():
            return current_link.resolve().name
        elif current_link.is_dir():
            metadata_path = current_link / "metadata.json"
            if metadata_path.exists():
                with metadata_path.open() as f:
                    data = json.load(f)
                    return data.get("version")
    return None


# ==============================================================================
# PRODUCTION MODEL CARD
# ==============================================================================


def create_production_model_card(
    version: str,
    environment: EnvironmentInfo,
    data_lineage: DataLineage,
    artifacts: list[ModelArtifact],
    metrics: dict[str, dict[str, float]],
    feature_importance: dict[str, dict[str, float]],
    hyperparameters: dict[str, object],
    best_model: dict[str, object],
) -> ProductionModelCard:
    """Crée une Model Card complète pour la production."""
    return ProductionModelCard(
        version=version,
        created_at=datetime.now().isoformat(),
        environment=environment,
        data_lineage=data_lineage,
        artifacts=artifacts,
        metrics=metrics,
        feature_importance=feature_importance,
        hyperparameters=hyperparameters,
        best_model=best_model,
        limitations=[
            "Entraîné sur données FFE France uniquement",
            "Performance dégradée pour ELO < 1000 ou > 2500",
            "Ne prédit pas les forfaits",
        ],
        use_cases=[
            "Prédiction résultat partie individuelle",
            "Composition optimale équipes interclubs",
            "Analyse probabiliste rencontres",
        ],
        conformance={
            "iso_42001": "AI Management System - Model Card",
            "iso_5259": "Data Quality for ML - Data Lineage",
            "iso_27001": "Information Security - Checksums",
            "iso_5055": "Code Quality - Strict Typing",
            "iso_29119": "Software Testing - Unit Tests",
        },
    )


def save_production_model_card(
    model_card: ProductionModelCard,
    version_dir: Path,
) -> Path:
    """Sauvegarde la Model Card au format JSON."""
    metadata_path = version_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(model_card.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("  Saved metadata.json (ISO 42001 Model Card)")
    return metadata_path


# ==============================================================================
# HIGH-LEVEL API
# ==============================================================================


def save_production_models(
    models: dict[str, object],
    metrics: dict[str, dict[str, float]],
    models_dir: Path,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    feature_names: list[str],
    hyperparameters: dict[str, object],
    label_encoders: dict[str, object],
    *,
    export_onnx: bool = False,
) -> Path:
    """Sauvegarde complète des modèles pour production avec conformité ISO."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"v{timestamp}"
    version_dir = models_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"SAVING PRODUCTION MODELS - {version}")
    logger.info(f"{'=' * 60}")

    # Environment
    logger.info("\n[1/6] Collecting environment info...")
    environment = get_environment_info()
    if environment.git_commit:
        logger.info(f"  Git: {environment.git_commit[:8]} ({environment.git_branch})")
    if environment.git_dirty:
        logger.warning("  WARNING: Repository has uncommitted changes!")

    # Data lineage
    logger.info("\n[2/6] Computing data lineage...")
    data_lineage = compute_data_lineage(
        train_path,
        valid_path,
        test_path,
        train_df,
        valid_df,
        test_df,
    )
    logger.info(
        f"  Train: {data_lineage.train_samples:,} samples (hash: {data_lineage.train_hash})"
    )
    logger.info(
        f"  Valid: {data_lineage.valid_samples:,} samples (hash: {data_lineage.valid_hash})"
    )
    logger.info(f"  Test:  {data_lineage.test_samples:,} samples (hash: {data_lineage.test_hash})")

    # Save models
    logger.info("\n[3/6] Saving model artifacts...")
    artifacts: list[ModelArtifact] = []
    for name, model in models.items():
        if model is None:
            continue
        artifact = save_model_artifact(
            model, name, version_dir, feature_names, export_onnx=export_onnx
        )
        if artifact:
            artifacts.append(artifact)

    # Save label encoders
    logger.info("\n[4/6] Saving label encoders...")
    encoders_path = version_dir / "label_encoders.joblib"
    joblib.dump(label_encoders, encoders_path)
    encoders_checksum = compute_file_checksum(encoders_path)
    logger.info(f"  Saved label_encoders.joblib (SHA256: {encoders_checksum[:12]}...)")

    # Feature importance
    logger.info("\n[5/6] Extracting feature importance...")
    feature_importance: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        if model is None:
            continue
        importance = extract_feature_importance(model, name, feature_names)
        if importance:
            feature_importance[name] = importance
            top_3 = list(importance.items())[:3]
            logger.info(f"  {name} top features: {', '.join(f'{k}={v:.3f}' for k, v in top_3)}")

    # Best model
    best_name = max(metrics.keys(), key=lambda k: metrics[k].get("auc_roc", 0))
    best_auc = metrics[best_name].get("auc_roc", 0)
    best_model = {"name": best_name, "auc": best_auc}

    # Model Card
    logger.info("\n[6/6] Creating production Model Card...")
    model_card = create_production_model_card(
        version=version,
        environment=environment,
        data_lineage=data_lineage,
        artifacts=artifacts,
        metrics=metrics,
        feature_importance=feature_importance,
        hyperparameters=hyperparameters,
        best_model=best_model,
    )
    save_production_model_card(model_card, version_dir)

    # Update current symlink
    current_link = models_dir / "current"
    if current_link.exists() or current_link.is_symlink():
        if current_link.is_symlink():
            current_link.unlink()
        elif current_link.is_dir():
            try:
                os.rmdir(current_link)
            except OSError:
                pass

    if platform.system() == "Windows":
        subprocess.run(  # nosec B603, B607 - trusted mklink command for symlink
            ["cmd", "/c", "mklink", "/J", str(current_link), str(version_dir)],
            capture_output=True,
            check=False,
        )
    else:
        current_link.symlink_to(version_dir.name)

    logger.info(f"\n  Updated 'current' -> {version}")
    logger.info(f"\n{'=' * 60}")
    logger.info(f"PRODUCTION MODELS SAVED: {version_dir}")
    logger.info(f"{'=' * 60}")

    return version_dir
