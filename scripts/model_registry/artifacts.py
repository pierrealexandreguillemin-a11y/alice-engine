"""Artefacts ML pour Model Registry - ISO 42001.

Ce module gère:
- Sauvegarde/chargement des modèles
- Validation d'intégrité (checksums)
- Export ONNX
- Feature importance

Conformité ISO/IEC 42001 (AI Management), ISO/IEC 27001 (Integrity).
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib

from scripts.model_registry.dataclasses import ModelArtifact
from scripts.model_registry.utils import compute_file_checksum

logger = logging.getLogger(__name__)

# Configuration
MODEL_FORMATS = {
    "CatBoost": ".cbm",
    "XGBoost": ".ubj",
    "LightGBM": ".txt",
}
ONNX_OPSET_VERSION = 15


# ==============================================================================
# FEATURE IMPORTANCE
# ==============================================================================


def extract_feature_importance(
    model: object,
    model_name: str,
    feature_names: list[str],
) -> dict[str, float]:
    """Extrait l'importance des features d'un modèle."""
    importance: dict[str, float] = {}

    try:
        if model_name == "CatBoost" and hasattr(model, "get_feature_importance"):
            importances = model.get_feature_importance()
            for i, name in enumerate(feature_names):
                if i < len(importances):
                    importance[name] = float(importances[i])

        elif model_name == "XGBoost" and hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for i, name in enumerate(feature_names):
                if i < len(importances):
                    importance[name] = float(importances[i])

        elif model_name == "LightGBM" and hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for i, name in enumerate(feature_names):
                if i < len(importances):
                    importance[name] = float(importances[i])

        # Normaliser
        if importance:
            total = sum(importance.values())
            if total > 0:
                importance = {k: v / total for k, v in importance.items()}

        # Trier par importance décroissante
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    except Exception as e:
        logger.warning(f"Could not extract feature importance for {model_name}: {e}")

    return importance


# ==============================================================================
# ONNX EXPORT
# ==============================================================================


def export_to_onnx(
    model: object,
    model_name: str,
    output_path: Path,
    feature_names: list[str],
    n_features: int,
) -> Path | None:
    """Exporte un modèle au format ONNX."""
    try:
        if model_name == "CatBoost":
            if hasattr(model, "save_model"):
                onnx_path = output_path.with_suffix(".onnx")
                model.save_model(
                    str(onnx_path),
                    format="onnx",
                    export_parameters={"onnx_domain": "ai.catboost"},
                )
                logger.info(f"  Exported {model_name} to ONNX: {onnx_path.name}")
                return onnx_path

        elif model_name in ("XGBoost", "LightGBM"):
            try:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType

                initial_type = [("float_input", FloatTensorType([None, n_features]))]
                onnx_model = convert_sklearn(
                    model,
                    initial_types=initial_type,
                    target_opset=ONNX_OPSET_VERSION,
                )
                onnx_path = output_path.with_suffix(".onnx")
                with onnx_path.open("wb") as f:
                    f.write(onnx_model.SerializeToString())
                logger.info(f"  Exported {model_name} to ONNX: {onnx_path.name}")
                return onnx_path
            except ImportError:
                logger.warning(f"skl2onnx not installed, skipping ONNX export for {model_name}")

    except Exception as e:
        logger.warning(f"ONNX export failed for {model_name}: {e}")

    return None


# ==============================================================================
# MODEL SAVING
# ==============================================================================


def save_model_artifact(
    model: object,
    model_name: str,
    version_dir: Path,
    feature_names: list[str],
    *,
    export_onnx: bool = False,
) -> ModelArtifact | None:
    """Sauvegarde un modèle avec checksums."""
    model_format = MODEL_FORMATS.get(model_name, ".joblib")
    model_path = version_dir / f"{model_name.lower()}{model_format}"

    try:
        # Sauvegarde native
        if model_name == "CatBoost" and hasattr(model, "save_model"):
            model.save_model(str(model_path))
        elif model_name == "XGBoost" and hasattr(model, "save_model"):
            model.save_model(str(model_path))
        elif model_name == "LightGBM" and hasattr(model, "booster_"):
            model.booster_.save_model(str(model_path))
        else:
            model_path = version_dir / f"{model_name.lower()}.joblib"
            joblib.dump(model, model_path)

        checksum = compute_file_checksum(model_path)
        size_bytes = model_path.stat().st_size

        onnx_path = None
        onnx_checksum = None
        if export_onnx:
            onnx_path = export_to_onnx(
                model,
                model_name,
                model_path,
                feature_names,
                len(feature_names),
            )
            if onnx_path and onnx_path.exists():
                onnx_checksum = compute_file_checksum(onnx_path)

        logger.info(f"  Saved {model_path.name} ({size_bytes:,} bytes, SHA256: {checksum[:12]}...)")

        return ModelArtifact(
            name=model_name,
            path=model_path,
            format=model_format,
            checksum=checksum,
            size_bytes=size_bytes,
            onnx_path=onnx_path,
            onnx_checksum=onnx_checksum,
        )

    except Exception as e:
        logger.exception(f"Failed to save {model_name}: {e}")
        return None


# ==============================================================================
# MODEL VALIDATION & LOADING
# ==============================================================================


def validate_model_integrity(artifact: ModelArtifact) -> bool:
    """Valide l'intégrité d'un modèle via checksum."""
    if not artifact.path.exists():
        logger.error(f"Model file not found: {artifact.path}")
        return False

    computed_checksum = compute_file_checksum(artifact.path)
    if computed_checksum != artifact.checksum:
        logger.error(
            f"Checksum mismatch for {artifact.name}: "
            f"expected {artifact.checksum[:12]}..., got {computed_checksum[:12]}..."
        )
        return False

    logger.info(f"  {artifact.name}: integrity verified")
    return True


def load_model_with_validation(
    artifact: ModelArtifact,
) -> object | None:
    """Charge un modèle avec validation d'intégrité."""
    if not validate_model_integrity(artifact):
        return None

    try:
        if artifact.format == ".cbm":
            from catboost import CatBoostClassifier

            model = CatBoostClassifier()
            model.load_model(str(artifact.path))
            return model

        elif artifact.format == ".ubj":
            from xgboost import XGBClassifier

            model = XGBClassifier()
            model.load_model(str(artifact.path))
            return model

        elif artifact.format == ".txt":
            import lightgbm as lgb

            booster = lgb.Booster(model_file=str(artifact.path))
            return booster

        else:
            return joblib.load(artifact.path)

    except Exception as e:
        logger.exception(f"Failed to load {artifact.name}: {e}")
        return None
