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
    try:
        importances = _get_raw_importances(model, model_name)
        if not importances:
            return {}

        importance = _build_importance_dict(importances, feature_names)
        importance = _normalize_importance(importance)
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    except Exception as e:
        logger.warning(f"Could not extract feature importance for {model_name}: {e}")
        return {}


def _get_raw_importances(model: object, model_name: str) -> list[float] | None:
    """Extrait les importances brutes selon le type de modele."""
    if model_name == "CatBoost" and hasattr(model, "get_feature_importance"):
        return list(model.get_feature_importance())
    if hasattr(model, "feature_importances_"):
        return list(model.feature_importances_)
    return None


def _build_importance_dict(importances: list[float], feature_names: list[str]) -> dict[str, float]:
    """Construit le dictionnaire d'importance."""
    return {
        name: float(importances[i]) for i, name in enumerate(feature_names) if i < len(importances)
    }


def _normalize_importance(importance: dict[str, float]) -> dict[str, float]:
    """Normalise les importances pour sommer a 1."""
    if not importance:
        return importance
    total = sum(importance.values())
    if total > 0:
        return {k: v / total for k, v in importance.items()}
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
    try:
        model_path, model_format = _save_model_native(model, model_name, version_dir)
        checksum = compute_file_checksum(model_path)
        size_bytes = model_path.stat().st_size

        onnx_path, onnx_checksum = _handle_onnx_export(
            model, model_name, model_path, feature_names, export_onnx
        )

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


def _save_model_native(model: object, model_name: str, version_dir: Path) -> tuple[Path, str]:
    """Sauvegarde le modele au format natif."""
    model_format = MODEL_FORMATS.get(model_name, ".joblib")
    model_path = version_dir / f"{model_name.lower()}{model_format}"

    if model_name in ("CatBoost", "XGBoost") and hasattr(model, "save_model"):
        model.save_model(str(model_path))
    elif model_name == "LightGBM" and hasattr(model, "booster_"):
        model.booster_.save_model(str(model_path))
    else:
        model_path = version_dir / f"{model_name.lower()}.joblib"
        model_format = ".joblib"
        joblib.dump(model, model_path)

    return model_path, model_format


def _handle_onnx_export(
    model: object,
    model_name: str,
    model_path: Path,
    feature_names: list[str],
    export_onnx: bool,
) -> tuple[Path | None, str | None]:
    """Gere l'export ONNX optionnel."""
    if not export_onnx:
        return None, None

    onnx_path = export_to_onnx(model, model_name, model_path, feature_names, len(feature_names))
    if onnx_path and onnx_path.exists():
        return onnx_path, compute_file_checksum(onnx_path)
    return None, None


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
