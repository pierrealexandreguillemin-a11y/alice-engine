#!/usr/bin/env python3
"""Model Registry pour ALICE - Normalisation modèles production.

Ce module centralise la sauvegarde, validation et chargement des modèles
avec conformité aux normes ISO pour la production.

Fonctionnalités:
- Checksums SHA-256 pour intégrité
- Git commit hash pour reproductibilité
- Data lineage pour traçabilité
- Validation au chargement
- Export ONNX optionnel
- Mécanisme de rollback
- Feature importance
- Statistiques données

Conformité:
- ISO/IEC 42001 (AI Management System)
- ISO/IEC 5259 (Data Quality for ML)
- ISO/IEC 27001 (Information Security)
- ISO/IEC 5055 (Code Quality)

Usage:
    from scripts.model_registry import ModelRegistry, save_production_model
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import platform
import secrets
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

# Logging
logger = logging.getLogger(__name__)


# ==============================================================================
# CONSTANTS
# ==============================================================================

CHECKSUM_ALGORITHM = "sha256"
MODEL_FORMATS = {
    "CatBoost": ".cbm",
    "XGBoost": ".ubj",
    "LightGBM": ".txt",
}
ONNX_OPSET_VERSION = 15

# P2: Retention policy
DEFAULT_MAX_VERSIONS = 10

# P2: Schema validation - colonnes requises pour les DataFrames ML
REQUIRED_TRAIN_COLUMNS: set[str] = {"resultat_blanc"}
REQUIRED_NUMERIC_COLUMNS: set[str] = {"blanc_elo", "noir_elo", "diff_elo"}


# ==============================================================================
# P2: SIGNATURE HMAC-SHA256 (ISO 27001 - Authenticity)
# ==============================================================================


def generate_signing_key() -> str:
    """Génère une clé de signature HMAC-SHA256 (32 bytes hex)."""
    return secrets.token_hex(32)


def compute_model_signature(file_path: Path, secret_key: str) -> str:
    """Calcule la signature HMAC-SHA256 d'un fichier modèle.

    Args:
    ----
        file_path: Chemin vers le fichier modèle
        secret_key: Clé secrète HMAC (hex string)

    Returns:
    -------
        Signature HMAC-SHA256 en hex
    """
    key_bytes = bytes.fromhex(secret_key)
    h = hmac.new(key_bytes, digestmod=hashlib.sha256)

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()


def verify_model_signature(file_path: Path, signature: str, secret_key: str) -> bool:
    """Vérifie la signature HMAC-SHA256 d'un fichier modèle.

    Args:
    ----
        file_path: Chemin vers le fichier modèle
        signature: Signature attendue (hex)
        secret_key: Clé secrète HMAC (hex string)

    Returns:
    -------
        True si signature valide, False sinon
    """
    if not file_path.exists():
        logger.error(f"File not found for signature verification: {file_path}")
        return False

    computed = compute_model_signature(file_path, secret_key)
    is_valid = hmac.compare_digest(computed, signature)

    if not is_valid:
        logger.error(f"Signature mismatch for {file_path.name}")

    return is_valid


def save_signing_key(key: str, key_path: Path) -> None:
    """Sauvegarde la clé de signature de manière sécurisée."""
    key_path.write_text(key)
    # Restrict permissions (Unix only)
    if platform.system() != "Windows":
        os.chmod(key_path, 0o600)
    logger.info(f"  Signing key saved to {key_path.name}")


def load_signing_key(key_path: Path) -> str | None:
    """Charge la clé de signature."""
    if not key_path.exists():
        logger.warning(f"Signing key not found: {key_path}")
        return None
    return key_path.read_text().strip()


# ==============================================================================
# P2: SCHEMA VALIDATION (ISO 5259 - Data Quality)
# ==============================================================================


@dataclass
class SchemaValidationResult:
    """Résultat de validation de schema DataFrame."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: set[str] | None = None,
    numeric_columns: set[str] | None = None,
    *,
    allow_missing: bool = False,
) -> SchemaValidationResult:
    """Valide le schema d'un DataFrame pour ML.

    Args:
    ----
        df: DataFrame à valider
        required_columns: Colonnes obligatoires
        numeric_columns: Colonnes qui doivent être numériques
        allow_missing: Autoriser les colonnes manquantes (warning au lieu d'erreur)

    Returns:
    -------
        SchemaValidationResult avec statut et messages
    """
    errors: list[str] = []
    warnings: list[str] = []

    if required_columns is None:
        required_columns = REQUIRED_TRAIN_COLUMNS

    if numeric_columns is None:
        numeric_columns = REQUIRED_NUMERIC_COLUMNS

    # Vérifier colonnes requises
    missing_required = required_columns - set(df.columns)
    if missing_required:
        msg = f"Missing required columns: {missing_required}"
        if allow_missing:
            warnings.append(msg)
        else:
            errors.append(msg)

    # Vérifier colonnes numériques
    present_numeric = numeric_columns & set(df.columns)
    for col in present_numeric:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' should be numeric, got {df[col].dtype}")

    # Vérifier valeurs nulles excessives (>50%)
    for col in df.columns:
        null_ratio = df[col].isnull().mean()
        if null_ratio > 0.5:
            warnings.append(f"Column '{col}' has {null_ratio:.1%} null values")

    # Vérifier DataFrame vide
    if len(df) == 0:
        errors.append("DataFrame is empty")

    is_valid = len(errors) == 0

    return SchemaValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def validate_train_valid_test_schema(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> SchemaValidationResult:
    """Valide la cohérence des schemas train/valid/test.

    Args:
    ----
        train_df: DataFrame d'entraînement
        valid_df: DataFrame de validation
        test_df: DataFrame de test

    Returns:
    -------
        SchemaValidationResult global
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Valider chaque DataFrame
    for name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        result = validate_dataframe_schema(df)
        errors.extend([f"[{name}] {e}" for e in result.errors])
        warnings.extend([f"[{name}] {w}" for w in result.warnings])

    # Vérifier cohérence des colonnes
    train_cols = set(train_df.columns)
    valid_cols = set(valid_df.columns)
    test_cols = set(test_df.columns)

    if train_cols != valid_cols:
        diff = train_cols.symmetric_difference(valid_cols)
        warnings.append(f"Column mismatch train/valid: {diff}")

    if train_cols != test_cols:
        diff = train_cols.symmetric_difference(test_cols)
        warnings.append(f"Column mismatch train/test: {diff}")

    # Vérifier ratio des splits
    total = len(train_df) + len(valid_df) + len(test_df)
    if total > 0:
        train_ratio = len(train_df) / total
        if train_ratio < 0.5:
            warnings.append(f"Train ratio is low: {train_ratio:.1%}")

    is_valid = len(errors) == 0

    return SchemaValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


# ==============================================================================
# P2: VERSION RETENTION POLICY (ISO 27001 - Data Lifecycle)
# ==============================================================================


def apply_retention_policy(
    models_dir: Path,
    max_versions: int = DEFAULT_MAX_VERSIONS,
    *,
    dry_run: bool = False,
) -> list[Path]:
    """Applique la politique de rétention des versions.

    Garde les N versions les plus récentes, supprime les anciennes.

    Args:
    ----
        models_dir: Répertoire des modèles
        max_versions: Nombre maximum de versions à conserver
        dry_run: Si True, liste seulement sans supprimer

    Returns:
    -------
        Liste des versions supprimées (ou à supprimer si dry_run)
    """
    versions = list_model_versions(models_dir)
    deleted: list[Path] = []

    if len(versions) <= max_versions:
        logger.info(f"Retention: {len(versions)}/{max_versions} versions, nothing to delete")
        return deleted

    # Versions à supprimer (les plus anciennes)
    to_delete = versions[max_versions:]

    for version_dir in to_delete:
        if dry_run:
            logger.info(f"  [DRY RUN] Would delete: {version_dir.name}")
        else:
            try:
                shutil.rmtree(version_dir)
                logger.info(f"  Deleted old version: {version_dir.name}")
            except OSError as e:
                logger.warning(f"  Failed to delete {version_dir.name}: {e}")
                continue
        deleted.append(version_dir)

    logger.info(f"Retention policy applied: kept {max_versions}, deleted {len(deleted)} versions")

    return deleted


def get_retention_status(
    models_dir: Path, max_versions: int = DEFAULT_MAX_VERSIONS
) -> dict[str, object]:
    """Retourne le statut de la politique de rétention.

    Args:
    ----
        models_dir: Répertoire des modèles
        max_versions: Nombre maximum de versions

    Returns:
    -------
        Dict avec current_count, max_versions, versions_to_delete, etc.
    """
    versions = list_model_versions(models_dir)
    to_delete_count = max(0, len(versions) - max_versions)

    return {
        "current_count": len(versions),
        "max_versions": max_versions,
        "versions_to_delete": to_delete_count,
        "oldest_version": versions[-1].name if versions else None,
        "newest_version": versions[0].name if versions else None,
        "retention_applied": to_delete_count == 0,
    }


# ==============================================================================
# DATACLASSES
# ==============================================================================


@dataclass
class DataLineage:
    """Traçabilité des données d'entraînement (ISO 5259)."""

    train_path: str
    valid_path: str
    test_path: str
    train_samples: int
    valid_samples: int
    test_samples: int
    train_hash: str
    valid_hash: str
    test_hash: str
    feature_count: int
    target_distribution: dict[str, float]
    created_at: str

    def to_dict(self) -> dict[str, object]:
        """Convertit en dictionnaire."""
        return {
            "train": {
                "path": self.train_path,
                "samples": self.train_samples,
                "hash": self.train_hash,
            },
            "valid": {
                "path": self.valid_path,
                "samples": self.valid_samples,
                "hash": self.valid_hash,
            },
            "test": {
                "path": self.test_path,
                "samples": self.test_samples,
                "hash": self.test_hash,
            },
            "feature_count": self.feature_count,
            "target_distribution": self.target_distribution,
            "created_at": self.created_at,
        }


@dataclass
class EnvironmentInfo:
    """Informations d'environnement pour reproductibilité."""

    python_version: str
    platform_system: str
    platform_release: str
    platform_machine: str
    git_commit: str | None
    git_branch: str | None
    git_dirty: bool
    packages: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        """Convertit en dictionnaire."""
        return {
            "python_version": self.python_version,
            "platform": {
                "system": self.platform_system,
                "release": self.platform_release,
                "machine": self.platform_machine,
            },
            "git": {
                "commit": self.git_commit,
                "branch": self.git_branch,
                "dirty": self.git_dirty,
            },
            "packages": self.packages,
        }


@dataclass
class ModelArtifact:
    """Artefact de modèle avec métadonnées de sécurité."""

    name: str
    path: Path
    format: str
    checksum: str
    size_bytes: int
    onnx_path: Path | None = None
    onnx_checksum: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convertit en dictionnaire."""
        result: dict[str, object] = {
            "name": self.name,
            "path": str(self.path),
            "format": self.format,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
        }
        if self.onnx_path:
            result["onnx"] = {
                "path": str(self.onnx_path),
                "checksum": self.onnx_checksum,
            }
        return result


@dataclass
class ProductionModelCard:
    """Model Card complète conforme ISO 42001."""

    version: str
    created_at: str
    environment: EnvironmentInfo
    data_lineage: DataLineage
    artifacts: list[ModelArtifact]
    metrics: dict[str, dict[str, float]]
    feature_importance: dict[str, dict[str, float]]
    hyperparameters: dict[str, object]
    best_model: dict[str, object]
    limitations: list[str] = field(default_factory=list)
    use_cases: list[str] = field(default_factory=list)
    conformance: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convertit en dictionnaire complet."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "environment": self.environment.to_dict(),
            "data_lineage": self.data_lineage.to_dict(),
            "artifacts": [a.to_dict() for a in self.artifacts],
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "hyperparameters": self.hyperparameters,
            "best_model": self.best_model,
            "limitations": self.limitations,
            "use_cases": self.use_cases,
            "conformance": self.conformance,
        }


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


def compute_file_checksum(file_path: Path) -> str:
    """Calcule le checksum SHA-256 d'un fichier."""
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """Calcule un hash déterministe d'un DataFrame."""
    # Utilise le hash pandas pour la cohérence
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[
        :16
    ]


def get_git_info() -> tuple[str | None, str | None, bool]:
    """Récupère les informations git du repository."""
    try:
        # Git commit hash
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        commit_hash = commit.stdout.strip() if commit.returncode == 0 else None

        # Git branch
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        branch_name = branch.stdout.strip() if branch.returncode == 0 else None

        # Git dirty (uncommitted changes)
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        is_dirty = bool(status.stdout.strip()) if status.returncode == 0 else False

        return commit_hash, branch_name, is_dirty
    except FileNotFoundError:
        return None, None, False


def get_package_versions() -> dict[str, str]:
    """Récupère les versions des packages ML."""
    versions: dict[str, str] = {}

    try:
        import catboost

        versions["catboost"] = catboost.__version__
    except ImportError:
        pass

    try:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    except ImportError:
        pass

    try:
        import lightgbm

        versions["lightgbm"] = lightgbm.__version__
    except ImportError:
        pass

    try:
        import sklearn

        versions["sklearn"] = sklearn.__version__
    except ImportError:
        pass

    try:
        import numpy

        versions["numpy"] = numpy.__version__
    except ImportError:
        pass

    try:
        import pandas

        versions["pandas"] = pandas.__version__
    except ImportError:
        pass

    return versions


def get_environment_info() -> EnvironmentInfo:
    """Collecte les informations d'environnement."""
    commit, branch, dirty = get_git_info()
    return EnvironmentInfo(
        python_version=sys.version,
        platform_system=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
        git_commit=commit,
        git_branch=branch,
        git_dirty=dirty,
        packages=get_package_versions(),
    )


def compute_data_lineage(
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "resultat_blanc",
) -> DataLineage:
    """Calcule la traçabilité des données."""
    # Distribution de la cible
    train_target = train_df[target_col] if target_col in train_df.columns else pd.Series()
    target_dist = {
        "positive_ratio": float(train_target.mean()) if len(train_target) > 0 else 0.0,
        "total_samples": len(train_df) + len(valid_df) + len(test_df),
    }

    return DataLineage(
        train_path=str(train_path),
        valid_path=str(valid_path),
        test_path=str(test_path),
        train_samples=len(train_df),
        valid_samples=len(valid_df),
        test_samples=len(test_df),
        train_hash=compute_dataframe_hash(train_df),
        valid_hash=compute_dataframe_hash(valid_df),
        test_hash=compute_dataframe_hash(test_df),
        feature_count=len(train_df.columns) - 1,  # Excluding target
        target_distribution=target_dist,
        created_at=datetime.now().isoformat(),
    )


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
            # CatBoost a un export ONNX natif
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
            # Utilise onnxmltools ou skl2onnx
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
            # Fallback joblib
            model_path = version_dir / f"{model_name.lower()}.joblib"
            joblib.dump(model, model_path)

        # Checksum
        checksum = compute_file_checksum(model_path)
        size_bytes = model_path.stat().st_size

        # ONNX export optionnel
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
# MODEL VALIDATION
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


# ==============================================================================
# ROLLBACK MECHANISM
# ==============================================================================


def list_model_versions(models_dir: Path) -> list[Path]:
    """Liste toutes les versions de modèles disponibles."""
    versions = []
    for item in models_dir.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            metadata_path = item / "metadata.json"
            if metadata_path.exists():
                versions.append(item)
    # Trier par date (plus récent en premier)
    return sorted(versions, key=lambda x: x.name, reverse=True)


def rollback_to_version(
    models_dir: Path,
    target_version: str,
) -> bool:
    """Rollback vers une version spécifique."""
    target_dir = models_dir / target_version
    if not target_dir.exists():
        logger.error(f"Version not found: {target_version}")
        return False

    # Mettre à jour le symlink "current"
    current_link = models_dir / "current"
    if current_link.exists() or current_link.is_symlink():
        if current_link.is_symlink():
            current_link.unlink()
        elif current_link.is_dir():
            # Windows junction
            try:
                os.rmdir(current_link)
            except OSError:
                shutil.rmtree(current_link)

    # Créer le nouveau lien
    if platform.system() == "Windows":
        subprocess.run(
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
            # Windows junction - lire le metadata
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
    # Version
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
        subprocess.run(
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
