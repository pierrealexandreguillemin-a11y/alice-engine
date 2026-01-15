"""Validation de données pour Model Registry - ISO 5259.

Ce module implémente:
- Validation de schema DataFrame
- Validation cohérence train/valid/test
- Politique de rétention des versions

Conformité ISO/IEC 5259 (Data Quality), ISO/IEC 27001 (Data Lifecycle).
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Plages de valeurs FFE (ISO 5259 - Data Quality)
ELO_MIN = 1000
ELO_MAX = 3000
ELO_WARNING_LOW = 1100
ELO_WARNING_HIGH = 2700

# Schema validation
REQUIRED_TRAIN_COLUMNS: set[str] = {"resultat_blanc", "blanc_elo", "noir_elo"}
REQUIRED_NUMERIC_COLUMNS: set[str] = {"blanc_elo", "noir_elo", "diff_elo"}

# Retention policy
DEFAULT_MAX_VERSIONS = 10


@dataclass
class SchemaValidationResult:
    """Résultat de validation de schema DataFrame."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _validate_required_columns(
    df: pd.DataFrame, required: set[str], allow_missing: bool
) -> tuple[list[str], list[str]]:
    """Valide les colonnes requises."""
    errors, warnings = [], []
    missing = required - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        (warnings if allow_missing else errors).append(msg)
    return errors, warnings


def _validate_numeric_columns(df: pd.DataFrame, numeric: set[str]) -> list[str]:
    """Valide que les colonnes sont numériques."""
    import pandas as pd  # Lazy import

    errors = []
    for col in numeric & set(df.columns):
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' should be numeric, got {df[col].dtype}")
    return errors


def _validate_null_ratios(df: pd.DataFrame) -> list[str]:
    """Détecte les colonnes avec trop de valeurs nulles."""
    warnings = []
    for col in df.columns:
        null_ratio = df[col].isnull().mean()
        if null_ratio > 0.5:
            warnings.append(f"Column '{col}' has {null_ratio:.1%} null values")
    return warnings


def _validate_elo_column(col_data: pd.Series, col_name: str) -> tuple[list[str], list[str]]:
    """Valide une colonne ELO."""
    errors, warnings = [], []
    below_min = (col_data < ELO_MIN).sum()
    above_max = (col_data > ELO_MAX).sum()
    if below_min > 0:
        errors.append(f"Column '{col_name}' has {below_min} values below ELO_MIN ({ELO_MIN})")
    if above_max > 0:
        errors.append(f"Column '{col_name}' has {above_max} values above ELO_MAX ({ELO_MAX})")
    pct_low = (col_data < ELO_WARNING_LOW).mean()
    pct_high = (col_data > ELO_WARNING_HIGH).mean()
    if pct_low > 0.1:
        warnings.append(f"Column '{col_name}': {pct_low:.1%} values below {ELO_WARNING_LOW}")
    if pct_high > 0.05:
        warnings.append(f"Column '{col_name}': {pct_high:.1%} values above {ELO_WARNING_HIGH}")
    return errors, warnings


def _validate_diff_elo_consistency(df: pd.DataFrame) -> list[str]:
    """Vérifie cohérence diff_elo."""
    if not {"diff_elo", "blanc_elo", "noir_elo"}.issubset(df.columns):
        return []
    computed = df["blanc_elo"] - df["noir_elo"]
    mismatch = (df["diff_elo"] != computed).sum()
    if mismatch > 0:
        return [f"diff_elo inconsistent with blanc_elo - noir_elo in {mismatch} rows"]
    return []


def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: set[str] | None = None,
    numeric_columns: set[str] | None = None,
    *,
    allow_missing: bool = False,
    validate_elo_ranges: bool = True,
) -> SchemaValidationResult:
    """Valide le schema d'un DataFrame pour ML.

    Args:
    ----
        df: DataFrame à valider
        required_columns: Colonnes obligatoires
        numeric_columns: Colonnes qui doivent être numériques
        allow_missing: Autoriser les colonnes manquantes (warning au lieu d'erreur)
        validate_elo_ranges: Valider les plages ELO FFE

    Returns:
    -------
        SchemaValidationResult avec statut et messages
    """
    import pandas as pd  # Lazy import

    errors: list[str] = []
    warnings: list[str] = []

    required = required_columns or REQUIRED_TRAIN_COLUMNS
    numeric = numeric_columns or REQUIRED_NUMERIC_COLUMNS

    # Colonnes requises
    req_err, req_warn = _validate_required_columns(df, required, allow_missing)
    errors.extend(req_err)
    warnings.extend(req_warn)

    # Colonnes numériques
    errors.extend(_validate_numeric_columns(df, numeric))

    # Valeurs nulles
    warnings.extend(_validate_null_ratios(df))

    # DataFrame vide
    if len(df) == 0:
        errors.append("DataFrame is empty")

    # Plages ELO FFE (ISO 5259)
    if validate_elo_ranges and len(df) > 0:
        for col in ["blanc_elo", "noir_elo"]:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    elo_err, elo_warn = _validate_elo_column(col_data, col)
                    errors.extend(elo_err)
                    warnings.extend(elo_warn)
        warnings.extend(_validate_diff_elo_consistency(df))

    return SchemaValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


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

    for name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        result = validate_dataframe_schema(df)
        errors.extend([f"[{name}] {e}" for e in result.errors])
        warnings.extend([f"[{name}] {w}" for w in result.warnings])

    train_cols = set(train_df.columns)
    valid_cols = set(valid_df.columns)
    test_cols = set(test_df.columns)

    if train_cols != valid_cols:
        diff = train_cols.symmetric_difference(valid_cols)
        warnings.append(f"Column mismatch train/valid: {diff}")

    if train_cols != test_cols:
        diff = train_cols.symmetric_difference(test_cols)
        warnings.append(f"Column mismatch train/test: {diff}")

    total = len(train_df) + len(valid_df) + len(test_df)
    if total > 0:
        train_ratio = len(train_df) / total
        if train_ratio < 0.5:
            warnings.append(f"Train ratio is low: {train_ratio:.1%}")

    is_valid = len(errors) == 0
    return SchemaValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


# ==============================================================================
# RETENTION POLICY (ISO 27001 - Data Lifecycle)
# ==============================================================================


def list_model_versions(models_dir: Path) -> list[Path]:
    """Liste toutes les versions de modèles disponibles."""
    versions = []
    for item in models_dir.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            metadata_path = item / "metadata.json"
            if metadata_path.exists():
                versions.append(item)
    return sorted(versions, key=lambda x: x.name, reverse=True)


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
    """Retourne le statut de la politique de rétention."""
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
