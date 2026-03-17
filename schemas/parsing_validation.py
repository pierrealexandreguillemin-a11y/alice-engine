"""Validation orchestration for raw parsed parquets (ISO 5259).

Validates echiquiers/joueurs DataFrames and persists reports.
Reuses DataLineage/ValidationReport from training_types.

ISO Compliance:
- ISO/IEC 5259:2024 - Data Quality for ML (Lineage, Validation)
- ISO/IEC 5055:2021 - Code Quality (<300 lines, SRP)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pandera.errors import SchemaErrors

if TYPE_CHECKING:
    import pandas as pd
    from pandera import DataFrameSchema

from schemas.parsing_schemas import EchiquiersRawSchema, JoueursRawSchema
from schemas.training_types import (
    DataLineage,
    ErrorSeverity,
    QualityMetrics,
    ValidationError,
    ValidationReport,
)

logger = logging.getLogger(__name__)

DEFAULT_REPORT_DIR = Path("reports/validation")


def validate_raw_echiquiers(
    df: pd.DataFrame,
    source_path: str = "data/echiquiers.parquet",
    report_dir: Path | None = None,
) -> ValidationReport:
    """Validate raw echiquiers DataFrame and persist report."""
    report = _validate_with_schema(
        df=df,
        schema=EchiquiersRawSchema,
        source_path=source_path,
        report_name="raw_echiquiers_report.json",
        report_dir=report_dir or DEFAULT_REPORT_DIR,
    )
    logger.info(
        "Echiquiers validation: valid=%s, errors=%d",
        report.is_valid,
        len(report.errors),
    )
    return report


def validate_raw_joueurs(
    df: pd.DataFrame,
    source_path: str = "data/joueurs.parquet",
    report_dir: Path | None = None,
) -> ValidationReport:
    """Validate raw joueurs DataFrame and persist report."""
    report = _validate_with_schema(
        df=df,
        schema=JoueursRawSchema,
        source_path=source_path,
        report_name="raw_joueurs_report.json",
        report_dir=report_dir or DEFAULT_REPORT_DIR,
    )
    logger.info(
        "Joueurs validation: valid=%s, errors=%d",
        report.is_valid,
        len(report.errors),
    )
    return report


def _validate_with_schema(
    df: pd.DataFrame,
    schema: DataFrameSchema,
    source_path: str,
    report_name: str,
    report_dir: Path,
) -> ValidationReport:
    """Run schema validation, build report, persist to disk."""
    lineage = DataLineage.from_dataframe(df, source_path)
    is_valid, errors = _run_validation(schema, df)
    error_counts = _count_by_severity(errors)
    metrics = QualityMetrics(
        total_rows=len(df),
        valid_rows=len(df) - sum(e.failure_count for e in errors),
        null_percentages={col: float(df[col].isna().mean() * 100) for col in df.columns},
        validation_rate=1.0 if is_valid else 0.0,
        critical_errors=error_counts.get("critical", 0),
        high_errors=error_counts.get("high", 0),
        medium_errors=error_counts.get("medium", 0),
        warnings=error_counts.get("warning", 0),
    )
    report = ValidationReport(
        lineage=lineage,
        metrics=metrics,
        errors=errors,
        is_valid=is_valid,
        schema_mode="raw_parsing",
    )
    _persist_report(report, report_dir, report_name)
    return report


def _run_validation(
    schema: DataFrameSchema,
    df: pd.DataFrame,
) -> tuple[bool, list[ValidationError]]:
    """Run Pandera schema validation, return (is_valid, errors)."""
    try:
        schema.validate(df, lazy=True)
        return True, []
    except SchemaErrors as exc:
        errors = [
            ValidationError(
                column=str(row.get("column", "dataframe")),
                check=str(row.get("check", "unknown")),
                failure_count=1,
                severity=ErrorSeverity.HIGH,
                recommendation=str(row.get("check", "")),
            )
            for _, row in exc.failure_cases.iterrows()
        ]
        return False, errors


def _count_by_severity(errors: list[ValidationError]) -> dict[str, int]:
    """Count errors by severity level."""
    counts: dict[str, int] = {}
    for error in errors:
        key = error.severity.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def _persist_report(
    report: ValidationReport,
    report_dir: Path,
    filename: str,
) -> None:
    """Save report as JSON to disk."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / filename
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Report saved: %s", report_path)
