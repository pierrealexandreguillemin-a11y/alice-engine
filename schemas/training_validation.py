"""Validation functions for ALICE Engine training data (ISO 5259).

ISO Compliance:
- ISO/IEC 5259:2024 - Data Quality for ML (Lineage, Validation)
- ISO/IEC 42001:2023 - AI Management System
"""

from typing import Any

import pandas as pd
import pandera as pa

from schemas.training_constants import (
    ECHIQUIER_JEUNES_HIGH_BOARDS,
    ECHIQUIER_MIN,
    ELO_MAX_N4_PLUS,
    MAX_SAMPLE_VALUES,
    NIVEAU_HIERARCHY_MAX,
    NIVEAU_N4,
    VALID_GAME_SCORES_ADULTES,
    VALID_GAME_SCORES_JEUNES_HIGH,
    VALID_GAME_SCORES_JEUNES_LOW,
)
from schemas.training_types import (
    DataLineage,
    ErrorSeverity,
    QualityMetrics,
    ValidationError,
    ValidationReport,
)


def validate_training_data(
    df: pd.DataFrame,
    strict: bool = False,
    lazy: bool = True,
) -> pa.errors.SchemaErrors | None:
    """Validate training DataFrame against FFE regulatory schema.

    Args:
    ----
        df: DataFrame to validate
        strict: Use strict FFE constraints
        lazy: If True, collect all errors; if False, fail on first error

    Returns:
    -------
        None if valid, SchemaErrors if invalid (when lazy=True)

    """
    from schemas.training_schemas import TrainingSchemaPermissive, TrainingSchemaStrict

    schema = TrainingSchemaStrict if strict else TrainingSchemaPermissive
    try:
        schema.validate(df, lazy=lazy)
        return None
    except pa.errors.SchemaErrors as err:
        return err


def validate_with_report(
    df: pd.DataFrame,
    source_path: str = "unknown",
    strict: bool = False,
) -> ValidationReport:
    """Validate DataFrame and return ISO 5259 compliant report.

    Args:
    ----
        df: DataFrame to validate
        source_path: Path to source file for lineage
        strict: Use strict FFE constraints

    Returns:
    -------
        ValidationReport with lineage, metrics, and errors

    """
    from schemas.training_schemas import TrainingSchemaPermissive, TrainingSchemaStrict

    # Create lineage
    lineage = DataLineage.from_dataframe(df, source_path)

    # Validate
    schema = TrainingSchemaStrict if strict else TrainingSchemaPermissive
    errors: list[ValidationError] = []
    is_valid = True

    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        is_valid = False
        # Parse errors into structured format
        for _, row in err.failure_cases.iterrows():
            severity = _classify_error_severity(row["check"], row["column"])
            errors.append(
                ValidationError(
                    column=str(row["column"]) if pd.notna(row["column"]) else "schema",
                    check=str(row["check"]),
                    failure_count=1,  # Each row is one failure
                    severity=severity,
                    sample_values=[row.get("failure_case")],
                    recommendation=_get_recommendation(row["check"]),
                )
            )

    # Aggregate errors by column/check
    aggregated_errors = _aggregate_errors(errors)

    # Calculate metrics
    null_pcts = {col: df[col].isna().mean() for col in df.columns}
    critical_count = sum(1 for e in aggregated_errors if e.severity == ErrorSeverity.CRITICAL)
    high_count = sum(1 for e in aggregated_errors if e.severity == ErrorSeverity.HIGH)
    medium_count = sum(1 for e in aggregated_errors if e.severity == ErrorSeverity.MEDIUM)
    warning_count = sum(1 for e in aggregated_errors if e.severity == ErrorSeverity.WARNING)

    error_rows = sum(e.failure_count for e in aggregated_errors)
    valid_rows = max(0, len(df) - error_rows)

    metrics = QualityMetrics(
        total_rows=len(df),
        valid_rows=valid_rows,
        null_percentages=null_pcts,
        validation_rate=valid_rows / len(df) if len(df) > 0 else 1.0,
        critical_errors=critical_count,
        high_errors=high_count,
        medium_errors=medium_count,
        warnings=warning_count,
    )

    return ValidationReport(
        lineage=lineage,
        metrics=metrics,
        errors=aggregated_errors,
        is_valid=is_valid,
        schema_mode="strict" if strict else "permissive",
    )


def _classify_error_severity(check: str, column: str) -> ErrorSeverity:
    """Classify error severity based on check type."""
    check_str = str(check).lower()

    # Critical: data integrity
    if "dtype" in check_str or "coerce" in check_str:
        return ErrorSeverity.CRITICAL
    if column in ["resultat_blanc", "resultat_noir", "blanc_elo", "noir_elo"]:
        return ErrorSeverity.CRITICAL

    # High: business logic
    if "diff_elo" in check_str or "equipe" in check_str:
        return ErrorSeverity.HIGH
    if "resultat" in check_str or "type_resultat" in check_str:
        return ErrorSeverity.HIGH

    # Medium: regulatory
    if "niveau" in check_str or "competition" in check_str:
        return ErrorSeverity.MEDIUM

    return ErrorSeverity.WARNING


def _get_recommendation(check: str) -> str:
    """Get remediation recommendation for check failure."""
    check_str = str(check).lower()

    if "diff_elo" in check_str:
        return "Recalculate diff_elo as blanc_elo - noir_elo"
    if "equipe" in check_str:
        return "Verify player team assignment matches match teams"
    if "resultat" in check_str:
        return "Check game result encoding matches FFE regulations"
    if "elo" in check_str:
        return "Verify Elo rating is within valid range [799, 2900]"
    if "niveau" in check_str:
        return "Check competition level encoding"

    return "Review data against FFE regulations"


def _aggregate_errors(errors: list[ValidationError]) -> list[ValidationError]:
    """Aggregate errors by column and check."""
    aggregated: dict[tuple[str, str], ValidationError] = {}

    for error in errors:
        key = (error.column, error.check)
        if key in aggregated:
            aggregated[key].failure_count += 1
            if len(aggregated[key].sample_values) < MAX_SAMPLE_VALUES:
                aggregated[key].sample_values.extend(error.sample_values)
        else:
            aggregated[key] = ValidationError(
                column=error.column,
                check=error.check,
                failure_count=1,
                severity=error.severity,
                sample_values=error.sample_values[:5],
                recommendation=error.recommendation,
            )

    return list(aggregated.values())


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_expected_score_range(type_competition: str, echiquier: int) -> list[float]:
    """Get valid score range based on competition type and board number."""
    if type_competition == "national_jeunes":
        if ECHIQUIER_MIN <= echiquier <= ECHIQUIER_JEUNES_HIGH_BOARDS:
            return VALID_GAME_SCORES_JEUNES_HIGH  # victoire=2 for boards 1-6
        return VALID_GAME_SCORES_JEUNES_LOW  # echiquiers 7-8: victoire=1
    elif type_competition == "scolaire":
        return VALID_GAME_SCORES_JEUNES_LOW
    return VALID_GAME_SCORES_ADULTES  # FIDE standard


def is_valid_niveau_for_elo(niveau: int, elo: int) -> bool:
    """Check if Elo is valid for given competition level.

    A02 Art. 3.7.j: Elo > 2400 interdit en N4 et divisions inferieures.
    """
    if niveau >= NIVEAU_N4 and niveau <= NIVEAU_HIERARCHY_MAX and elo > ELO_MAX_N4_PLUS:
        return False
    return True


def compute_quality_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute data quality summary for monitoring.

    Returns dict with key metrics for dashboards/logging.
    """
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "null_percentage": df.isna().mean().mean() * 100,
        "duplicate_rows": df.duplicated().sum(),
        "saison_range": [int(df["saison"].min()), int(df["saison"].max())],
        "elo_range": [
            int(min(df["blanc_elo"].min(), df["noir_elo"].min())),
            int(max(df["blanc_elo"].max(), df["noir_elo"].max())),
        ],
        "competition_types": df["type_competition"].unique().tolist(),
    }
