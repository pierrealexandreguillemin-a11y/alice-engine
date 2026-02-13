"""ISO 5259 data quality structures for ALICE Engine.

ISO Compliance:
- ISO/IEC 5259:2024 - Data Quality for ML (Lineage, Validation)
- ISO/IEC 42001:2023 - AI Management System
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from hashlib import sha256
from typing import Any, Self

import pandas as pd

from schemas.training_constants import QUALITY_VALIDATION_RATE_THRESHOLD


class ErrorSeverity(Enum):
    """ISO 5259 error categorization."""

    CRITICAL = "critical"  # Data unusable
    HIGH = "high"  # Significant quality issue
    MEDIUM = "medium"  # Minor quality issue
    WARNING = "warning"  # Informational
    INFO = "info"  # Metadata


@dataclass
class DataLineage:
    """ISO 5259 data lineage tracking."""

    source_path: str
    source_hash: str
    row_count: int
    column_count: int
    validation_timestamp: str
    schema_version: str = "1.1.0"

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, source_path: str) -> Self:
        """Create lineage from DataFrame."""
        # Compute hash of first 1000 rows for performance
        sample = df.head(1000).to_json()
        data_hash = sha256(sample.encode()).hexdigest()[:16]

        return cls(
            source_path=source_path,
            source_hash=data_hash,
            row_count=len(df),
            column_count=len(df.columns),
            validation_timestamp=datetime.now().isoformat(),
        )


@dataclass
class QualityMetrics:
    """ISO 5259 data quality metrics."""

    total_rows: int
    valid_rows: int
    null_percentages: dict[str, float]
    validation_rate: float
    critical_errors: int
    high_errors: int
    medium_errors: int
    warnings: int

    @property
    def is_acceptable(self) -> bool:
        """Check if quality meets minimum threshold (95% valid)."""
        return (
            self.validation_rate >= QUALITY_VALIDATION_RATE_THRESHOLD and self.critical_errors == 0
        )


@dataclass
class ValidationError:
    """Structured validation error."""

    column: str
    check: str
    failure_count: int
    severity: ErrorSeverity
    sample_values: list[Any] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class ValidationReport:
    """ISO 5259 compliant validation report."""

    lineage: DataLineage
    metrics: QualityMetrics
    errors: list[ValidationError]
    is_valid: bool
    schema_mode: str  # "strict" or "permissive"

    def to_dict(self) -> dict:
        """Export report as dictionary for serialization."""
        return {
            "lineage": {
                "source_path": self.lineage.source_path,
                "source_hash": self.lineage.source_hash,
                "row_count": self.lineage.row_count,
                "column_count": self.lineage.column_count,
                "validation_timestamp": self.lineage.validation_timestamp,
                "schema_version": self.lineage.schema_version,
            },
            "metrics": {
                "total_rows": self.metrics.total_rows,
                "valid_rows": self.metrics.valid_rows,
                "validation_rate": self.metrics.validation_rate,
                "critical_errors": self.metrics.critical_errors,
                "high_errors": self.metrics.high_errors,
                "medium_errors": self.metrics.medium_errors,
                "warnings": self.metrics.warnings,
            },
            "errors": [
                {
                    "column": e.column,
                    "check": e.check,
                    "failure_count": e.failure_count,
                    "severity": e.severity.value,
                    "recommendation": e.recommendation,
                }
                for e in self.errors
            ],
            "is_valid": self.is_valid,
            "schema_mode": self.schema_mode,
        }
