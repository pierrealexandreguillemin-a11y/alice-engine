"""Tests Training Types (Pydantic) - ISO 29119.

Document ID: ALICE-TEST-SCHEMAS-TYPES
Version: 1.0.0
Tests count: 17

ISO Compliance:
- ISO 29119 (Testing standards)
- ISO/IEC 5259:2024 (Data Quality for ML)
"""

from datetime import UTC, datetime

import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from schemas.training_types import (
    DataLineage,
    ErrorSeverity,
    QualityMetrics,
    ValidationError,
    ValidationReport,
)


class TestErrorSeverityEnum:
    """Test ErrorSeverity enum structure."""

    def test_has_critical(self) -> None:
        """Verify CRITICAL severity exists."""
        assert ErrorSeverity.CRITICAL.value == "critical"

    def test_has_high(self) -> None:
        """Verify HIGH severity exists."""
        assert ErrorSeverity.HIGH.value == "high"

    def test_has_medium(self) -> None:
        """Verify MEDIUM severity exists."""
        assert ErrorSeverity.MEDIUM.value == "medium"

    def test_has_warning(self) -> None:
        """Verify WARNING severity exists."""
        assert ErrorSeverity.WARNING.value == "warning"

    def test_no_info_member(self) -> None:
        """Verify INFO severity does NOT exist (removed in migration)."""
        members = [member.value for member in ErrorSeverity]
        assert "info" not in members
        assert len(members) == 4


class TestDataLineage:
    """Test DataLineage pydantic dataclass."""

    def test_from_dataframe_creates_lineage(self) -> None:
        """Verify from_dataframe classmethod works with minimal DataFrame."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        lineage = DataLineage.from_dataframe(df, source_path="test_data.csv")

        assert lineage.source_path == "test_data.csv"
        assert lineage.row_count == 3
        assert lineage.column_count == 2
        assert lineage.schema_version == "1.1.0"

    def test_validation_timestamp_is_timezone_aware(self) -> None:
        """Verify validation_timestamp contains timezone info."""
        df = pd.DataFrame({"col1": [1, 2]})
        lineage = DataLineage.from_dataframe(df, source_path="test_data.csv")

        # ISO format with UTC should contain "+00:00" or "Z"
        assert "+00:00" in lineage.validation_timestamp or lineage.validation_timestamp.endswith(
            "Z"
        )

    def test_source_hash_is_16_char_hex(self) -> None:
        """Verify source_hash is a 16-character hex string."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        lineage = DataLineage.from_dataframe(df, source_path="test_data.csv")

        assert len(lineage.source_hash) == 16
        # Verify it's hexadecimal
        int(lineage.source_hash, 16)  # Will raise ValueError if not hex


class TestQualityMetrics:
    """Test QualityMetrics pydantic dataclass."""

    def test_is_acceptable_true_when_valid(self) -> None:
        """Verify is_acceptable returns True when validation_rate >= 0.95 and no critical errors."""
        metrics = QualityMetrics(
            total_rows=100,
            valid_rows=95,
            null_percentages={"col1": 0.0},
            validation_rate=0.95,
            critical_errors=0,
            high_errors=2,
            medium_errors=3,
            warnings=5,
        )

        assert metrics.is_acceptable is True

    def test_is_acceptable_false_with_critical_errors(self) -> None:
        """Verify is_acceptable returns False when critical_errors > 0."""
        metrics = QualityMetrics(
            total_rows=100,
            valid_rows=99,
            null_percentages={"col1": 0.0},
            validation_rate=0.99,
            critical_errors=1,  # Has critical error
            high_errors=0,
            medium_errors=0,
            warnings=0,
        )

        assert metrics.is_acceptable is False

    def test_is_acceptable_false_with_low_validation_rate(self) -> None:
        """Verify is_acceptable returns False when validation_rate < 0.95."""
        metrics = QualityMetrics(
            total_rows=100,
            valid_rows=93,
            null_percentages={"col1": 0.0},
            validation_rate=0.93,  # Below threshold
            critical_errors=0,
            high_errors=0,
            medium_errors=0,
            warnings=0,
        )

        assert metrics.is_acceptable is False

    def test_pydantic_validation_rejects_wrong_types(self) -> None:
        """Verify Pydantic rejects invalid types (e.g., total_rows as string)."""
        with pytest.raises(PydanticValidationError):
            QualityMetrics(
                total_rows="abc",  # type: ignore[arg-type]  # Invalid type
                valid_rows=95,
                null_percentages={"col1": 0.0},
                validation_rate=0.95,
                critical_errors=0,
                high_errors=0,
                medium_errors=0,
                warnings=0,
            )


class TestValidationErrorDataclass:
    """Test ValidationError pydantic dataclass."""

    def test_create_with_required_fields(self) -> None:
        """Verify ValidationError can be created with all required fields."""
        error = ValidationError(
            column="test_column",
            check="not_null",
            failure_count=10,
            severity=ErrorSeverity.HIGH,
        )

        assert error.column == "test_column"
        assert error.check == "not_null"
        assert error.failure_count == 10
        assert error.severity == ErrorSeverity.HIGH

    def test_sample_values_defaults_to_empty_list(self) -> None:
        """Verify sample_values defaults to empty list when not provided."""
        error = ValidationError(
            column="test_column",
            check="range_check",
            failure_count=5,
            severity=ErrorSeverity.MEDIUM,
        )

        assert error.sample_values == []

    def test_create_with_optional_fields(self) -> None:
        """Verify ValidationError accepts optional fields."""
        error = ValidationError(
            column="test_column",
            check="range_check",
            failure_count=3,
            severity=ErrorSeverity.WARNING,
            sample_values=[1, 2, 3],
            recommendation="Review data source",
        )

        assert error.sample_values == [1, 2, 3]
        assert error.recommendation == "Review data source"


class TestValidationReport:
    """Test ValidationReport pydantic dataclass."""

    @pytest.fixture
    def sample_lineage(self) -> DataLineage:
        """Create sample DataLineage for testing."""
        return DataLineage(
            source_path="test_data.csv",
            source_hash="a1b2c3d4e5f67890",
            row_count=100,
            column_count=5,
            validation_timestamp=datetime.now(tz=UTC).isoformat(),
            schema_version="1.1.0",
        )

    @pytest.fixture
    def sample_metrics(self) -> QualityMetrics:
        """Create sample QualityMetrics for testing."""
        return QualityMetrics(
            total_rows=100,
            valid_rows=95,
            null_percentages={"col1": 0.05},
            validation_rate=0.95,
            critical_errors=0,
            high_errors=1,
            medium_errors=2,
            warnings=3,
        )

    @pytest.fixture
    def sample_errors(self) -> list[ValidationError]:
        """Create sample ValidationError list for testing."""
        return [
            ValidationError(
                column="col1",
                check="not_null",
                failure_count=5,
                severity=ErrorSeverity.HIGH,
                recommendation="Check data source",
            )
        ]

    def test_to_dict_returns_correct_structure(
        self,
        sample_lineage: DataLineage,
        sample_metrics: QualityMetrics,
        sample_errors: list[ValidationError],
    ) -> None:
        """Verify to_dict() returns correct dictionary structure."""
        report = ValidationReport(
            lineage=sample_lineage,
            metrics=sample_metrics,
            errors=sample_errors,
            is_valid=True,
            schema_mode="strict",
        )

        result = report.to_dict()

        # Verify top-level keys
        assert set(result.keys()) == {"lineage", "metrics", "errors", "is_valid", "schema_mode"}

        # Verify lineage structure
        assert result["lineage"]["source_path"] == "test_data.csv"
        assert result["lineage"]["source_hash"] == "a1b2c3d4e5f67890"
        assert result["lineage"]["row_count"] == 100
        assert result["lineage"]["column_count"] == 5

        # Verify metrics structure
        assert result["metrics"]["total_rows"] == 100
        assert result["metrics"]["valid_rows"] == 95
        assert result["metrics"]["validation_rate"] == 0.95

        # Verify errors structure
        assert len(result["errors"]) == 1
        assert result["errors"][0]["column"] == "col1"
        assert result["errors"][0]["severity"] == "high"

        # Verify report-level fields
        assert result["is_valid"] is True
        assert result["schema_mode"] == "strict"

    def test_is_valid_field_works(
        self,
        sample_lineage: DataLineage,
        sample_metrics: QualityMetrics,
        sample_errors: list[ValidationError],
    ) -> None:
        """Verify is_valid field is correctly stored and retrieved."""
        report_valid = ValidationReport(
            lineage=sample_lineage,
            metrics=sample_metrics,
            errors=[],
            is_valid=True,
            schema_mode="strict",
        )

        report_invalid = ValidationReport(
            lineage=sample_lineage,
            metrics=sample_metrics,
            errors=sample_errors,
            is_valid=False,
            schema_mode="permissive",
        )

        assert report_valid.is_valid is True
        assert report_invalid.is_valid is False
