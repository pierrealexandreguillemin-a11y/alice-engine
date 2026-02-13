"""Tests Validation Functions - ISO 29119.

Document ID: ALICE-TEST-SCHEMAS-VALIDATION
Version: 1.0.0
Tests: 23

Classes:
- TestRunSchemaValidation: Tests _run_schema_validation (4 tests)
- TestComputeQualityMetrics: Tests _compute_quality_metrics (4 tests)
- TestCountBySeverity: Tests _count_by_severity (3 tests)
- TestClassifyErrorSeverity: Tests _classify_error_severity (6 tests)
- TestAggregateErrors: Tests _aggregate_errors (6 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lines)
- ISO/IEC 5259:2024 - Data Quality for ML

Author: ALICE Engine Team
Last Updated: 2026-02-13
"""

import pandas as pd
from pandera import Check, Column, DataFrameSchema

from schemas.training_types import ErrorSeverity, QualityMetrics, ValidationError
from schemas.training_validation import (
    _aggregate_errors,
    _classify_error_severity,
    _compute_quality_metrics,
    _count_by_severity,
    _run_schema_validation,
)


class TestRunSchemaValidation:
    """Tests for _run_schema_validation function."""

    def test_run_schema_validation_valid_dataframe(self):
        """Test validation with valid DataFrame."""
        schema = DataFrameSchema({"col1": Column(int, checks=Check.ge(0))})
        df = pd.DataFrame({"col1": [1, 2, 3]})
        is_valid, errors = _run_schema_validation(schema, df)
        assert is_valid is True
        assert errors == []

    def test_run_schema_validation_invalid_dataframe(self):
        """Test validation with invalid DataFrame."""
        schema = DataFrameSchema({"col1": Column(int, checks=Check.ge(0))})
        df = pd.DataFrame({"col1": [-1, 2, -3]})
        is_valid, errors = _run_schema_validation(schema, df)
        assert is_valid is False
        assert len(errors) > 0
        assert all(isinstance(e, ValidationError) for e in errors)

    def test_run_schema_validation_error_structure(self):
        """Test that validation errors have proper structure."""
        schema = DataFrameSchema({"test_col": Column(int, checks=Check.ge(0))})
        df = pd.DataFrame({"test_col": [-5]})
        is_valid, errors = _run_schema_validation(schema, df)
        assert is_valid is False
        error = errors[0]
        assert error.column == "test_col"
        assert error.failure_count == 1
        assert isinstance(error.severity, ErrorSeverity)
        assert isinstance(error.sample_values, list)
        assert isinstance(error.recommendation, str)

    def test_run_schema_validation_dtype_error(self):
        """Test validation with dtype mismatch."""
        schema = DataFrameSchema({"col1": Column(int)})
        df = pd.DataFrame({"col1": ["not_int", "also_not_int"]})
        is_valid, errors = _run_schema_validation(schema, df)
        assert is_valid is False
        assert any("dtype" in e.check.lower() or "type" in e.check.lower() for e in errors)


class TestComputeQualityMetrics:
    """Tests for _compute_quality_metrics function."""

    def test_compute_quality_metrics_no_errors(self):
        """Test metrics computation with no errors."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        metrics = _compute_quality_metrics(df, [])
        assert isinstance(metrics, QualityMetrics)
        assert metrics.total_rows == 3
        assert metrics.valid_rows == 3
        assert metrics.validation_rate == 1.0
        assert metrics.critical_errors == 0

    def test_compute_quality_metrics_with_errors(self):
        """Test metrics computation with errors."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        errors = [
            ValidationError(
                column="col1", check="test", failure_count=2, severity=ErrorSeverity.CRITICAL
            ),
            ValidationError(
                column="col2", check="test2", failure_count=1, severity=ErrorSeverity.WARNING
            ),
        ]
        metrics = _compute_quality_metrics(df, errors)
        assert metrics.total_rows == 3
        assert metrics.valid_rows == 0
        assert metrics.validation_rate == 0.0
        assert metrics.critical_errors == 1
        assert metrics.warnings == 1

    def test_compute_quality_metrics_null_percentages(self):
        """Test null percentage calculation."""
        df = pd.DataFrame({"col1": [1, None, 3], "col2": [None, None, 6]})
        metrics = _compute_quality_metrics(df, [])
        assert "col1" in metrics.null_percentages
        assert "col2" in metrics.null_percentages
        assert abs(metrics.null_percentages["col1"] - 1 / 3) < 0.01
        assert abs(metrics.null_percentages["col2"] - 2 / 3) < 0.01

    def test_compute_quality_metrics_severity_distribution(self):
        """Test all severity levels are counted."""
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
        errors = [
            ValidationError(
                column="c1", check="t1", failure_count=1, severity=ErrorSeverity.CRITICAL
            ),
            ValidationError(column="c2", check="t2", failure_count=1, severity=ErrorSeverity.HIGH),
            ValidationError(
                column="c3", check="t3", failure_count=1, severity=ErrorSeverity.MEDIUM
            ),
            ValidationError(
                column="c4", check="t4", failure_count=1, severity=ErrorSeverity.WARNING
            ),
        ]
        metrics = _compute_quality_metrics(df, errors)
        assert metrics.critical_errors == 1
        assert metrics.high_errors == 1
        assert metrics.medium_errors == 1
        assert metrics.warnings == 1


class TestCountBySeverity:
    """Tests for _count_by_severity function."""

    def test_count_by_severity_empty_list(self):
        """Test counting with no errors."""
        counts = _count_by_severity([])
        assert counts["critical_errors"] == 0
        assert counts["high_errors"] == 0
        assert counts["medium_errors"] == 0
        assert counts["warnings"] == 0

    def test_count_by_severity_single_critical(self):
        """Test counting single critical error."""
        errors = [
            ValidationError(
                column="c1", check="t1", failure_count=1, severity=ErrorSeverity.CRITICAL
            )
        ]
        counts = _count_by_severity(errors)
        assert counts["critical_errors"] == 1
        assert counts["high_errors"] == 0

    def test_count_by_severity_mixed_levels(self):
        """Test counting with all severity levels."""
        errors = [
            ValidationError(
                column="c1", check="t1", failure_count=1, severity=ErrorSeverity.CRITICAL
            ),
            ValidationError(
                column="c2", check="t2", failure_count=1, severity=ErrorSeverity.CRITICAL
            ),
            ValidationError(column="c3", check="t3", failure_count=1, severity=ErrorSeverity.HIGH),
            ValidationError(
                column="c4", check="t4", failure_count=1, severity=ErrorSeverity.MEDIUM
            ),
            ValidationError(
                column="c5", check="t5", failure_count=1, severity=ErrorSeverity.MEDIUM
            ),
            ValidationError(
                column="c6", check="t6", failure_count=1, severity=ErrorSeverity.WARNING
            ),
        ]
        counts = _count_by_severity(errors)
        assert counts["critical_errors"] == 2
        assert counts["high_errors"] == 1
        assert counts["medium_errors"] == 2
        assert counts["warnings"] == 1


class TestClassifyErrorSeverity:
    """Tests for _classify_error_severity function."""

    def test_classify_error_severity_critical_column(self):
        """Test classification for critical columns."""
        assert _classify_error_severity("any_check", "resultat_blanc") == ErrorSeverity.CRITICAL
        assert _classify_error_severity("any_check", "resultat_noir") == ErrorSeverity.CRITICAL
        assert _classify_error_severity("any_check", "blanc_elo") == ErrorSeverity.CRITICAL
        assert _classify_error_severity("any_check", "noir_elo") == ErrorSeverity.CRITICAL

    def test_classify_error_severity_dtype_keyword(self):
        """Test classification for dtype errors."""
        assert _classify_error_severity("dtype('int64')", "other_column") == ErrorSeverity.CRITICAL
        assert _classify_error_severity("coerce_error", "other_column") == ErrorSeverity.CRITICAL

    def test_classify_error_severity_high_keywords(self):
        """Test classification for high severity keywords."""
        assert _classify_error_severity("diff_elo_invalid", "col") == ErrorSeverity.HIGH
        assert _classify_error_severity("equipe_mismatch", "col") == ErrorSeverity.HIGH
        assert _classify_error_severity("resultat_encoding", "col") == ErrorSeverity.HIGH

    def test_classify_error_severity_medium_keywords(self):
        """Test classification for medium severity keywords."""
        assert _classify_error_severity("niveau_invalid", "col") == ErrorSeverity.MEDIUM
        assert _classify_error_severity("competition_type", "col") == ErrorSeverity.MEDIUM

    def test_classify_error_severity_default_warning(self):
        """Test default classification as warning."""
        assert _classify_error_severity("unknown_check", "other_column") == ErrorSeverity.WARNING

    def test_classify_error_severity_case_insensitive(self):
        """Test that classification is case-insensitive."""
        assert _classify_error_severity("DTYPE", "col") == ErrorSeverity.CRITICAL
        assert _classify_error_severity("DIFF_ELO", "col") == ErrorSeverity.HIGH
        assert _classify_error_severity("NIVEAU", "col") == ErrorSeverity.MEDIUM


class TestAggregateErrors:
    """Tests for _aggregate_errors function."""

    def test_aggregate_errors_empty_list(self):
        """Test aggregation with no errors."""
        result = _aggregate_errors([])
        assert result == []

    def test_aggregate_errors_single_error(self):
        """Test aggregation with single error."""
        errors = [
            ValidationError(
                column="col1",
                check="check1",
                failure_count=1,
                severity=ErrorSeverity.WARNING,
                sample_values=["value1"],
                recommendation="Fix it",
            ),
        ]
        result = _aggregate_errors(errors)
        assert len(result) == 1
        assert result[0].column == "col1"
        assert result[0].failure_count == 1

    def test_aggregate_errors_same_column_check(self):
        """Test aggregation of errors with same column and check."""
        errors = [
            ValidationError(
                column="col1",
                check="check1",
                failure_count=1,
                severity=ErrorSeverity.WARNING,
                sample_values=["value1"],
            ),
            ValidationError(
                column="col1",
                check="check1",
                failure_count=1,
                severity=ErrorSeverity.WARNING,
                sample_values=["value2"],
            ),
        ]
        result = _aggregate_errors(errors)
        assert len(result) == 1
        assert result[0].failure_count == 2
        assert len(result[0].sample_values) == 2

    def test_aggregate_errors_different_keys(self):
        """Test aggregation with different columns and checks."""
        errors = [
            ValidationError(
                column="col1", check="check1", failure_count=1, severity=ErrorSeverity.WARNING
            ),
            ValidationError(
                column="col2", check="check1", failure_count=1, severity=ErrorSeverity.WARNING
            ),
            ValidationError(
                column="col1", check="check2", failure_count=1, severity=ErrorSeverity.WARNING
            ),
        ]
        result = _aggregate_errors(errors)
        assert len(result) == 3

    def test_aggregate_errors_sample_values_limit(self):
        """Test that sample values are limited to MAX_SAMPLE_VALUES (5)."""
        errors = [
            ValidationError(
                column="col1",
                check="check1",
                failure_count=1,
                severity=ErrorSeverity.WARNING,
                sample_values=[f"val{i}"],
            )
            for i in range(10)
        ]
        result = _aggregate_errors(errors)
        assert len(result) == 1
        assert len(result[0].sample_values) == 5

    def test_aggregate_errors_mixed_aggregation(self):
        """Test complex aggregation scenario."""
        errors = [
            ValidationError(
                column="col1", check="check1", failure_count=1, severity=ErrorSeverity.CRITICAL
            ),
            ValidationError(
                column="col1", check="check1", failure_count=1, severity=ErrorSeverity.CRITICAL
            ),
            ValidationError(
                column="col1", check="check2", failure_count=1, severity=ErrorSeverity.HIGH
            ),
            ValidationError(
                column="col2", check="check1", failure_count=1, severity=ErrorSeverity.WARNING
            ),
        ]
        result = _aggregate_errors(errors)
        assert len(result) == 3
        col1_check1 = next(e for e in result if e.column == "col1" and e.check == "check1")
        assert col1_check1.failure_count == 2
