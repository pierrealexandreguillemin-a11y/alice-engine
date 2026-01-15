"""Tests Schema - ISO 29119.

Document ID: ALICE-TEST-MODEL-SCHEMA
Version: 1.0.0
Tests: 1 classes

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd

from scripts.model_registry import (
    validate_dataframe_schema,
    validate_train_valid_test_schema,
)


class TestSchemaValidation:
    """Tests pour validation schema DataFrame (ISO 5259)."""

    def test_valid_schema(self) -> None:
        """Test schema valide."""
        df = pd.DataFrame(
            {
                "resultat_blanc": [1.0, 0.0, 1.0],
                "blanc_elo": [1500, 1600, 1700],
                "noir_elo": [1450, 1550, 1650],
                "diff_elo": [50, 50, 50],
            }
        )

        result = validate_dataframe_schema(df)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_missing_required_column(self) -> None:
        """Test colonne requise manquante."""
        df = pd.DataFrame(
            {
                "blanc_elo": [1500, 1600],
                "noir_elo": [1450, 1550],
            }
        )

        result = validate_dataframe_schema(df)

        assert result.is_valid is False
        assert any("resultat_blanc" in e for e in result.errors)

    def test_non_numeric_column(self) -> None:
        """Test colonne non-numérique."""
        df = pd.DataFrame(
            {
                "resultat_blanc": [1.0, 0.0],
                "blanc_elo": ["high", "low"],  # Should be numeric
                "noir_elo": [1450, 1550],
                "diff_elo": [50, 50],
            }
        )

        result = validate_dataframe_schema(df, validate_elo_ranges=False)

        assert result.is_valid is False
        assert any("blanc_elo" in e and "numeric" in e for e in result.errors)

    def test_empty_dataframe(self) -> None:
        """Test DataFrame vide."""
        df = pd.DataFrame({"resultat_blanc": [], "blanc_elo": [], "noir_elo": []})

        result = validate_dataframe_schema(df)

        assert result.is_valid is False
        assert any("empty" in e.lower() for e in result.errors)

    def test_high_null_ratio_warning(self) -> None:
        """Test warning pour taux de null élevé."""
        df = pd.DataFrame(
            {
                "resultat_blanc": [1.0, None, None, None, None],  # 80% null
                "blanc_elo": [1500, 1600, 1700, 1800, 1900],
                "noir_elo": [1450, 1550, 1650, 1750, 1850],
            }
        )

        result = validate_dataframe_schema(df, required_columns=set())

        assert len(result.warnings) > 0
        assert any("null" in w.lower() for w in result.warnings)

    def test_allow_missing_columns(self) -> None:
        """Test allow_missing=True."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})

        result = validate_dataframe_schema(df, allow_missing=True, validate_elo_ranges=False)

        assert result.is_valid is True
        assert len(result.warnings) > 0  # Warning instead of error

    def test_train_valid_test_consistency(self) -> None:
        """Test cohérence train/valid/test."""
        train = pd.DataFrame(
            {
                "resultat_blanc": [1.0] * 100,
                "blanc_elo": [1500] * 100,
                "noir_elo": [1400] * 100,
                "diff_elo": [100] * 100,
            }
        )
        valid = train.copy()
        test = train.head(20).copy()

        result = validate_train_valid_test_schema(train, valid, test)

        assert result.is_valid is True

    def test_elo_below_min_error(self) -> None:
        """Test erreur ELO sous minimum."""
        df = pd.DataFrame(
            {
                "resultat_blanc": [1.0, 0.0],
                "blanc_elo": [500, 1500],  # 500 < ELO_MIN (1000)
                "noir_elo": [1450, 1550],
            }
        )

        result = validate_dataframe_schema(df)

        assert result.is_valid is False
        assert any("below ELO_MIN" in e for e in result.errors)

    def test_elo_above_max_error(self) -> None:
        """Test erreur ELO au-dessus maximum."""
        df = pd.DataFrame(
            {
                "resultat_blanc": [1.0, 0.0],
                "blanc_elo": [1500, 3500],  # 3500 > ELO_MAX (3000)
                "noir_elo": [1450, 1550],
            }
        )

        result = validate_dataframe_schema(df)

        assert result.is_valid is False
        assert any("above ELO_MAX" in e for e in result.errors)

    def test_diff_elo_inconsistency_warning(self) -> None:
        """Test warning si diff_elo incohérent."""
        df = pd.DataFrame(
            {
                "resultat_blanc": [1.0, 0.0, 1.0],
                "blanc_elo": [1500, 1600, 1700],
                "noir_elo": [1450, 1550, 1650],
                "diff_elo": [50, 100, 50],  # 100 devrait être 50
            }
        )

        result = validate_dataframe_schema(df)

        assert len(result.warnings) > 0
        assert any("diff_elo inconsistent" in w for w in result.warnings)

    def test_skip_elo_validation(self) -> None:
        """Test désactivation validation ELO."""
        df = pd.DataFrame(
            {
                "resultat_blanc": [1.0],
                "blanc_elo": [500],  # Normalement erreur
                "noir_elo": [1500],
            }
        )

        result = validate_dataframe_schema(df, validate_elo_ranges=False)

        assert result.is_valid is True  # Pas d'erreur ELO
