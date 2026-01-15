"""Tests Training Features - ISO 29119/5259.

Document ID: ALICE-TEST-TRAINING-FEATURES
Version: 1.0.0
Tests: 12

Classes:
- TestPrepareFeatures: Tests préparation features (4 tests)
- TestPrepareNumericWithQualityFlags: Tests flags qualité (3 tests)
- TestEncodeCategoricalFeatures: Tests encodage (3 tests)
- TestEncodeWithExistingEncoder: Tests réencodage (2 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML (missing flags)
- ISO/IEC 5055:2021 - Code Quality (<150 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from scripts.training.features import (
    CATEGORICAL_FEATURES,
    _encode_categorical_features,
    _encode_with_existing_encoder,
    _prepare_numeric_with_quality_flags,
    prepare_features,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """DataFrame d'exemple pour les tests."""
    return pd.DataFrame(
        {
            "blanc_elo": [1600, 1700, 1500, 1650],
            "noir_elo": [1550, 1600, 1450, 1700],
            "diff_elo": [50, 100, 50, -50],
            "echiquier": [1, 2, 3, 4],
            "niveau": [1, 1, 2, 2],
            "ronde": [1, 1, 2, 2],
            "type_competition": ["national", "national", "regional", "regional"],
            "division": ["N1", "N1", "R1", "R1"],
            "ligue_code": ["IDF", "IDF", "ARA", "ARA"],
            "blanc_titre": ["", "FM", "", ""],
            "noir_titre": ["", "", "CM", ""],
            "jour_semaine": ["Samedi", "Dimanche", "Samedi", "Samedi"],
            "resultat_blanc": [1.0, 0.5, 0.0, 1.0],
        }
    )


class TestPrepareFeatures:
    """Tests pour prepare_features."""

    def test_returns_correct_shape(self, sample_dataframe: pd.DataFrame) -> None:
        """Retourne X, y avec bonnes dimensions."""
        X, y, encoders = prepare_features(sample_dataframe, fit_encoders=True)

        assert len(X) == len(sample_dataframe)
        assert len(y) == len(sample_dataframe)
        assert isinstance(encoders, dict)

    def test_target_is_binary(self, sample_dataframe: pd.DataFrame) -> None:
        """Target y est binaire (0 ou 1)."""
        _, y, _ = prepare_features(sample_dataframe, fit_encoders=True)

        assert set(y.unique()).issubset({0, 1})

    def test_creates_encoders_when_fit(self, sample_dataframe: pd.DataFrame) -> None:
        """Crée encodeurs en mode fit."""
        _, _, encoders = prepare_features(sample_dataframe, fit_encoders=True)

        assert len(encoders) > 0
        for col in encoders:
            assert col in CATEGORICAL_FEATURES

    def test_uses_existing_encoders(self, sample_dataframe: pd.DataFrame) -> None:
        """Utilise encodeurs existants en mode transform."""
        _, _, encoders = prepare_features(sample_dataframe, fit_encoders=True)

        X2, _, _ = prepare_features(sample_dataframe, label_encoders=encoders, fit_encoders=False)

        assert len(X2) == len(sample_dataframe)


class TestPrepareNumericWithQualityFlags:
    """Tests pour _prepare_numeric_with_quality_flags."""

    def test_no_missing_no_flags(self, sample_dataframe: pd.DataFrame) -> None:
        """Pas de flags si aucune donnée manquante."""
        X_num, flags = _prepare_numeric_with_quality_flags(sample_dataframe)

        assert len(flags.columns) == 0
        assert len(X_num) == len(sample_dataframe)

    def test_creates_missing_flags(self) -> None:
        """Crée flags pour données manquantes (ISO 5259)."""
        df = pd.DataFrame(
            {
                "blanc_elo": [1600, np.nan, 1500, 1650],
                "noir_elo": [1550, 1600, np.nan, 1700],
                "diff_elo": [50, 100, 50, -50],
                "echiquier": [1, 2, 3, 4],
                "niveau": [1, 1, 2, 2],
                "ronde": [1, 1, 2, 2],
            }
        )

        X_num, flags = _prepare_numeric_with_quality_flags(df)

        assert "blanc_elo_missing" in flags.columns
        assert "noir_elo_missing" in flags.columns
        assert flags["blanc_elo_missing"].sum() == 1
        assert flags["noir_elo_missing"].sum() == 1

    def test_imputes_with_median(self) -> None:
        """Impute valeurs manquantes avec médiane."""
        df = pd.DataFrame(
            {
                "blanc_elo": [1600, np.nan, 1500, 1700],
                "noir_elo": [1550, 1600, 1450, 1700],
                "diff_elo": [50, 100, 50, -50],
                "echiquier": [1, 2, 3, 4],
                "niveau": [1, 1, 2, 2],
                "ronde": [1, 1, 2, 2],
            }
        )

        X_num, _ = _prepare_numeric_with_quality_flags(df)

        assert not X_num["blanc_elo"].isna().any()
        # Médiane de [1600, 1500, 1700] = 1600
        assert X_num["blanc_elo"].iloc[1] == 1600


class TestEncodeCategoricalFeatures:
    """Tests pour _encode_categorical_features."""

    def test_encodes_categorical_columns(self, sample_dataframe: pd.DataFrame) -> None:
        """Encode les colonnes catégorielles."""
        encoders: dict[str, LabelEncoder] = {}
        X_cat = _encode_categorical_features(sample_dataframe, encoders, fit_encoders=True)

        assert len(X_cat.columns) > 0
        for col in X_cat.columns:
            assert X_cat[col].dtype in [np.int32, np.int64]

    def test_handles_missing_column(self) -> None:
        """Ignore colonnes manquantes."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        encoders: dict[str, LabelEncoder] = {}

        X_cat = _encode_categorical_features(df, encoders, fit_encoders=True)

        assert len(X_cat.columns) == 0

    def test_fills_na_with_unknown(self) -> None:
        """Remplace NaN par UNKNOWN."""
        df = pd.DataFrame(
            {
                "type_competition": ["national", None, "regional"],
                "division": ["N1", "N2", None],
            }
        )
        encoders: dict[str, LabelEncoder] = {}

        X_cat = _encode_categorical_features(df, encoders, fit_encoders=True)

        # Vérifie que les NaN ont été encodés
        assert not X_cat.isna().any().any()


class TestEncodeWithExistingEncoder:
    """Tests pour _encode_with_existing_encoder."""

    def test_returns_none_if_no_encoder(self, sample_dataframe: pd.DataFrame) -> None:
        """Retourne None si encodeur absent."""
        encoders: dict[str, LabelEncoder] = {}

        result = _encode_with_existing_encoder(sample_dataframe, "type_competition", encoders)

        assert result is None

    def test_handles_unknown_values(self) -> None:
        """Encode valeurs inconnues comme UNKNOWN."""
        # Créer encodeur avec classes connues
        le = LabelEncoder()
        le.fit(["national", "regional"])
        encoders = {"type_competition": le}

        # DataFrame avec valeur inconnue
        df = pd.DataFrame({"type_competition": ["national", "unknown_value", "regional"]})

        result = _encode_with_existing_encoder(df, "type_competition", encoders)

        assert result is not None
        assert len(result) == 3
        # UNKNOWN doit avoir été ajouté aux classes
        assert "UNKNOWN" in le.classes_
