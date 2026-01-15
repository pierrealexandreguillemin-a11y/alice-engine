"""Tests Features Preparation - ISO 29119.

Document ID: ALICE-TEST-TRAIN-FEATURES
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd
import pytest

from scripts.training import prepare_features


class TestPrepareFeatures:
    """Tests pour prepare_features."""

    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """DataFrame de test."""
        return pd.DataFrame(
            {
                "blanc_elo": [1500, 1600, 1700, 1800],
                "noir_elo": [1550, 1650, 1750, 1850],
                "diff_elo": [-50, -50, -50, -50],
                "echiquier": [1, 2, 3, 4],
                "niveau": [1, 1, 2, 2],
                "ronde": [1, 1, 1, 1],
                "type_competition": ["N1", "N1", "N2", "N2"],
                "division": ["A", "A", "B", "B"],
                "ligue_code": ["IDF", "IDF", "ARA", "ARA"],
                "blanc_titre": ["", "FM", "", "IM"],
                "noir_titre": ["", "", "FM", ""],
                "jour_semaine": ["samedi", "samedi", "dimanche", "dimanche"],
                "resultat_blanc": [1.0, 0.5, 0.0, 1.0],
            }
        )

    def test_prepare_features_fit_encoders(self, sample_dataframe: pd.DataFrame) -> None:
        """Test preparation avec fit_encoders=True."""
        X, y, encoders = prepare_features(sample_dataframe, fit_encoders=True)

        assert len(X) == 4
        assert len(y) == 4
        assert len(encoders) > 0
        assert "type_competition" in encoders

    def test_prepare_features_reuse_encoders(self, sample_dataframe: pd.DataFrame) -> None:
        """Test preparation avec encodeurs existants."""
        _, _, encoders = prepare_features(sample_dataframe, fit_encoders=True)

        new_df = sample_dataframe.copy()
        X, y, _ = prepare_features(new_df, label_encoders=encoders)

        assert len(X) == 4
        assert len(y) == 4

    def test_prepare_features_target_binary(self, sample_dataframe: pd.DataFrame) -> None:
        """Test target est binaire (victoire=1, autre=0)."""
        _, y, _ = prepare_features(sample_dataframe, fit_encoders=True)

        assert set(y.unique()).issubset({0, 1})
        assert y.sum() == 2

    def test_prepare_features_unknown_category(self, sample_dataframe: pd.DataFrame) -> None:
        """Test gestion categorie inconnue."""
        _, _, encoders = prepare_features(sample_dataframe, fit_encoders=True)

        new_df = sample_dataframe.copy()
        new_df.loc[0, "type_competition"] = "UNKNOWN_CATEGORY"

        X, _, _ = prepare_features(new_df, label_encoders=encoders)

        assert len(X) == 4
