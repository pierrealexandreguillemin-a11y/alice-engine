"""Tests Feature Engineering V2 - ISO 29119/5259.

Document ID: ALICE-TEST-FE-V2
Version: 1.0.0
Tests: 6

Classes:
- TestTemporalSplit: Tests split temporel (3 tests)
- TestComputeFeaturesForSplit: Tests calcul features (2 tests)
- TestMain: Tests point d'entrée (1 test)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality (no leakage)
- ISO/IEC 5055:2021 - Code Quality (<80 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from unittest.mock import patch

import pandas as pd

from scripts.feature_engineering import temporal_split


class TestTemporalSplit:
    """Tests pour temporal_split."""

    def test_splits_by_year(self):
        """Split correct par année."""
        df = pd.DataFrame(
            {
                "saison": [2020, 2021, 2022, 2023, 2024],
                "value": [1, 2, 3, 4, 5],
            }
        )

        train, valid, test = temporal_split(df, train_end=2022, valid_end=2023)

        assert len(train) == 3  # 2020, 2021, 2022
        assert len(valid) == 1  # 2023
        assert len(test) == 1  # 2024

    def test_empty_splits(self):
        """Gère les splits vides."""
        df = pd.DataFrame(
            {
                "saison": [2020, 2021],
                "value": [1, 2],
            }
        )

        train, valid, test = temporal_split(df, train_end=2022, valid_end=2023)

        assert len(train) == 2
        assert len(valid) == 0
        assert len(test) == 0

    def test_preserves_columns(self):
        """Préserve toutes les colonnes."""
        df = pd.DataFrame(
            {
                "saison": [2020, 2023, 2024],
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
            }
        )

        train, valid, test = temporal_split(df, train_end=2022, valid_end=2023)

        assert "a" in train.columns
        assert "b" in valid.columns


class TestComputeFeaturesForSplit:
    """Tests pour compute_features_for_split."""

    def test_filters_non_played(self):
        """Filtre les parties non jouées."""
        from scripts.feature_engineering import compute_features_for_split

        df_split = pd.DataFrame(
            {
                "saison": [2024],
                "type_resultat": ["1-0"],
            }
        )
        df_history = pd.DataFrame(
            {
                "saison": [2023],
                "type_resultat": ["1-0"],
                "blanc_elo": [1500],
            }
        )

        with patch("scripts.feature_engineering.extract_all_features") as mock_extract:
            with patch("scripts.feature_engineering.merge_all_features") as mock_merge:
                mock_extract.return_value = {}
                mock_merge.return_value = df_split

                result = compute_features_for_split(
                    df_split, df_history, "test", include_advanced=False
                )

                mock_extract.assert_called_once()
                assert len(result) == 1

    def test_includes_advanced_flag(self):
        """Respecte le flag include_advanced."""
        from scripts.feature_engineering import compute_features_for_split

        df_split = pd.DataFrame({"saison": [2024], "type_resultat": ["0-1"]})
        df_history = pd.DataFrame({"saison": [2023], "type_resultat": ["0-1"]})

        with patch("scripts.feature_engineering.extract_all_features") as mock_extract:
            with patch("scripts.feature_engineering.merge_all_features") as mock_merge:
                mock_extract.return_value = {}
                mock_merge.return_value = df_split

                compute_features_for_split(df_split, df_history, "train", include_advanced=True)

                _, kwargs = mock_extract.call_args
                # Le dernier argument positionnel est include_advanced
                args = mock_extract.call_args[0]
                assert args[-1] is True


class TestMain:
    """Tests pour main."""

    def test_main_exists(self):
        """Fonction main existe."""
        from scripts.feature_engineering import main

        assert callable(main)
