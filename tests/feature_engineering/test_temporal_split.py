"""Tests Temporal Split - ISO 29119.

Document ID: ALICE-TEST-FE-TEMPORAL
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd


class TestTemporalSplit:
    """Tests pour temporal_split() du module principal feature_engineering."""

    def test_temporal_split_basic(self) -> None:
        """Test split temporel basique."""
        from scripts.feature_engineering import temporal_split

        df = pd.DataFrame(
            {
                "saison": [2020, 2021, 2022, 2023, 2024, 2025],
                "data": ["a", "b", "c", "d", "e", "f"],
            }
        )

        train, valid, test = temporal_split(df, train_end=2022, valid_end=2023)

        assert len(train) == 3  # 2020, 2021, 2022
        assert len(valid) == 1  # 2023
        assert len(test) == 2  # 2024, 2025

    def test_temporal_split_train_contains_correct_years(self) -> None:
        """Test que train contient les bonnes annees."""
        from scripts.feature_engineering import temporal_split

        df = pd.DataFrame(
            {
                "saison": [2020, 2021, 2022, 2023, 2024],
                "value": range(5),
            }
        )

        train, _, _ = temporal_split(df, train_end=2022, valid_end=2023)

        assert set(train["saison"].unique()) == {2020, 2021, 2022}

    def test_temporal_split_valid_contains_correct_years(self) -> None:
        """Test que valid contient les bonnes annees."""
        from scripts.feature_engineering import temporal_split

        df = pd.DataFrame(
            {
                "saison": [2020, 2021, 2022, 2023, 2024],
                "value": range(5),
            }
        )

        _, valid, _ = temporal_split(df, train_end=2022, valid_end=2023)

        assert set(valid["saison"].unique()) == {2023}

    def test_temporal_split_test_contains_correct_years(self) -> None:
        """Test que test contient les bonnes annees."""
        from scripts.feature_engineering import temporal_split

        df = pd.DataFrame(
            {
                "saison": [2020, 2021, 2022, 2023, 2024, 2025],
                "value": range(6),
            }
        )

        _, _, test = temporal_split(df, train_end=2022, valid_end=2023)

        assert set(test["saison"].unique()) == {2024, 2025}

    def test_temporal_split_empty_valid(self) -> None:
        """Test avec valid vide (pas de donnees pour cette periode)."""
        from scripts.feature_engineering import temporal_split

        df = pd.DataFrame(
            {
                "saison": [2020, 2021, 2025],
                "value": range(3),
            }
        )

        train, valid, test = temporal_split(df, train_end=2022, valid_end=2023)

        assert len(train) == 2
        assert len(valid) == 0  # Pas de 2023
        assert len(test) == 1

    def test_temporal_split_empty_test(self) -> None:
        """Test avec test vide."""
        from scripts.feature_engineering import temporal_split

        df = pd.DataFrame(
            {
                "saison": [2020, 2021, 2022, 2023],
                "value": range(4),
            }
        )

        _, _, test = temporal_split(df, train_end=2022, valid_end=2023)

        assert len(test) == 0

    def test_temporal_split_custom_boundaries(self) -> None:
        """Test avec boundaries personnalisees."""
        from scripts.feature_engineering import temporal_split

        df = pd.DataFrame(
            {
                "saison": [2018, 2019, 2020, 2021, 2022],
                "value": range(5),
            }
        )

        train, valid, test = temporal_split(df, train_end=2019, valid_end=2020)

        assert len(train) == 2  # 2018, 2019
        assert len(valid) == 1  # 2020
        assert len(test) == 2  # 2021, 2022

    def test_temporal_split_no_data_leakage(self) -> None:
        """Test ISO 5259: pas de data leakage entre splits."""
        from scripts.feature_engineering import temporal_split

        df = pd.DataFrame(
            {
                "saison": [2020, 2021, 2022, 2023, 2024],
                "value": range(5),
            }
        )

        train, valid, test = temporal_split(df, train_end=2022, valid_end=2023)

        # Verifier qu'il n'y a pas de chevauchement
        train_years = set(train["saison"].unique())
        valid_years = set(valid["saison"].unique())
        test_years = set(test["saison"].unique())

        assert train_years.isdisjoint(valid_years)
        assert train_years.isdisjoint(test_years)
        assert valid_years.isdisjoint(test_years)
