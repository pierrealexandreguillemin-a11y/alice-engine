"""Tests for forfait filtering — ISO 5259 data quality.

Document ID: ALICE-TEST-FORFAIT-FILTER
Version: 1.0.0
Tests count: 8

Validates that forfait rows (resultat_blanc=2.0) are correctly excluded
from all feature computations, and that W/D/L rate helpers are correct.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML (forfait exclusion)
- ISO/IEC 5055:2021 - Code Quality (<300 lines)
"""

import pandas as pd
import pytest


class TestForfaitFilter:
    """Forfait exclusion — 3 tests."""

    @pytest.fixture()
    def sample_with_forfeits(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "resultat_blanc": [1.0, 0.5, 0.0, 2.0, 1.0, 2.0],
                "type_resultat": [
                    "victoire_blanc",
                    "nulle",
                    "victoire_noir",
                    "victoire_blanc",
                    "victoire_blanc",
                    "victoire_blanc",
                ],
                "blanc_elo": [1500, 1600, 1400, 1500, 1700, 1800],
            }
        )

    def test_excludes_forfeits(self, sample_with_forfeits: pd.DataFrame) -> None:
        from scripts.features.helpers import exclude_forfeits

        result = exclude_forfeits(sample_with_forfeits)
        assert len(result) == 4
        assert 2.0 not in result["resultat_blanc"].values

    def test_preserves_draws(self, sample_with_forfeits: pd.DataFrame) -> None:
        from scripts.features.helpers import exclude_forfeits

        result = exclude_forfeits(sample_with_forfeits)
        assert 0.5 in result["resultat_blanc"].values

    def test_returns_copy(self, sample_with_forfeits: pd.DataFrame) -> None:
        from scripts.features.helpers import exclude_forfeits

        result = exclude_forfeits(sample_with_forfeits)
        assert result is not sample_with_forfeits


class TestFilterPlayedGames:
    """Played games filter — 2 tests."""

    def test_filters_non_played(self) -> None:
        from scripts.features.helpers import filter_played_games

        df = pd.DataFrame(
            {
                "resultat_blanc": [1.0, 0.5, 0.0, 2.0, 1.0],
                "type_resultat": [
                    "victoire_blanc",
                    "nulle",
                    "non_joue",
                    "victoire_blanc",
                    "forfait_noir",
                ],
            }
        )
        result = filter_played_games(df)
        assert len(result) == 2  # Only victoire_blanc and nulle
        assert set(result["type_resultat"]) == {"victoire_blanc", "nulle"}

    def test_excludes_forfeits_too(self) -> None:
        from scripts.features.helpers import filter_played_games

        df = pd.DataFrame(
            {
                "resultat_blanc": [2.0],
                "type_resultat": ["victoire_blanc"],
            }
        )
        result = filter_played_games(df)
        assert len(result) == 0


class TestComputeWdlRates:
    """W/D/L rate computation — 3 tests."""

    def test_pure_wins(self) -> None:
        from scripts.features.helpers import compute_wdl_rates

        result = compute_wdl_rates(pd.Series([1.0, 1.0, 1.0]))
        assert result["win_rate"] == 1.0
        assert result["draw_rate"] == 0.0
        assert result["expected_score"] == 1.0

    def test_mixed_results(self) -> None:
        from scripts.features.helpers import compute_wdl_rates

        result = compute_wdl_rates(pd.Series([1.0, 0.5, 0.0, 1.0, 0.5]))
        assert abs(result["win_rate"] - 0.4) < 0.01
        assert abs(result["draw_rate"] - 0.4) < 0.01
        assert abs(result["expected_score"] - 0.6) < 0.01

    def test_empty_series(self) -> None:
        from scripts.features.helpers import compute_wdl_rates

        result = compute_wdl_rates(pd.Series([], dtype=float))
        assert result["win_rate"] == 0.0
