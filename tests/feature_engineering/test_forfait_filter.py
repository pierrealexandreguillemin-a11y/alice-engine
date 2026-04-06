"""Tests for forfait filtering — ISO 5259 data quality.

Document ID: ALICE-TEST-FORFAIT-FILTER
Version: 2.0.0
Tests count: 8

Validates that non-played games (forfeits, non_joue) are correctly excluded
via type_resultat, and that resultat_blanc=2.0 (victoire jeunes FFE) is
treated as a real win.

Fix 2026-03-25: resultat_blanc=2.0 is a real win (J02 §4.1), NOT a forfeit.
See docs/postmortem/2026-03-25-resultat-blanc-2.0-bug.md

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML (type_resultat-based filtering)
- ISO/IEC 5055:2021 - Code Quality (<300 lines)
"""

import pandas as pd
import pytest


class TestFilterPlayedGames:
    """Played games filter via type_resultat — 4 tests."""

    @pytest.fixture()
    def sample_mixed(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "resultat_blanc": [1.0, 0.5, 0.0, 2.0, 1.0, 0.0],
                "type_resultat": [
                    "victoire_blanc",
                    "nulle",
                    "non_joue",
                    "victoire_blanc",  # 2.0 = victoire jeunes, IS played
                    "forfait_noir",
                    "forfait_blanc",
                ],
            }
        )

    def test_filters_non_played(self, sample_mixed: pd.DataFrame) -> None:
        from scripts.features.helpers import filter_played_games

        result = filter_played_games(sample_mixed)
        assert len(result) == 3  # victoire_blanc(1.0) + nulle + victoire_blanc(2.0)

    def test_keeps_2_0_wins(self, sample_mixed: pd.DataFrame) -> None:
        """resultat_blanc=2.0 with type_resultat=victoire_blanc must be KEPT."""
        from scripts.features.helpers import filter_played_games

        result = filter_played_games(sample_mixed)
        assert 2.0 in result["resultat_blanc"].values

    def test_excludes_forfeits(self, sample_mixed: pd.DataFrame) -> None:
        from scripts.features.helpers import filter_played_games

        result = filter_played_games(sample_mixed)
        assert "forfait_noir" not in result["type_resultat"].values
        assert "forfait_blanc" not in result["type_resultat"].values
        assert "non_joue" not in result["type_resultat"].values

    def test_returns_copy(self, sample_mixed: pd.DataFrame) -> None:
        from scripts.features.helpers import filter_played_games

        result = filter_played_games(sample_mixed)
        assert result is not sample_mixed


class TestComputeWdlRates:
    """W/D/L rate computation — 4 tests."""

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

    def test_2_0_counted_as_win(self) -> None:
        """resultat_blanc=2.0 must be counted as win (jeunes FFE J02 §4.1)."""
        from scripts.features.helpers import compute_wdl_rates

        result = compute_wdl_rates(pd.Series([2.0, 0.5, 0.0]))
        assert abs(result["win_rate"] - 1 / 3) < 0.01
        assert abs(result["draw_rate"] - 1 / 3) < 0.01

    def test_empty_series(self) -> None:
        from scripts.features.helpers import compute_wdl_rates

        result = compute_wdl_rates(pd.Series([], dtype=float))
        assert result["win_rate"] == 0.0
