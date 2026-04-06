"""Tests Draw Priors — ISO 29119.

Document ID: ALICE-TEST-DRAW-PRIORS
Version: 1.0.0
Tests count: 6

Validates draw rate prior computation: avg_elo, elo_proximity,
per-player and per-team draw rates, and forfait exclusion.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML (forfait exclusion)
- ISO/IEC 5055:2021 - Code Quality (<300 lines)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.features.draw_priors import (
    DIFF_BINS,
    ELO_BINS,
    build_draw_rate_lookup,
    compute_draw_priors,
    compute_equipe_draw_rates,
    compute_player_draw_rates,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_games() -> pd.DataFrame:
    """Deterministic 100-row DataFrame with realistic chess data (rng=42).

    Columns: blanc_nom, noir_nom, blanc_elo, noir_elo, resultat_blanc,
             resultat_noir, equipe_dom, equipe_ext, type_resultat.
    Includes 10 forfait rows (resultat_blanc=2.0) to test exclusion.
    """
    rng = np.random.default_rng(42)
    n = 100

    elos = rng.integers(1000, 2500, size=(n, 2))
    blanc_elo = elos[:, 0].astype(float)
    noir_elo = elos[:, 1].astype(float)

    # Results: 0=noir wins, 0.5=draw, 1=blanc wins
    raw_results = rng.choice([0.0, 0.5, 1.0], size=n, p=[0.33, 0.17, 0.50])

    # Inject 10 forfait rows
    forfait_idx = rng.choice(n, size=10, replace=False)
    raw_results[forfait_idx] = 2.0

    # type_resultat mapping
    result_map = {0.0: "victoire_noir", 0.5: "nulle", 1.0: "victoire_blanc", 2.0: "victoire_blanc"}
    type_resultat = [result_map[r] for r in raw_results]

    players = [f"P{i}" for i in range(20)]
    teams = [f"Equipe{i}" for i in range(8)]

    blanc_noms = [players[i % 20] for i in range(n)]
    noir_noms = [players[(i + 1) % 20] for i in range(n)]
    equipe_dom = [teams[i % 8] for i in range(n)]
    equipe_ext = [teams[(i + 3) % 8] for i in range(n)]

    return pd.DataFrame(
        {
            "blanc_nom": blanc_noms,
            "noir_nom": noir_noms,
            "blanc_elo": blanc_elo,
            "noir_elo": noir_elo,
            "resultat_blanc": raw_results,
            "resultat_noir": 1.0 - np.where(raw_results == 2.0, 1.0, raw_results),
            "equipe_dom": equipe_dom,
            "equipe_ext": equipe_ext,
            "type_resultat": type_resultat,
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAvgElo:
    """avg_elo computation — 1 test."""

    def test_avg_elo_computed(self, sample_games: pd.DataFrame) -> None:
        """Test avg_elo = (blanc_elo + noir_elo) / 2 for all rows."""
        history = sample_games[sample_games["resultat_blanc"] != 2.0].copy()
        result = compute_draw_priors(history, history)

        expected = (history["blanc_elo"] + history["noir_elo"]) / 2
        # avg_elo must be present
        assert "avg_elo" in result.columns
        # Values match to float precision
        pd.testing.assert_series_equal(
            result["avg_elo"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            rtol=1e-5,
        )


class TestEloProximity:
    """elo_proximity range validation — 1 test."""

    def test_elo_proximity_range(self, sample_games: pd.DataFrame) -> None:
        """Test elo_proximity is always in [0, 1]."""
        history = sample_games[sample_games["resultat_blanc"] != 2.0].copy()
        result = compute_draw_priors(history, history)

        assert "elo_proximity" in result.columns
        assert result["elo_proximity"].between(0.0, 1.0).all(), "elo_proximity out of [0, 1] range"

    def test_elo_proximity_equal_elos(self) -> None:
        """Test proximity=1.0 when both players have same Elo."""
        df = pd.DataFrame(
            {
                "blanc_elo": [1500.0] * 30,
                "noir_elo": [1500.0] * 30,
                "resultat_blanc": [1.0] * 20 + [0.5] * 10,
                "equipe_dom": ["A"] * 30,
                "equipe_ext": ["B"] * 30,
                "type_resultat": ["victoire_blanc"] * 20 + ["nulle"] * 10,
            }
        )
        result = compute_draw_priors(df, df)
        assert (result["elo_proximity"] - 1.0).abs().max() < 1e-6


class TestDrawRatePrior:
    """draw_rate_prior lookup — 1 test."""

    def test_draw_rate_prior_not_null(self, sample_games: pd.DataFrame) -> None:
        """Test draw_rate_prior is non-null for most rows (fallback to global)."""
        history = sample_games[sample_games["resultat_blanc"] != 2.0].copy()
        result = compute_draw_priors(history, history)

        assert "draw_rate_prior" in result.columns
        null_frac = result["draw_rate_prior"].isna().mean()
        # Fallback to global rate means no NaN expected
        assert null_frac == 0.0, f"Unexpected NaN fraction: {null_frac:.2%}"


class TestPlayerDrawRates:
    """Per-player draw rates — 1 test."""

    def test_draw_rate_player_computed(self, sample_games: pd.DataFrame) -> None:
        """Test per-player draw rates are in [0, 1] and counts are positive."""
        result = compute_player_draw_rates(sample_games, min_games=5)

        assert not result.empty
        assert "joueur_nom" in result.columns
        assert "draw_rate" in result.columns
        assert "n_games" in result.columns

        assert result["draw_rate"].between(0.0, 1.0).all()
        assert (result["n_games"] >= 5).all()


class TestEquipeDrawRates:
    """Per-team draw rates — 1 test."""

    def test_draw_rate_equipe_computed(self, sample_games: pd.DataFrame) -> None:
        """Test per-team draw rates are in [0, 1] and counts are positive."""
        result = compute_equipe_draw_rates(sample_games, min_games=5)

        assert not result.empty
        assert "equipe" in result.columns
        assert "draw_rate" in result.columns

        assert result["draw_rate"].between(0.0, 1.0).all()
        assert (result["n_games"] >= 5).all()


class TestForfaitExclusion:
    """Forfait rows excluded from all computations — 1 test."""

    def test_forfeits_excluded(self, sample_games: pd.DataFrame) -> None:
        """Test forfeit rows (type_resultat=forfait_*) excluded from draw rates."""
        # P1/P2: all forfeits — should be excluded by filter_played_games
        df_forfeits_only = pd.DataFrame(
            {
                "blanc_nom": ["P1"] * 20,
                "noir_nom": ["P2"] * 20,
                "blanc_elo": [1500.0] * 20,
                "noir_elo": [1500.0] * 20,
                "resultat_blanc": [1.0] * 20,
                "resultat_noir": [0.0] * 20,
                "equipe_dom": ["A"] * 20,
                "equipe_ext": ["B"] * 20,
                "type_resultat": ["forfait_noir"] * 20,
            }
        )
        # P3/P4: 30 real draws
        df_real = pd.DataFrame(
            {
                "blanc_nom": ["P3"] * 30,
                "noir_nom": ["P4"] * 30,
                "blanc_elo": [1600.0] * 30,
                "noir_elo": [1600.0] * 30,
                "resultat_blanc": [0.5] * 30,
                "resultat_noir": [0.5] * 30,
                "equipe_dom": ["C"] * 30,
                "equipe_ext": ["D"] * 30,
                "type_resultat": ["nulle"] * 30,
            }
        )
        combined = pd.concat([df_forfeits_only, df_real], ignore_index=True)

        player_rates = compute_player_draw_rates(combined, min_games=5)
        p3_row = player_rates[player_rates["joueur_nom"] == "P3"]
        assert len(p3_row) == 1
        assert p3_row["draw_rate"].values[0] == pytest.approx(1.0)

        # P1 and P2 should be absent (all their games are forfeits)
        assert "P1" not in player_rates["joueur_nom"].values
        assert "P2" not in player_rates["joueur_nom"].values


class TestConstants:
    """Public constants exported for baselines.py."""

    def test_elo_bins_exported(self) -> None:
        """Test ELO_BINS is exported and has correct structure."""
        assert ELO_BINS[0] == 0
        assert ELO_BINS[-1] == 3500
        assert len(ELO_BINS) == 9  # 8 bands

    def test_diff_bins_exported(self) -> None:
        """Test DIFF_BINS is exported and has correct structure."""
        assert DIFF_BINS[0] == 0
        assert DIFF_BINS[-1] == 800
        assert len(DIFF_BINS) == 6  # 5 bands

    def test_build_draw_rate_lookup_public(self, sample_games: pd.DataFrame) -> None:
        """Test build_draw_rate_lookup returns required columns."""
        history = sample_games[sample_games["resultat_blanc"] != 2.0].copy()
        lookup = build_draw_rate_lookup(history)

        assert "elo_band" in lookup.columns
        assert "diff_band" in lookup.columns
        assert "draw_rate_prior" in lookup.columns
        assert lookup["draw_rate_prior"].between(0.0, 1.0).all()
