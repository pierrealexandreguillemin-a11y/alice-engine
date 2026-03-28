"""Tests for differential features — ISO 29119.

Document ID: ALICE-TEST-DIFFERENTIALS
Version: 1.0.0
Tests: 25
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestSafeDiff:
    """Tests for the _safe_diff helper function."""

    def test_basic_subtraction(self):
        from scripts.features.differentials import _safe_diff

        df = pd.DataFrame({"a_blanc": [0.7], "a_noir": [0.3]})
        result = _safe_diff(df, "a_blanc", "a_noir", "diff_a")
        assert result["diff_a"].iloc[0] == pytest.approx(0.4)

    def test_nan_propagation(self):
        from scripts.features.differentials import _safe_diff

        df = pd.DataFrame({"a_blanc": [np.nan], "a_noir": [0.3]})
        result = _safe_diff(df, "a_blanc", "a_noir", "diff_a")
        assert pd.isna(result["diff_a"].iloc[0])

    def test_missing_column_skips(self):
        from scripts.features.differentials import _safe_diff

        df = pd.DataFrame({"a_blanc": [0.7]})
        result = _safe_diff(df, "a_blanc", "MISSING", "diff_a")
        assert "diff_a" not in result.columns

    def test_equal_values_zero(self):
        from scripts.features.differentials import _safe_diff

        df = pd.DataFrame({"a_blanc": [0.5], "a_noir": [0.5]})
        result = _safe_diff(df, "a_blanc", "a_noir", "diff_a")
        assert result["diff_a"].iloc[0] == pytest.approx(0.0)


class TestPlayerDifferentials:
    """8 player differential features."""

    @pytest.fixture()
    def sample_df(self):
        return pd.DataFrame(
            {
                "expected_score_recent_blanc": [0.7, 0.4],
                "expected_score_recent_noir": [0.3, 0.6],
                "win_rate_recent_blanc": [0.6, 0.3],
                "win_rate_recent_noir": [0.4, 0.5],
                "draw_rate_blanc": [0.2, 0.15],
                "draw_rate_noir": [0.1, 0.3],
                "draw_rate_recent_blanc": [0.25, 0.1],
                "draw_rate_recent_noir": [0.15, 0.2],
                "win_rate_normal_blanc": [0.55, 0.4],
                "win_rate_normal_noir": [0.45, 0.5],
                "clutch_win_blanc": [0.1, -0.05],
                "clutch_win_noir": [-0.05, 0.1],
                "momentum_blanc": [0.3, -0.2],
                "momentum_noir": [-0.1, 0.4],
                "derniere_presence_blanc": [1, 4],
                "derniere_presence_noir": [2, 1],
            }
        )

    def test_diff_form(self, sample_df):
        from scripts.features.differentials import _player_differentials

        result = _player_differentials(sample_df.copy())
        assert result["diff_form"].iloc[0] == pytest.approx(0.4)
        assert result["diff_form"].iloc[1] == pytest.approx(-0.2)

    def test_diff_clutch(self, sample_df):
        from scripts.features.differentials import _player_differentials

        result = _player_differentials(sample_df.copy())
        assert result["diff_clutch"].iloc[0] == pytest.approx(0.15)
        assert result["diff_clutch"].iloc[1] == pytest.approx(-0.15)

    def test_diff_derniere_presence(self, sample_df):
        from scripts.features.differentials import _player_differentials

        result = _player_differentials(sample_df.copy())
        assert result["diff_derniere_presence"].iloc[0] == pytest.approx(-1.0)
        assert result["diff_derniere_presence"].iloc[1] == pytest.approx(3.0)

    def test_all_8_diffs_present(self, sample_df):
        from scripts.features.differentials import _player_differentials

        result = _player_differentials(sample_df.copy())
        expected = [
            "diff_form",
            "diff_win_rate_recent",
            "diff_draw_rate",
            "diff_draw_rate_recent",
            "diff_win_rate_normal",
            "diff_clutch",
            "diff_momentum",
            "diff_derniere_presence",
        ]
        for col in expected:
            assert col in result.columns, f"Missing: {col}"


class TestTeamDifferentials:
    """6 team differential features."""

    @pytest.fixture()
    def sample_df(self):
        return pd.DataFrame(
            {
                "position_dom": [3, 8],
                "position_ext": [8, 2],
                "points_cumules_dom": [12, 4],
                "points_cumules_ext": [6, 14],
                "profondeur_effectif_dom": [15, 8],
                "profondeur_effectif_ext": [8, 12],
                "noyau_stable_dom": [6, 3],
                "noyau_stable_ext": [4, 5],
                "win_rate_home_dom": [0.6, 0.4],
                "win_rate_home_ext": [0.3, 0.7],
                "draw_rate_home_dom": [0.2, 0.15],
                "draw_rate_home_ext": [0.1, 0.25],
            }
        )

    def test_diff_position(self, sample_df):
        from scripts.features.differentials import _team_differentials

        result = _team_differentials(sample_df.copy())
        assert result["diff_position"].iloc[0] == pytest.approx(-5.0)
        assert result["diff_position"].iloc[1] == pytest.approx(6.0)

    def test_diff_profondeur(self, sample_df):
        from scripts.features.differentials import _team_differentials

        result = _team_differentials(sample_df.copy())
        assert result["diff_profondeur"].iloc[0] == pytest.approx(7.0)

    def test_all_6_diffs_present(self, sample_df):
        from scripts.features.differentials import _team_differentials

        result = _team_differentials(sample_df.copy())
        expected = [
            "diff_position",
            "diff_points_cumules",
            "diff_profondeur",
            "diff_stabilite",
            "diff_win_rate_home",
            "diff_draw_rate_home",
        ]
        for col in expected:
            assert col in result.columns, f"Missing: {col}"


class TestInteractions:
    """6 board x match interaction features."""

    @pytest.fixture()
    def sample_df(self):
        return pd.DataFrame(
            {
                "diff_form": [0.4, -0.2],
                "zone_enjeu_dom": ["danger", "mi_tableau"],
                "echiquier": [1, 2],
                "est_domicile_blanc": [1, 0],
                "couleur_preferee_blanc": ["blanc", "noir"],
                "decalage_position_blanc": [2.0, -1.0],
                "match_important": [1, 0],
                "club_utilise_marge_100_dom": [0.3, 0.0],
                "flexibilite_echiquier_blanc": [0.8, 0.2],
                "joueur_promu_blanc": [1, 0],
                "diff_elo": [-150, 50],
            }
        )

    def test_form_in_danger(self, sample_df):
        from scripts.features.differentials import _board_match_interactions

        result = _board_match_interactions(sample_df.copy())
        assert result["form_in_danger"].iloc[0] == pytest.approx(0.4)
        assert result["form_in_danger"].iloc[1] == pytest.approx(0.0)

    def test_color_match_dom_odd_pref_blanc(self, sample_df):
        from scripts.features.differentials import _board_match_interactions

        result = _board_match_interactions(sample_df.copy())
        assert result["color_match"].iloc[0] == 1

    def test_color_match_ext_even_pref_noir(self, sample_df):
        from scripts.features.differentials import _board_match_interactions

        result = _board_match_interactions(sample_df.copy())
        assert result["color_match"].iloc[1] == 0

    def test_promu_vs_strong(self, sample_df):
        from scripts.features.differentials import _board_match_interactions

        result = _board_match_interactions(sample_df.copy())
        assert result["promu_vs_strong"].iloc[0] == pytest.approx(0.375)
        assert result["promu_vs_strong"].iloc[1] == pytest.approx(0.0)

    def test_decalage_important(self, sample_df):
        from scripts.features.differentials import _board_match_interactions

        result = _board_match_interactions(sample_df.copy())
        assert result["decalage_important"].iloc[0] == pytest.approx(2.0)
        assert result["decalage_important"].iloc[1] == pytest.approx(0.0)
