"""Tests Color Performance (legacy) — redirect to new W/D/L API - ISO 29119.

Document ID: ALICE-TEST-FE-COLOR-COMPAT
Version: 2.0.0
Tests count: 6

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-03-22
"""

import pandas as pd
import pytest

from scripts.features.performance import calculate_color_performance


class TestCalculateColorPerformance:
    """Tests pour calculate_color_performance() — API W/D/L V2."""

    def test_color_basic(self, sample_color_games: pd.DataFrame) -> None:
        """Test calcul performance couleur basique avec nouvelles colonnes."""
        result = calculate_color_performance(sample_color_games, min_games=10)

        assert not result.empty
        assert "win_rate_white" in result.columns
        assert "draw_rate_white" in result.columns
        assert "win_rate_black" in result.columns
        assert "draw_rate_black" in result.columns
        assert "win_adv_white" in result.columns
        assert "draw_adv_white" in result.columns
        assert "couleur_preferee" in result.columns
        assert "data_quality" in result.columns

    def test_color_joueur_x_prefere_blanc(self, sample_color_games: pd.DataFrame) -> None:
        """Test Joueur X préfère blancs (8W/10 blanc vs 4W/10 noir)."""
        result = calculate_color_performance(sample_color_games, min_games=10)

        x_row = result[result["joueur_nom"] == "Joueur X"]
        assert len(x_row) == 1

        assert x_row["win_rate_white"].values[0] == pytest.approx(0.8, abs=0.01)
        assert x_row["win_rate_black"].values[0] == pytest.approx(0.4, abs=0.01)
        assert x_row["win_adv_white"].values[0] == pytest.approx(0.4, abs=0.01)
        assert x_row["couleur_preferee"].values[0] == "blanc"
        assert x_row["data_quality"].values[0] == "complet"

    def test_color_joueur_y_neutre(self, sample_color_games: pd.DataFrame) -> None:
        """Test Joueur Y est neutre (5W/10 blanc, 5W/10 noir)."""
        result = calculate_color_performance(sample_color_games, min_games=10)

        y_row = result[result["joueur_nom"] == "Joueur Y"]
        assert len(y_row) == 1

        assert y_row["win_rate_white"].values[0] == pytest.approx(0.5, abs=0.01)
        assert y_row["win_rate_black"].values[0] == pytest.approx(0.5, abs=0.01)
        assert abs(y_row["win_adv_white"].values[0]) < 0.05
        assert y_row["couleur_preferee"].values[0] == "neutre"

    def test_color_min_games_filter(self, sample_color_games: pd.DataFrame) -> None:
        """Test filtre min_games fonctionne."""
        result = calculate_color_performance(sample_color_games, min_games=100)
        assert result.empty

    def test_color_counts(self, sample_color_games: pd.DataFrame) -> None:
        """Test comptage parties correct."""
        result = calculate_color_performance(sample_color_games, min_games=10)

        x_row = result[result["joueur_nom"] == "Joueur X"]
        assert x_row["nb_blancs"].values[0] == 10
        assert x_row["nb_noirs"].values[0] == 10

    def test_color_insufficient_data_no_fillna(self, sample_color_games: pd.DataFrame) -> None:
        """Test ISO 5259: données insuffisantes marquées NaN, pas estimées."""
        result = calculate_color_performance(sample_color_games, min_games=10, min_per_color=5)

        # Joueur Z: 10 blancs, 2 noirs => partiel_noirs
        z_row = result[result["joueur_nom"] == "Joueur Z"]
        assert len(z_row) == 1

        assert z_row["data_quality"].values[0] == "partiel_noirs"
        assert z_row["couleur_preferee"].values[0] == "donnees_insuffisantes"
        assert pd.isna(z_row["win_adv_white"].values[0])

    def test_color_empty_df(self) -> None:
        """Test avec DataFrame vide."""
        result = calculate_color_performance(pd.DataFrame())
        assert result.empty
