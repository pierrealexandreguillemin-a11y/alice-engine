"""Tests Calculate Recent Form — W/D/L decomposition - ISO 29119.

Document ID: ALICE-TEST-FEATURES-RECENT-FORM-FORM
Version: 2.0.0
Tests count: 8

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)
- ISO/IEC 42001:2023 - AI traceability (W/D/L decomposition)

Author: ALICE Engine Team
Last Updated: 2026-03-22
"""

import pandas as pd

from scripts.features.recent_form import calculate_recent_form

_NEW_COLS = {"win_rate_recent", "draw_rate_recent", "expected_score_recent"}


class TestCalculateRecentForm:
    """Tests pour calculate_recent_form (colonnes V2)."""

    def test_basic_form_returns_wdl_columns(self, sample_matches_for_form: pd.DataFrame) -> None:
        """Test calcul forme basique retourne colonnes W/D/L."""
        result = calculate_recent_form(sample_matches_for_form, window=5)

        assert not result.empty
        assert "joueur_nom" in result.columns
        assert "win_rate_recent" in result.columns
        assert "draw_rate_recent" in result.columns
        assert "expected_score_recent" in result.columns
        assert "nb_matchs_forme" in result.columns
        assert "win_trend" in result.columns
        assert "draw_trend" in result.columns

    def test_win_draw_rates_in_valid_range(self, sample_matches_for_form: pd.DataFrame) -> None:
        """Test que win_rate et draw_rate sont dans [0, 1]."""
        result = calculate_recent_form(sample_matches_for_form, window=5)

        assert result["win_rate_recent"].between(0.0, 1.0).all()
        assert result["draw_rate_recent"].between(0.0, 1.0).all()
        assert result["expected_score_recent"].between(0.0, 1.0).all()

    def test_joueur_a_win_trend_hausse(self, sample_matches_for_form: pd.DataFrame) -> None:
        """Test Joueur A avec victoires croissantes = win_trend hausse."""
        result = calculate_recent_form(sample_matches_for_form, window=5)

        joueur_a = result[result["joueur_nom"] == "Joueur A"]
        assert len(joueur_a) >= 1
        assert joueur_a.iloc[0]["win_trend"] == "hausse"

    def test_joueur_b_win_trend_baisse(self, sample_matches_for_form: pd.DataFrame) -> None:
        """Test Joueur B avec victoires decroissantes = win_trend baisse."""
        result = calculate_recent_form(sample_matches_for_form, window=5)

        joueur_b = result[result["joueur_nom"] == "Joueur B"]
        assert len(joueur_b) >= 1
        assert joueur_b.iloc[0]["win_trend"] == "baisse"

    def test_joueur_excluded_if_below_stratify_threshold(self) -> None:
        """Test joueur avec 2 matchs (< _STRATIFY_MIN_GAMES=3) est exclu."""
        df = pd.DataFrame(
            [
                {
                    "blanc_nom": "Joueur Rare",
                    "noir_nom": f"Adv{i}",
                    "resultat_blanc": 1.0,
                    "resultat_noir": 0.0,
                    "type_resultat": "victoire_blanc",
                    "date": f"2025-01-0{i}",
                }
                for i in range(1, 3)  # Only 2 games
            ]
        )
        result = calculate_recent_form(df, window=5)

        # Result is empty (no player meets the 3-game threshold)
        assert result.empty or "Joueur Rare" not in result.get("joueur_nom", pd.Series()).values

    def test_empty_dataframe_returns_empty(self) -> None:
        """Test DataFrame vide retourne DataFrame vide."""
        result = calculate_recent_form(pd.DataFrame(), window=5)

        assert result.empty

    def test_forfaits_excluded_win_rate_is_one(self, matches_with_forfaits: pd.DataFrame) -> None:
        """Test ISO 5259: forfaits exclus — Joueur D a win_rate=1.0."""
        result = calculate_recent_form(matches_with_forfaits, window=5)

        joueur_d = result[result["joueur_nom"] == "Joueur D"]
        assert len(joueur_d) == 1
        assert joueur_d.iloc[0]["win_rate_recent"] == 1.0

    def test_custom_window_includes_joueur_c(self, sample_matches_for_form: pd.DataFrame) -> None:
        """Test avec taille de fenetre window=3 inclut Joueur C."""
        result = calculate_recent_form(sample_matches_for_form, window=3)

        joueur_c = result[result["joueur_nom"] == "Joueur C"]
        assert len(joueur_c) == 1
