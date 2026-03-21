"""Tests Form and Head-to-Head - ISO 29119.

Document ID: ALICE-TEST-FE-FORM-H2H
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd

from scripts.features.advanced import calculate_head_to_head
from scripts.features.performance import calculate_recent_form


class TestCalculateRecentForm:
    """Tests pour calculate_recent_form()."""

    def test_form_basic(self, sample_color_games: pd.DataFrame) -> None:
        """Test calcul forme recente basique — colonnes W/D/L."""
        result = calculate_recent_form(sample_color_games, window=5)

        assert not result.empty
        assert "win_rate_recent" in result.columns
        assert "draw_rate_recent" in result.columns
        assert "expected_score_recent" in result.columns

    def test_form_trend_values(self, sample_color_games: pd.DataFrame) -> None:
        """Test que win_trend et draw_trend ont les bonnes valeurs."""
        result = calculate_recent_form(sample_color_games, window=5)

        for col in ("win_trend", "draw_trend"):
            for val in result[col].dropna().unique():
                assert val in ["hausse", "baisse", "stable"]

    def test_form_empty_df(self) -> None:
        """Test avec DataFrame vide."""
        result = calculate_recent_form(pd.DataFrame())
        assert result.empty


class TestCalculateHeadToHead:
    """Tests pour calculate_head_to_head() — W/D/L decomposition."""

    def test_h2h_columns(self, sample_h2h_games: pd.DataFrame) -> None:
        """Test colonnes de sortie W/D/L."""
        result = calculate_head_to_head(sample_h2h_games, min_games=3)

        assert not result.empty
        assert "joueur_a" in result.columns
        assert "joueur_b" in result.columns
        assert "nb_confrontations" in result.columns
        assert "h2h_win_rate" in result.columns
        assert "h2h_draw_rate" in result.columns
        assert "h2h_exists" in result.columns
        # old column must be gone
        assert "avantage_a" not in result.columns
        assert "score_a" not in result.columns
        assert "score_b" not in result.columns

    def test_h2h_win_rate_values(self, sample_h2h_games: pd.DataFrame) -> None:
        """Test que win_rate et draw_rate sont cohérents (somme <= 1)."""
        result = calculate_head_to_head(sample_h2h_games, min_games=3)

        for _, row in result.iterrows():
            assert 0.0 <= row["h2h_win_rate"] <= 1.0
            assert 0.0 <= row["h2h_draw_rate"] <= 1.0
            assert row["h2h_win_rate"] + row["h2h_draw_rate"] <= 1.0 + 1e-9

    def test_h2h_exists_flag(self, sample_h2h_games: pd.DataFrame) -> None:
        """Test que h2h_exists=True pour toutes les paires retournées."""
        result = calculate_head_to_head(sample_h2h_games, min_games=3)
        assert result["h2h_exists"].all()

    def test_h2h_confrontations_count(self, sample_h2h_games: pd.DataFrame) -> None:
        """Test décompte confrontations A vs B = 5."""
        result = calculate_head_to_head(sample_h2h_games, min_games=3)

        row = result[
            ((result["joueur_a"] == "Joueur A") & (result["joueur_b"] == "Joueur B"))
            | ((result["joueur_a"] == "Joueur B") & (result["joueur_b"] == "Joueur A"))
        ]
        assert len(row) == 1
        assert row["nb_confrontations"].values[0] == 5

    def test_h2h_win_rate_dominance(self, sample_h2h_games: pd.DataFrame) -> None:
        """Test que joueur_a (alphabetical first) a win_rate > 0.5 quand A domine."""
        result = calculate_head_to_head(sample_h2h_games, min_games=3)
        # "Joueur A" < "Joueur B" alphabetically — joueur_a = "Joueur A"
        # A wins 4/5 = 0.8
        row = result[(result["joueur_a"] == "Joueur A") & (result["joueur_b"] == "Joueur B")]
        assert len(row) == 1
        assert abs(row["h2h_win_rate"].values[0] - 0.8) < 1e-6

    def test_h2h_min_games_filter(self, sample_h2h_games: pd.DataFrame) -> None:
        """Test filtre min_games."""
        result = calculate_head_to_head(sample_h2h_games, min_games=10)
        assert result.empty

    def test_h2h_empty_df(self) -> None:
        """Test avec DataFrame vide."""
        result = calculate_head_to_head(pd.DataFrame())
        assert result.empty

    def test_h2h_forfait_excluded(self) -> None:
        """Test que les forfaits sont exclus du calcul H2H."""
        games = [
            {
                "blanc_nom": "Joueur A",
                "noir_nom": "Joueur B",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
            }
            for _ in range(4)
        ]
        # Add a forfait — should be excluded
        games.append(
            {
                "blanc_nom": "Joueur A",
                "noir_nom": "Joueur B",
                "resultat_blanc": 2.0,
                "resultat_noir": 0.0,
                "type_resultat": "forfait_noir",
            }
        )
        df = pd.DataFrame(games)
        result = calculate_head_to_head(df, min_games=3)
        # Only 4 real games, not 5
        row = result[(result["joueur_a"] == "Joueur A") & (result["joueur_b"] == "Joueur B")]
        assert len(row) == 1
        assert row["nb_confrontations"].values[0] == 4
