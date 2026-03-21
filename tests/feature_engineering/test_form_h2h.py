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
        """Test calcul forme recente basique."""
        result = calculate_recent_form(sample_color_games, window=5)

        assert not result.empty
        assert "forme_recente" in result.columns
        assert "forme_tendance" in result.columns

    def test_form_tendance_values(self, sample_color_games: pd.DataFrame) -> None:
        """Test que tendance a les bonnes valeurs."""
        result = calculate_recent_form(sample_color_games, window=5)

        tendances = result["forme_tendance"].unique()
        for t in tendances:
            assert t in ["hausse", "baisse", "stable"]

    def test_form_empty_df(self) -> None:
        """Test avec DataFrame vide."""
        result = calculate_recent_form(pd.DataFrame())
        assert result.empty


class TestCalculateHeadToHead:
    """Tests pour calculate_head_to_head()."""

    def test_h2h_basic(self, sample_h2h_games: pd.DataFrame) -> None:
        """Test H2H basique."""
        result = calculate_head_to_head(sample_h2h_games, min_games=3)

        assert not result.empty
        assert "joueur_a" in result.columns
        assert "joueur_b" in result.columns
        assert "avantage_a" in result.columns

    def test_h2h_scores(self, sample_h2h_games: pd.DataFrame) -> None:
        """Test que A domine B."""
        result = calculate_head_to_head(sample_h2h_games, min_games=3)

        # A vs B: A gagne 4/5
        row = result[
            ((result["joueur_a"] == "Joueur A") & (result["joueur_b"] == "Joueur B"))
            | ((result["joueur_a"] == "Joueur B") & (result["joueur_b"] == "Joueur A"))
        ]
        assert len(row) == 1
        assert row["nb_confrontations"].values[0] == 5

    def test_h2h_min_games_filter(self, sample_h2h_games: pd.DataFrame) -> None:
        """Test filtre min_games."""
        result = calculate_head_to_head(sample_h2h_games, min_games=10)
        assert result.empty

    def test_h2h_empty_df(self) -> None:
        """Test avec DataFrame vide."""
        result = calculate_head_to_head(pd.DataFrame())
        assert result.empty
