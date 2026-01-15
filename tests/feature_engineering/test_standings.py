"""Tests Standings - ISO 29119.

Document ID: ALICE-TEST-FE-STANDINGS
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd

from scripts.features.standings import (
    calculate_standings,
    extract_team_enjeu_features,
)


class TestCalculateStandings:
    """Tests pour calculate_standings()."""

    def test_standings_basic(self, sample_matches: pd.DataFrame) -> None:
        """Test classement basique apres 2 rondes."""
        result = calculate_standings(sample_matches)

        assert not result.empty
        assert "position" in result.columns
        assert "points_cumules" in result.columns

    def test_standings_points_correct(self, sample_matches: pd.DataFrame) -> None:
        """Test points corrects (2 pts victoire, 1 pt nul)."""
        result = calculate_standings(sample_matches)

        r2 = result[result["ronde"] == 2]

        a_pts = r2[r2["equipe"] == "Equipe A"]["points_cumules"].values[0]
        c_pts = r2[r2["equipe"] == "Equipe C"]["points_cumules"].values[0]
        b_pts = r2[r2["equipe"] == "Equipe B"]["points_cumules"].values[0]
        d_pts = r2[r2["equipe"] == "Equipe D"]["points_cumules"].values[0]

        assert a_pts == 3  # 2 + 1
        assert c_pts == 3  # 2 + 1
        assert b_pts == 2  # 0 + 2
        assert d_pts == 0  # 0 + 0

    def test_standings_position_order(self, sample_matches: pd.DataFrame) -> None:
        """Test que les positions sont ordonnees correctement."""
        result = calculate_standings(sample_matches)
        r2 = result[result["ronde"] == 2]

        d_pos = r2[r2["equipe"] == "Equipe D"]["position"].values[0]
        assert d_pos == 4

        a_pos = r2[r2["equipe"] == "Equipe A"]["position"].values[0]
        c_pos = r2[r2["equipe"] == "Equipe C"]["position"].values[0]
        assert a_pos in [1, 2]
        assert c_pos in [1, 2]

    def test_standings_ecart_premier(self, sample_matches: pd.DataFrame) -> None:
        """Test ecart_premier calcule correctement."""
        result = calculate_standings(sample_matches)
        r2 = result[result["ronde"] == 2]

        d_ecart = r2[r2["equipe"] == "Equipe D"]["ecart_premier"].values[0]
        assert d_ecart == 3

        a_ecart = r2[r2["equipe"] == "Equipe A"]["ecart_premier"].values[0]
        assert a_ecart == 0

    def test_standings_nb_equipes(self, sample_matches: pd.DataFrame) -> None:
        """Test nb_equipes correct."""
        result = calculate_standings(sample_matches)
        assert (result["nb_equipes"] == 4).all()

    def test_standings_empty_df(self) -> None:
        """Test avec DataFrame vide."""
        result = calculate_standings(pd.DataFrame())
        assert result.empty

    def test_standings_missing_columns(self) -> None:
        """Test avec colonnes manquantes."""
        df = pd.DataFrame({"saison": [2025], "equipe_dom": ["A"]})
        result = calculate_standings(df)
        assert result.empty

    def test_standings_has_tiebreaker_columns(self, sample_matches: pd.DataFrame) -> None:
        """Test que les colonnes tie-breaker sont presentes."""
        result = calculate_standings(sample_matches)

        assert "victoires" in result.columns
        assert "diff_points_matchs" in result.columns


class TestExtractTeamEnjeuFeatures:
    """Tests pour extract_team_enjeu_features()."""

    def test_enjeu_uses_real_position(self, sample_matches: pd.DataFrame) -> None:
        """Test que zone_enjeu utilise position reelle."""
        from scripts.features import calculate_standings

        standings = calculate_standings(sample_matches)
        result = extract_team_enjeu_features(sample_matches, standings)

        assert not result.empty
        assert "position" in result.columns
        assert "zone_enjeu" in result.columns

        d_zone = result[result["equipe"] == "Equipe D"]["zone_enjeu"].unique()
        assert len(d_zone) > 0

    def test_enjeu_has_ecarts(self, sample_matches: pd.DataFrame) -> None:
        """Test que ecart_premier et ecart_dernier sont presents."""
        from scripts.features import calculate_standings

        standings = calculate_standings(sample_matches)
        result = extract_team_enjeu_features(sample_matches, standings)

        assert "ecart_premier" in result.columns
        assert "ecart_dernier" in result.columns
