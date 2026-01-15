"""Tests Trajectory and Pressure - ISO 29119.

Document ID: ALICE-TEST-FE-TRAJ-PRESSURE
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd

from scripts.features.advanced import (
    calculate_elo_trajectory,
    calculate_pressure_performance,
)


class TestCalculateEloTrajectory:
    """Tests pour calculate_elo_trajectory()."""

    def test_trajectory_basic(self, sample_dated_games: pd.DataFrame) -> None:
        """Test trajectoire Elo basique."""
        result = calculate_elo_trajectory(sample_dated_games, window=4)

        assert not result.empty
        assert "elo_trajectory" in result.columns
        assert "momentum" in result.columns

    def test_trajectory_progression(self, sample_dated_games: pd.DataFrame) -> None:
        """Test detection progression Elo."""
        result = calculate_elo_trajectory(sample_dated_games, window=4)

        # Joueur Test: 1500 -> 1560 = +60 = progression
        row = result[result["joueur_nom"] == "Joueur Test"]
        if not row.empty:
            assert row["elo_trajectory"].values[0] == "progression"
            assert row["momentum"].values[0] > 0

    def test_trajectory_no_date(self) -> None:
        """Test sans colonne date."""
        df = pd.DataFrame({"blanc_nom": ["A"], "blanc_elo": [1500]})
        result = calculate_elo_trajectory(df)
        assert result.empty


class TestCalculatePressurePerformance:
    """Tests pour calculate_pressure_performance()."""

    def test_pressure_basic(self) -> None:
        """Test performance pression basique."""
        games = []
        # Matchs normaux (ronde < 7)
        for i in range(10):
            games.append(
                {
                    "ronde": 3,
                    "score_dom": 4,
                    "score_ext": 2,
                    "blanc_nom": "Joueur P",
                    "noir_nom": f"Adv {i}",
                    "resultat_blanc": 0.5,
                    "resultat_noir": 0.5,
                    "type_resultat": "nulle",
                }
            )
        # Matchs decisifs (ronde >= 7)
        for i in range(5):
            games.append(
                {
                    "ronde": 8,
                    "score_dom": 3,
                    "score_ext": 3,
                    "blanc_nom": "Joueur P",
                    "noir_nom": f"Adv D{i}",
                    "resultat_blanc": 1.0,
                    "resultat_noir": 0.0,
                    "type_resultat": "victoire_blanc",
                }
            )

        df = pd.DataFrame(games)
        result = calculate_pressure_performance(df, min_games=3)

        assert not result.empty
        p_row = result[result["joueur_nom"] == "Joueur P"]
        assert len(p_row) == 1
        # Score normal = 0.5, score pression = 1.0 => clutch
        assert p_row["pressure_type"].values[0] == "clutch"

    def test_pressure_empty_df(self) -> None:
        """Test avec DataFrame vide."""
        result = calculate_pressure_performance(pd.DataFrame())
        assert result.empty
