"""Tests Trajectory and Pressure - ISO 29119.

Document ID: ALICE-TEST-FE-TRAJ-PRESSURE
Version: 2.0.0
Tests count: 8

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)
- ISO/IEC 5259:2024 - Data Quality (no leakage in pressure definition)

Author: ALICE Engine Team
Last Updated: 2026-03-22
"""

import pandas as pd
import pytest

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
        result = calculate_elo_trajectory(sample_dated_games, window=6)

        # Joueur Test: 1500 -> 1560 = +60 = progression (window=6 for all 6 games)
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
        """Test que pressure_type=clutch quand win_rate pression > normal."""

        def _game(ronde: int, blanc: str, noir: str, res_b: float, res_n: float, t: str) -> dict:
            return {
                "ronde": ronde,
                "blanc_nom": blanc,
                "noir_nom": noir,
                "resultat_blanc": res_b,
                "resultat_noir": res_n,
                "type_resultat": t,
            }

        games = [_game(3, "Joueur P", f"Adv {i}", 0.5, 0.5, "nulle") for i in range(10)]
        games += [_game(8, "Joueur P", f"Adv D{i}", 1.0, 0.0, "victoire_blanc") for i in range(5)]
        df = pd.DataFrame(games)
        result = calculate_pressure_performance(df, min_games=3)

        assert not result.empty
        p_row = result[result["joueur_nom"] == "Joueur P"]
        assert len(p_row) == 1
        # win_rate normal = 0.0, win_rate pression = 1.0 => clutch_win = +1.0
        assert p_row["pressure_type"].values[0] == "clutch"

    def test_pressure_empty_df(self) -> None:
        """Test avec DataFrame vide."""
        result = calculate_pressure_performance(pd.DataFrame())
        assert result.empty

    def test_pressure_uses_zone_enjeu_not_score_dom(self) -> None:
        """Decisive = zone_enjeu_dom/ext IN (montee, danger), NOT score_dom/ext.

        All ronde=3 so ronde-fallback never fires.
        close score_dom=3/score_ext=3 must NOT trigger decisive.
        zone_enjeu_dom=montee must trigger decisive regardless of score.
        """
        games = []

        # Normal games: zone confort, various score_dom (including "close" ones)
        for i in range(8):
            games.append(
                {
                    "ronde": 3,
                    "zone_enjeu_dom": "confort",
                    "zone_enjeu_ext": "confort",
                    "score_dom": 3,  # "close" match score — must NOT trigger decisive
                    "score_ext": 3,
                    "blanc_nom": "Joueur Z",
                    "noir_nom": f"Adv N{i}",
                    "resultat_blanc": 0.0,
                    "resultat_noir": 1.0,
                    "type_resultat": "victoire_noir",
                }
            )

        # Pressure games: zone montee, non-close score_dom
        for i in range(5):
            games.append(
                {
                    "ronde": 3,
                    "zone_enjeu_dom": "montee",
                    "zone_enjeu_ext": "confort",
                    "score_dom": 5,  # clear win — must NOT prevent decisive
                    "score_ext": 1,
                    "blanc_nom": "Joueur Z",
                    "noir_nom": f"Adv P{i}",
                    "resultat_blanc": 1.0,
                    "resultat_noir": 0.0,
                    "type_resultat": "victoire_blanc",
                }
            )

        df = pd.DataFrame(games)
        result = calculate_pressure_performance(df, min_games=3)

        assert not result.empty, "Should compute pressure stats"
        row = result[result["joueur_nom"] == "Joueur Z"]
        assert len(row) == 1

        # Normal context: 8 losses → win_rate = 0.0
        # Pressure context (zone montee): 5 wins → win_rate = 1.0
        assert row["win_rate_normal"].values[0] == pytest.approx(0.0)
        assert row["win_rate_pression"].values[0] == pytest.approx(1.0)
        assert row["clutch_win"].values[0] == pytest.approx(1.0)
        assert row["pressure_type"].values[0] == "clutch"

    def test_clutch_wdl_decomposition(self) -> None:
        """clutch_win and clutch_draw are independent metrics (W/D decomposed).

        Normal: 0W/6D/4L → win=0.0, draw=0.6
        Pressure: 4W/1D/0L → win=0.8, draw=0.2
        clutch_win=+0.8, clutch_draw=-0.4
        """
        games = []

        # 6 draws + 4 losses in normal context
        for i in range(6):
            games.append(
                {
                    "ronde": 3,
                    "blanc_nom": "Joueur W",
                    "noir_nom": f"Adv N{i}",
                    "resultat_blanc": 0.5,
                    "resultat_noir": 0.5,
                    "type_resultat": "nulle",
                }
            )
        for i in range(4):
            games.append(
                {
                    "ronde": 3,
                    "blanc_nom": "Joueur W",
                    "noir_nom": f"Adv L{i}",
                    "resultat_blanc": 0.0,
                    "resultat_noir": 1.0,
                    "type_resultat": "victoire_noir",
                }
            )

        # 4 wins + 1 draw in pressure context (ronde >= 7)
        for i in range(4):
            games.append(
                {
                    "ronde": 8,
                    "blanc_nom": "Joueur W",
                    "noir_nom": f"Adv P{i}",
                    "resultat_blanc": 1.0,
                    "resultat_noir": 0.0,
                    "type_resultat": "victoire_blanc",
                }
            )
        games.append(
            {
                "ronde": 9,
                "blanc_nom": "Joueur W",
                "noir_nom": "Adv PD",
                "resultat_blanc": 0.5,
                "resultat_noir": 0.5,
                "type_resultat": "nulle",
            }
        )

        df = pd.DataFrame(games)
        result = calculate_pressure_performance(df, min_games=3)

        assert not result.empty
        row = result[result["joueur_nom"] == "Joueur W"]
        assert len(row) == 1

        assert row["win_rate_normal"].values[0] == pytest.approx(0.0)
        assert row["draw_rate_normal"].values[0] == pytest.approx(0.6)
        assert row["win_rate_pression"].values[0] == pytest.approx(0.8)
        assert row["draw_rate_pression"].values[0] == pytest.approx(0.2)
        assert row["clutch_win"].values[0] == pytest.approx(0.8)
        assert row["clutch_draw"].values[0] == pytest.approx(-0.4)

        # Both clutch metrics must be present independently
        assert "clutch_win" in row.columns
        assert "clutch_draw" in row.columns

    def test_pressure_zone_danger_triggers_decisive(self) -> None:
        """zone_enjeu_ext='danger' must also trigger is_decisive."""
        games = []

        # Normal: zone confort both sides
        for i in range(5):
            games.append(
                {
                    "ronde": 3,
                    "zone_enjeu_dom": "confort",
                    "zone_enjeu_ext": "confort",
                    "blanc_nom": "Joueur D",
                    "noir_nom": f"Adv N{i}",
                    "resultat_blanc": 0.5,
                    "resultat_noir": 0.5,
                    "type_resultat": "nulle",
                }
            )

        # Pressure: zone_enjeu_ext='danger' (ext team in relegation zone)
        for i in range(4):
            games.append(
                {
                    "ronde": 3,
                    "zone_enjeu_dom": "confort",
                    "zone_enjeu_ext": "danger",
                    "blanc_nom": "Joueur D",
                    "noir_nom": f"Adv P{i}",
                    "resultat_blanc": 1.0,
                    "resultat_noir": 0.0,
                    "type_resultat": "victoire_blanc",
                }
            )

        df = pd.DataFrame(games)
        result = calculate_pressure_performance(df, min_games=3)

        assert not result.empty
        row = result[result["joueur_nom"] == "Joueur D"]
        assert len(row) == 1
        # danger in ext → pressure games → win_rate_pression > win_rate_normal
        assert row["win_rate_pression"].values[0] > row["win_rate_normal"].values[0]

    def test_no_score_dom_column_used(self) -> None:
        """Function must work without score_dom/score_ext columns present."""
        games = []
        for i in range(6):
            games.append(
                {
                    "ronde": 3,
                    "blanc_nom": "Joueur NS",
                    "noir_nom": f"Adv {i}",
                    "resultat_blanc": 0.5,
                    "resultat_noir": 0.5,
                    "type_resultat": "nulle",
                }
            )
        for i in range(4):
            games.append(
                {
                    "ronde": 8,
                    "blanc_nom": "Joueur NS",
                    "noir_nom": f"Adv P{i}",
                    "resultat_blanc": 1.0,
                    "resultat_noir": 0.0,
                    "type_resultat": "victoire_blanc",
                }
            )

        df = pd.DataFrame(games)
        # Explicitly ensure score_dom is NOT present
        assert "score_dom" not in df.columns
        assert "score_ext" not in df.columns

        result = calculate_pressure_performance(df, min_games=3)
        assert not result.empty
        assert "joueur_nom" in result.columns
