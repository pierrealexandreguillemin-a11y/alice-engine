"""Tests club_behavior — W/D/L home rates + forfait exclusion - ISO 29119.

Document ID: ALICE-TEST-FE-CLUB-BEH
Version: 1.0.0
Tests count: 9

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)
- ISO/IEC 5259:2024 - Data Quality (forfait exclusion)
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.features.club_behavior import extract_club_behavior


@pytest.fixture
def sample_club_games() -> pd.DataFrame:
    """Fixture avec matchs home/away pour un club sur une saison."""
    rows = []
    # Club A: 3 home games — 2 wins, 1 draw
    for ronde, resultat, type_r in [
        (1, 1.0, "victoire_blanc"),
        (2, 1.0, "victoire_blanc"),
        (3, 0.5, "nulle"),
    ]:
        rows.append(
            {
                "saison": 2025,
                "ronde": ronde,
                "equipe_dom": "Club A",
                "equipe_ext": "Club B",
                "blanc_nom": f"Joueur A{ronde}",
                "noir_nom": f"Joueur B{ronde}",
                "blanc_elo": 1800,
                "noir_elo": 1750,
                "echiquier": 1,
                "resultat_blanc": resultat,
                "resultat_noir": 1.0 - resultat,
                "type_resultat": type_r,
            }
        )
    # Club A: 2 away games
    for ronde, resultat, type_r in [
        (4, 0.0, "victoire_blanc"),
        (5, 1.0, "victoire_noir"),
    ]:
        rows.append(
            {
                "saison": 2025,
                "ronde": ronde,
                "equipe_dom": "Club C",
                "equipe_ext": "Club A",
                "blanc_nom": f"Joueur C{ronde}",
                "noir_nom": f"Joueur A{ronde}",
                "blanc_elo": 1750,
                "noir_elo": 1800,
                "echiquier": 1,
                "resultat_blanc": resultat,
                "resultat_noir": 1.0 - resultat,
                "type_resultat": type_r,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_club_games_with_forfait(sample_club_games: pd.DataFrame) -> pd.DataFrame:
    """Fixture avec un forfait dans les matchs a domicile."""
    forfait = {
        "saison": 2025,
        "ronde": 6,
        "equipe_dom": "Club A",
        "equipe_ext": "Club D",
        "blanc_nom": "Joueur A6",
        "noir_nom": "Joueur D6",
        "blanc_elo": 1800,
        "noir_elo": 1750,
        "echiquier": 1,
        "resultat_blanc": 2.0,
        "resultat_noir": 0.0,
        "type_resultat": "forfait_noir",
    }
    return pd.concat([sample_club_games, pd.DataFrame([forfait])], ignore_index=True)


class TestExtractClubBehavior:
    """Tests pour extract_club_behavior() — W/D/L home rates."""

    def test_output_columns(self, sample_club_games: pd.DataFrame) -> None:
        """Test que toutes les colonnes attendues sont presentes."""
        result = extract_club_behavior(sample_club_games)
        assert not result.empty
        expected_cols = {
            "equipe",
            "saison",
            "nb_joueurs_utilises",
            "rotation_effectif",
            "noyau_stable",
            "profondeur_effectif",
            "renforce_fin_saison",
            "win_rate_home",
            "draw_rate_home",
            "club_utilise_marge_100",
        }
        assert expected_cols.issubset(set(result.columns))
        # Old column must be gone
        assert "avantage_dom_club" not in result.columns

    def test_win_rate_home_correct(self, sample_club_games: pd.DataFrame) -> None:
        """Test win_rate_home = 2/3 pour Club A (2 victoires sur 3 matchs dom)."""
        result = extract_club_behavior(sample_club_games)
        row = result[result["equipe"] == "Club A"]
        assert len(row) == 1
        assert abs(row["win_rate_home"].values[0] - 2 / 3) < 1e-3

    def test_draw_rate_home_correct(self, sample_club_games: pd.DataFrame) -> None:
        """Test draw_rate_home = 1/3 pour Club A (1 nulle sur 3 matchs dom)."""
        result = extract_club_behavior(sample_club_games)
        row = result[result["equipe"] == "Club A"]
        assert abs(row["draw_rate_home"].values[0] - 1 / 3) < 1e-3

    def test_rates_sum_le_one(self, sample_club_games: pd.DataFrame) -> None:
        """Test win_rate_home + draw_rate_home <= 1."""
        result = extract_club_behavior(sample_club_games)
        for _, row in result.iterrows():
            assert row["win_rate_home"] + row["draw_rate_home"] <= 1.0 + 1e-9

    def test_forfait_excluded_from_home_rates(
        self, sample_club_games_with_forfait: pd.DataFrame
    ) -> None:
        """Test que le forfait n'est PAS compté dans win_rate_home/draw_rate_home."""
        result = extract_club_behavior(sample_club_games_with_forfait)
        row = result[result["equipe"] == "Club A"]
        # Still 3 real home games, forfait excluded — win_rate_home = 2/3
        assert abs(row["win_rate_home"].values[0] - 2 / 3) < 1e-3
        assert abs(row["draw_rate_home"].values[0] - 1 / 3) < 1e-3

    def test_empty_df(self) -> None:
        """Test avec DataFrame vide."""
        result = extract_club_behavior(pd.DataFrame())
        assert result.empty

    def test_no_equipe_dom_col(self) -> None:
        """Test sans colonne equipe_dom retourne DataFrame vide."""
        df = pd.DataFrame([{"saison": 2025, "equipe_ext": "Club B"}])
        result = extract_club_behavior(df)
        assert result.empty

    def test_nb_joueurs_utilises(self, sample_club_games: pd.DataFrame) -> None:
        """Test nb_joueurs_utilises est positif."""
        result = extract_club_behavior(sample_club_games)
        row = result[result["equipe"] == "Club A"]
        assert row["nb_joueurs_utilises"].values[0] > 0

    def test_zero_home_games_returns_zero_rates(self) -> None:
        """Test club sans matchs dom retourne rates = 0.0."""
        rows = [
            {
                "saison": 2025,
                "ronde": 1,
                "equipe_dom": "Club X",
                "equipe_ext": "Club A",
                "blanc_nom": "Joueur X1",
                "noir_nom": "Joueur A1",
                "blanc_elo": 1800,
                "noir_elo": 1750,
                "echiquier": 1,
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
            }
        ]
        df = pd.DataFrame(rows)
        result = extract_club_behavior(df)
        row_a = result[result["equipe"] == "Club A"]
        if len(row_a) == 1:
            assert row_a["win_rate_home"].values[0] == 0.0
            assert row_a["draw_rate_home"].values[0] == 0.0
