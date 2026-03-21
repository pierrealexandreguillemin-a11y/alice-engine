"""Tests Color Performance V2 — W/D/L decomposition + rolling 3 saisons - ISO 29119.

Document ID: ALICE-TEST-FE-COLOR-V2
Version: 1.0.0
Tests count: 9

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML (rolling window, forfait exclusion)
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-03-22
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.features.color_perf import calculate_color_performance


def _make_game(
    blanc_nom: str,
    noir_nom: str,
    resultat_blanc: float,
    saison: int | None = None,
    type_resultat: str = "victoire_blanc",
) -> dict:
    """Helper to build a single game row."""
    row: dict = {
        "blanc_nom": blanc_nom,
        "noir_nom": noir_nom,
        "resultat_blanc": resultat_blanc,
        "resultat_noir": 1.0 - resultat_blanc if resultat_blanc != 0.5 else 0.5,
        "type_resultat": type_resultat,
    }
    if saison is not None:
        row["saison"] = saison
    return row


class TestWDLColorDecomposition:
    """Tests decomposition W/D/L par couleur."""

    def test_win_rate_white_computed(self) -> None:
        """Vérifie win_rate_white = nb_victoires_blanc / nb_blancs."""
        games = (
            [
                _make_game("Alice", f"Opp{i}", 1.0)
                for i in range(6)  # 6W
            ]
            + [
                _make_game("Alice", f"OppD{i}", 0.5, type_resultat="nulle")
                for i in range(2)  # 2D
            ]
            + [
                _make_game("Alice", f"OppL{i}", 0.0, type_resultat="victoire_noir")
                for i in range(2)  # 2L
            ]
            + [
                # Give Alice enough black games too (min_per_color=5)
                _make_game(f"OppB{i}", "Alice", 0.0, type_resultat="victoire_noir")
                for i in range(6)
            ]
            + [_make_game(f"OppBD{i}", "Alice", 0.5, type_resultat="nulle") for i in range(4)]
        )
        df = pd.DataFrame(games)

        result = calculate_color_performance(df, min_games=10, min_per_color=5)
        row = result[result["joueur_nom"] == "Alice"].iloc[0]

        assert row["win_rate_white"] == pytest.approx(0.6, abs=0.01)
        assert row["draw_rate_white"] == pytest.approx(0.2, abs=0.01)
        assert row["nb_blancs"] == 10

    def test_win_rate_black_computed(self) -> None:
        """Vérifie win_rate_black = nb_victoires_noir / nb_noirs."""
        # Bob: 7 wins as black, 3 losses as black; 10 games as white (5W/5D)
        games = (
            [_make_game(f"OppB{i}", "Bob", 0.0, type_resultat="victoire_noir") for i in range(7)]
            + [_make_game(f"OppBL{i}", "Bob", 1.0) for i in range(3)]
            + [_make_game("Bob", f"OppW{i}", 1.0) for i in range(5)]
            + [_make_game("Bob", f"OppWD{i}", 0.5, type_resultat="nulle") for i in range(5)]
        )
        df = pd.DataFrame(games)

        result = calculate_color_performance(df, min_games=10, min_per_color=5)
        row = result[result["joueur_nom"] == "Bob"].iloc[0]

        assert row["win_rate_black"] == pytest.approx(0.7, abs=0.01)
        assert row["draw_rate_black"] == pytest.approx(0.0, abs=0.01)
        assert row["nb_noirs"] == 10

    def test_win_adv_and_draw_adv_computed(self) -> None:
        """Vérifie win_adv_white = win_rate_white - win_rate_black."""
        # Carol: 8W/10 blanc, 3W/10 noir
        games = (
            [_make_game("Carol", f"Opp{i}", 1.0) for i in range(8)]
            + [
                _make_game("Carol", f"OppL{i}", 0.0, type_resultat="victoire_noir")
                for i in range(2)
            ]
            + [
                _make_game(f"OppB{i}", "Carol", 0.0, type_resultat="victoire_noir")
                for i in range(3)
            ]
            + [_make_game(f"OppBL{i}", "Carol", 1.0) for i in range(7)]
        )
        df = pd.DataFrame(games)

        result = calculate_color_performance(df, min_games=10, min_per_color=5)
        row = result[result["joueur_nom"] == "Carol"].iloc[0]

        assert row["win_adv_white"] == pytest.approx(0.8 - 0.3, abs=0.01)
        assert row["couleur_preferee"] == "blanc"

    def test_draw_rate_captured_separately_from_win_rate(self) -> None:
        """Vérifie que draw_rate est distinct du win_rate (ne pas confondre)."""
        # Dave: 5W + 5D en blanc, 5W + 5L en noir
        games = (
            [_make_game("Dave", f"Opp{i}", 1.0) for i in range(5)]
            + [_make_game("Dave", f"OppD{i}", 0.5, type_resultat="nulle") for i in range(5)]
            + [_make_game(f"OppB{i}", "Dave", 0.0, type_resultat="victoire_noir") for i in range(5)]
            + [_make_game(f"OppBL{i}", "Dave", 1.0) for i in range(5)]
        )
        df = pd.DataFrame(games)

        result = calculate_color_performance(df, min_games=10, min_per_color=5)
        row = result[result["joueur_nom"] == "Dave"].iloc[0]

        # White: 5W 5D 0L → win=0.5, draw=0.5
        assert row["win_rate_white"] == pytest.approx(0.5, abs=0.01)
        assert row["draw_rate_white"] == pytest.approx(0.5, abs=0.01)
        # Black: 5W 0D 5L → win=0.5, draw=0.0
        assert row["win_rate_black"] == pytest.approx(0.5, abs=0.01)
        assert row["draw_rate_black"] == pytest.approx(0.0, abs=0.01)
        # draw_adv_white = 0.5 - 0.0 = 0.5
        assert row["draw_adv_white"] == pytest.approx(0.5, abs=0.01)


class TestRolling3Seasons:
    """Tests fenêtre glissante 3 saisons."""

    def test_old_seasons_excluded(self) -> None:
        """Vérifie que les saisons hors fenêtre 3 ans sont exclues."""
        # Eve: 10 wins en blanc en 2020 (old), 0 wins en blanc en 2024-2026
        games_old = [_make_game("Eve", f"Opp{i}", 1.0, saison=2020) for i in range(10)]
        games_recent_white = [
            _make_game("Eve", f"OppR{i}", 0.0, saison=2025, type_resultat="victoire_noir")
            for i in range(5)
        ] + [
            _make_game("Eve", f"OppR2{i}", 0.0, saison=2026, type_resultat="victoire_noir")
            for i in range(5)
        ]
        games_black = [_make_game(f"OppB{i}", "Eve", 1.0, saison=2024) for i in range(6)] + [
            _make_game(f"OppB2{i}", "Eve", 1.0, saison=2025) for i in range(4)
        ]
        df = pd.DataFrame(games_old + games_recent_white + games_black)

        result = calculate_color_performance(df, min_games=10, min_per_color=5)
        row = result[result["joueur_nom"] == "Eve"].iloc[0]

        # Only recent seasons (2024-2026) counted → 0W/10 blanc
        assert row["win_rate_white"] == pytest.approx(0.0, abs=0.01)
        assert row["nb_blancs"] == 10  # 5+5 from 2025+2026 only

    def test_rolling_window_uses_max_saison(self) -> None:
        """Vérifie que la fenêtre est relative à max(saison), pas à l'année courante."""
        # Data from 2010-2014 only (ancient dataset). max=2014 → cutoff=2012
        games_2010 = [_make_game("Frank", f"Opp{i}", 1.0, saison=2010) for i in range(10)]
        games_2012_plus = [
            _make_game("Frank", f"OppR{i}", 0.0, saison=2013, type_resultat="victoire_noir")
            for i in range(10)
        ]
        games_black = [_make_game(f"OppB{i}", "Frank", 1.0, saison=2013) for i in range(10)]
        df = pd.DataFrame(games_2010 + games_2012_plus + games_black)

        result = calculate_color_performance(df, min_games=10, min_per_color=5)
        row = result[result["joueur_nom"] == "Frank"].iloc[0]

        # Cutoff=2012, only 2013 games counted. 0W/10 blanc
        assert row["win_rate_white"] == pytest.approx(0.0, abs=0.01)
        assert row["nb_blancs"] == 10

    def test_no_saison_column_uses_all_data(self) -> None:
        """Vérifie que l'absence de colonne saison utilise tout l'historique."""
        games = (
            [_make_game("Grace", f"Opp{i}", 1.0) for i in range(8)]
            + [
                _make_game("Grace", f"OppL{i}", 0.0, type_resultat="victoire_noir")
                for i in range(2)
            ]
            + [_make_game(f"OppB{i}", "Grace", 1.0) for i in range(10)]
        )
        df = pd.DataFrame(games)
        assert "saison" not in df.columns

        result = calculate_color_performance(df, min_games=10, min_per_color=5)
        row = result[result["joueur_nom"] == "Grace"].iloc[0]

        assert row["win_rate_white"] == pytest.approx(0.8, abs=0.01)
        assert row["nb_blancs"] == 10


class TestForfeitsExcluded:
    """Tests exclusion des forfaits."""

    def test_forfait_results_not_in_computation(self) -> None:
        """Vérifie que les parties type_resultat=forfait_blanc/noir sont exclues."""
        # Henri: 5 vraies victoires + 5 forfaits adverses (résultat=2.0)
        games_real = [_make_game("Henri", f"Opp{i}", 1.0) for i in range(5)] + [
            _make_game("Henri", f"OppL{i}", 0.0, type_resultat="victoire_noir") for i in range(5)
        ]
        games_forfait = [
            {
                "blanc_nom": "Henri",
                "noir_nom": f"OppF{i}",
                "resultat_blanc": 2.0,
                "resultat_noir": 0.0,
                "type_resultat": "forfait_noir",
            }
            for i in range(5)
        ]
        games_black = [_make_game(f"OppB{i}", "Henri", 1.0) for i in range(10)]
        df = pd.DataFrame(games_real + games_forfait + games_black)

        result = calculate_color_performance(df, min_games=10, min_per_color=5)
        row = result[result["joueur_nom"] == "Henri"].iloc[0]

        # Only 10 real games as white (5W + 5L) — forfaits excluded
        assert row["nb_blancs"] == 10
        assert row["win_rate_white"] == pytest.approx(0.5, abs=0.01)

    def test_non_joue_excluded(self) -> None:
        """Vérifie que type_resultat=non_joue est exclu."""
        games_real = [_make_game("Iris", f"Opp{i}", 1.0) for i in range(8)] + [
            _make_game("Iris", f"OppD{i}", 0.5, type_resultat="nulle") for i in range(2)
        ]
        games_non_joue = [
            {
                "blanc_nom": "Iris",
                "noir_nom": f"OppNJ{i}",
                "resultat_blanc": 0.0,
                "resultat_noir": 0.0,
                "type_resultat": "non_joue",
            }
            for i in range(10)
        ]
        games_black = [_make_game(f"OppB{i}", "Iris", 1.0) for i in range(10)]
        df = pd.DataFrame(games_real + games_non_joue + games_black)

        result = calculate_color_performance(df, min_games=10, min_per_color=5)
        row = result[result["joueur_nom"] == "Iris"].iloc[0]

        # Only 10 real games as white
        assert row["nb_blancs"] == 10
        assert row["win_rate_white"] == pytest.approx(0.8, abs=0.01)
