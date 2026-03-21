"""Tests Edge Cases Recent Form — W/D/L - ISO 29119.

Document ID: ALICE-TEST-FEATURES-RECENT-FORM-EDGE
Version: 2.0.0
Tests count: 5

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-03-22
"""

import pandas as pd

from scripts.features.recent_form import calculate_recent_form


class TestRecentFormEdgeCases:
    """Tests edge cases pour robustesse du calcul W/D/L."""

    def test_player_both_colors_aggregated(self) -> None:
        """Test joueur jouant blanc ET noir est agrege en une ligne."""
        df = pd.DataFrame(
            [
                {
                    "blanc_nom": "Joueur X",
                    "noir_nom": "A1",
                    "resultat_blanc": 1.0,
                    "resultat_noir": 0.0,
                    "type_resultat": "victoire_blanc",
                    "date": f"2025-01-0{i}",
                }
                for i in range(1, 6)
            ]
            + [
                {
                    "blanc_nom": f"A{i}",
                    "noir_nom": "Joueur X",
                    "resultat_blanc": 0.0,
                    "resultat_noir": 1.0,
                    "type_resultat": "victoire_noir",
                    "date": f"2025-01-1{i}",
                }
                for i in range(1, 6)
            ]
        )

        result = calculate_recent_form(df, window=5)

        joueur_x = result[result["joueur_nom"] == "Joueur X"]
        # Aggregated across both colors: win_rate_recent should reflect all wins
        assert len(joueur_x) == 1
        assert joueur_x.iloc[0]["win_rate_recent"] == 1.0

    def test_all_wins_gives_win_rate_one(self) -> None:
        """Test avec toutes victoires: win_rate=1.0, draw_rate=0.0."""
        df = pd.DataFrame(
            [
                {
                    "blanc_nom": "Joueur Y",
                    "noir_nom": f"Adversaire {i}",
                    "resultat_blanc": 1.0,
                    "resultat_noir": 0.0,
                    "type_resultat": "victoire_blanc",
                    "date": f"2025-01-{i:02d}",
                }
                for i in range(1, 8)
            ]
        )

        result = calculate_recent_form(df, window=5)

        joueur_y = result[result["joueur_nom"] == "Joueur Y"]
        assert joueur_y.iloc[0]["win_rate_recent"] == 1.0
        assert joueur_y.iloc[0]["draw_rate_recent"] == 0.0
        assert joueur_y.iloc[0]["win_trend"] == "stable"

    def test_all_draws_gives_expected_score_half(self) -> None:
        """Test toutes nulles: draw_rate=1.0, expected_score=0.5."""
        df = pd.DataFrame(
            [
                {
                    "blanc_nom": "Joueur Z",
                    "noir_nom": f"Adversaire {i}",
                    "resultat_blanc": 0.5,
                    "resultat_noir": 0.5,
                    "type_resultat": "nulle",
                    "date": f"2025-01-{i:02d}",
                }
                for i in range(1, 8)
            ]
        )

        result = calculate_recent_form(df, window=5)

        joueur_z = result[result["joueur_nom"] == "Joueur Z"]
        assert joueur_z.iloc[0]["draw_rate_recent"] == 1.0
        assert abs(joueur_z.iloc[0]["expected_score_recent"] - 0.5) < 1e-9
        assert joueur_z.iloc[0]["draw_trend"] == "stable"

    def test_without_date_column_still_works(self) -> None:
        """Test sans colonne date (pas de tri temporel)."""
        df = pd.DataFrame(
            [
                {
                    "blanc_nom": "Joueur W",
                    "noir_nom": f"Adversaire {i}",
                    "resultat_blanc": 1.0 if i % 2 == 0 else 0.0,
                    "resultat_noir": 0.0 if i % 2 == 0 else 1.0,
                    "type_resultat": "victoire_blanc" if i % 2 == 0 else "victoire_noir",
                }
                for i in range(1, 8)
            ]
        )

        result = calculate_recent_form(df, window=5)

        assert not result.empty
        assert "Joueur W" in result["joueur_nom"].values

    def test_missing_player_columns_returns_empty(self) -> None:
        """Test colonnes joueur manquantes retourne vide."""
        df = pd.DataFrame(
            {
                "autre_colonne": [1, 2, 3],
                "type_resultat": ["victoire_blanc", "nulle", "victoire_noir"],
                "resultat_blanc": [1.0, 0.5, 0.0],
            }
        )
        result = calculate_recent_form(df, window=5)

        assert result.empty
