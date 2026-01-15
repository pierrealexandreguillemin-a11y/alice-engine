"""Tests Edge Cases CE - ISO 29119.

Document ID: ALICE-TEST-CE-EDGE
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd

from scripts.features.ce import (
    calculate_scenario_features,
    calculate_urgency_features,
)


class TestCEEdgeCases:
    """Tests edge cases pour features CE."""

    def test_single_team(self) -> None:
        """Test avec une seule equipe."""
        df = pd.DataFrame(
            [
                {
                    "equipe": "Seul",
                    "position": 1,
                    "points_cumules": 10,
                    "nb_equipes": 1,
                    "ecart_premier": 0,
                    "ecart_dernier": 0,
                }
            ]
        )
        scenarios = calculate_scenario_features(df)
        urgency = calculate_urgency_features(df, ronde_actuelle=5)

        assert len(scenarios) == 1
        assert len(urgency) == 1

    def test_ronde_zero(self) -> None:
        """Test avec ronde 0 (debut de saison)."""
        df = pd.DataFrame(
            [
                {
                    "equipe": "Debut",
                    "position": 1,
                    "points_cumules": 0,
                    "nb_equipes": 8,
                    "ecart_premier": 0,
                    "ecart_dernier": 0,
                }
            ]
        )
        result = calculate_urgency_features(df, ronde_actuelle=0, nb_rondes_total=9)
        assert result.iloc[0]["rondes_restantes"] == 9

    def test_ronde_beyond_total(self) -> None:
        """Test avec ronde > total (edge case)."""
        df = pd.DataFrame(
            [
                {
                    "equipe": "Fini",
                    "position": 1,
                    "points_cumules": 18,
                    "nb_equipes": 8,
                    "ecart_premier": 0,
                    "ecart_dernier": 14,
                }
            ]
        )
        result = calculate_urgency_features(df, ronde_actuelle=10, nb_rondes_total=9)
        assert result.iloc[0]["rondes_restantes"] == 0

    def test_all_teams_same_points(self) -> None:
        """Test toutes equipes a egalite de points."""
        df = pd.DataFrame(
            [
                {
                    "equipe": f"Equipe{i}",
                    "position": i,
                    "points_cumules": 10,
                    "nb_equipes": 8,
                    "ecart_premier": 0,
                    "ecart_dernier": 0,
                }
                for i in range(1, 9)
            ]
        )
        scenarios = calculate_scenario_features(df)
        urgency = calculate_urgency_features(df, ronde_actuelle=7)

        assert len(scenarios) == 8
        assert len(urgency) == 8

    def test_negative_points(self) -> None:
        """Test avec points negatifs (edge case invalide mais robuste)."""
        df = pd.DataFrame(
            [
                {
                    "equipe": "Negatif",
                    "position": 8,
                    "points_cumules": -2,
                    "nb_equipes": 8,
                    "ecart_premier": 20,
                    "ecart_dernier": 0,
                }
            ]
        )
        scenarios = calculate_scenario_features(df)
        urgency = calculate_urgency_features(df, ronde_actuelle=7)

        assert len(scenarios) == 1
        assert len(urgency) == 1

    def test_large_nb_equipes(self) -> None:
        """Test avec grand nombre d'equipes."""
        df = pd.DataFrame(
            [
                {
                    "equipe": f"Equipe{i}",
                    "position": i,
                    "points_cumules": 100 - i,
                    "nb_equipes": 100,
                    "ecart_premier": i - 1,
                    "ecart_dernier": 100 - i,
                }
                for i in range(1, 101)
            ]
        )
        scenarios = calculate_scenario_features(df)
        urgency = calculate_urgency_features(df, ronde_actuelle=15, nb_rondes_total=20)

        assert len(scenarios) == 100
        assert len(urgency) == 100

    def test_scenario_values_are_valid(self, sample_standings: pd.DataFrame) -> None:
        """Test que les valeurs de scenario sont valides."""
        result = calculate_scenario_features(sample_standings)

        valid_scenarios = {"course_titre", "course_montee", "danger", "condamne", "mi_tableau"}
        for scenario in result["scenario"]:
            assert scenario in valid_scenarios, f"Scenario invalide: {scenario}"

    def test_urgency_values_are_valid(self, sample_standings: pd.DataFrame) -> None:
        """Test que les valeurs d'urgence sont valides."""
        result = calculate_urgency_features(sample_standings, ronde_actuelle=7)

        valid_levels = {"critique", "haute", "normale", "aucune"}
        for level in result["urgence_level"]:
            assert level in valid_levels, f"Niveau urgence invalide: {level}"
