"""Tests Scenario Features - ISO 29119.

Document ID: ALICE-TEST-CE-SCENARIO
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd

from scripts.features.ce import calculate_scenario_features


class TestCalculateScenarioFeatures:
    """Tests pour calculate_scenario_features()."""

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        """Test avec DataFrame vide retourne DataFrame vide."""
        result = calculate_scenario_features(empty_df)
        assert result.empty

    def test_missing_required_columns(self) -> None:
        """Test colonnes manquantes retourne DataFrame vide."""
        df = pd.DataFrame({"equipe": ["A"], "position": [1]})
        result = calculate_scenario_features(df)
        assert result.empty

    def test_returns_dataframe(self, sample_standings: pd.DataFrame) -> None:
        """Test retourne un DataFrame."""
        result = calculate_scenario_features(sample_standings)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self, sample_standings: pd.DataFrame) -> None:
        """Test colonnes requises presentes."""
        result = calculate_scenario_features(sample_standings)

        required_cols = ["equipe", "saison", "ronde", "position", "scenario", "urgence_score"]
        for col in required_cols:
            assert col in result.columns, f"Colonne manquante: {col}"

    def test_scenario_course_titre_position_1(self, sample_standings: pd.DataFrame) -> None:
        """Test scenario 'course_titre' pour position 1."""
        result = calculate_scenario_features(sample_standings)

        leader = result[result["equipe"] == "Leader"]
        assert leader.iloc[0]["scenario"] == "course_titre"

    def test_scenario_course_titre_position_2(self, sample_standings: pd.DataFrame) -> None:
        """Test scenario 'course_titre' pour position 2."""
        result = calculate_scenario_features(sample_standings)

        dauphin = result[result["equipe"] == "Dauphin"]
        assert dauphin.iloc[0]["scenario"] == "course_titre"

    def test_scenario_course_titre_ecart_2pts(self) -> None:
        """Test scenario 'course_titre' si ecart <= 2 pts du premier."""
        df = pd.DataFrame(
            [
                {
                    "equipe": "Proche",
                    "position": 3,
                    "points_cumules": 12,
                    "nb_equipes": 8,
                    "ecart_premier": 2,
                    "ecart_dernier": 6,
                }
            ]
        )
        result = calculate_scenario_features(df)
        assert result.iloc[0]["scenario"] == "course_titre"

    def test_scenario_course_montee(self) -> None:
        """Test scenario 'course_montee' pour positions 3-4."""
        df = pd.DataFrame(
            [
                {
                    "equipe": "Barrage",
                    "position": 3,
                    "points_cumules": 10,
                    "nb_equipes": 8,
                    "ecart_premier": 4,
                    "ecart_dernier": 6,
                }
            ]
        )
        result = calculate_scenario_features(df)
        assert result.iloc[0]["scenario"] == "course_montee"

    def test_scenario_danger_zone_relegation(self, sample_standings: pd.DataFrame) -> None:
        """Test scenario 'danger' pour zone relegation."""
        result = calculate_scenario_features(sample_standings)

        relegable = result[result["equipe"] == "Relegable"]
        assert relegable.iloc[0]["scenario"] == "danger"

    def test_scenario_danger_ecart_2pts_dernier(self, sample_standings: pd.DataFrame) -> None:
        """Test scenario 'danger' si ecart <= 2 pts du dernier."""
        result = calculate_scenario_features(sample_standings)

        danger = result[result["equipe"] == "Danger"]
        assert danger.iloc[0]["scenario"] == "danger"

    def test_scenario_condamne(self) -> None:
        """Test scenario 'condamne' (dernier avec gros ecart)."""
        df = pd.DataFrame(
            [
                {
                    "equipe": "Condamne",
                    "position": 8,
                    "points_cumules": 2,
                    "nb_equipes": 8,
                    "ecart_premier": 12,
                    "ecart_dernier": 0,
                }
            ]
        )
        result = calculate_scenario_features(df)
        assert result.iloc[0]["scenario"] == "condamne"

    def test_scenario_mi_tableau(self) -> None:
        """Test scenario 'mi_tableau' (ni course ni danger)."""
        df = pd.DataFrame(
            [
                {
                    "equipe": "Confort",
                    "position": 5,
                    "points_cumules": 8,
                    "nb_equipes": 10,
                    "ecart_premier": 6,
                    "ecart_dernier": 6,
                }
            ]
        )
        result = calculate_scenario_features(df)
        assert result.iloc[0]["scenario"] == "mi_tableau"

    def test_urgence_score_range(self, sample_standings: pd.DataFrame) -> None:
        """Test urgence_score dans [0, 1]."""
        result = calculate_scenario_features(sample_standings)

        for score in result["urgence_score"]:
            assert 0 <= score <= 1
