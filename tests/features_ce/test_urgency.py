"""Tests Urgency Features - ISO 29119.

Document ID: ALICE-TEST-CE-URGENCY
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd

from scripts.features.ce import calculate_urgency_features


class TestCalculateUrgencyFeatures:
    """Tests pour calculate_urgency_features()."""

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        """Test avec DataFrame vide retourne DataFrame vide."""
        result = calculate_urgency_features(empty_df, ronde_actuelle=7)
        assert result.empty

    def test_returns_dataframe(self, sample_standings: pd.DataFrame) -> None:
        """Test retourne un DataFrame."""
        result = calculate_urgency_features(sample_standings, ronde_actuelle=7)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self, sample_standings: pd.DataFrame) -> None:
        """Test colonnes requises presentes."""
        result = calculate_urgency_features(sample_standings, ronde_actuelle=7)

        required_cols = [
            "equipe",
            "montee_possible",
            "maintien_assure",
            "rondes_restantes",
            "urgence_level",
            "points_max_possibles",
        ]
        for col in required_cols:
            assert col in result.columns, f"Colonne manquante: {col}"

    def test_rondes_restantes_calculation(self, sample_standings: pd.DataFrame) -> None:
        """Test calcul rondes restantes correct."""
        result = calculate_urgency_features(sample_standings, ronde_actuelle=7, nb_rondes_total=9)
        assert all(result["rondes_restantes"] == 2)

    def test_rondes_restantes_zero_at_end(self, sample_standings: pd.DataFrame) -> None:
        """Test rondes restantes = 0 a la derniere ronde."""
        result = calculate_urgency_features(sample_standings, ronde_actuelle=9, nb_rondes_total=9)
        assert all(result["rondes_restantes"] == 0)

    def test_points_max_possibles(self, sample_standings: pd.DataFrame) -> None:
        """Test calcul points max possibles (2 pts par victoire)."""
        result = calculate_urgency_features(sample_standings, ronde_actuelle=7, nb_rondes_total=9)

        leader = result[result["equipe"] == "Leader"]
        assert leader.iloc[0]["points_max_possibles"] == 18

    def test_montee_possible_leader(self, sample_standings: pd.DataFrame) -> None:
        """Test montee_possible pour le leader."""
        result = calculate_urgency_features(sample_standings, ronde_actuelle=7)

        leader = result[result["equipe"] == "Leader"]
        assert bool(leader.iloc[0]["montee_possible"]) is True

    def test_maintien_assure_leader(self, sample_standings: pd.DataFrame) -> None:
        """Test maintien_assure pour le leader."""
        result = calculate_urgency_features(sample_standings, ronde_actuelle=7)

        leader = result[result["equipe"] == "Leader"]
        assert bool(leader.iloc[0]["maintien_assure"]) is True

    def test_maintien_not_assure_relegable(self, sample_standings: pd.DataFrame) -> None:
        """Test maintien non assure pour equipe relegable."""
        result = calculate_urgency_features(sample_standings, ronde_actuelle=7)

        relegable = result[result["equipe"] == "Relegable"]
        assert bool(relegable.iloc[0]["maintien_assure"]) is False

    def test_urgence_critique_last_round(self) -> None:
        """Test urgence 'critique' derniere ronde + non maintenu."""
        df = pd.DataFrame(
            [
                {
                    "equipe": "Critique",
                    "position": 7,
                    "points_cumules": 4,
                    "nb_equipes": 8,
                    "ecart_premier": 10,
                    "ecart_dernier": 1,
                }
            ]
        )
        result = calculate_urgency_features(df, ronde_actuelle=9, nb_rondes_total=9)
        assert result.iloc[0]["urgence_level"] == "critique"

    def test_urgence_haute_3_rounds_left(self, sample_standings: pd.DataFrame) -> None:
        """Test urgence 'haute' avec 3 rondes restantes en danger."""
        result = calculate_urgency_features(sample_standings, ronde_actuelle=6, nb_rondes_total=9)

        danger = result[result["equipe"] == "Danger"]
        assert danger.iloc[0]["urgence_level"] == "haute"

    def test_urgence_aucune_maintained_mid_table(self) -> None:
        """Test urgence 'aucune' si maintien assure et mi-tableau."""
        df = pd.DataFrame(
            [
                {
                    "equipe": "Tranquille",
                    "position": 5,
                    "points_cumules": 12,
                    "nb_equipes": 10,
                    "ecart_premier": 6,
                    "ecart_dernier": 10,
                }
            ]
        )
        result = calculate_urgency_features(df, ronde_actuelle=7, nb_rondes_total=9)
        assert result.iloc[0]["urgence_level"] == "aucune"

    def test_urgence_normale_default(self) -> None:
        """Test urgence 'normale' par defaut."""
        df = pd.DataFrame(
            [
                {
                    "equipe": "Normal",
                    "position": 4,
                    "points_cumules": 8,
                    "nb_equipes": 8,
                    "ecart_premier": 6,
                    "ecart_dernier": 4,
                }
            ]
        )
        result = calculate_urgency_features(df, ronde_actuelle=5, nb_rondes_total=9)
        assert result.iloc[0]["urgence_level"] == "normale"
