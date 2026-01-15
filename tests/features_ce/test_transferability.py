"""Tests Transferability - ISO 29119.

Document ID: ALICE-TEST-CE-TRANSFER
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd
import pytest

from scripts.features.ce import (
    calculate_scenario_features,
    calculate_transferability,
    calculate_urgency_features,
)


class TestCalculateTransferability:
    """Tests pour calculate_transferability()."""

    @pytest.fixture
    def scenarios_urgency_data(
        self, sample_standings: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fixture avec scenarios et urgency calcules."""
        scenarios = calculate_scenario_features(sample_standings)
        urgency = calculate_urgency_features(sample_standings, ronde_actuelle=7, nb_rondes_total=9)
        return scenarios, urgency

    def test_empty_scenarios(self, empty_df: pd.DataFrame) -> None:
        """Test avec scenarios vide retourne DataFrame vide."""
        urgency = pd.DataFrame([{"equipe": "A", "saison": 2025, "ronde": 7}])
        result = calculate_transferability(empty_df, urgency, ronde_actuelle=7)
        assert result.empty

    def test_empty_urgency(self, sample_standings: pd.DataFrame) -> None:
        """Test avec urgency vide retourne DataFrame vide."""
        scenarios = calculate_scenario_features(sample_standings)
        result = calculate_transferability(scenarios, pd.DataFrame(), ronde_actuelle=7)
        assert result.empty

    def test_returns_dataframe(
        self, scenarios_urgency_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test retourne un DataFrame."""
        scenarios, urgency = scenarios_urgency_data
        result = calculate_transferability(scenarios, urgency, ronde_actuelle=7)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(
        self, scenarios_urgency_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test colonnes requises presentes."""
        scenarios, urgency = scenarios_urgency_data
        result = calculate_transferability(scenarios, urgency, ronde_actuelle=7)

        required_cols = [
            "equipe",
            "scenario",
            "can_donate",
            "can_receive",
            "priority",
            "transfer_score",
            "reason",
        ]
        for col in required_cols:
            assert col in result.columns, f"Colonne manquante: {col}"

    def test_condamne_can_donate(self) -> None:
        """Test equipe condamnee peut donner joueurs."""
        scenarios = pd.DataFrame(
            [
                {
                    "equipe": "Condamne",
                    "saison": 2025,
                    "ronde": 7,
                    "position": 8,
                    "scenario": "condamne",
                    "urgence_score": 0.3,
                }
            ]
        )
        urgency = pd.DataFrame(
            [
                {
                    "equipe": "Condamne",
                    "saison": 2025,
                    "ronde": 7,
                    "montee_possible": False,
                    "maintien_assure": False,
                    "points_cumules": 2,
                }
            ]
        )
        result = calculate_transferability(scenarios, urgency, ronde_actuelle=7)

        condamne = result[result["equipe"] == "Condamne"]
        assert bool(condamne.iloc[0]["can_donate"]) is True
        assert bool(condamne.iloc[0]["can_receive"]) is False

    def test_course_titre_can_receive(self) -> None:
        """Test equipe en course titre peut recevoir renforts."""
        scenarios = pd.DataFrame(
            [
                {
                    "equipe": "Leader",
                    "saison": 2025,
                    "ronde": 7,
                    "position": 1,
                    "scenario": "course_titre",
                    "urgence_score": 0.9,
                }
            ]
        )
        urgency = pd.DataFrame(
            [
                {
                    "equipe": "Leader",
                    "saison": 2025,
                    "ronde": 7,
                    "montee_possible": True,
                    "maintien_assure": True,
                    "points_cumules": 14,
                }
            ]
        )
        result = calculate_transferability(scenarios, urgency, ronde_actuelle=7)

        leader = result[result["equipe"] == "Leader"]
        assert bool(leader.iloc[0]["can_receive"]) is True
        assert bool(leader.iloc[0]["can_donate"]) is False

    def test_danger_can_receive(self) -> None:
        """Test equipe en danger peut recevoir renforts."""
        scenarios = pd.DataFrame(
            [
                {
                    "equipe": "Danger",
                    "saison": 2025,
                    "ronde": 7,
                    "position": 6,
                    "scenario": "danger",
                    "urgence_score": 0.85,
                }
            ]
        )
        urgency = pd.DataFrame(
            [
                {
                    "equipe": "Danger",
                    "saison": 2025,
                    "ronde": 7,
                    "montee_possible": False,
                    "maintien_assure": False,
                    "points_cumules": 5,
                }
            ]
        )
        result = calculate_transferability(scenarios, urgency, ronde_actuelle=7)

        danger = result[result["equipe"] == "Danger"]
        assert bool(danger.iloc[0]["can_receive"]) is True

    def test_mi_tableau_neutral(self) -> None:
        """Test mi-tableau ni donneur ni receveur."""
        scenarios = pd.DataFrame(
            [
                {
                    "equipe": "Milieu",
                    "saison": 2025,
                    "ronde": 7,
                    "position": 5,
                    "scenario": "mi_tableau",
                    "urgence_score": 0.4,
                }
            ]
        )
        urgency = pd.DataFrame(
            [
                {
                    "equipe": "Milieu",
                    "saison": 2025,
                    "ronde": 7,
                    "montee_possible": True,
                    "maintien_assure": False,
                    "points_cumules": 8,
                }
            ]
        )
        result = calculate_transferability(scenarios, urgency, ronde_actuelle=7)

        milieu = result[result["equipe"] == "Milieu"]
        assert bool(milieu.iloc[0]["can_donate"]) is False
        assert bool(milieu.iloc[0]["can_receive"]) is False

    def test_transfer_score_donor_negative(self) -> None:
        """Test transfer_score negatif pour donneur."""
        scenarios = pd.DataFrame(
            [
                {
                    "equipe": "Donneur",
                    "saison": 2025,
                    "ronde": 7,
                    "position": 8,
                    "scenario": "condamne",
                    "urgence_score": 0.3,
                }
            ]
        )
        urgency = pd.DataFrame(
            [
                {
                    "equipe": "Donneur",
                    "saison": 2025,
                    "ronde": 7,
                    "montee_possible": False,
                    "maintien_assure": False,
                    "points_cumules": 2,
                }
            ]
        )
        result = calculate_transferability(scenarios, urgency, ronde_actuelle=7)

        assert result.iloc[0]["transfer_score"] < 0

    def test_transfer_score_receiver_positive(self) -> None:
        """Test transfer_score positif pour receveur."""
        scenarios = pd.DataFrame(
            [
                {
                    "equipe": "Receveur",
                    "saison": 2025,
                    "ronde": 7,
                    "position": 1,
                    "scenario": "course_titre",
                    "urgence_score": 0.9,
                }
            ]
        )
        urgency = pd.DataFrame(
            [
                {
                    "equipe": "Receveur",
                    "saison": 2025,
                    "ronde": 7,
                    "montee_possible": True,
                    "maintien_assure": True,
                    "points_cumules": 14,
                }
            ]
        )
        result = calculate_transferability(scenarios, urgency, ronde_actuelle=7)

        assert result.iloc[0]["transfer_score"] > 0

    def test_priority_course_titre_highest(self) -> None:
        """Test priorite 1 (max) pour course titre urgente."""
        scenarios = pd.DataFrame(
            [
                {
                    "equipe": "Urgent",
                    "saison": 2025,
                    "ronde": 8,
                    "position": 1,
                    "scenario": "course_titre",
                    "urgence_score": 0.9,
                }
            ]
        )
        urgency = pd.DataFrame(
            [
                {
                    "equipe": "Urgent",
                    "saison": 2025,
                    "ronde": 8,
                    "montee_possible": True,
                    "maintien_assure": True,
                    "points_cumules": 14,
                }
            ]
        )
        result = calculate_transferability(scenarios, urgency, ronde_actuelle=8)

        assert result.iloc[0]["priority"] == 1
