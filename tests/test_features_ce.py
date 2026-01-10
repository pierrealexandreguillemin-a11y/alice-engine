"""Module: test_features_ce.py - Tests CE Features.

Tests unitaires pour les features CE (Composition Engine).

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing (unit tests, boundary values)
- ISO/IEC 5259:2024 - Data Quality for ML (feature validation)
- ISO/IEC 25010:2023 - System Quality (testability)

Author: ALICE Engine Team
Last Updated: 2026-01-10
"""

import pandas as pd
import pytest

from scripts.features.ce import (
    calculate_scenario_features,
    calculate_transferability,
    calculate_urgency_features,
    suggest_transfers,
)

# ==============================================================================
# FIXTURES (ISO 29119-3: Test Design)
# ==============================================================================


@pytest.fixture
def sample_standings() -> pd.DataFrame:
    """Fixture DataFrame classement pour tests."""
    return pd.DataFrame(
        [
            {
                "equipe": "Leader",
                "saison": 2025,
                "ronde": 7,
                "position": 1,
                "points_cumules": 14,
                "nb_equipes": 8,
                "ecart_premier": 0,
                "ecart_dernier": 10,
            },
            {
                "equipe": "Dauphin",
                "saison": 2025,
                "ronde": 7,
                "position": 2,
                "points_cumules": 12,
                "nb_equipes": 8,
                "ecart_premier": 2,
                "ecart_dernier": 8,
            },
            {
                "equipe": "Milieu",
                "saison": 2025,
                "ronde": 7,
                "position": 4,
                "points_cumules": 8,
                "nb_equipes": 8,
                "ecart_premier": 6,
                "ecart_dernier": 4,
            },
            {
                "equipe": "Danger",
                "saison": 2025,
                "ronde": 7,
                "position": 6,
                "points_cumules": 5,
                "nb_equipes": 8,
                "ecart_premier": 9,
                "ecart_dernier": 1,
            },
            {
                "equipe": "Relégable",
                "saison": 2025,
                "ronde": 7,
                "position": 7,
                "points_cumules": 4,
                "nb_equipes": 8,
                "ecart_premier": 10,
                "ecart_dernier": 0,
            },
            {
                "equipe": "Dernier",
                "saison": 2025,
                "ronde": 7,
                "position": 8,
                "points_cumules": 2,
                "nb_equipes": 8,
                "ecart_premier": 12,
                "ecart_dernier": 0,
            },
        ]
    )


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Fixture DataFrame vide."""
    return pd.DataFrame()


# ==============================================================================
# TESTS: calculate_scenario_features
# ==============================================================================


class TestCalculateScenarioFeatures:
    """Tests pour calculate_scenario_features()."""

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        """Test avec DataFrame vide retourne DataFrame vide."""
        result = calculate_scenario_features(empty_df)
        assert result.empty

    def test_missing_required_columns(self) -> None:
        """Test colonnes manquantes retourne DataFrame vide."""
        df = pd.DataFrame({"equipe": ["A"], "position": [1]})  # Manque points_cumules, nb_equipes
        result = calculate_scenario_features(df)
        assert result.empty

    def test_returns_dataframe(self, sample_standings: pd.DataFrame) -> None:
        """Test retourne un DataFrame."""
        result = calculate_scenario_features(sample_standings)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self, sample_standings: pd.DataFrame) -> None:
        """Test colonnes requises présentes."""
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
        """Test scenario 'course_titre' si écart <= 2 pts du premier."""
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
        """Test scenario 'danger' pour zone relégation."""
        result = calculate_scenario_features(sample_standings)

        relegable = result[result["equipe"] == "Relégable"]
        assert relegable.iloc[0]["scenario"] == "danger"

    def test_scenario_danger_ecart_2pts_dernier(self, sample_standings: pd.DataFrame) -> None:
        """Test scenario 'danger' si écart <= 2 pts du dernier."""
        result = calculate_scenario_features(sample_standings)

        danger = result[result["equipe"] == "Danger"]
        assert danger.iloc[0]["scenario"] == "danger"

    def test_scenario_condamne(self) -> None:
        """Test scenario 'condamne' (dernier avec gros écart)."""
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


# ==============================================================================
# TESTS: calculate_urgency_features
# ==============================================================================


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
        """Test colonnes requises présentes."""
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
        """Test rondes restantes = 0 à la dernière ronde."""
        result = calculate_urgency_features(sample_standings, ronde_actuelle=9, nb_rondes_total=9)
        assert all(result["rondes_restantes"] == 0)

    def test_points_max_possibles(self, sample_standings: pd.DataFrame) -> None:
        """Test calcul points max possibles (2 pts par victoire)."""
        result = calculate_urgency_features(sample_standings, ronde_actuelle=7, nb_rondes_total=9)

        leader = result[result["equipe"] == "Leader"]
        # 14 pts + (2 rondes * 2 pts) = 18 pts max
        assert leader.iloc[0]["points_max_possibles"] == 18

    def test_montee_possible_leader(self, sample_standings: pd.DataFrame) -> None:
        """Test montee_possible pour le leader."""
        result = calculate_urgency_features(sample_standings, ronde_actuelle=7)

        leader = result[result["equipe"] == "Leader"]
        assert bool(leader.iloc[0]["montee_possible"]) is True

    def test_maintien_assure_leader(self, sample_standings: pd.DataFrame) -> None:
        """Test maintien_assure pour le leader (loin du dernier)."""
        result = calculate_urgency_features(sample_standings, ronde_actuelle=7)

        leader = result[result["equipe"] == "Leader"]
        assert bool(leader.iloc[0]["maintien_assure"]) is True

    def test_maintien_not_assure_relegable(self, sample_standings: pd.DataFrame) -> None:
        """Test maintien non assuré pour équipe relégable."""
        result = calculate_urgency_features(sample_standings, ronde_actuelle=7)

        relegable = result[result["equipe"] == "Relégable"]
        assert bool(relegable.iloc[0]["maintien_assure"]) is False

    def test_urgence_critique_last_round(self) -> None:
        """Test urgence 'critique' dernière ronde + non maintenu."""
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

    def test_urgence_critique_close_to_relegation_last_round(self) -> None:
        """Test urgence 'critique' proche relégation avec 1 ronde restante."""
        # Une équipe en position 6 avec 1 ronde restante, écart_dernier=2
        # Points_max_restants = 2, donc elle pourrait être rattrapée
        df = pd.DataFrame(
            [
                {
                    "equipe": "ProcheRelegation",
                    "position": 6,
                    "points_cumules": 6,
                    "nb_equipes": 8,
                    "ecart_premier": 8,
                    "ecart_dernier": 2,
                },
                {
                    "equipe": "Releguable",
                    "position": 7,
                    "points_cumules": 4,
                    "nb_equipes": 8,
                    "ecart_premier": 10,
                    "ecart_dernier": 0,
                },
                {
                    "equipe": "Dernier",
                    "position": 8,
                    "points_cumules": 4,
                    "nb_equipes": 8,
                    "ecart_premier": 10,
                    "ecart_dernier": 0,
                },
            ]
        )
        # Avec 1 ronde restante (ronde 8/9), points_max_restants = 2
        # marge = 2, maintien_assure = (2 > 2) = False
        result = calculate_urgency_features(df, ronde_actuelle=8, nb_rondes_total=9)
        proche = result[result["equipe"] == "ProcheRelegation"]
        # Position 6, écart_dernier=2, 1 ronde restante, non maintenu → critique
        assert proche.iloc[0]["urgence_level"] == "critique"

    def test_urgence_haute_3_rounds_left(self, sample_standings: pd.DataFrame) -> None:
        """Test urgence 'haute' avec 3 rondes restantes en danger."""
        result = calculate_urgency_features(sample_standings, ronde_actuelle=6, nb_rondes_total=9)

        danger = result[result["equipe"] == "Danger"]
        assert danger.iloc[0]["urgence_level"] == "haute"

    def test_urgence_aucune_maintained_mid_table(self) -> None:
        """Test urgence 'aucune' si maintien assuré et mi-tableau."""
        df = pd.DataFrame(
            [
                {
                    "equipe": "Tranquille",
                    "position": 5,
                    "points_cumules": 12,
                    "nb_equipes": 10,
                    "ecart_premier": 6,
                    "ecart_dernier": 10,  # Très loin du dernier
                }
            ]
        )
        # Avec 2 rondes restantes, 4 pts max, le leader ne peut rattraper
        # et le dernier ne peut nous rejoindre
        result = calculate_urgency_features(df, ronde_actuelle=7, nb_rondes_total=9)
        assert result.iloc[0]["urgence_level"] == "aucune"

    def test_urgence_normale_default(self) -> None:
        """Test urgence 'normale' par défaut."""
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


# ==============================================================================
# TESTS: Edge Cases (ISO 29119-4: Boundary Value Analysis)
# ==============================================================================


class TestCEEdgeCases:
    """Tests edge cases pour features CE."""

    def test_single_team(self) -> None:
        """Test avec une seule équipe."""
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
        """Test avec ronde 0 (début de saison)."""
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
        # rondes_restantes doit être 0 (max(0, 9-10) = 0)
        assert result.iloc[0]["rondes_restantes"] == 0

    def test_all_teams_same_points(self) -> None:
        """Test toutes équipes à égalité de points."""
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
        """Test avec points négatifs (edge case invalide mais robuste)."""
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
        # Ne doit pas lever d'exception
        scenarios = calculate_scenario_features(df)
        urgency = calculate_urgency_features(df, ronde_actuelle=7)

        assert len(scenarios) == 1
        assert len(urgency) == 1

    def test_large_nb_equipes(self) -> None:
        """Test avec grand nombre d'équipes."""
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
            assert scenario in valid_scenarios, f"Scénario invalide: {scenario}"

    def test_urgency_values_are_valid(self, sample_standings: pd.DataFrame) -> None:
        """Test que les valeurs d'urgence sont valides."""
        result = calculate_urgency_features(sample_standings, ronde_actuelle=7)

        valid_levels = {"critique", "haute", "normale", "aucune"}
        for level in result["urgence_level"]:
            assert level in valid_levels, f"Niveau urgence invalide: {level}"


# ==============================================================================
# TESTS: calculate_transferability (Transférabilité inter-équipes)
# ==============================================================================


class TestCalculateTransferability:
    """Tests pour calculate_transferability()."""

    @pytest.fixture
    def scenarios_urgency_data(
        self, sample_standings: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fixture avec scenarios et urgency calculés."""
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
        """Test colonnes requises présentes."""
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
        """Test équipe condamnée peut donner joueurs."""
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
        """Test équipe en course titre peut recevoir renforts."""
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
        """Test équipe en danger peut recevoir renforts."""
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
        """Test transfer_score négatif pour donneur."""
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
        """Test priorité 1 (max) pour course titre urgente."""
        scenarios = pd.DataFrame(
            [
                {
                    "equipe": "Urgent",
                    "saison": 2025,
                    "ronde": 8,
                    "position": 1,
                    "scenario": "course_titre",
                    "urgence_score": 0.9,  # >= 0.8 pour priorité 1
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


# ==============================================================================
# TESTS: suggest_transfers
# ==============================================================================


class TestSuggestTransfers:
    """Tests pour suggest_transfers()."""

    def test_empty_transferability(self) -> None:
        """Test avec transferability vide retourne liste vide."""
        result = suggest_transfers(pd.DataFrame())
        assert result == []

    def test_no_donors(self) -> None:
        """Test sans donneurs retourne liste vide."""
        transferability = pd.DataFrame(
            [
                {
                    "equipe": "Receveur",
                    "saison": 2025,
                    "can_donate": False,
                    "can_receive": True,
                    "scenario": "course_titre",
                    "priority": 1,
                    "reason": "Test",
                }
            ]
        )
        result = suggest_transfers(transferability)
        assert result == []

    def test_no_receivers(self) -> None:
        """Test sans receveurs retourne liste vide."""
        transferability = pd.DataFrame(
            [
                {
                    "equipe": "Donneur",
                    "saison": 2025,
                    "can_donate": True,
                    "can_receive": False,
                    "scenario": "condamne",
                    "priority": 5,
                    "reason": "Test",
                }
            ]
        )
        result = suggest_transfers(transferability)
        assert result == []

    def test_donor_receiver_match(self) -> None:
        """Test suggestion quand donneur et receveur matchent."""
        transferability = pd.DataFrame(
            [
                {
                    "equipe": "Donneur",
                    "saison": 2025,
                    "can_donate": True,
                    "can_receive": False,
                    "scenario": "condamne",
                    "priority": 5,
                    "reason": "Peut donner",
                },
                {
                    "equipe": "Receveur",
                    "saison": 2025,
                    "can_donate": False,
                    "can_receive": True,
                    "scenario": "course_titre",
                    "priority": 1,
                    "reason": "Peut recevoir",
                },
            ]
        )
        result = suggest_transfers(transferability)

        assert len(result) == 1
        assert result[0]["from_team"] == "Donneur"
        assert result[0]["to_team"] == "Receveur"

    def test_different_saisons_no_match(self) -> None:
        """Test pas de match entre saisons différentes."""
        transferability = pd.DataFrame(
            [
                {
                    "equipe": "Donneur",
                    "saison": 2024,
                    "can_donate": True,
                    "can_receive": False,
                    "scenario": "condamne",
                    "priority": 5,
                    "reason": "Test",
                },
                {
                    "equipe": "Receveur",
                    "saison": 2025,
                    "can_donate": False,
                    "can_receive": True,
                    "scenario": "course_titre",
                    "priority": 1,
                    "reason": "Test",
                },
            ]
        )
        result = suggest_transfers(transferability)
        assert result == []

    def test_suggestions_sorted_by_priority(self) -> None:
        """Test suggestions triées par priorité receveur."""
        transferability = pd.DataFrame(
            [
                {
                    "equipe": "Donneur",
                    "saison": 2025,
                    "can_donate": True,
                    "can_receive": False,
                    "scenario": "condamne",
                    "priority": 5,
                    "reason": "Test",
                },
                {
                    "equipe": "Receveur_Prio2",
                    "saison": 2025,
                    "can_donate": False,
                    "can_receive": True,
                    "scenario": "course_titre",
                    "priority": 2,
                    "reason": "Test",
                },
                {
                    "equipe": "Receveur_Prio1",
                    "saison": 2025,
                    "can_donate": False,
                    "can_receive": True,
                    "scenario": "course_titre",
                    "priority": 1,
                    "reason": "Test",
                },
            ]
        )
        result = suggest_transfers(transferability)

        assert len(result) == 2
        assert result[0]["to_team"] == "Receveur_Prio1"
        assert result[1]["to_team"] == "Receveur_Prio2"
