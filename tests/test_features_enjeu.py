"""Module: test_features_enjeu.py - Tests Zones Enjeu Features.

Tests unitaires pour le module enjeu.py - zones d'enjeu equipe.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing (unit tests, boundary, edge cases)
- ISO/IEC 5259:2024 - Data Quality for ML (feature validation, lineage)
- ISO/IEC 25010:2023 - System Quality (testability)

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

import pandas as pd
import pytest

from scripts.features.enjeu import extract_team_enjeu_fallback, extract_team_enjeu_features

# ==============================================================================
# FIXTURES (ISO 29119-3: Test Design)
# ==============================================================================


@pytest.fixture
def sample_standings() -> pd.DataFrame:
    """Fixture avec classement pour tester zones enjeu."""
    return pd.DataFrame(
        [
            # Nationale 1 - 10 equipes
            {
                "equipe": "N1 Equipe A",
                "saison": 2025,
                "competition": "Interclubs",
                "division": "Nationale 1",
                "groupe": "Groupe A",
                "ronde": 5,
                "position": 1,
                "points_cumules": 8,
                "nb_equipes": 10,
                "ecart_premier": 0,
                "ecart_dernier": 6,
            },
            {
                "equipe": "N1 Equipe B",
                "saison": 2025,
                "competition": "Interclubs",
                "division": "Nationale 1",
                "groupe": "Groupe A",
                "ronde": 5,
                "position": 5,
                "points_cumules": 5,
                "nb_equipes": 10,
                "ecart_premier": 3,
                "ecart_dernier": 3,
            },
            {
                "equipe": "N1 Equipe C",
                "saison": 2025,
                "competition": "Interclubs",
                "division": "Nationale 1",
                "groupe": "Groupe A",
                "ronde": 5,
                "position": 9,
                "points_cumules": 2,
                "nb_equipes": 10,
                "ecart_premier": 6,
                "ecart_dernier": 0,
            },
            # Nationale 4 - 8 equipes
            {
                "equipe": "N4 Equipe D",
                "saison": 2025,
                "competition": "Interclubs",
                "division": "Nationale 4",
                "groupe": "Groupe B",
                "ronde": 5,
                "position": 1,
                "points_cumules": 10,
                "nb_equipes": 8,
                "ecart_premier": 0,
                "ecart_dernier": 8,
            },
            {
                "equipe": "N4 Equipe E",
                "saison": 2025,
                "competition": "Interclubs",
                "division": "Nationale 4",
                "groupe": "Groupe B",
                "ronde": 5,
                "position": 8,
                "points_cumules": 2,
                "nb_equipes": 8,
                "ecart_premier": 8,
                "ecart_dernier": 0,
            },
        ]
    )


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    """Fixture avec matchs pour test fallback."""
    return pd.DataFrame(
        [
            {
                "equipe_dom": "N1 Club A",
                "equipe_ext": "N1 Club B",
                "saison": 2025,
                "ronde": 1,
            },
            {
                "equipe_dom": "N1 Club A",
                "equipe_ext": "N1 Club C",
                "saison": 2025,
                "ronde": 2,
            },
            {
                "equipe_dom": "N4 Club D",
                "equipe_ext": "N4 Club E",
                "saison": 2025,
                "ronde": 1,
            },
        ]
    )


# ==============================================================================
# TESTS: extract_team_enjeu_features (ISO 29119-4: Test Techniques)
# ==============================================================================


class TestExtractTeamEnjeuFeatures:
    """Tests pour extract_team_enjeu_features."""

    def test_basic_extraction(
        self, sample_matches: pd.DataFrame, sample_standings: pd.DataFrame
    ) -> None:
        """Test extraction basique avec classement valide."""
        result = extract_team_enjeu_features(sample_matches, sample_standings)

        assert not result.empty
        assert "equipe" in result.columns
        assert "zone_enjeu" in result.columns
        assert "niveau_hierarchique" in result.columns

    def test_zone_enjeu_values(
        self, sample_matches: pd.DataFrame, sample_standings: pd.DataFrame
    ) -> None:
        """Test valeurs zone_enjeu valides."""
        result = extract_team_enjeu_features(sample_matches, sample_standings)

        valid_zones = {"promotion", "maintien", "mi_tableau", "montee", "danger"}
        assert all(z in valid_zones for z in result["zone_enjeu"].unique())

    def test_first_position_is_promotion(
        self, sample_matches: pd.DataFrame, sample_standings: pd.DataFrame
    ) -> None:
        """Test position 1 = zone promotion/montee."""
        result = extract_team_enjeu_features(sample_matches, sample_standings)

        equipe_a = result[result["equipe"] == "N1 Equipe A"]
        assert len(equipe_a) == 1
        # Position 1 sur 10 = zone montee
        assert equipe_a.iloc[0]["zone_enjeu"] in ["promotion", "montee"]

    def test_last_position_is_relegation(
        self, sample_matches: pd.DataFrame, sample_standings: pd.DataFrame
    ) -> None:
        """Test derniere position = zone maintien/danger."""
        result = extract_team_enjeu_features(sample_matches, sample_standings)

        equipe_c = result[result["equipe"] == "N1 Equipe C"]
        assert len(equipe_c) == 1
        # Position 9 sur 10 = zone relegation
        assert equipe_c.iloc[0]["zone_enjeu"] in ["maintien", "danger"]

    def test_middle_position_is_mi_tableau(
        self, sample_matches: pd.DataFrame, sample_standings: pd.DataFrame
    ) -> None:
        """Test position milieu = mi_tableau."""
        result = extract_team_enjeu_features(sample_matches, sample_standings)

        equipe_b = result[result["equipe"] == "N1 Equipe B"]
        assert len(equipe_b) == 1
        # Position 5 sur 10 = mi-tableau
        assert equipe_b.iloc[0]["zone_enjeu"] == "mi_tableau"

    def test_niveau_hierarchique_calculated(
        self, sample_matches: pd.DataFrame, sample_standings: pd.DataFrame
    ) -> None:
        """Test calcul niveau hierarchique."""
        result = extract_team_enjeu_features(sample_matches, sample_standings)

        # N1 = niveau 2, N4 = niveau 5
        n1_equipe = result[result["equipe"] == "N1 Equipe A"]
        n4_equipe = result[result["equipe"] == "N4 Equipe D"]

        # N1 plus haut niveau que N4
        assert n1_equipe.iloc[0]["niveau_hierarchique"] < n4_equipe.iloc[0]["niveau_hierarchique"]

    def test_empty_df_returns_empty(self, sample_standings: pd.DataFrame) -> None:
        """Test DataFrame vide retourne DataFrame vide."""
        result = extract_team_enjeu_features(pd.DataFrame(), sample_standings)
        assert result.empty

    def test_empty_standings_uses_fallback(self, sample_matches: pd.DataFrame) -> None:
        """Test classement vide utilise fallback."""
        result = extract_team_enjeu_features(sample_matches, pd.DataFrame())

        # Fallback retourne des donnees avec is_fallback
        assert not result.empty

    def test_missing_columns_returns_empty(self) -> None:
        """Test colonnes manquantes (ronde/saison) retourne vide."""
        df = pd.DataFrame({"autre_colonne": [1, 2, 3]})
        result = extract_team_enjeu_features(df, pd.DataFrame())
        assert result.empty

    def test_all_required_columns_present(
        self, sample_matches: pd.DataFrame, sample_standings: pd.DataFrame
    ) -> None:
        """Test toutes les colonnes requises presentes."""
        result = extract_team_enjeu_features(sample_matches, sample_standings)

        required_cols = [
            "equipe",
            "saison",
            "competition",
            "division",
            "groupe",
            "ronde",
            "position",
            "points_cumules",
            "nb_equipes",
            "ecart_premier",
            "ecart_dernier",
            "zone_enjeu",
            "niveau_hierarchique",
        ]

        for col in required_cols:
            assert col in result.columns, f"Colonne manquante: {col}"


# ==============================================================================
# TESTS: extract_team_enjeu_fallback (ISO 29119-4: Test Techniques)
# ==============================================================================


class TestExtractTeamEnjeuFallback:
    """Tests pour extract_team_enjeu_fallback."""

    def test_fallback_basic(self, sample_matches: pd.DataFrame) -> None:
        """Test fallback extraction basique."""
        result = extract_team_enjeu_fallback(sample_matches)

        assert not result.empty
        assert "equipe" in result.columns
        assert "zone_enjeu" in result.columns
        assert "is_fallback" in result.columns

    def test_fallback_marks_as_fallback(self, sample_matches: pd.DataFrame) -> None:
        """Test ISO 5259: fallback marque comme estimation."""
        result = extract_team_enjeu_fallback(sample_matches)

        # Toutes les lignes doivent avoir is_fallback=True
        assert result["is_fallback"].all()

    def test_fallback_estimates_middle_position(self, sample_matches: pd.DataFrame) -> None:
        """Test fallback estime position mi-tableau."""
        result = extract_team_enjeu_fallback(sample_matches)

        # Position estimee = nb_equipes // 2
        for _, row in result.iterrows():
            expected_mid = row["nb_equipes"] // 2
            assert row["position"] == expected_mid

    def test_fallback_deduplicated(self, sample_matches: pd.DataFrame) -> None:
        """Test fallback deduplique par equipe/saison."""
        result = extract_team_enjeu_fallback(sample_matches)

        # Pas de doublons equipe/saison
        duplicates = result.duplicated(subset=["equipe", "saison"])
        assert not duplicates.any()

    def test_fallback_empty_df(self) -> None:
        """Test fallback avec DataFrame vide."""
        result = extract_team_enjeu_fallback(pd.DataFrame())
        assert result.empty

    def test_fallback_missing_equipe_columns(self) -> None:
        """Test fallback sans colonnes equipe."""
        df = pd.DataFrame({"autre": [1], "saison": [2025], "ronde": [1]})
        result = extract_team_enjeu_fallback(df)
        assert result.empty

    def test_fallback_counts_rondes(self, sample_matches: pd.DataFrame) -> None:
        """Test fallback compte les rondes par equipe."""
        result = extract_team_enjeu_fallback(sample_matches)

        # N1 Club A a 2 rondes (1 et 2)
        club_a = result[result["equipe"] == "N1 Club A"]
        if len(club_a) > 0:
            assert club_a.iloc[0]["nb_rondes"] == 2


# ==============================================================================
# TESTS: Zone Enjeu Logic (ISO 29119-4: Boundary Value Analysis)
# ==============================================================================


class TestZoneEnjeuLogic:
    """Tests logique zones enjeu."""

    def test_n1_promotion_zone(self) -> None:
        """Test N1: positions 1-2 = montee."""
        standings = pd.DataFrame(
            [
                {
                    "equipe": "Test",
                    "saison": 2025,
                    "competition": "Interclubs",
                    "division": "Nationale 1",
                    "groupe": "A",
                    "ronde": 5,
                    "position": 1,
                    "points_cumules": 10,
                    "nb_equipes": 10,
                    "ecart_premier": 0,
                    "ecart_dernier": 8,
                }
            ]
        )
        df = pd.DataFrame({"ronde": [1], "saison": [2025]})

        result = extract_team_enjeu_features(df, standings)

        assert result.iloc[0]["zone_enjeu"] in ["montee", "promotion"]

    def test_n1_relegation_zone(self) -> None:
        """Test N1: positions 9-10 = danger."""
        standings = pd.DataFrame(
            [
                {
                    "equipe": "Test",
                    "saison": 2025,
                    "competition": "Interclubs",
                    "division": "Nationale 1",
                    "groupe": "A",
                    "ronde": 5,
                    "position": 10,
                    "points_cumules": 2,
                    "nb_equipes": 10,
                    "ecart_premier": 8,
                    "ecart_dernier": 0,
                }
            ]
        )
        df = pd.DataFrame({"ronde": [1], "saison": [2025]})

        result = extract_team_enjeu_features(df, standings)

        assert result.iloc[0]["zone_enjeu"] in ["danger", "maintien"]

    def test_multiple_seasons(self) -> None:
        """Test avec plusieurs saisons."""
        standings = pd.DataFrame(
            [
                {
                    "equipe": "Test",
                    "saison": 2024,
                    "competition": "Interclubs",
                    "division": "Nationale 2",
                    "groupe": "A",
                    "ronde": 5,
                    "position": 1,
                    "points_cumules": 10,
                    "nb_equipes": 10,
                    "ecart_premier": 0,
                    "ecart_dernier": 8,
                },
                {
                    "equipe": "Test",
                    "saison": 2025,
                    "competition": "Interclubs",
                    "division": "Nationale 2",
                    "groupe": "A",
                    "ronde": 5,
                    "position": 8,
                    "points_cumules": 3,
                    "nb_equipes": 10,
                    "ecart_premier": 7,
                    "ecart_dernier": 1,
                },
            ]
        )
        df = pd.DataFrame({"ronde": [1, 1], "saison": [2024, 2025]})

        result = extract_team_enjeu_features(df, standings)

        # Devrait avoir 2 lignes pour les 2 saisons
        assert len(result) == 2

        # 2024 position 1 = montee, 2025 position 8 = danger
        s2024 = result[result["saison"] == 2024]
        s2025 = result[result["saison"] == 2025]

        assert s2024.iloc[0]["zone_enjeu"] in ["montee", "promotion"]
        assert s2025.iloc[0]["zone_enjeu"] in ["danger", "maintien"]


# ==============================================================================
# TESTS: Integration with FFE Rules (ISO 29119-4: Integration Testing)
# ==============================================================================


class TestEnjeuFfeIntegration:
    """Tests integration avec regles FFE."""

    def test_niveau_hierarchique_consistency(self) -> None:
        """Test coherence niveau hierarchique avec FFE."""
        from scripts.ffe_rules_features import get_niveau_equipe

        # Verifier que les niveaux sont coherents
        assert get_niveau_equipe("N1 Equipe Test") == 2  # Nationale 1
        assert get_niveau_equipe("N2 Equipe Test") == 3  # Nationale 2
        assert get_niveau_equipe("N3 Equipe Test") == 4  # Nationale 3
        assert get_niveau_equipe("N4 Equipe Test") == 5  # Nationale 4

    def test_zone_enjeu_uses_ffe_rules(self) -> None:
        """Test zone_enjeu utilise calculer_zone_enjeu FFE."""
        from scripts.ffe_rules_features import calculer_zone_enjeu

        # Verifier coherence avec module FFE
        zone = calculer_zone_enjeu(position=1, nb_equipes=10, division="Nationale 1")
        assert zone in ["montee", "promotion"]

        zone = calculer_zone_enjeu(position=10, nb_equipes=10, division="Nationale 1")
        assert zone in ["danger", "maintien"]
