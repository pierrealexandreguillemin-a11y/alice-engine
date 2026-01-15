"""Tests Zone Enjeu Logic - ISO 5259.

Document ID: ALICE-TEST-FEATURES-ENJEU-LOGIC
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd

from scripts.features.enjeu import extract_team_enjeu_features


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

        assert len(result) == 2

        s2024 = result[result["saison"] == 2024]
        s2025 = result[result["saison"] == 2025]

        assert s2024.iloc[0]["zone_enjeu"] in ["montee", "promotion"]
        assert s2025.iloc[0]["zone_enjeu"] in ["danger", "maintien"]


class TestEnjeuFfeIntegration:
    """Tests integration avec regles FFE."""

    def test_niveau_hierarchique_consistency(self) -> None:
        """Test coherence niveau hierarchique avec FFE."""
        from scripts.ffe_rules_features import get_niveau_equipe

        assert get_niveau_equipe("N1 Equipe Test") == 2
        assert get_niveau_equipe("N2 Equipe Test") == 3
        assert get_niveau_equipe("N3 Equipe Test") == 4
        assert get_niveau_equipe("N4 Equipe Test") == 5

    def test_zone_enjeu_uses_ffe_rules(self) -> None:
        """Test zone_enjeu utilise calculer_zone_enjeu FFE."""
        from scripts.ffe_rules_features import calculer_zone_enjeu

        zone = calculer_zone_enjeu(position=1, nb_equipes=10, division="Nationale 1")
        assert zone in ["montee", "promotion"]

        zone = calculer_zone_enjeu(position=10, nb_equipes=10, division="Nationale 1")
        assert zone in ["danger", "maintien"]
