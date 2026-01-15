"""Tests Reliability - ISO 29119.

Document ID: ALICE-TEST-FE-RELIABILITY
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd

from scripts.features.reliability import (
    extract_club_reliability,
    extract_player_reliability,
)


class TestReliabilityFeatures:
    """Tests pour features de fiabilite."""

    def test_club_reliability(self) -> None:
        """Test extract_club_reliability."""
        df = pd.DataFrame(
            [
                {"equipe_dom": "Club A", "equipe_ext": "Club B", "type_resultat": "victoire_blanc"},
                {"equipe_dom": "Club A", "equipe_ext": "Club C", "type_resultat": "forfait_blanc"},
                {"equipe_dom": "Club B", "equipe_ext": "Club A", "type_resultat": "nulle"},
            ]
        )

        result = extract_club_reliability(df)

        assert not result.empty
        assert "fiabilite_score" in result.columns

    def test_player_reliability(self) -> None:
        """Test extract_player_reliability."""
        df = pd.DataFrame(
            [
                {
                    "blanc_nom": "Player 1",
                    "noir_nom": "Player 2",
                    "type_resultat": "victoire_blanc",
                },
                {"blanc_nom": "Player 1", "noir_nom": "Player 3", "type_resultat": "forfait_blanc"},
                {"blanc_nom": "Player 2", "noir_nom": "Player 1", "type_resultat": "nulle"},
            ]
        )

        result = extract_player_reliability(df)

        assert not result.empty
        assert "taux_presence" in result.columns
        assert "joueur_fantome" in result.columns

    def test_reliability_empty_df(self) -> None:
        """Test avec DataFrame vide."""
        assert extract_club_reliability(pd.DataFrame()).empty
        assert extract_player_reliability(pd.DataFrame()).empty
