"""Tests ALI Edge Cases - ISO 29119.

Document ID: ALICE-TEST-FEATURES-ALI-EDGE-CASES
Version: 1.0.0

Tests edge cases pour features ALI.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd

from scripts.features.ali import calculate_presence_features, calculate_selection_patterns


class TestALIEdgeCases:
    """Tests edge cases pour features ALI."""

    def test_single_game_player(self) -> None:
        """Test joueur avec une seule partie."""
        df = pd.DataFrame(
            [{"saison": 2025, "ronde": 1, "blanc_nom": "Solo", "noir_nom": "X", "echiquier": 1}]
        )

        presence = calculate_presence_features(df)
        patterns = calculate_selection_patterns(df)

        assert len(presence) >= 1
        assert len(patterns) >= 1
        solo = patterns[patterns["joueur_nom"] == "Solo"]
        if not solo.empty:
            assert solo.iloc[0]["flexibilite_echiquier"] == 0.0

    def test_boundary_70_percent(self) -> None:
        """Test frontière exacte 70% (7 sur 10 rondes)."""
        df = pd.DataFrame(
            [
                {"saison": 2025, "ronde": i, "blanc_nom": "Border", "noir_nom": "X", "echiquier": 1}
                for i in range(1, 8)
            ]
            + [
                {"saison": 2025, "ronde": i, "blanc_nom": "Other", "noir_nom": "Y", "echiquier": 2}
                for i in range(1, 11)
            ]
        )

        result = calculate_presence_features(df)
        border = result[result["joueur_nom"] == "Border"]
        assert border.iloc[0]["regularite"] == "occasionnel"

    def test_boundary_30_percent(self) -> None:
        """Test frontière exacte 30% (3 sur 10 rondes)."""
        df = pd.DataFrame(
            [
                {"saison": 2025, "ronde": i, "blanc_nom": "Border", "noir_nom": "X", "echiquier": 1}
                for i in range(1, 4)
            ]
            + [
                {"saison": 2025, "ronde": i, "blanc_nom": "Other", "noir_nom": "Y", "echiquier": 2}
                for i in range(1, 11)
            ]
        )

        result = calculate_presence_features(df)
        border = result[result["joueur_nom"] == "Border"]
        assert border.iloc[0]["regularite"] == "occasionnel"

    def test_multiple_seasons(self) -> None:
        """Test avec plusieurs saisons."""
        df = pd.DataFrame(
            [
                {"saison": 2024, "ronde": 1, "blanc_nom": "A", "noir_nom": "X", "echiquier": 1},
                {"saison": 2025, "ronde": 1, "blanc_nom": "A", "noir_nom": "Y", "echiquier": 2},
            ]
        )

        result = calculate_presence_features(df)
        a_entries = result[result["joueur_nom"] == "A"]
        assert len(a_entries) == 2
        assert set(a_entries["saison"].unique()) == {2024, 2025}
