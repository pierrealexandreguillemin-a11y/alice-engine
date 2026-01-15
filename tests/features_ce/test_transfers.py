"""Tests Suggest Transfers - ISO 29119.

Document ID: ALICE-TEST-CE-SUGGEST
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd

from scripts.features.ce import suggest_transfers


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
        """Test pas de match entre saisons differentes."""
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
        """Test suggestions triees par priorite receveur."""
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
