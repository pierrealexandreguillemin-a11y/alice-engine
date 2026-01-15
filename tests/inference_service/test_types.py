"""Tests PlayerProbability Type - ISO 29119.

Document ID: ALICE-TEST-INFERENCE-SERVICE-TYPES
Version: 1.0.0

Tests pour PlayerProbability dataclass.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from services.inference import PlayerProbability


class TestPlayerProbability:
    """Tests pour PlayerProbability dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test création dataclass."""
        player = PlayerProbability(
            ffe_id="A12345",
            name="Test Player",
            elo=2000,
            probability=0.85,
            board=1,
            reasoning="Test reasoning",
        )

        assert player.ffe_id == "A12345"
        assert player.name == "Test Player"
        assert player.elo == 2000
        assert player.probability == 0.85
        assert player.board == 1
        assert player.reasoning == "Test reasoning"

    def test_dataclass_equality(self) -> None:
        """Test égalité dataclass."""
        p1 = PlayerProbability("A", "B", 2000, 0.5, 1, "R")
        p2 = PlayerProbability("A", "B", 2000, 0.5, 1, "R")

        assert p1 == p2
