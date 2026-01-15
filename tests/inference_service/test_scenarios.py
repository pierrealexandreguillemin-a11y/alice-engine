"""Tests Generate Scenarios and Edge Cases - ISO 29119.

Document ID: ALICE-TEST-INFERENCE-SERVICE-SCENARIOS
Version: 1.0.0

Tests pour generate_scenarios et edge cases.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from services.inference import InferenceService


class TestGenerateScenarios:
    """Tests pour generate_scenarios()."""

    def test_generate_scenarios_returns_list(self, service: InferenceService) -> None:
        """Test generate_scenarios retourne liste."""
        result = service.generate_scenarios([], scenario_count=5)

        assert isinstance(result, list)

    def test_generate_scenarios_empty_predictions(self, service: InferenceService) -> None:
        """Test avec predictions vides."""
        result = service.generate_scenarios([], scenario_count=10)

        assert result == []

    def test_generate_scenarios_with_predictions(
        self, service: InferenceService, sample_players: list[dict]
    ) -> None:
        """Test avec predictions valides."""
        predictions = service._fallback_prediction(sample_players[:4], 4)
        result = service.generate_scenarios(predictions, scenario_count=20)

        assert isinstance(result, list)


class TestInferenceEdgeCases:
    """Tests edge cases InferenceService."""

    def test_team_size_zero(self, service: InferenceService) -> None:
        """Test team_size = 0."""
        players = [{"ffe_id": "A", "name": "P", "elo": 2000}]
        result = service.predict_lineup("CLUB", 1, players, team_size=0)

        assert result == []

    def test_negative_elo(self, service: InferenceService) -> None:
        """Test Elo négatif (edge case)."""
        players = [
            {"ffe_id": "A", "name": "P1", "elo": -100},
            {"ffe_id": "B", "name": "P2", "elo": 1500},
        ]
        result = service._fallback_prediction(players, 2)

        assert result[0].elo == 1500
        assert result[1].elo == -100

    def test_very_large_team_size(
        self, service: InferenceService, sample_players: list[dict]
    ) -> None:
        """Test team_size très grand."""
        result = service.predict_lineup("CLUB", 1, sample_players, team_size=1000)

        assert len(result) == len(sample_players)
