"""Tests Predict Lineup and Fallback - ISO 29119.

Document ID: ALICE-TEST-INFERENCE-SERVICE-PREDICT
Version: 1.0.0

Tests pour predict_lineup et _fallback_prediction.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from services.inference import InferenceService, PlayerProbability


class TestPredictLineup:
    """Tests pour predict_lineup()."""

    def test_predict_lineup_fallback(
        self, service: InferenceService, sample_players: list[dict]
    ) -> None:
        """Test prédiction fallback (modèle non chargé)."""
        result = service.predict_lineup("CLUB123", 1, sample_players, team_size=8)

        assert len(result) == 8
        assert all(isinstance(p, PlayerProbability) for p in result)

    def test_predict_lineup_respects_team_size(
        self, service: InferenceService, sample_players: list[dict]
    ) -> None:
        """Test que team_size est respecté."""
        for size in [4, 6, 8, 10]:
            result = service.predict_lineup("CLUB", 1, sample_players, team_size=size)
            expected_size = min(size, len(sample_players))
            assert len(result) == expected_size

    def test_predict_lineup_sorted_by_elo(
        self, service: InferenceService, sample_players: list[dict]
    ) -> None:
        """Test que joueurs sont triés par Elo décroissant (fallback)."""
        result = service.predict_lineup("CLUB", 1, sample_players, team_size=8)

        assert result[0].elo >= result[1].elo
        assert result[0].name == "Carlsen Magnus"
        assert result[0].elo == 2850

    def test_predict_lineup_assigns_boards(
        self, service: InferenceService, sample_players: list[dict]
    ) -> None:
        """Test que les échiquiers sont assignés."""
        result = service.predict_lineup("CLUB", 1, sample_players, team_size=8)

        boards = [p.board for p in result]
        assert boards == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_predict_lineup_has_reasoning(
        self, service: InferenceService, sample_players: list[dict]
    ) -> None:
        """Test que reasoning est présent (fallback)."""
        result = service.predict_lineup("CLUB", 1, sample_players, team_size=4)

        for player in result:
            assert "Fallback" in player.reasoning
            assert player.probability == 0.5


class TestFallbackPrediction:
    """Tests pour _fallback_prediction()."""

    def test_fallback_empty_players(self, service: InferenceService) -> None:
        """Test fallback avec liste joueurs vide."""
        result = service._fallback_prediction([], 8)

        assert result == []

    def test_fallback_fewer_players_than_team_size(self, service: InferenceService) -> None:
        """Test fallback avec moins de joueurs que team_size."""
        players = [
            {"ffe_id": "A", "name": "P1", "elo": 2000},
            {"ffe_id": "B", "name": "P2", "elo": 1900},
        ]
        result = service._fallback_prediction(players, 8)

        assert len(result) == 2
        assert result[0].elo == 2000
        assert result[1].elo == 1900

    def test_fallback_missing_elo(self, service: InferenceService) -> None:
        """Test fallback avec joueurs sans Elo."""
        players = [
            {"ffe_id": "A", "name": "P1"},
            {"ffe_id": "B", "name": "P2", "elo": 1500},
        ]
        result = service._fallback_prediction(players, 8)

        assert len(result) == 2
        assert result[0].elo == 1500
        assert result[1].elo == 0

    def test_fallback_missing_fields(self, service: InferenceService) -> None:
        """Test fallback avec champs manquants."""
        players = [
            {},
            {"elo": 2000},
        ]
        result = service._fallback_prediction(players, 8)

        assert len(result) == 2
        assert result[0].ffe_id == ""
        assert result[0].name == ""
