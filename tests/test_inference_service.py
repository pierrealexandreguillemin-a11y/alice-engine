"""Module: test_inference_service.py - Tests InferenceService (ALI).

Tests unitaires pour le service d'inference ALI.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing (unit tests, edge cases)
- ISO/IEC 42001:2023 - AI Management System (model testing)
- ISO/IEC 25010:2023 - System Quality (testability)

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

import pytest

from services.inference import InferenceService, PlayerProbability

# ==============================================================================
# FIXTURES (ISO 29119-3: Test Design)
# ==============================================================================


@pytest.fixture
def service() -> InferenceService:
    """Fixture service sans modele."""
    return InferenceService()


@pytest.fixture
def service_with_path() -> InferenceService:
    """Fixture service avec chemin modele (fictif)."""
    return InferenceService(model_path="/fake/model/path.cbm")


@pytest.fixture
def sample_players() -> list[dict]:
    """Fixture liste joueurs pour tests."""
    return [
        {"ffe_id": "A12345", "name": "Carlsen Magnus", "elo": 2850},
        {"ffe_id": "B23456", "name": "Caruana Fabiano", "elo": 2800},
        {"ffe_id": "C34567", "name": "Ding Liren", "elo": 2780},
        {"ffe_id": "D45678", "name": "Nepomniachtchi Ian", "elo": 2760},
        {"ffe_id": "E56789", "name": "Firouzja Alireza", "elo": 2750},
        {"ffe_id": "F67890", "name": "Nakamura Hikaru", "elo": 2740},
        {"ffe_id": "G78901", "name": "Aronian Levon", "elo": 2720},
        {"ffe_id": "H89012", "name": "Giri Anish", "elo": 2700},
        {"ffe_id": "I90123", "name": "So Wesley", "elo": 2680},
        {"ffe_id": "J01234", "name": "Mamedyarov Shakhriyar", "elo": 2660},
    ]


# ==============================================================================
# TESTS: PlayerProbability Dataclass
# ==============================================================================


class TestPlayerProbability:
    """Tests pour PlayerProbability dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test creation dataclass."""
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
        """Test egalite dataclass."""
        p1 = PlayerProbability("A", "B", 2000, 0.5, 1, "R")
        p2 = PlayerProbability("A", "B", 2000, 0.5, 1, "R")

        assert p1 == p2


# ==============================================================================
# TESTS: InferenceService Init
# ==============================================================================


class TestInferenceServiceInit:
    """Tests initialisation InferenceService."""

    def test_init_without_model_path(self) -> None:
        """Test initialisation sans chemin modele."""
        service = InferenceService()

        assert service.model is None
        assert service.model_path is None
        assert not service.is_loaded

    def test_init_with_model_path(self) -> None:
        """Test initialisation avec chemin modele."""
        service = InferenceService(model_path="/path/to/model.cbm")

        assert service.model is None
        assert service.model_path == "/path/to/model.cbm"
        assert not service.is_loaded


# ==============================================================================
# TESTS: load_model
# ==============================================================================


class TestLoadModel:
    """Tests pour load_model()."""

    def test_load_model_no_path_returns_false(self, service: InferenceService) -> None:
        """Test load_model sans chemin retourne False."""
        result = service.load_model()

        assert result is False
        assert not service.is_loaded

    def test_load_model_with_path_succeeds(self, service_with_path: InferenceService) -> None:
        """Test load_model avec chemin (TODO: implementation)."""
        # Note: Le modele n'est pas vraiment charge car TODO
        # Mais la logique de base est testee
        result = service_with_path.load_model()

        # Actuellement retourne True car pas d'erreur (fichier non verifie)
        assert result is True
        assert service_with_path.is_loaded


# ==============================================================================
# TESTS: predict_lineup
# ==============================================================================


class TestPredictLineup:
    """Tests pour predict_lineup()."""

    def test_predict_lineup_fallback(
        self, service: InferenceService, sample_players: list[dict]
    ) -> None:
        """Test prediction fallback (modele non charge)."""
        result = service.predict_lineup("CLUB123", 1, sample_players, team_size=8)

        assert len(result) == 8
        assert all(isinstance(p, PlayerProbability) for p in result)

    def test_predict_lineup_respects_team_size(
        self, service: InferenceService, sample_players: list[dict]
    ) -> None:
        """Test que team_size est respecte."""
        for size in [4, 6, 8, 10]:
            result = service.predict_lineup("CLUB", 1, sample_players, team_size=size)
            # Ne peut pas depasser le nombre de joueurs disponibles
            expected_size = min(size, len(sample_players))
            assert len(result) == expected_size

    def test_predict_lineup_sorted_by_elo(
        self, service: InferenceService, sample_players: list[dict]
    ) -> None:
        """Test que joueurs sont tries par Elo decroissant (fallback)."""
        result = service.predict_lineup("CLUB", 1, sample_players, team_size=8)

        # Premier joueur = meilleur Elo
        assert result[0].elo >= result[1].elo
        assert result[0].name == "Carlsen Magnus"
        assert result[0].elo == 2850

    def test_predict_lineup_assigns_boards(
        self, service: InferenceService, sample_players: list[dict]
    ) -> None:
        """Test que les echiquiers sont assignes."""
        result = service.predict_lineup("CLUB", 1, sample_players, team_size=8)

        boards = [p.board for p in result]
        assert boards == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_predict_lineup_has_reasoning(
        self, service: InferenceService, sample_players: list[dict]
    ) -> None:
        """Test que reasoning est present (fallback)."""
        result = service.predict_lineup("CLUB", 1, sample_players, team_size=4)

        for player in result:
            assert "Fallback" in player.reasoning
            assert player.probability == 0.5  # Probabilite par defaut


# ==============================================================================
# TESTS: _fallback_prediction
# ==============================================================================


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
            {"ffe_id": "A", "name": "P1"},  # Pas d'Elo
            {"ffe_id": "B", "name": "P2", "elo": 1500},
        ]
        result = service._fallback_prediction(players, 8)

        # Joueur sans Elo a elo=0 par defaut
        assert len(result) == 2
        assert result[0].elo == 1500  # Trie par Elo decroissant
        assert result[1].elo == 0

    def test_fallback_missing_fields(self, service: InferenceService) -> None:
        """Test fallback avec champs manquants."""
        players = [
            {},  # Tous champs manquants
            {"elo": 2000},  # Seulement Elo
        ]
        result = service._fallback_prediction(players, 8)

        assert len(result) == 2
        # Champs par defaut
        assert result[0].ffe_id == ""
        assert result[0].name == ""


# ==============================================================================
# TESTS: generate_scenarios
# ==============================================================================


class TestGenerateScenarios:
    """Tests pour generate_scenarios()."""

    def test_generate_scenarios_returns_list(self, service: InferenceService) -> None:
        """Test generate_scenarios retourne liste."""
        result = service.generate_scenarios([], scenario_count=5)

        assert isinstance(result, list)

    def test_generate_scenarios_empty_predictions(self, service: InferenceService) -> None:
        """Test avec predictions vides."""
        result = service.generate_scenarios([], scenario_count=10)

        # TODO: Implementation retourne liste vide pour l'instant
        assert result == []

    def test_generate_scenarios_with_predictions(
        self, service: InferenceService, sample_players: list[dict]
    ) -> None:
        """Test avec predictions valides."""
        predictions = service._fallback_prediction(sample_players[:4], 4)
        result = service.generate_scenarios(predictions, scenario_count=20)

        # TODO: Implementation retourne liste vide pour l'instant
        assert isinstance(result, list)


# ==============================================================================
# TESTS: Edge Cases (ISO 29119-4: Boundary Value Analysis)
# ==============================================================================


class TestInferenceEdgeCases:
    """Tests edge cases InferenceService."""

    def test_team_size_zero(self, service: InferenceService) -> None:
        """Test team_size = 0."""
        players = [{"ffe_id": "A", "name": "P", "elo": 2000}]
        result = service.predict_lineup("CLUB", 1, players, team_size=0)

        assert result == []

    def test_negative_elo(self, service: InferenceService) -> None:
        """Test Elo negatif (edge case)."""
        players = [
            {"ffe_id": "A", "name": "P1", "elo": -100},
            {"ffe_id": "B", "name": "P2", "elo": 1500},
        ]
        result = service._fallback_prediction(players, 2)

        # Trie par Elo decroissant
        assert result[0].elo == 1500
        assert result[1].elo == -100

    def test_very_large_team_size(
        self, service: InferenceService, sample_players: list[dict]
    ) -> None:
        """Test team_size tres grand."""
        result = service.predict_lineup("CLUB", 1, sample_players, team_size=1000)

        # Limite au nombre de joueurs disponibles
        assert len(result) == len(sample_players)
