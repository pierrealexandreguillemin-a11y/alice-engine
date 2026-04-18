"""E2E tests for POST /compose (ISO 29119).

Document ID: ALICE-TEST-COMPOSE-E2E
Version: 1.0.0
ISO Compliance:
- ISO/IEC 29119 - Software Testing (fixtures, classes, structured IDs)
- ISO/IEC 25010 - System Quality (integration testing)
- ISO/IEC 42001 - AI Management (model stub testing)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scripts.serving.model_loader import ModelBundle
from services.inference import PredictionResult


def _make_mock_bundle() -> ModelBundle:
    """Build a ModelBundle with mocked sub-models.

    All three GBM stubs return [[0.30, 0.20, 0.50]] so that meta-feature
    assembly and temperature scaling produce a valid (1, 3) probability row.
    The MLP returns [[0.29, 0.21, 0.50]] which sums to 1.0.
    fallback_mode=False so the stacking path is exercised.
    """
    mock_lgb = MagicMock()
    mock_xgb = MagicMock()
    mock_cb = MagicMock()
    mock_mlp = MagicMock()

    gbm_proba = np.array([[0.30, 0.20, 0.50]])
    mlp_proba = np.array([[0.29, 0.21, 0.50]])

    for m in (mock_lgb, mock_xgb, mock_cb):
        m.predict_proba = MagicMock(return_value=gbm_proba)
    mock_mlp.predict_proba = MagicMock(return_value=mlp_proba)

    return ModelBundle(
        lgb_model=mock_lgb,
        xgb_model=mock_xgb,
        cb_model=mock_cb,
        mlp_model=mock_mlp,
        temperature=1.02,
        draw_rate_lookup=None,
        encoders=None,
        fallback_mode=False,
        version="test-e2e",
    )


def _mock_predict_board(
    player_elo: int,
    opponent_elo: int,
    features: np.ndarray,
    draw_rate_lookup=None,
) -> PredictionResult:
    """Deterministic stub for StackingInferenceService.predict_board.

    Returns fixed probabilities so tests never touch real model files
    or draw_rate_lookup parquets.
    """
    return PredictionResult(p_loss=0.30, p_draw=0.20, p_win=0.50, e_score=0.60)


@pytest.fixture
def client():
    """Create TestClient with fully mocked model bundle and inference.

    Two patches are applied:
    1. app.main.load_models — prevents real HF Hub download during lifespan.
    2. StackingInferenceService.predict_board — avoids draw_rate_lookup lookup
       which requires a real parquet file.
    """
    mock_bundle = _make_mock_bundle()

    with (
        patch("scripts.serving.model_loader.load_models", return_value=mock_bundle),
        patch(
            "services.inference.StackingInferenceService.predict_board",
            side_effect=_mock_predict_board,
        ),
    ):
        # Import app *after* patching so lifespan picks up the mock bundle
        import importlib

        import app.main as _main_module

        importlib.reload(_main_module)

        from app.main import app as _app

        with TestClient(_app) as tc:
            # Ensure state is set even if lifespan mock missed
            _app.state.model_bundle = mock_bundle
            _app.state.feature_store = None
            yield tc


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestComposeE2E:
    """E2E tests for the /compose endpoint.

    Covers: smoke, board fields, metadata shape, health endpoint.
    ISO 29119: each test has a single assertion goal.
    """

    def test_smoke_compose(self, client: TestClient) -> None:
        """Valid request returns 200 with compositions list.

        @id ALICE-TEST-CE-001
        @version 1.0.0
        """
        response = client.post(
            "/api/v1/compose",
            json={
                "club_id": "TEST01",
                "joueurs_disponibles": [
                    "A00001",
                    "A00002",
                    "A00003",
                    "A00004",
                    "A00005",
                    "A00006",
                    "A00007",
                    "A00008",
                ],
                "ronde": 3,
                "division": "N3",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "compositions" in data
        assert "metadata" in data
        assert len(data["compositions"]) >= 1

    def test_compose_has_boards(self, client: TestClient) -> None:
        """Compositions include board assignments with probabilities.

        @id ALICE-TEST-CE-002
        @version 1.0.0
        """
        response = client.post(
            "/api/v1/compose",
            json={
                "club_id": "TEST01",
                "joueurs_disponibles": ["A00001", "A00002", "A00003", "A00004"],
                "ronde": 1,
                "division": "N3",
            },
        )
        assert response.status_code == 200
        data = response.json()
        boards = data["compositions"][0]["boards"]
        assert len(boards) == 4
        for b in boards:
            assert "p_win" in b
            assert "p_draw" in b
            assert "p_loss" in b
            assert "e_score" in b

    def test_compose_probabilities_valid(self, client: TestClient) -> None:
        """All board probabilities are in [0, 1] and sum approximately to 1.

        @id ALICE-TEST-CE-003
        @version 1.0.0
        """
        response = client.post(
            "/api/v1/compose",
            json={
                "club_id": "TEST01",
                "joueurs_disponibles": ["A00001", "A00002"],
                "ronde": 1,
                "division": "N3",
            },
        )
        assert response.status_code == 200
        boards = response.json()["compositions"][0]["boards"]
        for b in boards:
            total = b["p_win"] + b["p_draw"] + b["p_loss"]
            assert 0.98 <= total <= 1.02, f"Probabilities don't sum to 1: {total}"
            assert 0.0 <= b["e_score"] <= 1.0

    def test_compose_metadata(self, client: TestClient) -> None:
        """Metadata includes model version and ALI mode.

        @id ALICE-TEST-CE-004
        @version 1.0.0
        """
        response = client.post(
            "/api/v1/compose",
            json={
                "club_id": "TEST01",
                "joueurs_disponibles": ["A00001"],
                "ronde": 1,
                "division": "N3",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["ali_mode"] == "elo_fallback"
        assert "model_version" in data["metadata"]

    def test_compose_board_count_capped_at_8(self, client: TestClient) -> None:
        """Board count is capped at 8 even with more than 8 players.

        @id ALICE-TEST-CE-005
        @version 1.0.0
        """
        players = [f"A{i:05d}" for i in range(12)]
        response = client.post(
            "/api/v1/compose",
            json={
                "club_id": "TEST01",
                "joueurs_disponibles": players,
                "ronde": 1,
                "division": "N1",
            },
        )
        assert response.status_code == 200
        boards = response.json()["compositions"][0]["boards"]
        assert len(boards) <= 8

    def test_health(self, client: TestClient) -> None:
        """Health endpoint works and reports model status.

        @id ALICE-TEST-CE-006
        @version 1.0.0
        """
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert "models_loaded" in data


# ---------------------------------------------------------------------------
# Input validation tests (no model needed — schema layer only)
# ---------------------------------------------------------------------------


class TestComposeValidation:
    """Test input validation on /compose.

    Validates that Pydantic / FastAPI rejects malformed payloads at 422.
    ISO 27034: input sanitization gates.
    """

    def test_missing_club_id(self, client: TestClient) -> None:
        """Missing required field returns 422.

        @id ALICE-TEST-CV-001
        @version 1.0.0
        """
        response = client.post(
            "/api/v1/compose",
            json={
                "joueurs_disponibles": ["A00001"],
                "ronde": 1,
                "division": "N3",
            },
        )
        assert response.status_code == 422

    def test_empty_players(self, client: TestClient) -> None:
        """Empty player list returns 422.

        @id ALICE-TEST-CV-002
        @version 1.0.0
        """
        response = client.post(
            "/api/v1/compose",
            json={
                "club_id": "TEST01",
                "joueurs_disponibles": [],
                "ronde": 1,
                "division": "N3",
            },
        )
        assert response.status_code == 422

    def test_invalid_strategy(self, client: TestClient) -> None:
        """Invalid mode_strategie value returns 422.

        @id ALICE-TEST-CV-003
        @version 1.0.0
        """
        response = client.post(
            "/api/v1/compose",
            json={
                "club_id": "TEST01",
                "joueurs_disponibles": ["A00001"],
                "ronde": 1,
                "division": "N3",
                "mode_strategie": "invalid",
            },
        )
        assert response.status_code == 422

    def test_ronde_out_of_range(self, client: TestClient) -> None:
        """Ronde > 20 returns 422.

        @id ALICE-TEST-CV-004
        @version 1.0.0
        """
        response = client.post(
            "/api/v1/compose",
            json={
                "club_id": "TEST01",
                "joueurs_disponibles": ["A00001"],
                "ronde": 99,
                "division": "N3",
            },
        )
        assert response.status_code == 422

    def test_missing_ronde(self, client: TestClient) -> None:
        """Missing ronde field returns 422.

        @id ALICE-TEST-CV-005
        @version 1.0.0
        """
        response = client.post(
            "/api/v1/compose",
            json={
                "club_id": "TEST01",
                "joueurs_disponibles": ["A00001"],
                "division": "N3",
            },
        )
        assert response.status_code == 422

    def test_missing_division(self, client: TestClient) -> None:
        """Missing division field returns 422.

        @id ALICE-TEST-CV-006
        @version 1.0.0
        """
        response = client.post(
            "/api/v1/compose",
            json={
                "club_id": "TEST01",
                "joueurs_disponibles": ["A00001"],
                "ronde": 1,
            },
        )
        assert response.status_code == 422
