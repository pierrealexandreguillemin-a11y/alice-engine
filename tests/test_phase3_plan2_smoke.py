"""Smoke E2E Plan 2 : /compose avec ALI ScenarioGenerator active.

ISO 29119 : integration smoke test.
Verifie : pipeline complet pool -> enrich -> fit copula -> TopK + MC ->
inference Phase 2 x 20 -> CE -> ComposeResponse avec lineage_hash.

Document ID: ALICE-TEST-P2-TASK10
Version: 1.0.0
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from scripts.serving.model_loader import ModelBundle
from services.inference import PredictionResult

J = Path("data/joueurs.parquet")
E = Path("data/echiquiers.parquet")

pytestmark = pytest.mark.skipif(
    not (J.exists() and E.exists()),
    reason="data parquets absent du runner",
)


def _make_mock_bundle() -> ModelBundle:
    """Mock ModelBundle so lifespan does not hit HF Hub."""
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
        version="test-plan2-smoke",
    )


def _mock_predict_board(
    player_elo: int,
    opponent_elo: int,
    features: np.ndarray,
    draw_rate_lookup=None,
) -> PredictionResult:
    """Deterministic stub for StackingInferenceService.predict_board."""
    return PredictionResult(p_loss=0.30, p_draw=0.20, p_win=0.50, e_score=0.60)


@pytest.fixture(scope="module")
def client():
    """TestClient with lifespan complete (ALI generator wired, models mocked).

    feature_store is loaded from data/feature_store/ (build_feature_store.py
    has been run). aggregation._assemble_features requires feature_store !=
    None even si predict_board est mocké — c'est le path strict-mode.
    """
    mock_bundle = _make_mock_bundle()
    with (
        patch("scripts.serving.model_loader.load_models", return_value=mock_bundle),
        patch(
            "services.inference.StackingInferenceService.predict_board",
            side_effect=_mock_predict_board,
        ),
    ):
        import importlib

        import app.main as _main_module

        importlib.reload(_main_module)
        from app.main import app as _app

        # FeatureStore minimal pour smoke ALI (skip si parquets absents)
        from services.feature_store import FeatureStore

        fs_path = Path("data/feature_store")
        if not (fs_path / "training_mean.parquet").exists():
            pytest.skip(
                "data/feature_store/training_mean.parquet absent — "
                "run scripts/build_feature_store.py first"
            )
        feature_store = FeatureStore(fs_path)
        feature_store.load()

        with TestClient(_app) as tc:
            _app.state.model_bundle = mock_bundle
            _app.state.feature_store = feature_store
            yield tc


def _find_eligible_clubs(min_pool: int = 40) -> tuple[str, str]:
    """Find user_club and opponent_club with >= min_pool joueurs each.

    ALI needs a viable opponent pool (>=40 joueurs) to produce 20 distinct
    scenarios for team_size=8. Avec min_pool=20-30, dedup TopK+MC produit
    parfois 16-18 scenarios uniques (combinatoire trop restreinte) ⇒ FAIL
    R-ALI-02 ScenarioSet.validate(). User club needs >= 12 FFE-IDs so
    /compose can build a pool and cap boards at 8.
    """
    df = pd.read_parquet(J)
    counts = df.groupby("club", dropna=False).size()
    eligible = counts[counts >= min_pool].index.tolist()
    if len(eligible) < 2:
        pytest.skip(f"besoin de >=2 clubs avec >={min_pool} joueurs")
    return str(eligible[0]), str(eligible[1])


def _get_first_n_players(club_id: str, n: int) -> list[str]:
    """Return first N nr_ffe IDs of a club (letter + 5 digits, schema-valid)."""
    df = pd.read_parquet(J)
    sub = df[df["club"] == club_id].head(n)
    return sub["nr_ffe"].astype(str).tolist()


def test_compose_with_ali_returns_lineage_hash(client):
    """ALI mode actif : metadata.lineage_hash present + ali_mode=scenario_generator."""
    user_club, opp_club = _find_eligible_clubs()
    user_players = _get_first_n_players(user_club, 12)

    response = client.post(
        "/api/v1/compose",
        json={
            "club_id": user_club,
            "joueurs_disponibles": user_players,
            "ronde": 5,
            "division": "N3",
            "mode_strategie": "agressif",
            "opponent_club_id": opp_club,
            "round_date": "2024-11-15",
            "saison": 2024,
            "current_round": 5,
            "nb_rondes_total": 11,
        },
    )

    assert response.status_code == 200, f"got {response.status_code}: {response.text}"
    data = response.json()
    assert "metadata" in data
    metadata = data["metadata"]
    assert "lineage_hash" in metadata, f"metadata = {metadata}"
    assert len(metadata["lineage_hash"]) == 64
    assert metadata.get("ali_mode") == "scenario_generator"
    assert metadata.get("n_scenarios") == 20


def test_compose_without_opponent_uses_fallback(client):
    """No opponent_club_id -> Phase 2 elo_fallback (backward compat)."""
    user_club, _ = _find_eligible_clubs()
    user_players = _get_first_n_players(user_club, 8)

    response = client.post(
        "/api/v1/compose",
        json={
            "club_id": user_club,
            "joueurs_disponibles": user_players,
            "ronde": 5,
            "division": "N3",
            "mode_strategie": "agressif",
        },
    )
    assert response.status_code == 200
    data = response.json()
    metadata = data.get("metadata", {})
    assert metadata.get("ali_mode") == "elo_fallback"
    assert "lineage_hash" not in metadata


def test_compose_with_ali_latency_under_5s(client):
    """ALI /compose latency < 5s (cible prod = 2s, marge runtime test)."""
    user_club, opp_club = _find_eligible_clubs()
    user_players = _get_first_n_players(user_club, 12)

    start = time.perf_counter()
    response = client.post(
        "/api/v1/compose",
        json={
            "club_id": user_club,
            "joueurs_disponibles": user_players,
            "ronde": 5,
            "division": "N3",
            "mode_strategie": "agressif",
            "opponent_club_id": opp_club,
            "round_date": "2024-11-15",
            "saison": 2024,
            "current_round": 5,
            "nb_rondes_total": 11,
        },
    )
    elapsed = time.perf_counter() - start
    assert response.status_code == 200
    assert elapsed < 5.0, f"latence {elapsed:.2f}s > 5s"


# ---------------------------------------------------------------------------
# D-P2-06 fix : inference loops over 20 scenarios, weighted average
# Spec §2 §4.7 : E[score_board_k] = Σ_s weight_s × E[score_board_k|scenario_s]
# ---------------------------------------------------------------------------


def _ali_compose_request(user_club: str, opp_club: str, players: list[str]) -> dict:
    """Build a valid ALI /compose payload (D-P2-06 test helper)."""
    return {
        "club_id": user_club,
        "joueurs_disponibles": players,
        "ronde": 5,
        "division": "N3",
        "mode_strategie": "agressif",
        "opponent_club_id": opp_club,
        "round_date": "2024-11-15",
        "saison": 2024,
        "current_round": 5,
        "nb_rondes_total": 11,
    }


def test_compose_aggregates_over_20_scenarios(client):
    """D-P2-06 fix : inference boucle sur 20 scenarios, agrege weighted avg.

    Spec §2 pipeline : per (scenario, board) inference call.
    Spec §4.7 : weighted sum over all 20 scenarios.
    """
    user_club, opp_club = _find_eligible_clubs()
    user_players = _get_first_n_players(user_club, 12)

    response = client.post(
        "/api/v1/compose", json=_ali_compose_request(user_club, opp_club, user_players)
    )
    assert response.status_code == 200, f"got {response.status_code}: {response.text}"
    data = response.json()

    # Probabilites agregees : [0,1] et somme ~1
    composition = data["compositions"][0]
    for board in composition["boards"]:
        assert 0.0 <= board["p_win"] <= 1.0
        assert 0.0 <= board["p_draw"] <= 1.0
        assert 0.0 <= board["p_loss"] <= 1.0
        total = board["p_win"] + board["p_draw"] + board["p_loss"]
        assert abs(total - 1.0) < 0.01, f"board {board['board']} proba sum {total}"
        assert 0.0 <= board["e_score"] <= 1.0

    # Metadata confirme agregation 20 scenarios
    metadata = data["metadata"]
    assert metadata.get("n_scenarios") == 20
    assert metadata.get("ali_mode") == "scenario_generator"
    assert composition["adversaire"].endswith("20-scenario weighted avg)")


def test_compose_aggregation_deterministic(client):
    """Meme request (seed ALI fixe) -> memes aggregated probabilities.

    ISO 42001 : determinisme inference = reproductibilite.
    """
    user_club, opp_club = _find_eligible_clubs()
    user_players = _get_first_n_players(user_club, 12)
    payload = _ali_compose_request(user_club, opp_club, user_players)

    r1 = client.post("/api/v1/compose", json=payload)
    r2 = client.post("/api/v1/compose", json=payload)
    assert r1.status_code == 200
    assert r2.status_code == 200

    boards1 = r1.json()["compositions"][0]["boards"]
    boards2 = r2.json()["compositions"][0]["boards"]
    assert len(boards1) == len(boards2)
    for b1, b2 in zip(boards1, boards2, strict=True):
        assert b1["p_win"] == b2["p_win"]
        assert b1["p_draw"] == b2["p_draw"]
        assert b1["p_loss"] == b2["p_loss"]
        assert b1["e_score"] == b2["e_score"]
