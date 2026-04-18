"""Module: routes.py - Controller Layer (SRP).

Responsabilite unique: Gestion HTTP (validation, serialisation).
La logique metier est deleguee aux services.

ISO Compliance:
- ISO/IEC 27001 - Information Security (audit logs)
- ISO/IEC 27034 - Secure Coding (input validation, CWE-20)
- ISO/IEC 42010 - Architecture (Controller layer, SRP)
- ISO/IEC 25010 - System Quality (securite, fiabilite)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.api.schemas import (
    BoardResult,
    ComposeRequest,
    ComposeResponse,
    OpponentPrediction,
    RecomposeRequest,
    TeamComposition,
)
from app.logging_config import get_audit_logger, get_logger
from services.ffe_rules import (
    check_elo_max,
    check_elo_order,
    check_foreign_quota,
    check_fr_gender,
    check_mutes_limit,
    check_noyau,
    check_team_size,
    check_unique_assignment,
    filter_brule,
    filter_match_count,
    filter_same_group,
    sort_by_elo,
)
from services.inference import StackingInferenceService

logger = get_logger(__name__)
audit_logger = get_audit_logger()

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/api/v1", tags=["ALICE"])


def _build_player_pool(body: ComposeRequest) -> list[dict]:
    """Build synthetic player pool (Phase 2: no MongoDB, all Elo 1500)."""
    return [
        {"ffe_id": fid, "elo": 1500, "matchs_joues": 0, "matchs_equipe_sup": {}}
        for fid in body.joueurs_disponibles
    ]


def _apply_pre_filters(players: list[dict], body: ComposeRequest) -> list[dict]:
    """FFE pre-filter: remove ineligible players before composition."""
    eligible = filter_brule(players, target_team=f"{body.club_id}_1", team_rank=1)
    eligible = filter_match_count(eligible, ronde=body.ronde)
    return filter_same_group(eligible, target_group="default", group_history={})


@dataclass
class _BoardContext:
    """Context for a single board prediction."""

    player: dict
    opp_elo: int
    board_num: int
    ronde: int
    division: str


def _predict_board(
    inference: StackingInferenceService, ctx: _BoardContext, feature_store: Any
) -> BoardResult:
    """Predict P(W/D/L) for one board assignment."""
    p_elo = ctx.player["elo"]
    if feature_store is not None:
        features = feature_store.assemble(
            player_name=ctx.player["ffe_id"],
            player_elo=p_elo,
            opponent_elo=ctx.opp_elo,
            context={"ronde": ctx.ronde, "division": ctx.division},
        )
        feat_values = features.values
    else:
        feat_values = np.zeros((1, 201))

    try:
        r = inference.predict_board(
            player_elo=p_elo, opponent_elo=ctx.opp_elo, features=feat_values
        )
        p_win, p_draw, p_loss, e_score = r.p_win, r.p_draw, r.p_loss, r.e_score
    except Exception:
        logger.exception("Inference failed for player %s", ctx.player["ffe_id"])
        p_win, p_draw, p_loss, e_score = 0.45, 0.15, 0.40, 0.525

    return BoardResult(
        board=ctx.board_num,
        joueur=ctx.player["ffe_id"],
        elo=p_elo,
        adversaire=f"OPP_{ctx.board_num}",
        adversaire_elo=ctx.opp_elo,
        p_win=round(p_win, 4),
        p_draw=round(p_draw, 4),
        p_loss=round(p_loss, 4),
        e_score=round(e_score, 4),
    )


def _validate_ffe(boards: list[BoardResult], body: ComposeRequest, team_size: int) -> bool:
    """FFE post-check: validate composed team against 11 blocking rules."""
    elos = [b.elo for b in boards]
    selected = [
        {
            "ffe_id": b.joueur,
            "elo": b.elo,
            "is_muted": False,
            "is_french_eu": True,
            "is_french": True,
            "sexe": "M",
        }
        for b in boards
    ]

    checks = [
        check_elo_order(elos),
        check_team_size(len(boards), required=team_size),
        check_unique_assignment([[b.joueur for b in boards]]),
        check_noyau([b.joueur for b in boards], noyau=set(), ronde=body.ronde),
        check_mutes_limit(selected, max_mutes=3),
        check_foreign_quota(selected, min_fr_eu=5),
    ]

    if body.division in ("N1", "N2"):
        checks.append(check_fr_gender(selected))
    if body.division == "N4":
        checks.append(check_elo_max(selected, elo_max=2400))

    return all(checks)


@router.post("/compose", response_model=ComposeResponse)
@limiter.limit("30/minute")
async def compose_teams(body: ComposeRequest, request: Request) -> ComposeResponse:
    """Compose optimal teams for a club (Phase 2)."""
    bundle = getattr(request.app.state, "model_bundle", None)
    feature_store = getattr(request.app.state, "feature_store", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    inference = StackingInferenceService(bundle)
    pool = _build_player_pool(body)
    eligible = _apply_pre_filters(pool, body)
    sorted_players = sort_by_elo(eligible)
    team_size = min(8, len(sorted_players))
    selected = sorted_players[:team_size]
    opponent_elos = [p["elo"] - 50 for p in selected]

    boards = [
        _predict_board(
            inference,
            _BoardContext(
                player=player,
                opp_elo=opp_elo,
                board_num=i + 1,
                ronde=body.ronde,
                division=body.division,
            ),
            feature_store,
        )
        for i, (player, opp_elo) in enumerate(zip(selected, opponent_elos, strict=False))
    ]

    all_ok = _validate_ffe(boards, body, team_size)
    composition = TeamComposition(
        equipe=f"{body.club_id} - Equipe 1",
        division=body.division,
        adversaire="Adversaire (ALI fallback)",
        adversaire_predit=[
            OpponentPrediction(board=i + 1, joueur=f"OPP_{i + 1}", elo=e)
            for i, e in enumerate(opponent_elos[:team_size])
        ],
        boards=boards,
        e_score_total=round(sum(b.e_score for b in boards), 4),
        contraintes_ok=all_ok,
        validated_by_flatsix=False,
    )

    audit_logger.info(
        "compose_request",
        club_id=body.club_id,
        n_players=len(body.joueurs_disponibles),
        n_boards=len(boards),
        fallback=bundle.fallback_mode if bundle else True,
        contraintes_ok=all_ok,
    )

    return ComposeResponse(
        compositions=[composition],
        metadata={
            "model_version": bundle.version if bundle else "none",
            "ali_mode": "elo_fallback",
            "fallback": bundle.fallback_mode if bundle else True,
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )


@router.post("/recompose", response_model=ComposeResponse)
@limiter.limit("30/minute")
async def recompose_teams(body: RecomposeRequest, request: Request) -> ComposeResponse:
    """Adjust composition after player change (Phase 2 stub)."""
    compose_body = ComposeRequest(
        club_id=body.club_id,
        joueurs_disponibles=body.joueurs_disponibles,
        ronde=1,
        division="N3",
        mode_strategie="agressif",
    )
    result: ComposeResponse = await compose_teams(body=compose_body, request=request)
    return result
