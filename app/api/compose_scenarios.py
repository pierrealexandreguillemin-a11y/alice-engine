"""Scenario + fallback aggregation helpers for /compose (ISO 5055).

Split from compose_helpers.py to keep both modules <= 300 lines.

Document ID: ALICE-COMPOSE-SCENARIOS
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from app.api.schemas import BoardResult, ComposeRequest, OpponentPrediction
from app.logging_config import get_logger
from services.ali.aggregation import (
    AggregatedBoard,
    ScenarioAggregationCtx,
    aggregate_from_scenarios,
)
from services.ali.types import CompetitionContext, PlayerCandidate

if TYPE_CHECKING:
    from fastapi import Request

    from services.ali.scenario import ScenarioSet
    from services.inference import StackingInferenceService


logger = get_logger(__name__)


@dataclass
class BoardContext:
    """Context for a single board prediction."""

    player: PlayerCandidate
    opp_elo: int
    board_num: int
    ronde: int
    division: str


def predict_board(
    inference: StackingInferenceService, ctx: BoardContext, feature_store: Any
) -> BoardResult:
    """Predict P(W/D/L) for one board assignment (Phase 2 fallback path)."""
    if feature_store is not None:
        feat = np.asarray(
            feature_store.assemble(
                player_name=ctx.player.nr_ffe,
                player_elo=ctx.player.elo,
                opponent_elo=ctx.opp_elo,
                context={"ronde": ctx.ronde, "division": ctx.division},
            ).values
        )
    else:
        feat = np.zeros((1, 201))
    try:
        r = inference.predict_board(
            player_elo=ctx.player.elo, opponent_elo=ctx.opp_elo, features=feat
        )
        p_win, p_draw, p_loss, e_score = r.p_win, r.p_draw, r.p_loss, r.e_score
    except Exception:
        logger.exception("Inference failed for player %s", ctx.player.nr_ffe)
        p_win, p_draw, p_loss, e_score = 0.45, 0.15, 0.40, 0.525
    return BoardResult(
        board=ctx.board_num,
        joueur=ctx.player.nr_ffe,
        elo=ctx.player.elo,
        adversaire=f"OPP_{ctx.board_num}",
        adversaire_elo=ctx.opp_elo,
        p_win=round(p_win, 4),
        p_draw=round(p_draw, 4),
        p_loss=round(p_loss, 4),
        e_score=round(e_score, 4),
    )


def make_competition_context(body: ComposeRequest, team_size: int) -> CompetitionContext:
    """Build CompetitionContext for the ScenarioGenerator from a ComposeRequest."""
    return CompetitionContext(
        competition_code="A02",
        niveau=body.division,
        ronde=body.ronde,
        team_size=team_size,
        noyau_min=50,
        max_mutes=3,
        elo_max=2400 if body.division == "N4" else None,
    )


def try_generate_scenarios(
    request: Request,
    body: ComposeRequest,
    team_size: int,
) -> ScenarioSet | None:
    """Invoke ALI ScenarioGenerator if available. None on failure (ISO 25010 degraded path)."""
    generator = getattr(request.app.state, "scenario_generator", None)
    if generator is None or body.opponent_club_id is None:
        return None
    round_date = body.round_date or datetime.now(UTC).strftime("%Y-%m-%d")
    saison = body.saison or datetime.now(UTC).year
    current_round = body.current_round or body.ronde
    nb_rondes_total = body.nb_rondes_total or 11
    ctx = make_competition_context(body, team_size)
    try:
        result: ScenarioSet = generator.generate(
            opponent_club_id=body.opponent_club_id,
            round_date=round_date,
            context=ctx,
            saison=saison,
            current_round=current_round,
            nb_rondes_total=nb_rondes_total,
            overrides=body.player_overrides,
        )
    except Exception:
        logger.exception("ali_generator_failed club=%s", body.opponent_club_id)
        return None
    return result


def to_board_and_pred(agg: AggregatedBoard) -> tuple[BoardResult, OpponentPrediction]:
    """Adapt aggregated service result -> API schemas (routes controller role)."""
    board = BoardResult(
        board=agg.board,
        joueur=agg.user_ffe_id,
        elo=agg.user_elo,
        adversaire=agg.mode_opponent_ffe,
        adversaire_elo=agg.mean_opponent_elo,
        p_win=agg.p_win,
        p_draw=agg.p_draw,
        p_loss=agg.p_loss,
        e_score=agg.e_score,
    )
    pred = OpponentPrediction(
        board=agg.board, joueur=agg.mode_opponent_ffe, elo=agg.mean_opponent_elo
    )
    return board, pred


def default_opponent(
    selected: list[PlayerCandidate], team_size: int
) -> tuple[list[int], list[OpponentPrediction]]:
    """Phase 2 fallback: opponent Elo = user Elo - 50 (ALICE-TEST-CE-001 style)."""
    opponent_elos = [int(p.elo) - 50 for p in selected[:team_size]]
    predictions = [
        OpponentPrediction(board=i + 1, joueur=f"OPP_{i + 1}", elo=e)
        for i, e in enumerate(opponent_elos)
    ]
    return opponent_elos, predictions


def build_boards(
    inference: StackingInferenceService,
    selected: list[PlayerCandidate],
    opponent_elos: list[int],
    body: ComposeRequest,
    feature_store: Any,
) -> list[BoardResult]:
    """Run StackingInferenceService.predict_board for each (player, opp_elo) pair."""
    return [
        predict_board(
            inference,
            BoardContext(
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


@dataclass
class ComposeCtx:
    """Context bundle for compose flow (ISO 5055 PLR0913 compliance)."""

    request: Request
    body: ComposeRequest
    selected: list[PlayerCandidate]
    team_size: int
    inference: StackingInferenceService
    feature_store: Any


def compose_boards_and_predictions(
    ctx: ComposeCtx,
) -> tuple[ScenarioSet | None, list[BoardResult], list[OpponentPrediction], str, str]:
    """Resolve opponent + per-board predictions. ALI path (D-P2-06) or Elo fallback."""
    scenario_set = try_generate_scenarios(ctx.request, ctx.body, ctx.team_size)
    if scenario_set is not None:
        # Adapt PlayerCandidate -> dict expected by ScenarioAggregationCtx.
        user_lineup_dicts: list[dict[str, Any]] = [
            {"ffe_id": p.nr_ffe, "elo": p.elo} for p in ctx.selected
        ]
        agg_ctx = ScenarioAggregationCtx(
            scenario_set=scenario_set,
            user_lineup=user_lineup_dicts,
            team_size=ctx.team_size,
            ronde=ctx.body.ronde,
            division=ctx.body.division,
        )
        aggregated = aggregate_from_scenarios(ctx.inference, ctx.feature_store, agg_ctx)
        boards = []
        preds = []
        for agg in aggregated:
            b, p = to_board_and_pred(agg)
            boards.append(b)
            preds.append(p)
        return (
            scenario_set,
            boards,
            preds,
            "scenario_generator",
            f"{ctx.body.opponent_club_id} (ALI 20-scenario weighted avg)",
        )
    opp_elos, preds = default_opponent(ctx.selected, ctx.team_size)
    boards = build_boards(ctx.inference, ctx.selected, opp_elos, ctx.body, ctx.feature_store)
    return None, boards, preds, "elo_fallback", "Adversaire (ALI fallback)"


def build_metadata(
    bundle: Any, ali_mode: str, duration_ms: float, scenario_set: ScenarioSet | None
) -> dict[str, Any]:
    """Assemble response metadata with lineage_hash when ALI active."""
    metadata: dict[str, Any] = {
        "model_version": bundle.version if bundle else "none",
        "ali_mode": ali_mode,
        "fallback": bundle.fallback_mode if bundle else True,
        "timestamp": datetime.now(UTC).isoformat(),
        "duration_ms": round(duration_ms, 2),
    }
    if scenario_set is not None:
        metadata["lineage_hash"] = scenario_set.lineage_hash
        metadata["n_scenarios"] = len(scenario_set.scenarios)
    return metadata
