"""ALI scenario aggregation — D-P2-06 fix.

Spec `docs/superpowers/specs/2026-04-19-phase3-ali-monte-carlo-design.md` §2 §4.7

Boucle pipeline : pour chaque (scenario, board) -> StackingInferenceService.predict_board
puis agregation ponderee :
    P_board_k(W/D/L) = Sum_s scenario_s.weight * P_board_k_scenario_s(W/D/L)
    E[score_board_k] = P_board_k(win) + 0.5 * P_board_k(draw)

ISO 5055 : SRP, module dedie, fonctions < 50 lignes.
ISO 5259 : lineage_hash propage via scenario_set.
ISO 42001 : weights traceable per-board (explicability).

Document ID: ALICE-ALI-AGGREGATION
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from services.ali.scenario import ScenarioSet
    from services.inference import StackingInferenceService

logger = logging.getLogger(__name__)


@dataclass
class AggregatedBoard:
    """Result of per-board aggregation over N scenarios.

    Plain-data container (ISO 5055 SRP). routes.py wraps this into the API schema.
    """

    board: int
    user_ffe_id: str
    user_elo: int
    mode_opponent_ffe: str
    mean_opponent_elo: int
    p_win: float
    p_draw: float
    p_loss: float
    e_score: float


@dataclass
class ScenarioAggregationCtx:
    """Context bundle for scenario aggregation (ISO 5055 PLR0913 compliance).

    `strict` controls behaviour when ML inference fails:
    - False (prod default) : log + return Elo-baseline fallback (degraded observable, ISO 25010)
    - True (backtest)      : raise (fail-fast, ISO 42001 explainability + ISO 29119 test integrity)
    """

    scenario_set: ScenarioSet
    user_lineup: list[dict[str, Any]]
    team_size: int
    ronde: int
    division: str
    strict: bool = False


def _assemble_features(
    player: dict[str, Any], opp_elo: int, ronde: int, division: str, feature_store: Any
) -> np.ndarray[Any, Any]:
    """Build feature vector for one (user, opponent) board pair."""
    if feature_store is None:
        return np.zeros((1, 201))
    features = feature_store.assemble(
        player_name=player["ffe_id"],
        player_elo=player["elo"],
        opponent_elo=opp_elo,
        context={"ronde": ronde, "division": division},
    )
    return np.asarray(features.values)


def _safe_predict(
    inference: StackingInferenceService,
    player_elo: int,
    opp_elo: int,
    feat: np.ndarray[Any, Any],
    strict: bool = False,
) -> tuple[float, float, float]:
    """Call inference.predict_board with controlled fallback semantics.

    @param strict: if True, re-raise on inference failure (backtest mode,
        ISO 42001 + 29119 fail-fast). If False, log + return Elo-baseline
        fallback (prod degraded observable, ISO 25010).

    Returns (p_win, p_draw, p_loss). e_score is recomputed downstream.
    """
    try:
        r = inference.predict_board(player_elo=player_elo, opponent_elo=opp_elo, features=feat)
        return r.p_win, r.p_draw, r.p_loss
    except Exception:
        logger.exception(
            "ALI aggregation: inference failed for player_elo=%s opp_elo=%s strict=%s",
            player_elo,
            opp_elo,
            strict,
        )
        if strict:
            raise
        return 0.45, 0.15, 0.40


def aggregate_one_board(
    inference: StackingInferenceService,
    feature_store: Any,
    user_player: dict[str, Any],
    board_idx: int,
    ctx: ScenarioAggregationCtx,
) -> AggregatedBoard:
    """Run inference over all N scenarios for one board and aggregate.

    D-P2-06 fix : loops over ALL ctx.scenario_set.scenarios, not just top.
    """
    p_loss_agg = p_draw_agg = p_win_agg = 0.0
    opp_elo_agg = 0.0
    top_candidates: dict[str, float] = {}
    for scenario in ctx.scenario_set.scenarios:
        w = scenario.weight
        assignments = sorted(scenario.lineup.assignments, key=lambda a: a.board)
        if board_idx >= len(assignments):
            continue
        opp = assignments[board_idx].player
        opp_elo = int(opp.elo)
        feat = _assemble_features(user_player, opp_elo, ctx.ronde, ctx.division, feature_store)
        p_win, p_draw, p_loss = _safe_predict(
            inference,
            user_player["elo"],
            opp_elo,
            feat,
            strict=ctx.strict,
        )
        p_loss_agg += w * p_loss
        p_draw_agg += w * p_draw
        p_win_agg += w * p_win
        opp_elo_agg += w * opp_elo
        top_candidates[opp.nr_ffe] = top_candidates.get(opp.nr_ffe, 0.0) + w

    e_score = p_win_agg + 0.5 * p_draw_agg
    mode_id = max(top_candidates.items(), key=lambda kv: kv[1])[0] if top_candidates else "OPP"
    return AggregatedBoard(
        board=board_idx + 1,
        user_ffe_id=user_player["ffe_id"],
        user_elo=int(user_player["elo"]),
        mode_opponent_ffe=mode_id,
        mean_opponent_elo=int(round(opp_elo_agg)),
        p_win=round(p_win_agg, 4),
        p_draw=round(p_draw_agg, 4),
        p_loss=round(p_loss_agg, 4),
        e_score=round(e_score, 4),
    )


def aggregate_from_scenarios(
    inference: StackingInferenceService,
    feature_store: Any,
    ctx: ScenarioAggregationCtx,
) -> list[AggregatedBoard]:
    """Aggregate per-board predictions over ALL scenarios weighted by scenario.weight.

    Spec §2 §4.7 : E[score_board_k] = Σ_s weight_s × E[score_board_k|scenario_s]
    D-P2-06 fix : utilise tous les 20 scenarios au lieu du top-weighted seul.
    """
    n_boards = min(ctx.team_size, len(ctx.user_lineup))
    return [
        aggregate_one_board(inference, feature_store, ctx.user_lineup[k], k, ctx)
        for k in range(n_boards)
    ]
