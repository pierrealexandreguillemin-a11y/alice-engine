"""Module: routes.py - Controller Layer (SRP).

Responsabilite unique: Gestion HTTP (validation, serialisation).
La logique metier est deleguee a app.api.compose_helpers.

ISO Compliance:
- ISO/IEC 27001 - Information Security (audit logs)
- ISO/IEC 27034 - Secure Coding (input validation, CWE-20)
- ISO/IEC 42010 - Architecture (Controller layer, SRP)
- ISO/IEC 25010 - System Quality (securite, fiabilite)
- ISO/IEC 5055 - File <= 300 lignes (D-P3-11 split compose_helpers)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.api.compose_helpers import (
    apply_pre_filters,
    build_player_pool,
    validate_ffe,
)
from app.api.compose_scenarios import (
    ComposeCtx,
    build_metadata,
    compose_boards_and_predictions,
    make_competition_context,
)
from app.api.schemas import (
    ComposeRequest,
    ComposeResponse,
    RecomposeRequest,
    TeamComposition,
)
from app.logging_config import get_audit_logger, get_logger
from services.audit import OperationType
from services.inference import StackingInferenceService

if TYPE_CHECKING:
    from services.ali.scenario import ScenarioSet
    from services.ffe.rule_engine import RuleEngine

logger = get_logger(__name__)
audit_logger = get_audit_logger()

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/api/v1", tags=["ALICE"])


async def _audit_compose(
    request: Request,
    body: ComposeRequest,
    scenario_set: ScenarioSet | None,
    contraintes_ok: bool,
    duration_ms: float,
) -> None:
    """Emit audit entry with lineage_hash and rule metadata (ISO 27001 A.8.15)."""
    audit = getattr(request.app.state, "audit_logger", None)
    if audit is None:
        return
    engine = getattr(request.app.state, "rule_engine", None)
    query: dict[str, Any] = {
        "club_id": body.club_id,
        "ronde": body.ronde,
        "division": body.division,
        "contraintes_ok": contraintes_ok,
    }
    if scenario_set is not None:
        query["lineage_hash"] = scenario_set.lineage_hash
        query["n_scenarios"] = len(scenario_set.scenarios)
        query["opponent_club_id"] = scenario_set.opponent_club_id
    if engine is not None:
        query["rule_uuids"] = [r.uuid_rfc4122 for r in engine.rules]
    await audit.log(
        operation_type=OperationType.READ,
        collection="compose",
        query=query,
        result_count=1,
        duration_ms=duration_ms,
        success=True,
    )


@router.post("/compose", response_model=ComposeResponse)
@limiter.limit("30/minute")
async def compose_teams(body: ComposeRequest, request: Request) -> ComposeResponse:
    """Compose optimal teams. Phase 3 ALI path active if opponent_club_id fourni."""
    t_start = datetime.now(UTC)
    bundle = getattr(request.app.state, "model_bundle", None)
    feature_store = getattr(request.app.state, "feature_store", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    inference = StackingInferenceService(bundle)
    engine: RuleEngine | None = getattr(request.app.state, "rule_engine", None)
    pool = build_player_pool(body)
    team_size_target = 8
    pre_ctx = make_competition_context(body, team_size_target)
    eligible = apply_pre_filters(engine, pool, pre_ctx)
    sorted_players = sorted(eligible, key=lambda p: -p.elo)
    team_size = min(team_size_target, len(sorted_players))
    selected = sorted_players[:team_size]

    compose_ctx = ComposeCtx(
        request=request,
        body=body,
        selected=selected,
        team_size=team_size,
        inference=inference,
        feature_store=feature_store,
    )
    scenario_set, boards, opp_preds, ali_mode, opp_label = compose_boards_and_predictions(
        compose_ctx
    )
    validate_ctx = make_competition_context(body, team_size)
    all_ok = validate_ffe(engine, boards, selected, validate_ctx)

    composition = TeamComposition(
        equipe=f"{body.club_id} - Equipe 1",
        division=body.division,
        adversaire=opp_label,
        adversaire_predit=opp_preds[:team_size],
        boards=boards,
        e_score_total=round(sum(b.e_score for b in boards), 4),
        contraintes_ok=all_ok,
        validated_by_flatsix=False,
    )

    duration_ms = (datetime.now(UTC) - t_start).total_seconds() * 1000
    audit_logger.info(
        "compose_request",
        club_id=body.club_id,
        n_players=len(body.joueurs_disponibles),
        n_boards=len(boards),
        ali_mode=ali_mode,
        lineage_hash=(scenario_set.lineage_hash if scenario_set else None),
        fallback=bundle.fallback_mode if bundle else True,
        contraintes_ok=all_ok,
        duration_ms=duration_ms,
    )
    await _audit_compose(request, body, scenario_set, all_ok, duration_ms)

    metadata = build_metadata(bundle, ali_mode, duration_ms, scenario_set)
    return ComposeResponse(compositions=[composition], metadata=metadata)


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
        opponent_club_id=None,
        round_date=None,
        saison=None,
        current_round=None,
        nb_rondes_total=None,
        player_overrides=None,
    )
    result: ComposeResponse = await compose_teams(body=compose_body, request=request)
    return result
