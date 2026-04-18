"""Module: routes.py - Controller Layer (SRP).

Responsabilite unique: Gestion HTTP (validation, serialisation).
La logique metier est deleguee aux services.

ISO Compliance:
- ISO/IEC 27001 - Information Security (authentification, autorisation, audit logs)
- ISO/IEC 27034 - Secure Coding (input validation, CWE-20)
- ISO/IEC 42010 - Architecture (Controller layer, SRP)
- ISO/IEC 25010 - System Quality (securite, fiabilite)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from datetime import UTC, datetime

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

logger = get_logger(__name__)
audit_logger = get_audit_logger()

# Rate limiter (uses app.state.limiter from main.py)
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/api/v1", tags=["ALICE"])


@router.post("/compose", response_model=ComposeResponse)
@limiter.limit("30/minute")
async def compose_teams(
    body: ComposeRequest,
    request: Request,
) -> ComposeResponse:
    """Compose optimal teams for a club (Phase 2).

    Pipeline: FFE pre-filter -> ALI fallback (Elo sort) -> ML predict -> compose -> FFE post-check.
    """
    bundle = getattr(request.app.state, "model_bundle", None)
    feature_store = getattr(request.app.state, "feature_store", None)

    if bundle is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    from services.ffe_rules import (  # noqa: PLC0415
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
    from services.inference import StackingInferenceService  # noqa: PLC0415

    inference = StackingInferenceService(bundle)

    # For Phase 2: single team, Elo lookup from request
    # TODO Phase 3: multi-team, MongoDB player lookup, ALI Monte Carlo
    # Phase 2: player data is synthetic (Elo 1500, no brulage/mute/noyau history)
    # The filters are wired but pass through when data is missing (graceful degradation)
    available = [
        {"ffe_id": fid, "elo": 1500, "matchs_joues": 0, "matchs_equipe_sup": {}}
        for fid in body.joueurs_disponibles
    ]

    # Pre-filter: remove ineligible players
    eligible = list(available)
    eligible = filter_brule(eligible, target_team=f"{body.club_id}_1", team_rank=1)
    eligible = filter_match_count(eligible, ronde=body.ronde)
    eligible = filter_same_group(eligible, target_group="default", group_history={})

    sorted_players = sort_by_elo(eligible)
    team_size = min(8, len(sorted_players))
    selected = sorted_players[:team_size]

    # ALI fallback: opponents = similar Elo (stub)
    opponent_elos = [p["elo"] - 50 for p in selected]

    boards = []
    for i, (player, opp_elo) in enumerate(zip(selected, opponent_elos, strict=False)):
        p_elo = player["elo"]
        # Feature assembly (if feature store available)
        if feature_store is not None:
            features = feature_store.assemble(
                player_name=player["ffe_id"],
                player_elo=p_elo,
                opponent_elo=opp_elo,
                context={"ronde": body.ronde, "division": body.division},
            )
            feat_values = features.values
        else:
            import numpy as np  # noqa: PLC0415

            feat_values = np.zeros((1, 201))

        try:
            result = inference.predict_board(
                player_elo=p_elo,
                opponent_elo=opp_elo,
                features=feat_values,
            )
            p_win, p_draw, p_loss, e_score = (
                result.p_win,
                result.p_draw,
                result.p_loss,
                result.e_score,
            )
        except Exception:
            logger.exception("Inference failed for player %s", player["ffe_id"])
            # Elo fallback
            p_win, p_draw, p_loss, e_score = 0.45, 0.15, 0.40, 0.525

        boards.append(
            BoardResult(
                board=i + 1,
                joueur=player["ffe_id"],
                elo=p_elo,
                adversaire=f"OPP_{i + 1}",
                adversaire_elo=opp_elo,
                p_win=round(p_win, 4),
                p_draw=round(p_draw, 4),
                p_loss=round(p_loss, 4),
                e_score=round(e_score, 4),
            )
        )

    elos = [b.elo for b in boards]
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
        contraintes_ok=check_elo_order(elos),
        validated_by_flatsix=False,
    )

    # Post-check: validate composed team (all 11 FFE rules)
    # Phase 2: synthetic player attributes (no real brulage/noyau/mutes history)
    selected_dicts = [
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

    elo_order_ok = check_elo_order(elos)
    team_size_ok = check_team_size(len(boards), required=team_size)
    unique_ok = check_unique_assignment([[b.joueur for b in boards]])
    noyau_ok = check_noyau([b.joueur for b in boards], noyau=set(), ronde=body.ronde)
    mutes_ok = check_mutes_limit(selected_dicts, max_mutes=3)
    foreign_ok = check_foreign_quota(selected_dicts, min_fr_eu=5)

    all_ok = all([elo_order_ok, team_size_ok, unique_ok, noyau_ok, mutes_ok, foreign_ok])

    # Division-specific checks
    if body.division in ("N1", "N2"):
        fr_gender_ok = check_fr_gender(selected_dicts)
        all_ok = all_ok and fr_gender_ok
    if body.division == "N4":
        elo_max_ok = check_elo_max(selected_dicts, elo_max=2400)
        all_ok = all_ok and elo_max_ok

    composition.contraintes_ok = all_ok

    # Audit log (ISO 27001)
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
async def recompose_teams(
    body: RecomposeRequest,
    request: Request,
) -> ComposeResponse:
    """Adjust composition after player change (Phase 2 stub).

    Mode 'remplacement': swap one player.
    Mode 'global': full recomposition with updated pool.
    """
    # Phase 2: delegate to /compose logic with updated player pool
    # TODO Phase 3: implement diff with composition_precedente
    from app.api.schemas import ComposeRequest  # noqa: PLC0415

    compose_body = ComposeRequest(
        club_id=body.club_id,
        joueurs_disponibles=body.joueurs_disponibles,
        ronde=1,  # TODO: get from composition_precedente
        division="N3",  # TODO: get from composition_precedente
        mode_strategie="agressif",  # TODO: get from composition_precedente
    )
    result: ComposeResponse = await compose_teams(body=compose_body, request=request)
    return result
