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

from fastapi import APIRouter, Header, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.api.schemas import (
    ErrorResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)
from app.config import settings
from app.logging_config import get_audit_logger, get_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()

# Rate limiter (uses app.state.limiter from main.py)
limiter = Limiter(key_func=get_remote_address)

# TODO: Importer les services quand ils seront implementes
# from services.inference import InferenceService
# from services.composer import ComposerService

router = APIRouter(prefix="/api/v1", tags=["ALICE"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        404: {"model": ErrorResponse, "description": "Club not found"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
@limiter.limit("30/minute")
async def predict_lineup(request: PredictRequest, http_request: Request) -> PredictResponse:
    """Predict opponent lineup and optimize user composition.

    1. ALI (Adversarial Lineup Inference): Predit la composition adverse
    2. CE (Composition Engine): Optimise la composition utilisateur

    @param request: Donnees du match et joueurs disponibles
    @returns: Composition adverse predite + composition optimale recommandee

    @see CDC_ALICE.md - F.1 et F.2
    """
    # TODO: Implementer la logique avec les services
    # inference_service = InferenceService()
    # composer_service = ComposerService()

    # Pour l'instant, reponse placeholder
    return PredictResponse(
        success=True,
        version=settings.app_version,
        predicted_opponent_lineup=[],
        scenarios=[],
        recommended_lineup=[],
        expected_match_score=0.0,
        confidence=0.0,
        alternatives=[],
        metadata={
            "processing_time_ms": 0,
            "model_version": "not_loaded",
            "data_points_used": 0,
        },
    )


@router.get(
    "/models/{club_id}",
    response_model=ModelInfoResponse,
)
async def get_model_info(club_id: str) -> ModelInfoResponse:
    """Return model information for a club.

    @param club_id: ID FFE du club
    @returns: Type de modele, version, metriques
    """
    # TODO: Implementer la logique
    return ModelInfoResponse(
        club_id=club_id,
        model_type="global-fallback",
        model_version="not_trained",
        reason="Model not yet trained",
    )


@router.post(
    "/train",
    response_model=TrainResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
@limiter.limit("5/minute")
async def trigger_training(
    request: TrainRequest,
    http_request: Request,
    x_api_key: str | None = Header(None),
) -> TrainResponse:
    """Trigger model retraining (protected by API key).

    @param request: Configuration d'entrainement
    @param x_api_key: Cle API requise

    @see ISO 27034 - Auth/Authz
    @see ISO 27001 - Audit logging
    """
    client_ip = get_remote_address(http_request)

    # Verification API key (ISO 27001 - audit auth attempts)
    if not x_api_key or x_api_key != settings.api_key:
        audit_logger.warning(
            "auth_failed",
            action="training_request",
            client_ip=client_ip,
            reason="invalid_api_key",
        )
        raise HTTPException(
            status_code=401,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Invalid or missing API key",
            },
        )

    # Audit log succes (ISO 27001)
    audit_logger.info(
        "auth_success",
        action="training_request",
        client_ip=client_ip,
        club_id=request.club_id or "global",
    )

    # TODO: Lancer l'entrainement en background
    return TrainResponse(
        success=True,
        message="Training job queued",
        job_id=f"train-{request.club_id or 'global'}-placeholder",
        estimated_duration="Unknown",
    )
