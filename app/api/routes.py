# app/api/routes.py
"""
Routes API - Controller Layer (SRP)

Responsabilite unique: Gestion HTTP (validation, serialisation)
La logique metier est deleguee aux services.

@see ISO 42010 - Architecture (Controller)
@see ISO 27034 - Input validation
"""

from fastapi import APIRouter, Header, HTTPException

from app.api.schemas import (
    ErrorResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)
from app.config import settings

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
async def predict_lineup(request: PredictRequest):
    """
    Endpoint principal - Prediction de composition adverse et optimisation.

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
async def get_model_info(club_id: str):
    """
    Information sur le modele utilise pour un club.

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
async def trigger_training(
    request: TrainRequest,
    x_api_key: str | None = Header(None),
):
    """
    Declencher un re-entrainement du modele (protege par API key).

    @param request: Configuration d'entrainement
    @param x_api_key: Cle API requise

    @see ISO 27034 - Auth/Authz
    """
    # Verification API key
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Invalid or missing API key",
            },
        )

    # TODO: Lancer l'entrainement en background
    return TrainResponse(
        success=True,
        message="Training job queued",
        job_id=f"train-{request.club_id or 'global'}-placeholder",
        estimated_duration="Unknown",
    )
