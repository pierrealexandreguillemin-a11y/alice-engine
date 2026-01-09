"""Module: schemas.py - Validation des donnees.

Tous les modeles de requetes/reponses sont definis ici.
Validation stricte des entrees pour eviter injections.

ISO Compliance:
- ISO/IEC 5259:2024 - Data Quality for ML (schema validation)
- ISO/IEC 25012 - Data Quality (exactitude, coherence)
- ISO/IEC 27034 - Secure Coding (input validation, CWE-20)
- ISO/IEC 5055 - Code Quality (types stricts)

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ============================================================
# SCHEMAS JOUEUR
# ============================================================


class PlayerInput(BaseModel):
    """Joueur disponible pour la composition."""

    ffe_id: str = Field(..., description="ID FFE du joueur (ex: A12345)")
    elo: int = Field(..., ge=500, le=3000, description="Classement Elo")
    name: str | None = Field(None, description="Nom complet")
    category: str | None = Field(None, description="Categorie (Sen, Vet, Jun...)")
    is_female: bool = Field(False, description="Joueuse")
    is_reserve: bool = Field(False, description="Joueur reserve")
    is_muted: bool = Field(False, description="Joueur mute")
    matches_played: int | None = Field(None, ge=0, description="Matchs joues")

    _FFE_ID_LENGTH = 6  # Format: 1 lettre + 5 chiffres

    @field_validator("ffe_id")
    @classmethod
    def validate_ffe_id(cls, v: str) -> str:
        """Validate FFE ID format (letter + 5 digits)."""
        v = v.upper().strip()
        if len(v) != cls._FFE_ID_LENGTH or not v[0].isalpha() or not v[1:].isdigit():
            raise ValueError("FFE ID doit etre au format A12345")
        return v


class PlayerPrediction(BaseModel):
    """Joueur predit dans la composition adverse."""

    board: int = Field(..., ge=1, le=8, description="Numero echiquier")
    ffe_id: str
    name: str | None = None
    elo: int
    probability: float = Field(..., ge=0, le=1, description="Probabilite presence")
    reasoning: str | None = Field(None, description="Explication de la prediction")


class PlayerRecommendation(BaseModel):
    """Joueur recommande dans la composition optimale."""

    board: int = Field(..., ge=1, le=8)
    ffe_id: str
    name: str | None = None
    elo: int
    expected_score: float = Field(..., ge=0, le=1, description="Score attendu (0-1)")
    win_probability: float | None = None
    draw_probability: float | None = None
    loss_probability: float | None = None
    opponent: dict[str, Any] | None = Field(None, description="Adversaire predit")


# ============================================================
# SCHEMAS SCENARIO
# ============================================================


class Scenario(BaseModel):
    """Scenario de composition adverse."""

    id: int
    probability: float = Field(..., ge=0, le=1)
    lineup: list[dict[str, Any]]


class Alternative(BaseModel):
    """Composition alternative recommandee."""

    rank: int = Field(..., ge=1)
    description: str | None = None
    lineup: list[PlayerRecommendation]
    expected_score: float
    tradeoff: str | None = Field(None, description="Compromis explique")


# ============================================================
# SCHEMAS CONTRAINTES
# ============================================================


class Constraints(BaseModel):
    """Contraintes de composition (regles FFE simplifiees)."""

    team_size: int = Field(8, ge=4, le=12, description="Nombre de joueurs")
    elo_descending: bool = Field(True, description="Ordre Elo decroissant")
    min_females: int = Field(0, ge=0, description="Minimum joueuses")
    max_muted: int | None = Field(None, ge=0, description="Maximum mutes")


class Options(BaseModel):
    """Options de calcul."""

    scenario_count: int = Field(20, ge=1, le=100, description="Nombre scenarios")
    include_alternatives: bool = Field(True, description="Inclure alternatives")
    alternative_count: int = Field(3, ge=1, le=10, description="Nombre alternatives")


# ============================================================
# SCHEMAS REQUETE/REPONSE PREDICT
# ============================================================


class PredictRequest(BaseModel):
    """Requete de prediction - Endpoint POST /predict."""

    club_id: str = Field(..., description="ID FFE du club utilisateur")
    opponent_club_id: str = Field(..., description="ID FFE du club adverse")
    competition_id: str | None = Field(None, description="ID competition")
    round_number: int = Field(..., ge=1, le=20, description="Numero de ronde")
    available_players: list[PlayerInput] = Field(
        ..., min_length=1, description="Joueurs disponibles"
    )
    constraints: Constraints | None = None
    options: Options | None = None


class PredictResponse(BaseModel):
    """Reponse de prediction - Endpoint POST /predict."""

    success: bool
    version: str
    warning: str | None = None

    # ALI - Prediction adverse
    predicted_opponent_lineup: list[PlayerPrediction]
    scenarios: list[Scenario] = Field(default_factory=list)

    # CE - Composition optimale
    recommended_lineup: list[PlayerRecommendation]
    expected_match_score: float
    score_range: dict[str, float] | None = None
    confidence: float = Field(..., ge=0, le=1)
    alternatives: list[Alternative] = Field(default_factory=list)

    # Metadata
    warnings: list[dict[str, str]] = Field(default_factory=list)
    metadata: dict[str, Any]


# ============================================================
# SCHEMAS MODELE
# ============================================================


class ModelInfoResponse(BaseModel):
    """Information sur le modele."""

    club_id: str
    model_type: str  # "club-specific" ou "global-fallback"
    model_version: str
    last_trained_at: datetime | None = None
    training_data_points: int | None = None
    accuracy: float | None = None
    features: list[str] | None = None
    reason: str | None = None  # Si fallback


# ============================================================
# SCHEMAS ENTRAINEMENT
# ============================================================


class TrainRequest(BaseModel):
    """Requete d'entrainement."""

    club_id: str | None = Field(None, description="Si absent, modele global")
    force_retrain: bool = Field(False, description="Forcer meme si donnees insuffisantes")


class TrainResponse(BaseModel):
    """Reponse d'entrainement."""

    success: bool
    message: str
    job_id: str
    estimated_duration: str | None = None


# ============================================================
# SCHEMAS ERREUR
# ============================================================


class ErrorDetail(BaseModel):
    """Detail d'une erreur."""

    field: str | None = None
    error: str


class ErrorResponse(BaseModel):
    """Reponse d'erreur standardisee."""

    success: bool = False
    error: dict[str, Any] = Field(
        ...,
        examples=[
            {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request",
                "details": [],
            }
        ],
    )
