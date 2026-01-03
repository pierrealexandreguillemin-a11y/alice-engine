# app/api/schemas.py
"""
Schemas Pydantic - Validation des donnees (ISO 25012)

Tous les modeles de requetes/reponses sont definis ici.
Validation stricte des entrees pour eviter injections.

@see ISO 25012 - Qualite des donnees
@see ISO 27034 - Input validation (CWE-20)
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


# ============================================================
# SCHEMAS JOUEUR
# ============================================================

class PlayerInput(BaseModel):
    """Joueur disponible pour la composition."""

    ffe_id: str = Field(..., description="ID FFE du joueur (ex: A12345)")
    elo: int = Field(..., ge=500, le=3000, description="Classement Elo")
    name: Optional[str] = Field(None, description="Nom complet")
    category: Optional[str] = Field(None, description="Categorie (Sen, Vet, Jun...)")
    is_female: bool = Field(False, description="Joueuse")
    is_reserve: bool = Field(False, description="Joueur reserve")
    is_muted: bool = Field(False, description="Joueur mute")
    matches_played: Optional[int] = Field(None, ge=0, description="Matchs joues")

    @field_validator("ffe_id")
    @classmethod
    def validate_ffe_id(cls, v: str) -> str:
        """Valide le format FFE ID (lettre + 5 chiffres)."""
        v = v.upper().strip()
        if len(v) != 6 or not v[0].isalpha() or not v[1:].isdigit():
            raise ValueError("FFE ID doit etre au format A12345")
        return v


class PlayerPrediction(BaseModel):
    """Joueur predit dans la composition adverse."""

    board: int = Field(..., ge=1, le=8, description="Numero echiquier")
    ffe_id: str
    name: Optional[str] = None
    elo: int
    probability: float = Field(..., ge=0, le=1, description="Probabilite presence")
    reasoning: Optional[str] = Field(None, description="Explication de la prediction")


class PlayerRecommendation(BaseModel):
    """Joueur recommande dans la composition optimale."""

    board: int = Field(..., ge=1, le=8)
    ffe_id: str
    name: Optional[str] = None
    elo: int
    expected_score: float = Field(..., ge=0, le=1, description="Score attendu (0-1)")
    win_probability: Optional[float] = None
    draw_probability: Optional[float] = None
    loss_probability: Optional[float] = None
    opponent: Optional[Dict[str, Any]] = Field(None, description="Adversaire predit")


# ============================================================
# SCHEMAS SCENARIO
# ============================================================

class Scenario(BaseModel):
    """Scenario de composition adverse."""

    id: int
    probability: float = Field(..., ge=0, le=1)
    lineup: List[Dict[str, Any]]


class Alternative(BaseModel):
    """Composition alternative recommandee."""

    rank: int = Field(..., ge=1)
    description: Optional[str] = None
    lineup: List[PlayerRecommendation]
    expected_score: float
    tradeoff: Optional[str] = Field(None, description="Compromis explique")


# ============================================================
# SCHEMAS CONTRAINTES
# ============================================================

class Constraints(BaseModel):
    """Contraintes de composition (regles FFE simplifiees)."""

    team_size: int = Field(8, ge=4, le=12, description="Nombre de joueurs")
    elo_descending: bool = Field(True, description="Ordre Elo decroissant")
    min_females: int = Field(0, ge=0, description="Minimum joueuses")
    max_muted: Optional[int] = Field(None, ge=0, description="Maximum mutes")


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
    competition_id: Optional[str] = Field(None, description="ID competition")
    round_number: int = Field(..., ge=1, le=20, description="Numero de ronde")
    available_players: List[PlayerInput] = Field(
        ..., min_length=1, description="Joueurs disponibles"
    )
    constraints: Optional[Constraints] = Field(default_factory=Constraints)
    options: Optional[Options] = Field(default_factory=Options)


class PredictResponse(BaseModel):
    """Reponse de prediction - Endpoint POST /predict."""

    success: bool
    version: str
    warning: Optional[str] = None

    # ALI - Prediction adverse
    predicted_opponent_lineup: List[PlayerPrediction]
    scenarios: List[Scenario] = Field(default_factory=list)

    # CE - Composition optimale
    recommended_lineup: List[PlayerRecommendation]
    expected_match_score: float
    score_range: Optional[Dict[str, float]] = None
    confidence: float = Field(..., ge=0, le=1)
    alternatives: List[Alternative] = Field(default_factory=list)

    # Metadata
    warnings: List[Dict[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Any]


# ============================================================
# SCHEMAS MODELE
# ============================================================

class ModelInfoResponse(BaseModel):
    """Information sur le modele."""

    club_id: str
    model_type: str  # "club-specific" ou "global-fallback"
    model_version: str
    last_trained_at: Optional[datetime] = None
    training_data_points: Optional[int] = None
    accuracy: Optional[float] = None
    features: Optional[List[str]] = None
    reason: Optional[str] = None  # Si fallback


# ============================================================
# SCHEMAS ENTRAINEMENT
# ============================================================

class TrainRequest(BaseModel):
    """Requete d'entrainement."""

    club_id: Optional[str] = Field(None, description="Si absent, modele global")
    force_retrain: bool = Field(False, description="Forcer meme si donnees insuffisantes")


class TrainResponse(BaseModel):
    """Reponse d'entrainement."""

    success: bool
    message: str
    job_id: str
    estimated_duration: Optional[str] = None


# ============================================================
# SCHEMAS ERREUR
# ============================================================

class ErrorDetail(BaseModel):
    """Detail d'une erreur."""

    field: Optional[str] = None
    error: str


class ErrorResponse(BaseModel):
    """Reponse d'erreur standardisee."""

    success: bool = False
    error: Dict[str, Any] = Field(
        ...,
        example={
            "code": "VALIDATION_ERROR",
            "message": "Invalid request",
            "details": [],
        },
    )
