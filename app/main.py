"""Module: main.py - Point d'entree FastAPI ALICE Engine.

API de prediction de compositions d'equipes d'echecs.
Gestion du lifecycle applicatif et middleware.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (API gouvernance)
- ISO/IEC 42010 - Architecture (application entry point)
- ISO/IEC 25010 - System Quality (fiabilite, performance)
- ISO/IEC 27001 - Information Security (CORS, middleware)

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle.

    Startup: Chargement des modeles ML
    Shutdown: Nettoyage des ressources
    """
    # Startup
    print(f"[ALICE] Demarrage v{settings.app_version}")
    print(f"[ALICE] Mode debug: {settings.debug}")
    # TODO: Charger le modele XGBoost/CatBoost ici

    yield

    # Shutdown
    print("[ALICE] Arret en cours...")


# Creation de l'application FastAPI
app = FastAPI(
    title=settings.app_name,
    description="Adversarial Lineup Inference & Composition Engine",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routes
app.include_router(api_router)


@app.get("/", tags=["Info"])
async def root() -> dict[str, str]:
    """Return service information."""
    return {
        "service": "ALICE",
        "fullName": "Adversarial Lineup Inference & Composition Engine",
        "version": settings.app_version,
        "status": "running",
        "documentation": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, Any]:
    """Check service health for monitoring (UptimeRobot, Render).

    @see ISO 25010 - Fiabilite
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": settings.app_version,
        "checks": {
            "api": "ok",
            "model": "not_loaded",  # TODO: Verifier modele charge
            "mongodb": "not_configured",  # TODO: Verifier connexion
        },
    }
