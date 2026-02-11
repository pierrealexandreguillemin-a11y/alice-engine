"""Module: main.py - Point d'entree FastAPI ALICE Engine.

API de prediction de compositions d'equipes d'echecs.
Gestion du lifecycle applicatif et middleware.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (API gouvernance)
- ISO/IEC 42010 - Architecture (application entry point)
- ISO/IEC 25010 - System Quality (fiabilite, performance)
- ISO/IEC 27001 - Information Security (CORS, rate limiting, audit logs)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.routes import router as api_router
from app.config import settings
from app.logging_config import configure_logging, get_logger

# Configure logging au demarrage (ISO 27001)
configure_logging(debug=settings.debug, json_format=not settings.debug)
logger = get_logger(__name__)

# Rate limiter (ISO 27001 - protection contre abus)
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle.

    Startup: Chargement des modeles ML + audit logger
    Shutdown: Nettoyage des ressources
    """
    # Startup
    logger.info(
        "service_started",
        version=settings.app_version,
        debug=settings.debug,
        log_level=settings.log_level,
    )

    # Audit logger (ISO 27001:2022 A.8.15)
    audit_logger = None
    if settings.audit_enabled and settings.mongodb_uri:
        from motor.motor_asyncio import AsyncIOMotorClient  # noqa: PLC0415

        from services.audit import AuditConfig, AuditLogger  # noqa: PLC0415

        audit_client = AsyncIOMotorClient(settings.mongodb_uri)
        audit_db = audit_client[settings.mongodb_database]
        audit_config = AuditConfig(
            enabled=settings.audit_enabled,
            collection_name=settings.audit_collection,
            batch_size=settings.audit_batch_size,
            flush_interval_s=settings.audit_flush_interval_s,
        )
        audit_logger = AuditLogger(db=audit_db, config=audit_config)
        await audit_logger.start()
        app.state.audit_logger = audit_logger

    # TODO: Charger le modele XGBoost/CatBoost ici

    yield

    # Shutdown
    if audit_logger is not None:
        await audit_logger.stop()
    logger.info("service_stopped", version=settings.app_version)


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

# Rate limiter state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware - ISO 27001: Restrictif en production
_cors_origins: list[str] = settings.cors_origins if settings.cors_origins else []
if settings.debug and not _cors_origins:
    _cors_origins = ["*"]  # Dev mode only
elif not _cors_origins:
    # Production sans origins = même origine seulement (plus sécurisé)
    _cors_origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"] if not settings.debug else ["*"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"] if not settings.debug else ["*"],
)

# Inclusion des routes
app.include_router(api_router)


@app.get("/", tags=["Info"])
@limiter.limit("60/minute")
async def root(request: Request) -> dict[str, str]:
    """Return service information."""
    return {
        "service": "ALICE",
        "fullName": "Adversarial Lineup Inference & Composition Engine",
        "version": settings.app_version,
        "status": "running",
        "documentation": "/docs",
    }


@app.get("/health", tags=["Health"])
@limiter.limit("120/minute")
async def health_check(request: Request) -> dict[str, Any]:
    """Check service health for monitoring (UptimeRobot, Render).

    ISO 25010 - Fiabilite
    ISO 27001 - Verification securite (integrite modeles, config)
    """
    # Verification modeles (ISO 27001 - integrite)
    model_path = Path(settings.model_path)
    model_status = "ok" if model_path.exists() else "not_configured"

    # Verification config securite
    security_checks = {
        "api_key_configured": bool(settings.api_key),
        "cors_restricted": not settings.debug or bool(settings.cors_origins),
        "debug_mode": settings.debug,
    }

    # Statut global
    all_ok = model_status == "ok" and security_checks["api_key_configured"]
    status = "healthy" if all_ok else "degraded"

    return {
        "status": status,
        "timestamp": datetime.now(UTC).isoformat(),
        "version": settings.app_version,
        "checks": {
            "api": "ok",
            "model": model_status,
            "mongodb": "not_configured",  # TODO: Verifier connexion
        },
        "security": security_checks,
    }
