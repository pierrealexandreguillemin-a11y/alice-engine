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
from motor.motor_asyncio import AsyncIOMotorClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.routes import router as api_router
from app.config import settings
from app.logging_config import configure_logging, get_logger
from scripts.serving.model_loader import load_models
from services.ali.cache import ALIDataCache
from services.ali.generator import ScenarioGenerator
from services.ali.history import HistoryEnricher
from services.ali.pool_loader import PlayerPoolLoader
from services.ali.verifiability import VerifiabilityClassifier
from services.audit import AuditConfig, AuditLogger
from services.feature_store import FeatureStore
from services.ffe.rule_engine import RuleEngine

# Configure logging au demarrage (ISO 27001)
configure_logging(debug=settings.debug, json_format=not settings.debug)
logger = get_logger(__name__)

# Rate limiter (ISO 27001 - protection contre abus)
limiter = Limiter(key_func=get_remote_address)


async def _init_audit_logger(app: FastAPI) -> AuditLogger | None:
    """Start async audit logger if MongoDB URI configured (ISO 27001 A.8.15)."""
    if not (settings.audit_enabled and settings.mongodb_uri):
        return None
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
    return audit_logger


def _init_model_bundle(app: FastAPI) -> None:
    """Load the stacking ModelBundle from HF cache (ISO 42001)."""
    try:
        model_bundle = load_models(
            cache_dir=Path(settings.model_cache_dir),
            hf_repo_id=settings.hf_repo_id,
            download=not settings.debug,
        )
        app.state.model_bundle = model_bundle
        logger.info("model_loaded", mode="fallback" if model_bundle.fallback_mode else "full")
    except Exception:
        logger.exception("model_load_failed")
        app.state.model_bundle = None


def _init_feature_store(app: FastAPI) -> None:
    """Load the feature store parquet bundle if present (ISO 5259)."""
    try:
        feature_store = FeatureStore(Path(settings.feature_store_path))
        feature_store.load()
        app.state.feature_store = feature_store
        logger.info("feature_store_loaded", age_hours=feature_store.age_hours)
    except Exception:
        logger.exception("feature_store_load_failed")
        app.state.feature_store = None


def _init_ali_generator(app: FastAPI) -> None:
    """Wire Phase 3 ALI ScenarioGenerator (Plan 2 Task 8). Degrades on failure.

    Loads ALIDataCache + RuleEngine + VerifiabilityClassifier, then
    instantiates ScenarioGenerator. Stores None on app.state if parquets
    or rule files are absent (Phase 2 behaviour preserved).
    """
    try:
        ali_cache = ALIDataCache.load_from_parquets(
            Path(settings.joueurs_parquet),
            Path(settings.echiquiers_parquet),
        )
        rule_engine = RuleEngine.from_json_file(
            Path(settings.ffe_rules_dir) / "a02.json",
        )
        classifier = VerifiabilityClassifier.from_json_file(
            Path(settings.ffe_rules_dir) / "alice_verifiability.json",
        )
        pool_loader = PlayerPoolLoader(ali_cache)
        history_enricher = HistoryEnricher(
            ali_cache,
            decay_lambda=settings.recency_decay_lambda,
        )
        generator = ScenarioGenerator(
            engine=rule_engine,
            classifier=classifier,
            cache=ali_cache,
            pool_loader=pool_loader,
            history_enricher=history_enricher,
            decay_lambda=settings.recency_decay_lambda,
        )
        app.state.ali_cache = ali_cache
        app.state.rule_engine = rule_engine
        app.state.verifiability_classifier = classifier
        app.state.scenario_generator = generator
        logger.info(
            "ali_ready",
            joueurs=len(ali_cache.joueurs_total),
            rules=len(rule_engine.rules),
            lineage=rule_engine.lineage_hash()[:12],
            decay_lambda=settings.recency_decay_lambda,
        )
    except Exception:
        logger.exception("ali_init_failed")
        app.state.ali_cache = None
        app.state.rule_engine = None
        app.state.verifiability_classifier = None
        app.state.scenario_generator = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle (startup wiring + shutdown cleanup)."""
    logger.info(
        "service_started",
        version=settings.app_version,
        debug=settings.debug,
        log_level=settings.log_level,
    )
    audit_logger = await _init_audit_logger(app)
    _init_model_bundle(app)
    _init_feature_store(app)
    _init_ali_generator(app)

    yield

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
    ISO 42001 - AI Management (model status reporting)
    """
    # Verification config securite
    security_checks = {
        "api_key_configured": bool(settings.api_key),
        "cors_restricted": not settings.debug or bool(settings.cors_origins),
        "debug_mode": settings.debug,
    }

    # Model bundle status (ISO 42001)
    model_bundle = getattr(request.app.state, "model_bundle", None)
    feature_store = getattr(request.app.state, "feature_store", None)
    models_loaded = model_bundle is not None
    fallback_mode: bool | None = None
    model_version: str | None = None
    if model_bundle is not None:
        fallback_mode = model_bundle.fallback_mode
        model_version = model_bundle.version

    # Statut global
    status = "healthy" if models_loaded else "degraded"

    return {
        "status": status,
        "timestamp": datetime.now(UTC).isoformat(),
        "version": settings.app_version,
        "models_loaded": models_loaded,
        "fallback_mode": fallback_mode,
        "model_version": model_version,
        "feature_store_loaded": feature_store is not None,
        "feature_store_age_hours": round(feature_store.age_hours, 1) if feature_store else None,
        "checks": {
            "api": "ok",
            "model": "loaded" if models_loaded else "not_loaded",
            "feature_store": "loaded" if feature_store is not None else "not_loaded",
            "mongodb": "not_configured",  # TODO: Verifier connexion
        },
        "security": security_checks,
    }
