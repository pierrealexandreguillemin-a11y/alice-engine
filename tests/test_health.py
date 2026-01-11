"""Module: test_health.py - Tests Health Endpoint.

Tests endpoint /health - validation demarrage API.

ISO Compliance:
- ISO/IEC 29119 - Software Testing (integration tests)
- ISO/IEC 25010 - System Quality (disponibilite)

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health_endpoint_returns_200():
    """Test: GET /health retourne 200.

    @see ISO 25010 - Fiabilite
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_endpoint_returns_status():
    """Test: GET /health contient status valide.

    ISO 27001: Health check retourne 'degraded' si config securite incomplete
    (api_key non configure en environnement test).
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    data = response.json()
    # Status is 'degraded' in test env (no api_key), 'healthy' in prod with full config
    assert data["status"] in ("healthy", "degraded")
    assert "timestamp" in data
    assert "version" in data
    assert "security" in data  # ISO 27001: security checks exposed


@pytest.mark.asyncio
async def test_root_endpoint_returns_service_info():
    """Test: GET / retourne les informations du service."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "ALICE"
    assert "version" in data
    assert data["documentation"] == "/docs"
