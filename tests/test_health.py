# tests/test_health.py
"""
Tests endpoint /health - ISO 29119

Premier test: Valider que l'API demarre correctement.
"""

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.mark.asyncio
async def test_health_endpoint_returns_200():
    """
    Test: GET /health retourne 200.

    @see ISO 25010 - Fiabilite
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_endpoint_returns_healthy_status():
    """
    Test: GET /health contient status=healthy.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_root_endpoint_returns_service_info():
    """
    Test: GET / retourne les informations du service.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "ALICE"
    assert "version" in data
    assert data["documentation"] == "/docs"
