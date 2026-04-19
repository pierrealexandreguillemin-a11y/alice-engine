"""Tests PlayerPoolLoader — F7 survivor filter (Brown 1992).

ISO 29119 : fixture session-scoped ali_data_cache.
ISO 24027 : verify F7 survivor filter excludes inactive licences.

Document ID: ALICE-TEST-POOL-LOADER
Version: 1.0.0
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from services.ali.pool_loader import PlayerPoolLoader

J = Path("data/joueurs.parquet")
E = Path("data/echiquiers.parquet")

pytestmark = pytest.mark.skipif(
    not (J.exists() and E.exists()),
    reason="data parquets absent du runner",
)


def test_loader_returns_players_for_club(ali_data_cache):
    loader = PlayerPoolLoader(ali_data_cache)
    first_club = next(iter(ali_data_cache.joueurs_by_club.keys()))
    candidates = loader.load_pool(first_club, date.today().isoformat())
    assert len(candidates) > 0
    assert all(c.club == first_club for c in candidates)


def test_loader_survivor_filter_excludes_inactive(ali_data_cache):
    loader = PlayerPoolLoader(ali_data_cache)
    first_club = next(iter(ali_data_cache.joueurs_by_club.keys()))
    candidates = loader.load_pool(first_club, date.today().isoformat())
    assert all(c.licence_active for c in candidates)


def test_loader_overrides_are_appended(ali_data_cache):
    loader = PlayerPoolLoader(ali_data_cache)
    first_club = next(iter(ali_data_cache.joueurs_by_club.keys()))
    overrides = [
        {
            "nr_ffe": "OVERRIDE_999",
            "nom": "Guest",
            "prenom": "X",
            "elo": 2200,
            "club": first_club,
            "mute": False,
            "genre": "M",
            "categorie": "SE",
            "licence_active": True,
        }
    ]
    candidates = loader.load_pool(first_club, date.today().isoformat(), overrides=overrides)
    assert any(c.nr_ffe == "OVERRIDE_999" for c in candidates)


def test_loader_unknown_club_returns_empty(ali_data_cache):
    loader = PlayerPoolLoader(ali_data_cache)
    candidates = loader.load_pool("UNKNOWN_CLUB_XYZ_999", date.today().isoformat())
    assert candidates == []


def test_loader_overrides_replace_existing_by_nr_ffe(ali_data_cache):
    loader = PlayerPoolLoader(ali_data_cache)
    first_club = next(iter(ali_data_cache.joueurs_by_club.keys()))
    existing = ali_data_cache.joueurs_by_club[first_club].iloc[0]
    overrides = [
        {
            "nr_ffe": str(existing["nr_ffe"]),
            "nom": str(existing["nom"]),
            "prenom": str(existing.get("prenom", "")),
            "elo": 9999,
            "club": first_club,
            "mute": False,
            "genre": str(existing.get("genre", "M")),
            "categorie": str(existing.get("categorie", "SE")),
            "licence_active": True,
        }
    ]
    candidates = loader.load_pool(first_club, date.today().isoformat(), overrides=overrides)
    matching = [c for c in candidates if c.nr_ffe == str(existing["nr_ffe"])]
    assert len(matching) == 1
    assert matching[0].elo == 9999
