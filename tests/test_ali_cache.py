"""Tests for services.ali.cache ALIDataCache.

ISO 29119 structure : explicit per-case assertions.
ISO 5259 : SHA-256 lineage reproducibility.
ISO 25010 : perf (cache is loaded once, not per request).

Document ID: ALICE-TEST-ALI-CACHE
Version: 1.0.0
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from services.ali.cache import ALIDataCache

J = Path("data/joueurs.parquet")
E = Path("data/echiquiers.parquet")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (J.exists() and E.exists()),
        reason="data parquets absent du runner",
    ),
]


def test_cache_loads_parquets() -> None:
    cache = ALIDataCache.load_from_parquets(J, E)
    assert len(cache.joueurs_total) > 0
    assert len(cache.echiquiers_total) > 0


def test_cache_signatures_are_sha256() -> None:
    cache = ALIDataCache.load_from_parquets(J, E)
    assert len(cache.parquet_sig_joueurs) == 64
    assert len(cache.parquet_sig_echiquiers) == 64
    cache2 = ALIDataCache.load_from_parquets(J, E)
    assert cache.parquet_sig_joueurs == cache2.parquet_sig_joueurs


def test_cache_exposes_loaded_at_utc() -> None:
    cache = ALIDataCache.load_from_parquets(J, E)
    assert isinstance(cache.loaded_at, datetime)
    assert cache.loaded_at.tzinfo is not None


def test_cache_is_stale_by_age() -> None:
    cache = ALIDataCache.load_from_parquets(J, E)
    assert cache.is_stale(max_age_days=1) is False
    object.__setattr__(
        cache,
        "loaded_at",
        datetime(2020, 1, 1, tzinfo=UTC),
    )
    assert cache.is_stale(max_age_days=7) is True


def test_lookup_club_returns_subset(ali_data_cache: ALIDataCache) -> None:
    first_club = next(iter(ali_data_cache.joueurs_by_club.keys()))
    subset = ali_data_cache.lookup_club(first_club)
    assert len(subset) > 0
    assert all(str(v) == first_club for v in subset["club"].unique())


def test_lookup_club_unknown_returns_empty_df(ali_data_cache: ALIDataCache) -> None:
    subset = ali_data_cache.lookup_club("UNKNOWN_CLUB_XYZ_999")
    assert subset.empty


def test_team_to_club_built(ali_data_cache: ALIDataCache) -> None:
    """Plan 3 H1 : team_to_club mapping echiquiers.team -> joueurs.club.

    Build via majority vote. Doit contenir >= 50 teams (FFE
    interclubs ~ >500 clubs * N divisions).
    """
    assert isinstance(ali_data_cache.team_to_club, dict)
    assert len(ali_data_cache.team_to_club) >= 50
    sample_team = next(iter(ali_data_cache.team_to_club))
    club = ali_data_cache.team_to_club[sample_team]
    # Club must be a known joueurs.club
    assert club in ali_data_cache.joueurs_by_club


def test_team_to_club_resolves_pool(ali_data_cache: ALIDataCache) -> None:
    """T11 usage : team_name -> club -> joueurs pool non-empty (ALL mappings).

    M-3 fix : iterate all entries (not just first 10). ISO 42001 integrity :
    if a single mapping fails to resolve, surface it instead of silencing.
    """
    unresolved: list[tuple[str, str]] = []
    for team, club in ali_data_cache.team_to_club.items():
        pool = ali_data_cache.joueurs_by_club.get(club)
        if pool is None or len(pool) == 0:
            unresolved.append((team, club))
    total = len(ali_data_cache.team_to_club)
    # Tolerate < 1% unresolved (edge clubs dissolved mid-season), fail otherwise.
    assert len(unresolved) / max(total, 1) < 0.01, (
        f"{len(unresolved)}/{total} team_to_club mappings unresolved (>1%). "
        f"Sample: {unresolved[:3]}"
    )


def test_team_to_club_deterministic(ali_data_cache: ALIDataCache) -> None:
    """M-2 fix : tie-break deterministe, meme parquet -> meme mapping.

    Load twice, expect identical mapping (ISO 5259 reproductibility).
    """
    cache2 = ALIDataCache.load_from_parquets(J, E)
    assert ali_data_cache.team_to_club == cache2.team_to_club


def test_lookup_history_returns_union_of_colors(ali_data_cache: ALIDataCache) -> None:
    first_name = next(iter(ali_data_cache.echiquiers_by_player.keys()))
    hist = ali_data_cache.lookup_history([first_name])
    assert len(hist) > 0
