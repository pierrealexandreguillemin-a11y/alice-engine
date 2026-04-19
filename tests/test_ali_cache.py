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

pytestmark = pytest.mark.skipif(
    not (J.exists() and E.exists()),
    reason="data parquets absent du runner",
)


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
