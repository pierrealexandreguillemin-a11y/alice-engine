"""Tests for PlayerPoolLoader.load_pool exclude_players (Phase 4a T6).

Fast unit tests with a fake cache (no parquet load). Verifies the
exclude_players filter (D-P3-19 top-down pool draining) and backward
compatibility (exclude_players=None identical to before).

Document ID: ALICE-TEST-POOL-LOADER-EXCLUDE
Version: 1.0.0
"""

from __future__ import annotations

import pandas as pd

from services.ali.pool_loader import PlayerPoolLoader


class _FakeCache:
    """Minimal ALIDataCache stand-in exposing only lookup_club."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def lookup_club(self, club_id: str) -> pd.DataFrame:  # noqa: ARG002
        return self._df


def _club_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nr_ffe": [f"P{i:03d}" for i in range(n)],
            "nom": [f"Nom{i}" for i in range(n)],
            "prenom": [f"Pre{i}" for i in range(n)],
            "elo": [2000 - i * 10 for i in range(n)],
            "club": ["CLUB"] * n,
            "mute": [False] * n,
            "genre": ["M"] * n,
            "categorie": ["SE"] * n,
        }
    )


def _loader(n: int = 10) -> PlayerPoolLoader:
    return PlayerPoolLoader(_FakeCache(_club_df(n)))  # type: ignore[arg-type]


def test_exclude_none_is_backward_compatible() -> None:
    loader = _loader(10)
    pool = loader.load_pool("CLUB", "2024-11-15")
    assert len(pool) == 10
    assert {c.nr_ffe for c in pool} == {f"P{i:03d}" for i in range(10)}


def test_exclude_subset_removes_only_excluded() -> None:
    loader = _loader(10)
    excluded = {"P000", "P001", "P002", "P003", "P004"}
    pool = loader.load_pool("CLUB", "2024-11-15", exclude_players=excluded)
    assert len(pool) == 5
    assert {c.nr_ffe for c in pool}.isdisjoint(excluded)


def test_exclude_all_returns_empty() -> None:
    loader = _loader(10)
    excluded = {f"P{i:03d}" for i in range(10)}
    pool = loader.load_pool("CLUB", "2024-11-15", exclude_players=excluded)
    assert pool == []


def test_exclude_applies_after_overrides() -> None:
    loader = _loader(3)
    overrides = [{"nr_ffe": "EXTRA", "elo": 2400, "club": "CLUB"}]
    pool = loader.load_pool(
        "CLUB",
        "2024-11-15",
        overrides=overrides,
        exclude_players={"EXTRA", "P000"},
    )
    nrs = {c.nr_ffe for c in pool}
    assert "EXTRA" not in nrs  # override player is still excludable
    assert "P000" not in nrs
    assert nrs == {"P001", "P002"}
