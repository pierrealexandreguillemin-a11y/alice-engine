"""Tests for PlayerPoolLoader sexe data-flow (Phase 4a M1).

Fast unit tests with a fake cache (no parquet load). Verifies that the
loader derives `PlayerCandidate.sexe` from the parquet `genre` column,
closing the latent default-"M" gap (C7 §3.7.i gender quota read `.sexe`,
which the loader never populated -> every candidate looked male).

Document ID: ALICE-TEST-POOL-LOADER-SEXE
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


def _mixed_genre_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nr_ffe": ["P000", "P001", "P002"],
            "nom": ["A", "B", "C"],
            "prenom": ["x", "y", "z"],
            "elo": [2000, 1900, 1800],
            "club": ["CLUB"] * 3,
            "mute": [False] * 3,
            "genre": ["M", "F", "F"],
            "categorie": ["SE"] * 3,
        }
    )


def test_row_candidate_derives_sexe_from_genre() -> None:
    loader = PlayerPoolLoader(_FakeCache(_mixed_genre_df()))  # type: ignore[arg-type]
    pool = loader.load_pool("CLUB", "2024-11-15")
    by_nr = {c.nr_ffe: c for c in pool}
    assert by_nr["P000"].sexe == "M"
    assert by_nr["P001"].sexe == "F"
    assert by_nr["P002"].sexe == "F"
    # genre kept in sync (single source of truth, no divergence)
    assert all(c.sexe == c.genre for c in pool)


def test_override_candidate_derives_sexe_from_genre() -> None:
    loader = PlayerPoolLoader(_FakeCache(_mixed_genre_df()))  # type: ignore[arg-type]
    overrides = [{"nr_ffe": "EXTRA", "elo": 2400, "club": "CLUB", "genre": "F"}]
    pool = loader.load_pool("CLUB", "2024-11-15", overrides=overrides)
    extra = next(c for c in pool if c.nr_ffe == "EXTRA")
    assert extra.sexe == "F"
    assert extra.genre == "F"
