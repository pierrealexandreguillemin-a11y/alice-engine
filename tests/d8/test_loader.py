"""Tests scripts/d8/loader (ISO 29119).

Document ID: ALICE-D8-TEST-LOADER
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from scripts.d8.loader import (
    compute_file_sha256,
    filter_eligible_matches,
    load_match_eligible,
)


@pytest.fixture
def tiny_echiquiers(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "saison": [2024, 2024, 2024, 2023],
            "ronde": [1, 1, 5, 1],
            "equipe_dom": ["USR", "OPP", "USR", "USR"],
            "equipe_ext": ["OPP", "USR", "OPP", "OPP"],
            "joueur_nom": ["A", "B", "C", "D"],
            "echiquier": [1, 1, 1, 1],
            "niveau": ["N3", "N3", "N3", "N3"],
        },
    )
    p = tmp_path / "echiquiers.parquet"
    df.to_parquet(p)
    return p


def test_compute_sha256_known_input() -> None:
    text = b"hello"
    expected = hashlib.sha256(text).hexdigest()
    assert len(expected) == 64


def test_filter_eligible_min_ronde_1(tiny_echiquiers: Path) -> None:
    df = pd.read_parquet(tiny_echiquiers)
    df_invalid = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "saison": [2024],
                    "ronde": [0],
                    "equipe_dom": ["X"],
                    "equipe_ext": ["Y"],
                    "joueur_nom": ["E"],
                    "echiquier": [1],
                    "niveau": ["N3"],
                },
            ),
        ],
    )
    eligible = filter_eligible_matches(df_invalid, saison=2024)
    assert all(m.ronde >= 1 for m in eligible)


def test_filter_eligible_saison_filter(tiny_echiquiers: Path) -> None:
    df = pd.read_parquet(tiny_echiquiers)
    matches_2024 = filter_eligible_matches(df, saison=2024)
    assert all(m.saison == 2024 for m in matches_2024)
    assert all(m.saison != 2023 for m in matches_2024)


def test_load_match_eligible_returns_list(tiny_echiquiers: Path) -> None:
    matches = load_match_eligible(tiny_echiquiers, saison=2024)
    assert isinstance(matches, list)
    assert len(matches) >= 1


def test_load_match_eligible_empty_saison(tiny_echiquiers: Path) -> None:
    matches = load_match_eligible(tiny_echiquiers, saison=1999)
    assert matches == []


def test_compute_file_sha256_path(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_bytes(b"alice")
    sha = compute_file_sha256(p)
    expected = hashlib.sha256(b"alice").hexdigest()
    assert sha == expected


def test_load_match_unique_match_ids(tiny_echiquiers: Path) -> None:
    matches = load_match_eligible(tiny_echiquiers, saison=2024)
    ids = [(m.saison, m.ronde, m.equipe_dom, m.equipe_ext) for m in matches]
    assert len(ids) == len(set(ids))


def test_filter_eligible_dom_ext_present() -> None:
    df = pd.DataFrame(
        {
            "saison": [2024],
            "ronde": [1],
            "equipe_dom": [None],
            "equipe_ext": ["OPP"],
            "joueur_nom": ["A"],
            "echiquier": [1],
            "niveau": ["N3"],
        },
    )
    eligible = filter_eligible_matches(df, saison=2024)
    assert eligible == []


def test_load_match_eligible_path_missing(tmp_path: Path) -> None:
    p = tmp_path / "missing.parquet"
    with pytest.raises(FileNotFoundError):
        load_match_eligible(p, saison=2024)


def test_compute_file_sha256_missing(tmp_path: Path) -> None:
    p = tmp_path / "absent.txt"
    with pytest.raises(FileNotFoundError):
        compute_file_sha256(p)


def test_filter_eligible_invalid_saison_type(tiny_echiquiers: Path) -> None:
    df = pd.read_parquet(tiny_echiquiers)
    with pytest.raises(TypeError):
        filter_eligible_matches(df, saison="2024")  # type: ignore[arg-type]


def test_load_match_eligible_returns_match_spec(tiny_echiquiers: Path) -> None:
    from scripts.d8.loader import MatchSpec

    matches = load_match_eligible(tiny_echiquiers, saison=2024)
    assert all(isinstance(m, MatchSpec) for m in matches)


def test_match_spec_frozen(tiny_echiquiers: Path) -> None:
    matches = load_match_eligible(tiny_echiquiers, saison=2024)
    if matches:
        with pytest.raises(Exception):  # noqa: B017, PT011 - frozen dataclass FrozenInstanceError
            matches[0].saison = 1999  # type: ignore[misc]


def test_filter_eligible_with_niveau_passthrough(tiny_echiquiers: Path) -> None:
    df = pd.read_parquet(tiny_echiquiers)
    matches = filter_eligible_matches(df, saison=2024)
    assert all(hasattr(m, "niveau") for m in matches)


def test_filter_eligible_dedup_pairs(tiny_echiquiers: Path) -> None:
    """Each match (saison, ronde, dom, ext) should appear once after dedup."""
    df = pd.read_parquet(tiny_echiquiers)
    matches = filter_eligible_matches(df, saison=2024)
    ids = [(m.saison, m.ronde, m.equipe_dom, m.equipe_ext) for m in matches]
    assert len(ids) == len(set(ids))
