"""Tests for Phase 4a joint-conditional adverse exclusion (T5a).

Fast unit tests with fake PlayerCandidate lists (no parquet, no real cache).
Verifies superior-team selection ordering + top-down exclusion via the
shipped AdverseCESolver, and Q7 complete_or_nothing fail-fast on INFEASIBLE.

Document ID: ALICE-TEST-JOINT-CONDITIONAL
Version: 1.0.0
"""

from __future__ import annotations

import pytest

from services.ali.joint_conditional import compute_adverse_exclusions, superior_teams
from services.ali.types import PlayerCandidate, TeamSpec


def _player(nr: str, elo: int) -> PlayerCandidate:
    return PlayerCandidate(
        nr_ffe=nr,
        nom=nr,
        prenom="X",
        elo=elo,
        club="CLUB",
        mute=False,
        genre="M",
        categorie="SE",
        licence_active=True,
    )


def _pool(n: int) -> list[PlayerCandidate]:
    return [_player(f"P{i:03d}", 2400 - i * 20) for i in range(n)]


def _teams() -> list[TeamSpec]:
    return [
        TeamSpec(team_name="CLUB 1", division="N1", board_count=8),
        TeamSpec(team_name="CLUB 2", division="N3", board_count=8, target_team=True),
        TeamSpec(team_name="CLUB 3", division="D1", board_count=8),
    ]


def test_superior_teams_returns_teams_before_target() -> None:
    sup = superior_teams(_teams(), target_team="CLUB 2")
    assert [t.team_name for t in sup] == ["CLUB 1"]


def test_superior_teams_target_first_is_empty() -> None:
    sup = superior_teams(_teams(), target_team="CLUB 1")
    assert sup == []


def test_superior_teams_unknown_target_raises() -> None:
    with pytest.raises(ValueError, match="not in simultaneous_teams"):
        superior_teams(_teams(), target_team="GHOST")


def test_compute_exclusions_target_first_returns_empty() -> None:
    excl = compute_adverse_exclusions(pool=_pool(24), teams=_teams(), target_team="CLUB 1", seed=42)
    assert excl == set()


def test_compute_exclusions_excludes_one_superior_board_count() -> None:
    excl = compute_adverse_exclusions(pool=_pool(24), teams=_teams(), target_team="CLUB 2", seed=42)
    # exactly one superior team (CLUB 1), 8 boards -> 8 players consumed
    assert len(excl) == 8
    assert excl.issubset({f"P{i:03d}" for i in range(24)})


def test_compute_exclusions_two_superior_teams() -> None:
    teams = [
        TeamSpec(team_name="CLUB 1", division="N1", board_count=8),
        TeamSpec(team_name="CLUB 2", division="N2", board_count=8),
        TeamSpec(team_name="CLUB 3", division="N3", board_count=8, target_team=True),
    ]
    excl = compute_adverse_exclusions(pool=_pool(30), teams=teams, target_team="CLUB 3", seed=42)
    assert len(excl) == 16  # two superior teams x 8 boards, disjoint (top-down drain)


def test_compute_exclusions_infeasible_raises() -> None:
    # superior team needs 8 boards but only 4 players in pool -> INFEASIBLE
    teams = [
        TeamSpec(team_name="CLUB 1", division="N1", board_count=8),
        TeamSpec(team_name="CLUB 2", division="N3", board_count=4, target_team=True),
    ]
    with pytest.raises(RuntimeError, match="infeasible|INFEASIBLE"):
        compute_adverse_exclusions(pool=_pool(4), teams=teams, target_team="CLUB 2", seed=42)


def test_compute_exclusions_deterministic() -> None:
    a = compute_adverse_exclusions(pool=_pool(24), teams=_teams(), target_team="CLUB 2", seed=42)
    b = compute_adverse_exclusions(pool=_pool(24), teams=_teams(), target_team="CLUB 2", seed=42)
    assert a == b


def _mute_player(nr: str, elo: int, *, mute: bool) -> PlayerCandidate:
    return PlayerCandidate(
        nr_ffe=nr,
        nom=nr,
        prenom="X",
        elo=elo,
        club="CLUB",
        mute=mute,
        genre="M",
        categorie="SE",
        licence_active=True,
    )


def test_compute_exclusions_enforces_mute_quota_c5() -> None:
    # Superior team (4 boards) over a pool whose top-4 Elo are all mute.
    # Without C5 (A02 §3.7.j max 3 mutes) the max-Elo solve fields the 4
    # mutes; with C5 it must field at most 3 mutes + the non-mute P004.
    pool = [
        _mute_player("P000", 2000, mute=True),
        _mute_player("P001", 1980, mute=True),
        _mute_player("P002", 1960, mute=True),
        _mute_player("P003", 1940, mute=True),
        _mute_player("P004", 1920, mute=False),
    ]
    teams = [
        TeamSpec(team_name="CLUB 1", division="N1", board_count=4),
        TeamSpec(team_name="CLUB 2", division="N3", board_count=4, target_team=True),
    ]
    excl = compute_adverse_exclusions(pool=pool, teams=teams, target_team="CLUB 2", seed=42)
    assert len(excl) == 4  # one superior team x 4 boards
    mutes_excluded = sum(1 for p in pool if p.nr_ffe in excl and p.mute)
    assert mutes_excluded <= 3  # C5 mute quota enforced on the adverse mirror
    # the only non-mute is forced in (otherwise a 4th mute would be needed)
    assert "P004" in excl


def test_compute_exclusions_c5_infeasible_when_only_mutes() -> None:
    # 4 boards, 4 players ALL mute, no non-mute substitute -> the only reason
    # this is INFEASIBLE is the C5 quota (max 3 mutes). Proves C5 is load-bearing:
    # without C5 the solver would happily field 4 mutes (FEASIBLE).
    pool = [_mute_player(f"M{i:03d}", 2000 - i * 20, mute=True) for i in range(4)]
    teams = [
        TeamSpec(team_name="CLUB 1", division="N1", board_count=4),
        TeamSpec(team_name="CLUB 2", division="N3", board_count=4, target_team=True),
    ]
    with pytest.raises(RuntimeError, match="infeasible|INFEASIBLE"):
        compute_adverse_exclusions(pool=pool, teams=teams, target_team="CLUB 2", seed=42)
