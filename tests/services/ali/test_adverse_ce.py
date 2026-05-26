"""Tests for services.ali.adverse_ce AdverseCESolver (Phase 4a T1).

ISO 29119 : class-based structured tests + bare functions.
ISO 25059 : >= 15 test cases, >= 90% coverage on services.ali.adverse_ce.
ISO 5055 : SRP per test class.

Tests cover :
 - TestTypes (T1.1) : TeamSpec + AdverseCESolution frozen invariants.
 - TestFeasibleSolves (T1.3) : 2/3/5 teams feasible, A02 §3.7.b ordering.
 - TestInfeasibleSolves (T1.4) : pool too small, pool drained, noyau impossible.
 - TestDeterminism (T1.5a) : seed -> lineage hash + solution invariants.
 - TestPerformance (T1.5b) : 10 teams under 5000ms wall.
 - TestSolverInternals : status mapping + lineage hash + Pydantic validation.

Document ID: ALICE-TEST-ALI-ADVERSE-CE
Version: 1.0.0
Count: 17 test cases
"""

from __future__ import annotations

import dataclasses
import time

import pytest
from pydantic import ValidationError

from services.ali.adverse_ce import (
    AdverseCEInput,
    AdverseCESolver,
    compute_lineage_hash,
)
from services.ali.types import AdverseCESolution, PlayerCandidate, TeamSpec


def _make_pool(n: int, base_elo: int = 2000) -> list[PlayerCandidate]:
    """Build a synthetic pool of n PlayerCandidate with descending Elos.

    base_elo, base_elo - 10, ..., base_elo - (n-1)*10.
    All other fields default-eligible (mute=False, licence_active=True).
    """
    return [
        PlayerCandidate(
            nr_ffe=f"P{i:05d}",
            nom=f"NOM{i}",
            prenom=f"Pre{i}",
            elo=base_elo - i * 10,
            club="TestClub",
            mute=False,
            genre="M",
            categorie="SE",
            licence_active=True,
        )
        for i in range(n)
    ]


def _avg_team_elo(sol: AdverseCESolution, pool: list[PlayerCandidate]) -> float:
    """Compute average Elo of players in an AdverseCESolution."""
    elo_by_nr = {p.nr_ffe: p.elo for p in pool}
    elos = [elo_by_nr[nr] for nr, _ in sol.assignments]
    return sum(elos) / len(elos) if elos else 0.0


class TestTypes:
    """T1.1 : TeamSpec + AdverseCESolution frozen dataclass invariants."""

    def test_teamspec_frozen(self) -> None:
        t = TeamSpec(team_name="T1", division="N3", board_count=8, target_team=True)
        assert dataclasses.is_dataclass(t)
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.board_count = 16  # type: ignore[misc]

    def test_adverse_ce_solution_frozen(self) -> None:
        sol = AdverseCESolution(
            team_name="T1",
            assignments=(("P00001", 0), ("P00002", 1)),
            solver_status="OPTIMAL",
            wall_time_ms=123,
            objective_value=4000.0,
        )
        assert dataclasses.is_dataclass(sol)
        with pytest.raises(dataclasses.FrozenInstanceError):
            sol.solver_status = "INFEASIBLE"  # type: ignore[misc]


class TestFeasibleSolves:
    """T1.3 : feasible CP-SAT solves on synthetic pools."""

    def test_feasible_2_team_each_4_boards(self) -> None:
        pool = _make_pool(16)
        teams = [
            TeamSpec(team_name="T1", division="N4", board_count=4, target_team=True),
            TeamSpec(team_name="T2", division="N4", board_count=4),
        ]
        solver = AdverseCESolver()
        sols = solver.solve(AdverseCEInput(pool=pool, teams=teams))

        assert len(sols) == 2
        assert sols[0].solver_status in {"OPTIMAL", "FEASIBLE"}
        assert sols[1].solver_status in {"OPTIMAL", "FEASIBLE"}
        assert len(sols[0].assignments) == 4
        assert len(sols[1].assignments) == 4
        # No double-assignment across teams
        assigned_t1 = {nr for nr, _ in sols[0].assignments}
        assigned_t2 = {nr for nr, _ in sols[1].assignments}
        assert assigned_t1.isdisjoint(assigned_t2)

    def test_feasible_3_team_top_down_ordering(self) -> None:
        # 24 players, 3 teams x 8 boards. Top-down => T1 gets strongest pool.
        pool = _make_pool(24)
        teams = [
            TeamSpec(team_name="T1", division="N2", board_count=8, target_team=True),
            TeamSpec(team_name="T2", division="N3", board_count=8),
            TeamSpec(team_name="T3", division="N4", board_count=8),
        ]
        solver = AdverseCESolver()
        sols = solver.solve(AdverseCEInput(pool=pool, teams=teams))
        assert all(s.solver_status in {"OPTIMAL", "FEASIBLE"} for s in sols)
        # A02 §3.7.b : T1 avg Elo > T2 avg Elo > T3 avg Elo (top-down drains pool)
        avg_t1 = _avg_team_elo(sols[0], pool)
        avg_t2 = _avg_team_elo(sols[1], pool)
        avg_t3 = _avg_team_elo(sols[2], pool)
        assert avg_t1 > avg_t2 > avg_t3

    def test_feasible_5_team_each_8_boards(self) -> None:
        pool = _make_pool(40)
        teams = [TeamSpec(team_name=f"T{i}", division="N4", board_count=8) for i in range(1, 6)]
        solver = AdverseCESolver()
        sols = solver.solve(AdverseCEInput(pool=pool, teams=teams))
        assert len(sols) == 5
        assert all(s.solver_status in {"OPTIMAL", "FEASIBLE"} for s in sols)
        total_assigned = sum(len(s.assignments) for s in sols)
        assert total_assigned == 40

    def test_feasible_elo_descending_per_board(self) -> None:
        # A02 §3.6.e : within a team, Elo must descend by board index.
        pool = _make_pool(8)
        teams = [TeamSpec(team_name="T1", division="N4", board_count=4, target_team=True)]
        solver = AdverseCESolver()
        sols = solver.solve(AdverseCEInput(pool=pool, teams=teams))
        elo_by_nr = {p.nr_ffe: p.elo for p in pool}
        # Sort assignments by board_idx and verify monotonic non-increasing Elo
        sorted_by_board = sorted(sols[0].assignments, key=lambda a: a[1])
        elos = [elo_by_nr[nr] for nr, _ in sorted_by_board]
        assert all(elos[i] >= elos[i + 1] for i in range(len(elos) - 1))


class TestInfeasibleSolves:
    """T1.4 : UNSAT edge cases (pool too small, drained, noyau impossible)."""

    def test_unsat_pool_too_small(self) -> None:
        pool = _make_pool(3)
        teams = [TeamSpec(team_name="T1", division="N3", board_count=8, target_team=True)]
        solver = AdverseCESolver()
        sols = solver.solve(AdverseCEInput(pool=pool, teams=teams))
        assert sols[0].solver_status == "INFEASIBLE"
        assert sols[0].assignments == ()
        assert sols[0].objective_value == 0.0

    def test_unsat_pool_drained_by_higher_teams(self) -> None:
        # 8 players + 2 teams x 8 boards => T1 takes all 8, T2 has 0.
        pool = _make_pool(8)
        teams = [
            TeamSpec(team_name="T1", division="N3", board_count=8, target_team=True),
            TeamSpec(team_name="T2", division="N4", board_count=8),
        ]
        solver = AdverseCESolver()
        sols = solver.solve(AdverseCEInput(pool=pool, teams=teams))
        assert sols[0].solver_status in {"OPTIMAL", "FEASIBLE"}
        assert sols[1].solver_status == "INFEASIBLE"
        assert sols[1].assignments == ()

    def test_unsat_noyau_impossible(self) -> None:
        # 8 players in pool, noyau requires nr_ffe NOT in pool -> INFEASIBLE.
        pool = _make_pool(8)
        teams = [TeamSpec(team_name="T1", division="N3", board_count=8, target_team=True)]
        noyau = {"T1": {"PFAKE001", "PFAKE002", "PFAKE003", "PFAKE004"}}
        solver = AdverseCESolver()
        sols = solver.solve(AdverseCEInput(pool=pool, teams=teams, historical_noyau=noyau))
        assert sols[0].solver_status == "INFEASIBLE"


class TestDeterminism:
    """T1.5a : deterministic via seed + lineage hash."""

    def test_same_seed_same_lineage_hash(self) -> None:
        pool = _make_pool(16)
        teams = [TeamSpec(team_name="T1", division="N3", board_count=8)]
        p1 = AdverseCEInput(pool=pool, teams=teams, seed=42)
        p2 = AdverseCEInput(pool=pool, teams=teams, seed=42)
        assert compute_lineage_hash(p1) == compute_lineage_hash(p2)

    def test_different_seed_different_hash(self) -> None:
        pool = _make_pool(16)
        teams = [TeamSpec(team_name="T1", division="N3", board_count=8)]
        p1 = AdverseCEInput(pool=pool, teams=teams, seed=42)
        p2 = AdverseCEInput(pool=pool, teams=teams, seed=99)
        assert compute_lineage_hash(p1) != compute_lineage_hash(p2)

    def test_same_seed_same_solution(self) -> None:
        # Two solves with identical input + seed produce identical assignment sets.
        pool = _make_pool(16)
        teams = [TeamSpec(team_name="T1", division="N3", board_count=8)]
        payload = AdverseCEInput(pool=pool, teams=teams, seed=42)
        solver_a = AdverseCESolver()
        solver_b = AdverseCESolver()
        sols_a = solver_a.solve(payload)
        sols_b = solver_b.solve(payload)
        # Compare assigned (nr_ffe -> board_idx) maps
        map_a = dict(sols_a[0].assignments)
        map_b = dict(sols_b[0].assignments)
        assert map_a == map_b


class TestPerformance:
    """T1.5b : performance budget for 10-team multi-solve."""

    def test_solve_under_5000ms_10_teams(self) -> None:
        # 80 players, 10 teams x 8 boards. Budget : 500 ms/team => 5000 ms total.
        pool = _make_pool(80)
        teams = [TeamSpec(team_name=f"T{i}", division="N4", board_count=8) for i in range(1, 11)]
        solver = AdverseCESolver()
        start = time.perf_counter()
        sols = solver.solve(AdverseCEInput(pool=pool, teams=teams, max_time_sec=2.0))
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert len(sols) == 10
        assert all(s.solver_status in {"OPTIMAL", "FEASIBLE"} for s in sols)
        assert elapsed_ms < 5000, f"10-team solve took {elapsed_ms:.0f}ms (budget 5000ms)"


class TestSolverInternals:
    """Status mapping, Pydantic validation, lineage hash determinism."""

    def test_pydantic_rejects_empty_pool(self) -> None:
        teams = [TeamSpec(team_name="T1", division="N3", board_count=8)]
        with pytest.raises(ValidationError):
            AdverseCEInput(pool=[], teams=teams)

    def test_pydantic_rejects_empty_teams(self) -> None:
        pool = _make_pool(8)
        with pytest.raises(ValidationError):
            AdverseCEInput(pool=pool, teams=[])

    def test_pydantic_rejects_invalid_time_budget(self) -> None:
        pool = _make_pool(8)
        teams = [TeamSpec(team_name="T1", division="N3", board_count=8)]
        with pytest.raises(ValidationError):
            AdverseCEInput(pool=pool, teams=teams, max_time_sec=0.0)
        with pytest.raises(ValidationError):
            AdverseCEInput(pool=pool, teams=teams, max_time_sec=100.0)

    def test_noyau_feasible_with_50pct_in_pool(self) -> None:
        # 16 players in pool, noyau requires 4 of them (50% of 8 boards) -> FEASIBLE.
        pool = _make_pool(16)
        teams = [TeamSpec(team_name="T1", division="N3", board_count=8, target_team=True)]
        noyau = {"T1": {pool[0].nr_ffe, pool[1].nr_ffe, pool[2].nr_ffe, pool[3].nr_ffe}}
        solver = AdverseCESolver()
        sols = solver.solve(AdverseCEInput(pool=pool, teams=teams, historical_noyau=noyau))
        assert sols[0].solver_status in {"OPTIMAL", "FEASIBLE"}
        # At least 4 of the assigned players are in the noyau
        assigned_nrs = {nr for nr, _ in sols[0].assignments}
        assert len(assigned_nrs & noyau["T1"]) >= 4
