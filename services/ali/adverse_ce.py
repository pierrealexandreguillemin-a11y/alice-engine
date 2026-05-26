"""AdverseCESolver - OR-Tools CP-SAT solver mirror for Phase 4a multi-team adverse CE.

ISO 5055 SRP strict + <=300L. ISO 27034 Pydantic. ISO 29119 deterministic seed.
ISO 42001 lineage hash (helpers). ISO 42010 ADR-016 mirror.

Self-contained per Phase 4a Q1 (Sandi Metz Squint Test). Mirrors A02 §3.6.e
(ordre Elo descendant), §3.7.b (top-down team ordering), §3.7.f (noyau >=50%),
§3.7.h/i/j (foreign/gender/mute via TeamConstraints, NEW v1.1).
Top-down ancestral sampling Bayesien (Pearl 1988) : team_1 d'abord, pool drains.

Upstream filtering : §3.7.c brule + §3.7.d same_group + licence_active = pool_loader.

Document ID: ALICE-ALI-ADVERSE-CE
Version: 1.1.0
Count: 1 per (saison, opponent_club_id, ronde_date, target_team) call
"""

from __future__ import annotations

import time

from ortools.sat.python import cp_model
from pydantic import BaseModel, ConfigDict, Field

from services.ali.adverse_ce_helpers import (
    TeamConstraints,
    add_extended_constraints,
    add_symmetry_breaking,
    compute_greedy_hint,
    compute_lineage_hash,
)
from services.ali.types import AdverseCESolution, PlayerCandidate, TeamSpec

__all__ = [
    "AdverseCEInput",
    "AdverseCESolver",
    "TeamConstraints",
    "compute_lineage_hash",
]


class AdverseCEInput(BaseModel):
    """Pydantic v2 input validation for AdverseCESolver.solve() (ISO 27034).

    `pool` : candidate players for opponent club (already filtered for
    eligibility per A02 §3.7.c brule + §3.7.d same_group upstream).
    `teams` : ordered TeamSpec list (index 0 = team_1, top priority).
    `historical_noyau` : team_name -> set of nr_ffe in noyau (A02 §3.7.f).
    `team_constraints` : team_name -> TeamConstraints (A02 §3.7.h/i/j) NEW v1.1.
    `max_time_sec` : per-team CP-SAT time budget (default 2.0s).
    `seed` : deterministic search seed (ISO 29119).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    pool: list[PlayerCandidate] = Field(..., min_length=1)
    teams: list[TeamSpec] = Field(..., min_length=1)
    historical_noyau: dict[str, set[str]] = Field(default_factory=dict)
    team_constraints: dict[str, TeamConstraints] = Field(default_factory=dict)
    max_time_sec: float = Field(default=2.0, gt=0, le=30.0)
    seed: int = Field(default=42, ge=0)


_STATUS_MAP: dict[int, str] = {
    int(cp_model.OPTIMAL): "OPTIMAL",
    int(cp_model.FEASIBLE): "FEASIBLE",
    int(cp_model.INFEASIBLE): "INFEASIBLE",
    int(cp_model.UNKNOWN): "UNKNOWN",
    int(cp_model.MODEL_INVALID): "UNKNOWN",
}


def _infeasible_solution(team_name: str, wall_ms: int = 0) -> AdverseCESolution:
    """Build an empty INFEASIBLE result (no assignments)."""
    return AdverseCESolution(
        team_name=team_name,
        assignments=(),
        solver_status="INFEASIBLE",
        wall_time_ms=wall_ms,
        objective_value=0.0,
    )


def _create_assign_vars(
    model: cp_model.CpModel, n_players: int, n_boards: int
) -> dict[tuple[int, int], cp_model.IntVar]:
    """Create boolean assignment vars `a_{p_idx}_{b_idx}` for all (player, board)."""
    return {
        (p_idx, b_idx): model.new_bool_var(f"a_{p_idx}_{b_idx}")
        for p_idx in range(n_players)
        for b_idx in range(n_boards)
    }


def _add_assignment_constraints(
    model: cp_model.CpModel,
    assign: dict[tuple[int, int], cp_model.IntVar],
    n_players: int,
    n_boards: int,
) -> None:
    """C1 (one player per board) + C2 (one board per player) constraints."""
    for b_idx in range(n_boards):
        model.add_exactly_one(assign[(p_idx, b_idx)] for p_idx in range(n_players))
    for p_idx in range(n_players):
        model.add_at_most_one(assign[(p_idx, b_idx)] for b_idx in range(n_boards))


def _add_elo_descending_constraint(
    model: cp_model.CpModel,
    assign: dict[tuple[int, int], cp_model.IntVar],
    pool: list[PlayerCandidate],
    n_boards: int,
) -> None:
    """C3 : A02 §3.6.e ordre Elo descendant per board."""
    n_players = len(pool)
    for b_idx in range(n_boards - 1):
        elo_b = sum(assign[(p_idx, b_idx)] * pool[p_idx].elo for p_idx in range(n_players))
        elo_b_plus_1 = sum(
            assign[(p_idx, b_idx + 1)] * pool[p_idx].elo for p_idx in range(n_players)
        )
        model.add(elo_b >= elo_b_plus_1)


def _add_noyau_constraint(
    model: cp_model.CpModel,
    assign: dict[tuple[int, int], cp_model.IntVar],
    pool: list[PlayerCandidate],
    n_boards: int,
    noyau: set[str],
) -> None:
    """C4 : A02 §3.7.f Noyau >= 50% (only applied if noyau non-empty, ronde > 1)."""
    if not noyau:
        return
    min_noyau = n_boards // 2
    noyau_count = sum(
        assign[(p_idx, b_idx)]
        for p_idx in range(len(pool))
        for b_idx in range(n_boards)
        if pool[p_idx].nr_ffe in noyau
    )
    model.add(noyau_count >= min_noyau)


def _set_max_elo_objective(
    model: cp_model.CpModel,
    assign: dict[tuple[int, int], cp_model.IntVar],
    pool: list[PlayerCandidate],
    n_boards: int,
) -> None:
    """Objective : maximize total fielded Elo (proxy for strength)."""
    model.maximize(
        sum(
            assign[(p_idx, b_idx)] * pool[p_idx].elo
            for p_idx in range(len(pool))
            for b_idx in range(n_boards)
        )
    )


def _build_model(
    team: TeamSpec,
    pool: list[PlayerCandidate],
    noyau: set[str],
    constraints: TeamConstraints | None,
) -> tuple[cp_model.CpModel, dict[tuple[int, int], cp_model.IntVar]]:
    """Build CP-SAT model for one team with A02 constraints C1-C7 + symmetry.

    Returns (model, assign_vars_dict). Pure constructor : no solve here.
    Delegates each constraint to a helper for ISO 5055 SRP + radon <= B.
    """
    model = cp_model.CpModel()
    n_players = len(pool)
    n_boards = team.board_count
    assign = _create_assign_vars(model, n_players, n_boards)
    _add_assignment_constraints(model, assign, n_players, n_boards)
    _add_elo_descending_constraint(model, assign, pool, n_boards)
    _add_noyau_constraint(model, assign, pool, n_boards, noyau)
    if constraints is not None:
        add_extended_constraints(model, assign, pool, n_boards, constraints)
    add_symmetry_breaking(model, assign, pool, n_boards)
    _set_max_elo_objective(model, assign, pool, n_boards)
    return model, assign


def _apply_warm_start(
    model: cp_model.CpModel,
    assign: dict[tuple[int, int], cp_model.IntVar],
    pool: list[PlayerCandidate],
    team: TeamSpec,
) -> None:
    """Inject greedy Elo-descending hint into CP-SAT model (Perron 2024)."""
    hint = compute_greedy_hint(pool, team)
    for (p_idx, b_idx), value in hint.items():
        model.add_hint(assign[(p_idx, b_idx)], value)


def _extract_assignments(
    solver: cp_model.CpSolver,
    pool: list[PlayerCandidate],
    assign: dict[tuple[int, int], cp_model.IntVar],
    n_boards: int,
) -> tuple[tuple[str, int], ...]:
    """Extract (nr_ffe, board_idx) tuples from solved CP-SAT model."""
    return tuple(
        (pool[p_idx].nr_ffe, b_idx)
        for p_idx in range(len(pool))
        for b_idx in range(n_boards)
        if solver.value(assign[(p_idx, b_idx)]) == 1
    )


class AdverseCESolver:
    """OR-Tools CP-SAT mirror solver per A02 §3.6.e/§3.7.b/c/d/f/h/i/j, top-down ordering.

    Stateless across calls : `solve()` reconstructs model each invocation
    (no shared state, ISO 5055 SRP).
    """

    def solve(self, payload: AdverseCEInput) -> list[AdverseCESolution]:
        """Solve CE-adverse for all teams top-down. One AdverseCESolution per team.

        Top-down loop : team_1 solved first (best Elos), assigned players
        excluded from pool for team_2, ... per A02 §3.7.b.
        """
        return self._solve_top_down(payload)

    def _solve_top_down(self, payload: AdverseCEInput) -> list[AdverseCESolution]:
        """Top-down ancestral sampling : assigned players drain the pool."""
        solutions: list[AdverseCESolution] = []
        assigned_players: set[str] = set()
        for team in payload.teams:
            pool_available = [p for p in payload.pool if p.nr_ffe not in assigned_players]
            sol = self._solve_one_team(
                team=team,
                pool=pool_available,
                historical_noyau=payload.historical_noyau,
                team_constraints=payload.team_constraints,
                max_time_sec=payload.max_time_sec,
                seed=payload.seed,
            )
            solutions.append(sol)
            assigned_players.update(p_nr for p_nr, _ in sol.assignments)
        return solutions

    def _solve_one_team(  # noqa: PLR0913
        self,
        team: TeamSpec,
        pool: list[PlayerCandidate],
        historical_noyau: dict[str, set[str]],
        team_constraints: dict[str, TeamConstraints],
        max_time_sec: float,
        seed: int,
    ) -> AdverseCESolution:
        """OR-Tools CP-SAT solve for ONE team. A02 §3.6.e/§3.7.b/c/d/f/h/i/j constraints."""
        if len(pool) < team.board_count:
            return _infeasible_solution(team.team_name)

        noyau = historical_noyau.get(team.team_name, set())
        constraints = team_constraints.get(team.team_name)
        model, assign = _build_model(team, pool, noyau, constraints)
        _apply_warm_start(model, assign, pool, team)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max_time_sec
        solver.parameters.random_seed = seed
        solver.parameters.num_search_workers = 1  # ISO 42001 cross-platform determinism
        start = time.perf_counter()
        status = solver.solve(model)
        wall_ms = int((time.perf_counter() - start) * 1000)

        status_str = _STATUS_MAP.get(int(status), "UNKNOWN")

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return AdverseCESolution(
                team_name=team.team_name,
                assignments=(),
                solver_status=status_str,
                wall_time_ms=wall_ms,
                objective_value=0.0,
            )

        assignments = _extract_assignments(solver, pool, assign, team.board_count)
        return AdverseCESolution(
            team_name=team.team_name,
            assignments=assignments,
            solver_status=status_str,
            wall_time_ms=wall_ms,
            objective_value=solver.objective_value,
        )
