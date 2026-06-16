"""Joint-conditional adverse exclusion for Phase 4a (D-P3-19).

When an opponent club fields several teams the same weekend, A02 §3.7.b forces
its players into teams by descending force. This module solves the *superior*
teams (those ranked above the target) top-down via the shipped AdverseCESolver
and returns the set of players they consume, so the target team is sampled from
the residual pool. Pure orchestration: the only side effect is the injected
CP-SAT solve (deterministic via seed).

MVP scope (Part 2a): single max-Elo allocation per superior team — captures the
dominant top-down conditioning signal (D8 Phase A acceptance §2.3). The richer
mixture over K diverse preference-scored allocations is deferred (debt
D-2026-06-16-adverse-allocation-mixture-preference-diversification, Q5 Phase 4c
contingency). Noyau A02 §3.7.f wiring deferred (debt
D-2026-06-16-adverse-noyau-wiring); empty historical_noyau here.

Q7 complete_or_nothing: any superior team that is INFEASIBLE/UNKNOWN raises
RuntimeError (no silent Phase 3 fallback).

Document ID: ALICE-ALI-JOINT-CONDITIONAL
Version: 1.0.0
Count: 1 set of excluded nr_ffe per (opponent_club, target_team) call
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from services.ali.adverse_ce import AdverseCEInput, AdverseCESolver

if TYPE_CHECKING:
    from services.ali.types import PlayerCandidate, TeamSpec

__all__ = ["compute_adverse_exclusions", "superior_teams"]

_FEASIBLE = ("OPTIMAL", "FEASIBLE")


def superior_teams(teams: list[TeamSpec], target_team: str) -> list[TeamSpec]:
    """Teams ranked above `target_team` (caller provides top-down force order).

    `teams` is ordered team_1..team_N by descending force (A02 §3.7.b), matching
    AdverseCEInput's contract. Returns every team strictly before the target.

    Raises
    ------
        ValueError: if `target_team` is not present in `teams`.

    """
    out: list[TeamSpec] = []
    for t in teams:
        if t.team_name == target_team:
            return out
        out.append(t)
    raise ValueError(f"target_team {target_team!r} not in simultaneous_teams")


def compute_adverse_exclusions(
    pool: list[PlayerCandidate],
    teams: list[TeamSpec],
    target_team: str,
    seed: int,
    max_time_sec: float = 2.0,
) -> set[str]:
    """Solve superior teams top-down; return union of consumed nr_ffe.

    Returns an empty set when the target is team_1 (no superior team).

    Raises
    ------
        ValueError: target_team absent from `teams`.
        RuntimeError: a superior team is INFEASIBLE/UNKNOWN (Q7 fail-fast).

    """
    sup = superior_teams(teams, target_team)
    if not sup:
        return set()

    payload = AdverseCEInput(
        pool=pool,
        teams=sup,
        historical_noyau={},
        max_time_sec=max_time_sec,
        seed=seed,
    )
    solutions = AdverseCESolver().solve(payload)

    excluded: set[str] = set()
    for sol in solutions:
        if sol.solver_status not in _FEASIBLE:
            raise RuntimeError(
                f"adverse CE infeasible/timeout for team {sol.team_name!r}: "
                f"{sol.solver_status} (Q7 complete_or_nothing, no Phase 3 fallback)"
            )
        excluded.update(nr for nr, _ in sol.assignments)
    return excluded
