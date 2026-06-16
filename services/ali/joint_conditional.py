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
contingency).

FFE quota wiring (M1, 2026-06-16): only C5 (mute, A02 §3.7.g/j max 3) is
enforced via per-team TeamConstraints. C6 (foreign §3.7.h) and the FR component
of C7 (§3.7.i FR female) are NOT modelled — nationality is absent from the FFE
dataset (joueurs.parquet has no nationality column), so they are left
unconstrained as a documented V1 limitation (adverse Model Card; debt
D-2026-06-16-adverse-ffe-constraints-inert). Noyau A02 §3.7.f wiring deferred
(debt D-2026-06-16-adverse-noyau-wiring); empty historical_noyau here.

Q7 complete_or_nothing: any superior team that is INFEASIBLE/UNKNOWN raises
RuntimeError (no silent Phase 3 fallback).

Document ID: ALICE-ALI-JOINT-CONDITIONAL
Version: 1.0.0
Count: 1 set of excluded nr_ffe per (opponent_club, target_team) call
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from services.ali.adverse_ce import AdverseCEInput, AdverseCESolver, TeamConstraints

if TYPE_CHECKING:
    from services.ali.types import PlayerCandidate, TeamSpec

__all__ = ["compute_adverse_exclusions", "superior_teams"]

_FEASIBLE = ("OPTIMAL", "FEASIBLE")

# A02 §3.7.g max mutes per team. Mirrors the project's canonical user-side A02
# rule (scripts.ffe_rules.competition.get_regles_a02().max_mutes == 3) so the
# opponent is held under the SAME constraint as the user club (symmetry). Kept
# local (not imported) to preserve ADR-016 self-containment. Flat 3 is exact for
# N1-N3 (FFE A02 §3.7.g, web-verified) ; lower divisions diverge (DECISIONS.md
# note "N4=1" vs A02 PDF "no limit" — unreconciled, conservative approximation
# tracked in debt D-2026-06-16-adverse-ffe-constraints-inert).
_A02_MAX_MUTES = 3


def _adverse_team_constraints(teams: list[TeamSpec]) -> dict[str, TeamConstraints]:
    """Derive the wireable A02 quotas per superior team (Phase 4a M1).

    Only C5 (mute, §3.7.g/j) is enforced: nationality is absent from the FFE
    dataset, so C6 (foreign §3.7.h) and the FR component of C7 (§3.7.i FR
    female) cannot be modelled and are left unconstrained (V1 limitation,
    documented in the adverse Model Card). C4 noyau is deferred separately.
    """
    return {t.team_name: TeamConstraints(max_mutes=_A02_MAX_MUTES) for t in teams}


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
        team_constraints=_adverse_team_constraints(sup),
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
