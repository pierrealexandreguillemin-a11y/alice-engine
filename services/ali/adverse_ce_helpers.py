"""AdverseCE helpers - internal CP-SAT primitives (lineage / symmetry / hint / quotas).

ISO 5055 : SRP split from adverse_ce.py (300L cap, helpers <= 200L).
ISO 27034 : Inputs validated upstream by AdverseCEInput Pydantic.
ISO 29119 : Pure functions, deterministic, testable in isolation.
ISO 42001 : Lineage hash via canonical JSON (sort_keys, fixed separators).
ISO 42010 : Reference ADR-016 ALI conditioned multi-team adverse CE mirror.

Phase 4a T1 debt resolution - mute/foreign/gender quotas (A02 §3.7.h/i/j),
JSON lineage hash (collision-resistant), lex-leader symmetry breaking on
Elo ties (Margot 2010, Gent 2006), greedy Elo-descending warm-start hint
(Perron 2024 OR-Tools 9.x docs).

Document ID: ALICE-ALI-ADVERSE-CE-HELPERS
Version: 1.0.0
Count: 1 module
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover
    from ortools.sat.python import cp_model

    from services.ali.adverse_ce import AdverseCEInput
    from services.ali.types import PlayerCandidate, TeamSpec


class TeamConstraints(BaseModel):
    """Per-team A02 §3.7.h/i/j quotas (Phase 4a T1 debt 1).

    All fields optional : `None` means unconstrained (backward-compat).
    Used by AdverseCESolver to enforce mute / foreign / gender quotas as
    additional CP-SAT constraints (C5/C6/C7) on top of C1-C4.

    - `max_mutes` : A02 §3.7.j max number of mute players on team.
    - `min_fr_eu` : A02 §3.7.h min number of FR/EU players on team.
    - `min_fr_gender_female` : A02 §3.7.i min number of FR female players
      (Top16/N1/N2 brackets only; None elsewhere).
    """

    max_mutes: int | None = Field(default=None, ge=0)
    min_fr_eu: int | None = Field(default=None, ge=0)
    min_fr_gender_female: int | None = Field(default=None, ge=0)


def compute_lineage_hash(payload: AdverseCEInput) -> str:
    """SHA-256 lineage hash via canonical JSON for ISO 5259/42001 traceability.

    Switched from |-join to JSON canonical form (Phase 4a T1 debt 2) :
    collision-resistant against arbitrary `team_name` content (`|`, `:`, `,`).
    JSON uses `sort_keys=True` and fixed separators to guarantee stability.
    """
    sorted_pool = sorted(payload.pool, key=lambda p: p.nr_ffe)
    canonical = {
        "pool": [{"nr_ffe": p.nr_ffe, "elo": p.elo} for p in sorted_pool],
        "teams": [
            {"name": t.team_name, "division": t.division, "boards": t.board_count}
            for t in payload.teams
        ],
        "seed": payload.seed,
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _add_mute_quota(
    model: cp_model.CpModel,
    assign: dict[tuple[int, int], cp_model.IntVar],
    pool: list[PlayerCandidate],
    n_boards: int,
    max_mutes: int,
) -> None:
    """C5 : A02 §3.7.j mute quota."""
    mute_count = sum(
        assign[(p_idx, b_idx)]
        for p_idx in range(len(pool))
        for b_idx in range(n_boards)
        if pool[p_idx].mute is True
    )
    model.add(mute_count <= max_mutes)


def _add_foreign_quota(
    model: cp_model.CpModel,
    assign: dict[tuple[int, int], cp_model.IntVar],
    pool: list[PlayerCandidate],
    n_boards: int,
    min_fr_eu: int,
) -> None:
    """C6 : A02 §3.7.h foreign quota."""
    fr_eu_count = sum(
        assign[(p_idx, b_idx)]
        for p_idx in range(len(pool))
        for b_idx in range(n_boards)
        if pool[p_idx].is_french_eu is True
    )
    model.add(fr_eu_count >= min_fr_eu)


def _add_gender_quota(
    model: cp_model.CpModel,
    assign: dict[tuple[int, int], cp_model.IntVar],
    pool: list[PlayerCandidate],
    n_boards: int,
    min_female: int,
) -> None:
    """C7 : A02 §3.7.i FR gender quota (Top16/N1/N2)."""
    female_count = sum(
        assign[(p_idx, b_idx)]
        for p_idx in range(len(pool))
        for b_idx in range(n_boards)
        if pool[p_idx].sexe == "F"
    )
    model.add(female_count >= min_female)


def add_extended_constraints(
    model: cp_model.CpModel,
    assign: dict[tuple[int, int], cp_model.IntVar],
    pool: list[PlayerCandidate],
    n_boards: int,
    constraints: TeamConstraints,
) -> None:
    """C5/C6/C7 : A02 §3.7.j (mute) + §3.7.h (foreign) + §3.7.i (gender).

    Phase 4a T1 debt 1 : pulled from upstream pool_loader filtering into
    solver responsibility (user doctrine no-deferral, T4 work absorbed).
    No-op if all constraint fields are None (backward-compat).
    """
    if constraints.max_mutes is not None:
        _add_mute_quota(model, assign, pool, n_boards, constraints.max_mutes)
    if constraints.min_fr_eu is not None:
        _add_foreign_quota(model, assign, pool, n_boards, constraints.min_fr_eu)
    if constraints.min_fr_gender_female is not None:
        _add_gender_quota(model, assign, pool, n_boards, constraints.min_fr_gender_female)


def add_symmetry_breaking(
    model: cp_model.CpModel,
    assign: dict[tuple[int, int], cp_model.IntVar],
    pool: list[PlayerCandidate],
    n_boards: int,
) -> None:
    """Lex-leader symmetry breaking on Elo-tied players (Margot 2010, Gent 2006).

    Phase 4a T1 debt 3 : among players with identical Elo, lower-index goes to
    lower-numbered board. Encoded via big-M relaxation : only active when both
    players are assigned, otherwise relaxed.
    """
    n_players = len(pool)
    by_elo: dict[int, list[int]] = {}
    for p_idx in range(n_players):
        by_elo.setdefault(pool[p_idx].elo, []).append(p_idx)
    for tied_indices in by_elo.values():
        if len(tied_indices) <= 1:
            continue
        for ii in range(len(tied_indices) - 1):
            p_i, p_j = tied_indices[ii], tied_indices[ii + 1]
            board_i = sum(b * assign[(p_i, b)] for b in range(n_boards))
            board_j = sum(b * assign[(p_j, b)] for b in range(n_boards))
            assigned_i = sum(assign[(p_i, b)] for b in range(n_boards))
            assigned_j = sum(assign[(p_j, b)] for b in range(n_boards))
            # If both assigned (assigned_i = assigned_j = 1), force board_i <= board_j.
            # Else relax via big-M = n_boards.
            model.add(board_i <= board_j + n_boards * (2 - assigned_i - assigned_j))


def compute_greedy_hint(pool: list[PlayerCandidate], team: TeamSpec) -> dict[tuple[int, int], int]:
    """Greedy Elo-descending warm-start hint for CP-SAT (Perron 2024).

    Phase 4a T1 debt 4 : produce initial feasible assignment hint by sorting
    pool by Elo descending and assigning top board_count players to boards
    0..board_count-1. Non-binding (add_hint) : solver may improve.
    """
    n_players = len(pool)
    sorted_idx = sorted(range(n_players), key=lambda i: -pool[i].elo)
    hint: dict[tuple[int, int], int] = {
        (p_idx, b_idx): 0 for p_idx in range(n_players) for b_idx in range(team.board_count)
    }
    for b_idx in range(min(team.board_count, len(sorted_idx))):
        hint[(sorted_idx[b_idx], b_idx)] = 1
    return hint
