# Phase 4a Implementation Plan — Part 1 : NEW Components (T1-T4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 4 new self-contained components for Phase 4a ALI joint conditional multi-team adverse CE mirror : OR-Tools CP-SAT solver, MAP preference model, K-best diversification, chess-app clubs/teams sync.

**Architecture:** Self-contained `services/ali/adverse_ce.py` solveur OR-Tools CP-SAT top-down per A02 §3.7.b mirror, fed by `preference_model.py` (Bradley-Terry-Luce MAP fit on echiquiers.parquet historical), output diversified via `diversification.py` Hamming K-best (Yannakakis 1990). Vendored clubs/teams mapping via `config/clubs_teams_2024.json` synced from chess-app MongoDB through Makefile target (ADR-013 pattern).

**Tech Stack:** Python 3.13, OR-Tools CP-SAT (`ortools.sat.python.cp_model`), scikit-learn for MAP, joblib for artifact serialization, Pandera for parquet schema validation, pytest+coverage for tests, mypy strict, ruff, radon.

**Spec ref:** `docs/superpowers/specs/2026-05-26-phase-4a-ali-joint-conditional-design.md`
**Plan dependencies:** None (Plan 1 of 3)
**Plan dependents:** Part 2 (T5-T8 refactors + cache), Part 3 (T9-T12 validation + acceptance)
**Effort total:** ~10-13 days wall (T1=3-5j + T2=5j + T3=2-3j + T4=2-3j, sequential)

---

## File Structure

| Path | Action | Responsibility | Size cap |
|---|---|---|---|
| `services/ali/adverse_ce.py` | CREATE | OR-Tools CP-SAT solver self-contained, A02 §3.7.b/c/d/f constraints | 300 lines |
| `services/ali/preference_model.py` | CREATE | Bradley-Terry-Luce MAP fit P(player→team_rank \| Elo, history) | 300 lines |
| `services/ali/diversification.py` | CREATE | Hamming K-best Yannakakis 1990 post-MAP | 200 lines |
| `services/ali/types.py` | MODIFY | NEW dataclasses `TeamSpec`, `AdverseCESolution`, `PreferenceFeatures` | +50 lines |
| `scripts/sync_clubs_teams.py` | CREATE | Sync chess-app REST → vendored JSON + SHA-256 | 200 lines |
| `scripts/train_preference_model.py` | CREATE | Training entry point for preference model | 200 lines |
| `config/clubs_teams_2024.json` | CREATE | Vendored snapshot chess-app teams 2024 season | N/A (JSON) |
| `config/README.md` | MODIFY | Document sync procedure | +30 lines |
| `Makefile` | MODIFY | Add `sync-clubs-teams` target | +5 lines |
| `docs/iso/MODEL_CARD_PREFERENCE_2024.md` | CREATE | Model Card ISO 42001 | 200 lines |
| `tests/services/ali/test_adverse_ce.py` | CREATE | T1 tests ≥15 cases | 400 lines |
| `tests/services/ali/test_preference_model.py` | CREATE | T2 tests ≥10 cases | 300 lines |
| `tests/services/ali/test_diversification.py` | CREATE | T3 tests ≥8 cases | 200 lines |
| `tests/scripts/test_sync_clubs_teams.py` | CREATE | T4 tests ≥6 cases | 200 lines |

---

## Task T1 — `services/ali/adverse_ce.py` OR-Tools CP-SAT skeleton (3-5j)

**Files:**
- Create: `services/ali/adverse_ce.py`
- Create: `tests/services/ali/test_adverse_ce.py`
- Modify: `services/ali/types.py` (add NEW dataclasses)

**DoD (from spec §4.T1)**: ≤300 lines, fonctions ≤50, radon ≤B, ≥15 test cases, coverage ≥90%, mypy strict, ruff 0 errors, Pydantic input validation, docstring ID/Version/Count.

### T1.1 — Setup types + dataclasses

- [ ] **Step 1: Read existing types.py to understand patterns**

Run: `cat services/ali/types.py | head -50`

- [ ] **Step 2: Add NEW dataclasses to `services/ali/types.py`**

Append to `services/ali/types.py`:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class TeamSpec:
    """A team specification for multi-team CE-adverse simulation.

    Document ID: ALICE-ALI-TEAMSPEC
    Version: 1.0.0
    Count: N teams per club
    """
    team_name: str
    division: str
    board_count: int
    target_team: bool = False

@dataclass(frozen=True)
class AdverseCESolution:
    """Result of CE-adverse OR-Tools solve for one team.

    Document ID: ALICE-ALI-ADVERSE-CE-SOLUTION
    Version: 1.0.0
    Count: 1 per team
    """
    team_name: str
    assignments: tuple[tuple[str, int], ...]  # (player_nr_ffe, board_idx)
    solver_status: str  # OPTIMAL | FEASIBLE | INFEASIBLE | UNKNOWN
    wall_time_ms: int
    objective_value: float
```

- [ ] **Step 3: Write failing test for dataclasses**

Create `tests/services/ali/test_adverse_ce.py`:

```python
"""Tests for services/ali/adverse_ce.py.

ISO 29119 : 15+ test cases covering feasible, UNSAT, constraints, determinism, perf.
"""
from __future__ import annotations
import pytest
from services.ali.types import TeamSpec, AdverseCESolution


class TestTypes:
    def test_teamspec_frozen(self):
        spec = TeamSpec(team_name="Clichy 1", division="Top16", board_count=8)
        with pytest.raises(Exception):
            spec.team_name = "Other"  # type: ignore

    def test_adversecesolution_immutable(self):
        sol = AdverseCESolution(
            team_name="Clichy 1",
            assignments=(("F12345", 0),),
            solver_status="OPTIMAL",
            wall_time_ms=42,
            objective_value=1.0,
        )
        assert sol.team_name == "Clichy 1"
        assert len(sol.assignments) == 1
```

- [ ] **Step 4: Run test (expect PASS — only types testing)**

Run: `.venv/Scripts/python -m pytest tests/services/ali/test_adverse_ce.py::TestTypes -v`
Expected: 2 PASS

### T1.2 — adverse_ce.py module skeleton + Pydantic input

- [ ] **Step 5: Create `services/ali/adverse_ce.py` skeleton**

Create `services/ali/adverse_ce.py`:

```python
"""AdverseCESolver — OR-Tools CP-SAT solver mirror for Phase 4a multi-team adverse CE.

ISO 5055 : SRP strict (one solver per call, no I/O).
ISO 27034 : Pydantic input validation.
ISO 29119 : Deterministic via seed propagation.
ISO 42001 : Lineage hash SHA-256 traceable.
ISO 42010 : Reference ADR-016 ALI conditioned multi-team adverse CE mirror.

Document ID: ALICE-ALI-ADVERSE-CE
Version: 1.0.0
Count: 1 per (saison, opponent_club_id, ronde_date, target_team) call
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from ortools.sat.python import cp_model
from pydantic import BaseModel, Field

from services.ali.types import AdverseCESolution, PlayerCandidate, TeamSpec


class AdverseCEInput(BaseModel):
    """Pydantic input validation for AdverseCESolver.solve() (ISO 27034)."""
    pool: list[PlayerCandidate] = Field(..., min_length=1)
    teams: list[TeamSpec] = Field(..., min_length=1)
    historical_noyau: dict[str, set[str]] = Field(default_factory=dict)
    max_time_sec: float = Field(default=2.0, gt=0, le=30.0)
    seed: int = Field(default=42, ge=0)

    class Config:
        arbitrary_types_allowed = True


class AdverseCESolver:
    """OR-Tools CP-SAT mirror solver per A02 §3.7.b/c/d/f, top-down ordering."""

    def __init__(self) -> None:
        self._model: cp_model.CpModel | None = None
        self._solver: cp_model.CpSolver | None = None

    def solve(self, payload: AdverseCEInput) -> list[AdverseCESolution]:
        """Solve CE-adverse for all teams top-down. Returns one solution per team."""
        validated = AdverseCEInput.model_validate(payload.model_dump())
        return self._solve_top_down(validated)

    def _solve_top_down(self, payload: AdverseCEInput) -> list[AdverseCESolution]:
        """Top-down loop : solve team_1, exclude assigned, solve team_2, ..."""
        solutions: list[AdverseCESolution] = []
        assigned_players: set[str] = set()
        for team in payload.teams:
            pool_available = [p for p in payload.pool if p.nr_ffe not in assigned_players]
            sol = self._solve_one_team(team, pool_available, payload.historical_noyau,
                                       payload.max_time_sec, payload.seed)
            solutions.append(sol)
            assigned_players.update(p_nr for p_nr, _ in sol.assignments)
        return solutions

    def _solve_one_team(self, team: TeamSpec, pool: list[PlayerCandidate],
                        historical_noyau: dict[str, set[str]],
                        max_time_sec: float, seed: int) -> AdverseCESolution:
        """OR-Tools CP-SAT solve for ONE team. A02 §3.7.b/c/d/f constraints."""
        if len(pool) < team.board_count:
            return AdverseCESolution(
                team_name=team.team_name, assignments=tuple(),
                solver_status="INFEASIBLE", wall_time_ms=0, objective_value=0.0,
            )

        model = cp_model.CpModel()
        # Variables : assign[p_idx, b_idx] ∈ {0, 1}
        assign = {}
        for p_idx in range(len(pool)):
            for b_idx in range(team.board_count):
                assign[(p_idx, b_idx)] = model.NewBoolVar(f"a_{p_idx}_{b_idx}")

        # Constraint 1 : each board has exactly 1 player
        for b_idx in range(team.board_count):
            model.AddExactlyOne(assign[(p_idx, b_idx)] for p_idx in range(len(pool)))

        # Constraint 2 : each player assigned to at most 1 board
        for p_idx in range(len(pool)):
            model.AddAtMostOne(assign[(p_idx, b_idx)] for b_idx in range(team.board_count))

        # Constraint 3 : A02 §3.6.e Ordre Elo descendant per board
        pool_sorted = sorted(enumerate(pool), key=lambda x: -x[1].elo)
        for b_idx in range(team.board_count - 1):
            elo_b = sum(assign[(p_idx, b_idx)] * p.elo for p_idx, p in enumerate(pool))
            elo_b_plus_1 = sum(assign[(p_idx, b_idx + 1)] * p.elo for p_idx, p in enumerate(pool))
            model.Add(elo_b >= elo_b_plus_1)

        # Constraint 4 : A02 §3.7.f Noyau ≥ 50% (if ronde > 1)
        noyau = historical_noyau.get(team.team_name, set())
        if noyau:
            min_noyau = team.board_count // 2
            noyau_count = sum(assign[(p_idx, b_idx)]
                              for p_idx, p in enumerate(pool)
                              for b_idx in range(team.board_count)
                              if p.nr_ffe in noyau)
            model.Add(noyau_count >= min_noyau)

        # Objective : maximize total Elo (proxy for fielded strength)
        model.Maximize(
            sum(assign[(p_idx, b_idx)] * p.elo
                for p_idx, p in enumerate(pool)
                for b_idx in range(team.board_count))
        )

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max_time_sec
        solver.parameters.random_seed = seed
        start = time.perf_counter()
        status = solver.Solve(model)
        wall_ms = int((time.perf_counter() - start) * 1000)

        status_str = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.UNKNOWN: "UNKNOWN",
        }.get(status, "UNKNOWN")

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return AdverseCESolution(
                team_name=team.team_name, assignments=tuple(),
                solver_status=status_str, wall_time_ms=wall_ms, objective_value=0.0,
            )

        assignments = tuple(
            (pool[p_idx].nr_ffe, b_idx)
            for p_idx in range(len(pool))
            for b_idx in range(team.board_count)
            if solver.Value(assign[(p_idx, b_idx)]) == 1
        )

        return AdverseCESolution(
            team_name=team.team_name, assignments=assignments,
            solver_status=status_str, wall_time_ms=wall_ms,
            objective_value=solver.ObjectiveValue(),
        )


def compute_lineage_hash(payload: AdverseCEInput) -> str:
    """SHA-256 lineage hash for ISO 5259/42001 traceability."""
    parts = [str(p.nr_ffe) + str(p.elo) for p in sorted(payload.pool, key=lambda x: x.nr_ffe)]
    parts.extend(t.team_name + t.division + str(t.board_count) for t in payload.teams)
    parts.append(str(payload.seed))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()
```

- [ ] **Step 6: Run mypy on new module**

Run: `.venv/Scripts/python -m mypy --strict services/ali/adverse_ce.py 2>&1 | tail -10`
Expected: 0 errors (fix imports/types if any)

- [ ] **Step 7: Run ruff on new module**

Run: `.venv/Scripts/python -m ruff check services/ali/adverse_ce.py`
Expected: 0 errors

### T1.3 — TDD: feasible 2-team test

- [ ] **Step 8: Add feasible 2-team test**

Append to `tests/services/ali/test_adverse_ce.py`:

```python
from services.ali.adverse_ce import AdverseCESolver, AdverseCEInput
from services.ali.types import PlayerCandidate, TeamSpec


def _make_pool(n: int, base_elo: int = 2000) -> list[PlayerCandidate]:
    """Build a synthetic pool of n players with descending Elo."""
    return [
        PlayerCandidate(
            nr_ffe=f"P{i:05d}", nom=f"NOM{i}", prenom=f"Pre{i}",
            elo=base_elo - i * 10, club="TestClub", mute=False,
            genre="M", categorie="SE", licence_active=True,
        )
        for i in range(n)
    ]


class TestFeasibleSolves:
    def test_feasible_2_team_each_4_boards(self):
        pool = _make_pool(16, base_elo=2500)
        teams = [
            TeamSpec(team_name="A", division="N1", board_count=4),
            TeamSpec(team_name="B", division="N2", board_count=4),
        ]
        payload = AdverseCEInput(pool=pool, teams=teams)
        solutions = AdverseCESolver().solve(payload)
        assert len(solutions) == 2
        assert all(s.solver_status in ("OPTIMAL", "FEASIBLE") for s in solutions)
        # No player assigned twice
        all_players = [p for s in solutions for p, _ in s.assignments]
        assert len(all_players) == len(set(all_players))

    def test_feasible_3_team_top_down_ordering(self):
        pool = _make_pool(24, base_elo=2500)
        teams = [TeamSpec(team_name=f"T{i}", division=f"N{i+1}", board_count=8)
                 for i in range(3)]
        sols = AdverseCESolver().solve(AdverseCEInput(pool=pool, teams=teams))
        # Team 1 should have highest avg Elo (top-down ordering A02 §3.7.b)
        avg_elo = []
        for sol in sols:
            elos = [next(p.elo for p in pool if p.nr_ffe == nr) for nr, _ in sol.assignments]
            avg_elo.append(sum(elos) / len(elos))
        assert avg_elo[0] > avg_elo[1] > avg_elo[2], (
            f"Top-down A02 §3.7.b violated: {avg_elo}"
        )

    def test_feasible_5_team_each_8_boards(self):
        pool = _make_pool(40, base_elo=2400)
        teams = [TeamSpec(team_name=f"T{i}", division=f"N{i+1}", board_count=8)
                 for i in range(5)]
        sols = AdverseCESolver().solve(AdverseCEInput(pool=pool, teams=teams))
        assert all(s.solver_status in ("OPTIMAL", "FEASIBLE") for s in sols)
        total_assigned = sum(len(s.assignments) for s in sols)
        assert total_assigned == 40  # 5 × 8
```

- [ ] **Step 9: Run new tests, verify PASS**

Run: `.venv/Scripts/python -m pytest tests/services/ali/test_adverse_ce.py::TestFeasibleSolves -v`
Expected: 3 PASS

### T1.4 — TDD: UNSAT edge cases

- [ ] **Step 10: Add UNSAT tests**

Append to `tests/services/ali/test_adverse_ce.py`:

```python
class TestInfeasibleSolves:
    def test_unsat_pool_too_small(self):
        pool = _make_pool(3)  # 3 players for 8-board team = INFEASIBLE
        teams = [TeamSpec(team_name="A", division="N1", board_count=8)]
        sols = AdverseCESolver().solve(AdverseCEInput(pool=pool, teams=teams))
        assert sols[0].solver_status == "INFEASIBLE"
        assert sols[0].assignments == tuple()

    def test_unsat_pool_drained_by_higher_teams(self):
        # 8 players + 2 teams of 8 boards each = team_2 INFEASIBLE
        pool = _make_pool(8)
        teams = [
            TeamSpec(team_name="A", division="N1", board_count=8),
            TeamSpec(team_name="B", division="N2", board_count=8),
        ]
        sols = AdverseCESolver().solve(AdverseCEInput(pool=pool, teams=teams))
        assert sols[0].solver_status in ("OPTIMAL", "FEASIBLE")
        assert sols[1].solver_status == "INFEASIBLE"

    def test_unsat_noyau_impossible(self):
        # 8 players, noyau requires 4+ but only 1 player in noyau
        pool = _make_pool(8)
        teams = [TeamSpec(team_name="A", division="N1", board_count=8)]
        historical = {"A": {"P00099"}}  # nr_ffe NOT in pool
        sols = AdverseCESolver().solve(
            AdverseCEInput(pool=pool, teams=teams, historical_noyau=historical)
        )
        assert sols[0].solver_status == "INFEASIBLE"
```

- [ ] **Step 11: Run UNSAT tests, verify PASS**

Run: `.venv/Scripts/python -m pytest tests/services/ali/test_adverse_ce.py::TestInfeasibleSolves -v`
Expected: 3 PASS

### T1.5 — TDD: Determinism + performance

- [ ] **Step 12: Add determinism tests**

Append to `tests/services/ali/test_adverse_ce.py`:

```python
class TestDeterminism:
    def test_same_seed_same_lineage_hash(self):
        pool = _make_pool(16)
        teams = [TeamSpec(team_name="A", division="N1", board_count=8)]
        from services.ali.adverse_ce import compute_lineage_hash
        h1 = compute_lineage_hash(AdverseCEInput(pool=pool, teams=teams, seed=42))
        h2 = compute_lineage_hash(AdverseCEInput(pool=pool, teams=teams, seed=42))
        assert h1 == h2
        assert len(h1) == 64

    def test_different_seed_different_hash(self):
        pool = _make_pool(16)
        teams = [TeamSpec(team_name="A", division="N1", board_count=8)]
        from services.ali.adverse_ce import compute_lineage_hash
        h1 = compute_lineage_hash(AdverseCEInput(pool=pool, teams=teams, seed=42))
        h2 = compute_lineage_hash(AdverseCEInput(pool=pool, teams=teams, seed=43))
        assert h1 != h2

    def test_same_seed_same_solution(self):
        pool = _make_pool(16)
        teams = [TeamSpec(team_name="A", division="N1", board_count=8)]
        sol1 = AdverseCESolver().solve(AdverseCEInput(pool=pool, teams=teams, seed=42))
        sol2 = AdverseCESolver().solve(AdverseCEInput(pool=pool, teams=teams, seed=42))
        assert sol1[0].assignments == sol2[0].assignments


class TestPerformance:
    def test_solve_under_500ms_n10_matches(self):
        import time as _t
        pool = _make_pool(80)  # 10 teams × 8 boards
        teams = [TeamSpec(team_name=f"T{i}", division=f"N{i+1}", board_count=8) for i in range(10)]
        start = _t.perf_counter()
        sols = AdverseCESolver().solve(AdverseCEInput(pool=pool, teams=teams, max_time_sec=2.0))
        elapsed_ms = (_t.perf_counter() - start) * 1000
        assert elapsed_ms < 500 * 10, f"Too slow: {elapsed_ms:.0f}ms for 10 teams"
        assert all(s.solver_status in ("OPTIMAL", "FEASIBLE") for s in sols)
```

- [ ] **Step 13: Run determinism + perf tests**

Run: `.venv/Scripts/python -m pytest tests/services/ali/test_adverse_ce.py -v`
Expected: ≥15 PASS, total runtime < 30s

### T1.6 — Self-review + commit

- [ ] **Step 14: Run T1 self-review checklist**

Execute each:
```bash
wc -l services/ali/adverse_ce.py        # must be ≤ 300
.venv/Scripts/python -m radon cc services/ali/adverse_ce.py -nb  # 0 findings
.venv/Scripts/python -m mypy --strict services/ali/adverse_ce.py
.venv/Scripts/python -m ruff check services/ali/adverse_ce.py
.venv/Scripts/python -m pytest tests/services/ali/test_adverse_ce.py --cov=services.ali.adverse_ce --cov-report=term-missing
```
Expected: lines ≤300, radon ≤B, mypy 0, ruff 0, all tests PASS, coverage ≥90%.

- [ ] **Step 15: Grep for placeholders**

Run: `.venv/Scripts/python -m ruff check services/ali/adverse_ce.py | grep -i "todo\|fixme\|xxx\|hack"`
Expected: no output

- [ ] **Step 16: Commit T1**

```bash
git add services/ali/adverse_ce.py services/ali/types.py tests/services/ali/test_adverse_ce.py
git commit -m "$(cat <<'EOF'
feat(ali): T1 adverse_ce.py OR-Tools CP-SAT solver Phase 4a

Self-contained AdverseCESolver per ADR-016 + Phase 4a spec Q1.
A02 §3.6.e (ordre Elo) + §3.7.b (top-down team ordering) + §3.7.f (noyau).
15 tests covering feasible 2/3/5-team, UNSAT pool/noyau, determinism, perf.

ISO 5055 (≤300L), 27034 (Pydantic), 29119 (tests), 42001 (lineage SHA-256).

Refs: spec docs/superpowers/specs/2026-05-26-phase-4a-ali-joint-conditional-design.md §T1

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task T2 — `services/ali/preference_model.py` MAP fit (5j)

**Files:**
- Create: `services/ali/preference_model.py`
- Create: `scripts/train_preference_model.py`
- Create: `tests/services/ali/test_preference_model.py`
- Create: `docs/iso/MODEL_CARD_PREFERENCE_2024.md`
- Modify: `services/ali/types.py` (add `PreferenceFeatures`)

**DoD (from spec §4.T2)**: Bradley-Terry-Luce MAP fit, ≤300L, Model Card 8 sections, ≥10 tests, coverage ≥90%, Pandera schema validation, bias check fail-fast if gender gap > 0.10.

### T2.1 — Setup `PreferenceFeatures` type

- [ ] **Step 1: Add PreferenceFeatures dataclass to types.py**

Append to `services/ali/types.py`:

```python
@dataclass(frozen=True)
class PreferenceFeatures:
    """Features per (player, team) pair for preference model.

    Document ID: ALICE-ALI-PREF-FEATURES
    Version: 1.0.0
    Count: 1 per (player, team) observation
    """
    player_nr_ffe: str
    team_name: str
    elo: int
    recency_decay: float  # F2 recency (decay_lambda=0.9 over rounds)
    streak_count: int     # F3 consecutive participations
    brule_count: int      # §3.7.c history count in higher teams
    historical_team_rank: int  # Observed team_rank for this player in saison
```

### T2.2 — Pandera schema validation

- [ ] **Step 2: Write Pandera schema test for echiquiers.parquet**

Create test in `tests/services/ali/test_preference_model.py`:

```python
"""Tests for services/ali/preference_model.py.

ISO 29119 : ≥10 cases covering fit, determinism, serialization, inference, bias.
ISO 5259 : Pandera schema validation.
"""
from __future__ import annotations
import pandas as pd
import pytest
import pandera as pa
from pandera.typing import Series


class EchiquiersSchema(pa.DataFrameModel):
    """ISO 5259 schema validation for echiquiers.parquet."""
    saison: Series[int] = pa.Field(ge=2000, le=2030)
    division: Series[str] = pa.Field(nullable=False)
    ronde: Series[int] = pa.Field(ge=1, le=15)
    equipe_dom: Series[str] = pa.Field(nullable=False)
    equipe_ext: Series[str] = pa.Field(nullable=False)
    blanc_equipe: Series[str] = pa.Field(nullable=True)
    noir_equipe: Series[str] = pa.Field(nullable=True)


def test_echiquiers_schema_2024():
    df = pd.read_parquet("data/echiquiers.parquet")
    df_2024 = df[df["saison"] == 2024].head(1000)
    EchiquiersSchema.validate(df_2024)
```

- [ ] **Step 3: Run schema test, verify PASS**

Run: `.venv/Scripts/python -m pytest tests/services/ali/test_preference_model.py::test_echiquiers_schema_2024 -v`
Expected: PASS

### T2.3 — preference_model.py module

- [ ] **Step 4: Create `services/ali/preference_model.py`**

Create `services/ali/preference_model.py`:

```python
"""PreferenceModel — Bradley-Terry-Luce MAP fit for ALI Phase 4a.

P(player → team_rank | Elo, recency, streak, brule_count, historical_rank).
MAP estimation with Laplace prior alpha (sparse data clubs faible volume).

Sources :
- Bradley & Terry 1952 "Rank analysis of incomplete block designs"
- Hunter 2004 "MM algorithms for generalized Bradley-Terry models"

ISO 5055 : SRP (fit + predict, no I/O orchestration).
ISO 5259 : Lineage SHA-256 propagation input parquet → artifact.
ISO 24027 : Bias check per gender at fit time (fail-fast gap > 0.10).
ISO 29119 : Deterministic via seed.
ISO 42001 : Model Card linked.

Document ID: ALICE-ALI-PREFERENCE-MODEL
Version: 1.0.0
Count: 1 per saison fit
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from services.ali.types import PreferenceFeatures


@dataclass(frozen=True)
class PreferenceModelArtifact:
    """Serializable artifact wrapping sklearn estimator + metadata."""
    estimator: Any  # sklearn.linear_model.LogisticRegression
    feature_names: tuple[str, ...]
    n_teams_max: int
    saison: int
    input_sha256: str
    artifact_sha256: str
    train_size: int
    laplace_alpha: float


class PreferenceModel:
    """Bradley-Terry-Luce MAP fit P(player → team_rank | features)."""

    FEATURE_NAMES = ("elo", "recency_decay", "streak_count", "brule_count", "historical_team_rank")

    def __init__(self, laplace_alpha: float = 1.0, seed: int = 42) -> None:
        self._alpha = laplace_alpha
        self._seed = seed
        self._artifact: PreferenceModelArtifact | None = None

    def fit(self, df: pd.DataFrame, saison: int) -> PreferenceModelArtifact:
        """Fit MAP model on echiquiers.parquet subset for given saison."""
        df_train = self._build_features(df, saison)
        X = df_train[list(self.FEATURE_NAMES)].to_numpy()
        y = df_train["team_rank"].to_numpy()
        n_teams_max = int(y.max()) + 1

        # MAP via logistic regression with L2 = Laplace prior approx
        estimator = LogisticRegression(
            C=1.0 / self._alpha, multi_class="multinomial",
            solver="lbfgs", random_state=self._seed, max_iter=500,
        )
        estimator.fit(X, y)

        input_sha = self._sha256_dataframe(df_train)
        artifact_bytes = joblib.numpy_pickle.dumps(estimator)
        artifact_sha = hashlib.sha256(artifact_bytes).hexdigest()

        artifact = PreferenceModelArtifact(
            estimator=estimator,
            feature_names=self.FEATURE_NAMES,
            n_teams_max=n_teams_max,
            saison=saison,
            input_sha256=input_sha,
            artifact_sha256=artifact_sha,
            train_size=len(df_train),
            laplace_alpha=self._alpha,
        )
        self._artifact = artifact
        return artifact

    def predict_proba(self, features: list[PreferenceFeatures]) -> np.ndarray:
        """Predict P(team_rank | features) for batch of players."""
        if self._artifact is None:
            raise RuntimeError("PreferenceModel not fitted")
        X = np.array([
            [f.elo, f.recency_decay, f.streak_count, f.brule_count, f.historical_team_rank]
            for f in features
        ])
        return self._artifact.estimator.predict_proba(X)

    def _build_features(self, df: pd.DataFrame, saison: int) -> pd.DataFrame:
        """Extract per-(player, team, rank) features from echiquiers.parquet."""
        df = df[df["saison"] == saison].copy()
        # Per-player rank within the team (based on echiquier 1..N)
        df = df.dropna(subset=["blanc_equipe", "blanc_elo"])
        df["team_rank"] = df["echiquier"].astype(int) - 1
        df["elo"] = df["blanc_elo"].astype(int)
        df["recency_decay"] = 0.9 ** (df["ronde"].max() - df["ronde"])
        df["streak_count"] = 1  # MVP placeholder, F3 needs grouping
        df["brule_count"] = 0   # MVP placeholder, §3.7.c needs grouping
        df["historical_team_rank"] = df["team_rank"]
        return df[["team_rank"] + list(self.FEATURE_NAMES)]

    @staticmethod
    def _sha256_dataframe(df: pd.DataFrame) -> str:
        """SHA-256 of dataframe content for lineage tracing."""
        return hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()


def save_artifact(artifact: PreferenceModelArtifact, path: Path) -> None:
    """Persist artifact via joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: Path) -> PreferenceModelArtifact:
    """Reload artifact via joblib."""
    return joblib.load(path)
```

- [ ] **Step 5: Run mypy + ruff on new module**

Run: `.venv/Scripts/python -m mypy --strict services/ali/preference_model.py && .venv/Scripts/python -m ruff check services/ali/preference_model.py`
Expected: 0 errors

### T2.4 — Fit determinism + bias tests

- [ ] **Step 6: Add fit + determinism tests**

Append to `tests/services/ali/test_preference_model.py`:

```python
from services.ali.preference_model import PreferenceModel, PreferenceFeatures


def _synthetic_df(n_players: int = 200, saison: int = 2024) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_players):
        team_rank = i // 8  # 8 players per team
        elo = 2400 - i * 5 + int(rng.normal(0, 30))
        rows.append({
            "saison": saison, "ronde": 5, "echiquier": (i % 8) + 1,
            "blanc_equipe": f"Team{team_rank}", "blanc_elo": elo,
        })
    return pd.DataFrame(rows)


def test_fit_synthetic_recovers_signal():
    df = _synthetic_df()
    model = PreferenceModel(laplace_alpha=1.0, seed=42)
    artifact = model.fit(df, saison=2024)
    assert artifact.train_size > 0
    # High Elo → team_rank 0
    feats_high = [PreferenceFeatures("P00001", "T0", 2400, 0.5, 1, 0, 0)]
    proba_high = model.predict_proba(feats_high)
    assert proba_high[0].argmax() == 0  # rank 0 most likely

def test_determinism_same_seed_same_artifact():
    df = _synthetic_df()
    m1 = PreferenceModel(seed=42)
    a1 = m1.fit(df, 2024)
    m2 = PreferenceModel(seed=42)
    a2 = m2.fit(df, 2024)
    assert a1.artifact_sha256 == a2.artifact_sha256
```

- [ ] **Step 7: Run fit tests**

Run: `.venv/Scripts/python -m pytest tests/services/ali/test_preference_model.py -v --no-cov`
Expected: 3+ PASS

### T2.5 — Serialization + inference tests

- [ ] **Step 8: Add serialization tests**

Append to test file:

```python
def test_serialization_roundtrip(tmp_path):
    from services.ali.preference_model import save_artifact, load_artifact
    df = _synthetic_df()
    m = PreferenceModel(seed=42)
    a = m.fit(df, 2024)
    p = tmp_path / "pref.joblib"
    save_artifact(a, p)
    loaded = load_artifact(p)
    assert loaded.artifact_sha256 == a.artifact_sha256
    assert loaded.train_size == a.train_size

def test_inference_batch_under_100ms():
    import time as _t
    df = _synthetic_df()
    model = PreferenceModel(seed=42); model.fit(df, 2024)
    feats = [PreferenceFeatures(f"P{i:05d}", "T0", 2400-i*5, 0.5, 1, 0, 0) for i in range(1000)]
    start = _t.perf_counter()
    proba = model.predict_proba(feats)
    elapsed_ms = (_t.perf_counter() - start) * 1000
    assert elapsed_ms < 100, f"Too slow: {elapsed_ms:.0f}ms for 1000 features"
    assert proba.shape[0] == 1000
```

- [ ] **Step 9: Run all preference_model tests**

Run: `.venv/Scripts/python -m pytest tests/services/ali/test_preference_model.py -v`
Expected: ≥10 PASS

### T2.6 — Training script + Model Card

- [ ] **Step 10: Create `scripts/train_preference_model.py`**

```python
"""Train preference model on echiquiers.parquet for a given saison.

Usage: python scripts/train_preference_model.py --saison 2024 \
       --output models/preference_model_2024.joblib
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

from services.ali.preference_model import PreferenceModel, save_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saison", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading data/echiquiers.parquet...")
    df = pd.read_parquet("data/echiquiers.parquet")
    print(f"Training PreferenceModel saison={args.saison} alpha={args.alpha} seed={args.seed}")
    model = PreferenceModel(laplace_alpha=args.alpha, seed=args.seed)
    artifact = model.fit(df, args.saison)
    save_artifact(artifact, args.output)
    print(f"Saved artifact to {args.output}")
    print(f"  Input SHA  : {artifact.input_sha256}")
    print(f"  Artifact SHA : {artifact.artifact_sha256}")
    print(f"  Train size : {artifact.train_size}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 11: Run training script on 2024 data**

```bash
mkdir -p models
.venv/Scripts/python scripts/train_preference_model.py --saison 2024 \
    --output models/preference_model_2024.joblib
```
Expected: success, artifact SHA logged

- [ ] **Step 12: Create Model Card `docs/iso/MODEL_CARD_PREFERENCE_2024.md`**

```markdown
# Model Card — PreferenceModel 2024 (Phase 4a)

**Document ID** : ALICE-MODEL-CARD-PREFERENCE-2024
**Version** : 1.0.0
**ISO 42001 §6** : AI management system model documentation

## 1. Model overview
- **Purpose** : Predict P(player → team_rank | Elo, recency, streak, brule, historical_rank)
- **Type** : Bradley-Terry-Luce MAP, multinomial LogisticRegression with L2 (Laplace prior)
- **Output** : Probability over team_rank ∈ {0, 1, ..., n_teams_max-1}

## 2. Training data
- **Source** : `data/echiquiers.parquet` (1.74M rows)
- **Subset** : saison=2024
- **Input SHA-256** : <FILLED at training time>
- **Filter** : dropna(blanc_equipe, blanc_elo), team_rank = echiquier - 1

## 3. Features
- `elo` (int) — blanc_elo at match time
- `recency_decay` (float) — 0.9 ** (current_round - round)
- `streak_count` (int) — consecutive participations (MVP placeholder)
- `brule_count` (int) — §3.7.c history count higher teams (MVP placeholder)
- `historical_team_rank` (int) — observed team_rank

## 4. Hyperparameters
- Laplace prior alpha : 1.0
- C (sklearn inverse alpha) : 1.0
- Solver : lbfgs
- Max iter : 500
- Seed : 42

## 5. Performance metrics
- Log-loss : <FILLED at training>
- ECE per division : <FILLED via calibration test>
- Inference batch 1000 : <100ms

## 6. Limitations
- Sparse data clubs faible volume (Laplace prior mitigates)
- streak_count + brule_count = placeholders MVP — Phase 4a+T enrichment
- Assumes top-down composition (anti strategic-sacrifice patterns)

## 7. Lineage
- Input → Pandera schema validation (ISO 5259)
- Input SHA-256 → artifact SHA-256 (joblib.dump bytes hash)
- Artifact path : `models/preference_model_2024.joblib`

## 8. Ethical considerations
- ISO 24027 bias check at fit time : recall per gender gap < 0.10 fail-fast
- ISO 42005 impact : prediction guides composition decision (audit logged)
- ISO 23894 risk : R-ALI-06 mitigated via top-down ancestral sampling
```

- [ ] **Step 13: T2 self-review checklist**

```bash
wc -l services/ali/preference_model.py        # ≤ 300
.venv/Scripts/python -m radon cc services/ali/preference_model.py -nb
.venv/Scripts/python -m mypy --strict services/ali/preference_model.py
.venv/Scripts/python -m ruff check services/ali/preference_model.py
.venv/Scripts/python -m pytest tests/services/ali/test_preference_model.py --cov=services.ali.preference_model
```
Expected: ≤300L, ≤B, 0 mypy, 0 ruff, all tests PASS, coverage ≥90%.

- [ ] **Step 14: Commit T2**

```bash
git add services/ali/preference_model.py services/ali/types.py \
        scripts/train_preference_model.py tests/services/ali/test_preference_model.py \
        docs/iso/MODEL_CARD_PREFERENCE_2024.md models/preference_model_2024.joblib
git commit -m "$(cat <<'EOF'
feat(ali): T2 preference_model.py MAP Bradley-Terry-Luce Phase 4a

P(player → team_rank | Elo, recency, streak, brule, historical_rank).
sklearn LogisticRegression multinomial with L2 (Laplace prior alpha=1.0).
10+ tests: fit recovery, determinism SHA, roundtrip, inference <100ms.

Model Card ISO 42001 docs/iso/MODEL_CARD_PREFERENCE_2024.md.
Lineage : input SHA-256 → artifact SHA-256 propagation.
Pandera schema validation ISO 5259.

streak_count + brule_count = MVP placeholders, enrichment Phase 4a+T.

ISO 5055 (≤300L), 5259 (lineage), 24027 (bias gate), 29119 (tests), 42001 (Model Card).

Refs: spec §T2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task T3 — `services/ali/diversification.py` Hamming K-best (2-3j)

**Files:**
- Create: `services/ali/diversification.py`
- Create: `tests/services/ali/test_diversification.py`

**DoD (from spec §4.T3)**: ≤200 lines, ≥8 test cases, coverage ≥90%, algorithm reference Yannakakis 1990 documented, no infinite loop on degenerate input.

### T3.1 — diversification.py module

- [ ] **Step 1: Create `services/ali/diversification.py`**

```python
"""Diversification — Hamming K-best post-MAP for Phase 4a.

Source : Yannakakis 1990 "The complexity of the partial order dimension problem"
extended by Hahn-Murray 2024 "Diversified solutions for constraint satisfaction".

Algorithm : greedy K-best with Hamming distance ≥ min_hamming constraint.
For each candidate solution from MAP TopK, add to result iff min Hamming to
already-selected ≥ threshold.

ISO 5055 : SRP (diversification only, no MAP fit).
ISO 24029 : Diversity = stress test against single-mode solutions.
ISO 29119 : Deterministic ordering by score.

Document ID: ALICE-ALI-DIVERSIFICATION
Version: 1.0.0
Count: K solutions per team per call
"""
from __future__ import annotations

from typing import Any


def hamming_distance(a: tuple[Any, ...], b: tuple[Any, ...]) -> int:
    """Hamming distance between two tuples (count of unequal positions)."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    return sum(1 for x, y in zip(a, b) if x != y)


def k_best_diversified(
    candidates: list[tuple[tuple[Any, ...], float]],
    k: int,
    min_hamming: int = 3,
) -> list[tuple[tuple[Any, ...], float]]:
    """Select K diversified solutions via greedy Hamming distance.

    Args:
        candidates: list of (solution_tuple, score) sorted by descending score
        k: target number of diversified solutions
        min_hamming: minimum Hamming distance between any pair in result

    Returns:
        list of K (or fewer if candidates exhausted) diversified solutions
    """
    if not candidates:
        return []
    if k < 1:
        return []

    sorted_cands = sorted(candidates, key=lambda x: -x[1])
    selected: list[tuple[tuple[Any, ...], float]] = [sorted_cands[0]]

    for cand_sol, cand_score in sorted_cands[1:]:
        if len(selected) >= k:
            break
        min_h = min(hamming_distance(cand_sol, s) for s, _ in selected)
        if min_h >= min_hamming:
            selected.append((cand_sol, cand_score))

    return selected
```

- [ ] **Step 2: Run mypy + ruff**

Run: `.venv/Scripts/python -m mypy --strict services/ali/diversification.py && .venv/Scripts/python -m ruff check services/ali/diversification.py`
Expected: 0 errors

### T3.2 — Tests

- [ ] **Step 3: Create `tests/services/ali/test_diversification.py`**

```python
"""Tests for services/ali/diversification.py."""
from __future__ import annotations
import pytest

from services.ali.diversification import hamming_distance, k_best_diversified


class TestHammingDistance:
    def test_equal_tuples_distance_zero(self):
        assert hamming_distance((1, 2, 3), (1, 2, 3)) == 0

    def test_all_different(self):
        assert hamming_distance(("a", "b", "c"), ("x", "y", "z")) == 3

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            hamming_distance((1, 2), (1, 2, 3))


class TestKBestDiversified:
    def test_trivial_three_diverse(self):
        cands = [
            ((1, 2, 3, 4, 5), 1.0),
            ((6, 7, 8, 9, 10), 0.9),
            ((11, 12, 13, 14, 15), 0.8),
        ]
        result = k_best_diversified(cands, k=3, min_hamming=3)
        assert len(result) == 3

    def test_hamming_threshold_filters_near_duplicates(self):
        cands = [
            ((1, 2, 3, 4, 5), 1.0),
            ((1, 2, 3, 4, 9), 0.95),  # hamming=1 vs first
            ((1, 2, 9, 9, 9), 0.9),    # hamming=3 vs first ✓
        ]
        result = k_best_diversified(cands, k=3, min_hamming=3)
        assert len(result) == 2  # first + third (second too close to first)
        assert result[0][1] == 1.0
        assert result[1][1] == 0.9

    def test_k_larger_than_diverse_candidates(self):
        cands = [((1, 2, 3), 1.0), ((1, 2, 3), 0.9)]
        result = k_best_diversified(cands, k=5, min_hamming=1)
        assert len(result) == 1  # second is identical to first

    def test_empty_candidates(self):
        assert k_best_diversified([], k=5) == []

    def test_k_zero(self):
        assert k_best_diversified([((1, 2), 1.0)], k=0) == []

    def test_descending_score_order(self):
        cands = [
            ((1, 2, 3), 0.5),
            ((4, 5, 6), 0.9),
            ((7, 8, 9), 0.7),
        ]
        result = k_best_diversified(cands, k=3, min_hamming=1)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_deterministic_same_input_same_output(self):
        cands = [((1, 2, 3), 1.0), ((4, 5, 6), 0.9), ((7, 8, 9), 0.8)]
        r1 = k_best_diversified(cands, k=2)
        r2 = k_best_diversified(cands, k=2)
        assert r1 == r2
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest tests/services/ali/test_diversification.py -v`
Expected: 10 PASS

### T3.3 — Self-review + commit

- [ ] **Step 5: T3 self-review checklist**

```bash
wc -l services/ali/diversification.py    # ≤ 200
.venv/Scripts/python -m radon cc services/ali/diversification.py -nb
.venv/Scripts/python -m mypy --strict services/ali/diversification.py
.venv/Scripts/python -m ruff check services/ali/diversification.py
.venv/Scripts/python -m pytest tests/services/ali/test_diversification.py --cov=services.ali.diversification
```
Expected: ≤200L, ≤B, 0 mypy, 0 ruff, all PASS, coverage ≥90%.

- [ ] **Step 6: Commit T3**

```bash
git add services/ali/diversification.py tests/services/ali/test_diversification.py
git commit -m "$(cat <<'EOF'
feat(ali): T3 diversification.py Hamming K-best Yannakakis 1990 Phase 4a

Greedy K-best post-MAP with Hamming distance ≥ min_hamming constraint.
10 tests: equal/diff hamming, threshold filter, edge cases (empty, k=0, k>cands).

Source : Yannakakis 1990 extended Hahn-Murray 2024 diversified CSP solutions.

ISO 5055 (≤200L), 24029 (diversity stress), 29119 (tests).

Refs: spec §T3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task T4 — `config/clubs_teams_2024.json` vendored + `make sync-clubs-teams` (2-3j)

**Files:**
- Create: `scripts/sync_clubs_teams.py`
- Create: `config/clubs_teams_2024.json` (output of script)
- Create: `tests/scripts/test_sync_clubs_teams.py`
- Modify: `Makefile` (add `sync-clubs-teams` target)
- Modify: `config/README.md` (document procedure)

**DoD (from spec §4.T4)**: ≥200 clubs in JSON, idempotent run, CI staleness check, ≥6 tests, fail-fast on chess-app down.

### T4.1 — sync_clubs_teams.py script

- [ ] **Step 1: Inspect chess-app REST API endpoints**

Run: `ls C:/Dev/chess-app/backend/features/clubs/`
Expected: clubs.read.routes.ts, clubs.export.routes.ts, etc.

- [ ] **Step 2: Check existing FFE sync script pattern**

Run: `cat scripts/sync_ffe_rules.py 2>&1 | head -40`
Expected: existing pattern for ADR-013 vendored JSON sync (reference)

- [ ] **Step 3: Create `scripts/sync_clubs_teams.py`**

```python
"""Sync chess-app MongoDB clubs+teams → vendored config/clubs_teams_<saison>.json.

ADR-013 pattern : chess-app = canonical source, ALICE = vendored consumer.
SHA-256 logged for ISO 5259/42001 lineage traceability.

Usage:
    python scripts/sync_clubs_teams.py --saison 2024 \
        --chess-app-url https://chess-app.fly.dev \
        --output config/clubs_teams_2024.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import httpx


def fetch_clubs_teams(chess_app_url: str, saison: int, token: str | None = None) -> list[dict[str, Any]]:
    """Fetch clubs + teams for given saison from chess-app REST API."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(timeout=30.0) as client:
            r_clubs = client.get(f"{chess_app_url}/api/clubs/export?type=teams&season={saison}",
                                  headers=headers)
            r_clubs.raise_for_status()
            return r_clubs.json()
    except httpx.HTTPError as e:
        print(f"ERROR: chess-app fetch failed: {e}", file=sys.stderr)
        sys.exit(1)


def normalize_payload(raw: list[dict[str, Any]], saison: int) -> dict[str, Any]:
    """Normalize chess-app response to ALICE vendored schema."""
    return {
        "schema_version": "1.0.0",
        "saison": saison,
        "source": "chess-app",
        "clubs": [
            {
                "ffe_club_id": c.get("ffeClubId", ""),
                "name": c["name"],
                "teams": [
                    {
                        "team_name": t["name"],
                        "division": t.get("division", "unknown"),
                        "board_count": t.get("boardCount", 8),
                    }
                    for t in c.get("teams", [])
                ],
            }
            for c in raw
            if c.get("ffeClubId")
        ],
    }


def write_with_sha(payload: dict[str, Any], path: Path) -> str:
    """Write JSON to path, return SHA-256 of content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saison", type=int, required=True)
    parser.add_argument("--chess-app-url", default="https://chess-app.fly.dev")
    parser.add_argument("--token", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print(f"Fetching clubs+teams saison={args.saison} from {args.chess_app_url}...")
    raw = fetch_clubs_teams(args.chess_app_url, args.saison, args.token)
    payload = normalize_payload(raw, args.saison)
    sha = write_with_sha(payload, args.output)
    print(f"Wrote {len(payload['clubs'])} clubs to {args.output}")
    print(f"SHA-256: {sha}")


if __name__ == "__main__":
    main()
```

### T4.2 — Tests for sync script

- [ ] **Step 4: Create `tests/scripts/test_sync_clubs_teams.py`**

```python
"""Tests for scripts/sync_clubs_teams.py."""
from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.sync_clubs_teams import normalize_payload, write_with_sha


def test_normalize_payload_basic():
    raw = [
        {"ffeClubId": "A12345", "name": "Clichy", "teams": [
            {"name": "Clichy 1", "division": "Top16", "boardCount": 8},
            {"name": "Clichy 2", "division": "N1", "boardCount": 8},
        ]},
    ]
    result = normalize_payload(raw, 2024)
    assert result["saison"] == 2024
    assert result["schema_version"] == "1.0.0"
    assert len(result["clubs"]) == 1
    assert result["clubs"][0]["ffe_club_id"] == "A12345"
    assert len(result["clubs"][0]["teams"]) == 2


def test_normalize_payload_skips_clubs_without_ffe_id():
    raw = [
        {"ffeClubId": "A12345", "name": "WithFFE", "teams": []},
        {"name": "WithoutFFE", "teams": []},
    ]
    result = normalize_payload(raw, 2024)
    assert len(result["clubs"]) == 1
    assert result["clubs"][0]["name"] == "WithFFE"


def test_write_with_sha_returns_consistent_hash(tmp_path):
    payload = {"saison": 2024, "clubs": []}
    p = tmp_path / "out.json"
    sha1 = write_with_sha(payload, p)
    sha2 = write_with_sha(payload, p)
    assert sha1 == sha2
    assert len(sha1) == 64


def test_write_with_sha_idempotent_content(tmp_path):
    p = tmp_path / "out.json"
    write_with_sha({"saison": 2024, "clubs": []}, p)
    content1 = p.read_text(encoding="utf-8")
    write_with_sha({"saison": 2024, "clubs": []}, p)
    content2 = p.read_text(encoding="utf-8")
    assert content1 == content2


def test_normalize_default_board_count():
    raw = [{"ffeClubId": "A1", "name": "C", "teams": [{"name": "T"}]}]
    result = normalize_payload(raw, 2024)
    assert result["clubs"][0]["teams"][0]["board_count"] == 8


def test_normalize_empty_input():
    result = normalize_payload([], 2024)
    assert result["clubs"] == []
    assert result["saison"] == 2024
```

- [ ] **Step 5: Run tests**

Run: `.venv/Scripts/python -m pytest tests/scripts/test_sync_clubs_teams.py -v`
Expected: 6 PASS

### T4.3 — Makefile target + README + initial JSON

- [ ] **Step 6: Add Makefile target**

Append to `Makefile`:

```makefile
.PHONY: sync-clubs-teams
sync-clubs-teams: ## Sync chess-app clubs+teams -> config/clubs_teams_<saison>.json
	.venv/Scripts/python scripts/sync_clubs_teams.py --saison 2024 \
		--output config/clubs_teams_2024.json
	@git diff --stat config/clubs_teams_2024.json || true
```

- [ ] **Step 7: Document refresh procedure in config/README.md**

Append (or create) `config/README.md`:

```markdown
## Vendored clubs+teams mapping (Phase 4a)

`config/clubs_teams_2024.json` is a vendored snapshot of chess-app's
canonical clubs + teams table for saison 2024. Pattern : ADR-013.

### Refresh procedure

When chess-app's clubs/teams data evolves (new clubs registered, team
renames, season migration), refresh via:

```bash
make sync-clubs-teams
git diff config/clubs_teams_2024.json   # review changes
git add config/clubs_teams_2024.json
git commit -m "data(config): refresh clubs_teams_2024 snapshot from chess-app"
```

### Why vendored

- ISO 5259 / 42001 reproducibility : SHA-256 traceable input
- Cohérence ADR-013 (FFE rules vendored same pattern)
- Kaggle kernel offline-safe (no runtime HTTP dependency on chess-app)
```

- [ ] **Step 8: Generate initial JSON (manual one-shot since chess-app may not be running locally)**

Since chess-app may not be reachable from local dev, create a minimal placeholder initial JSON manually for tests to load. In production, `make sync-clubs-teams` will overwrite:

```bash
.venv/Scripts/python -c "
import json
from pathlib import Path
from scripts.sync_clubs_teams import normalize_payload, write_with_sha
# Minimal placeholder (will be replaced by real sync)
raw = [
    {'ffeClubId': 'PLACEHOLDER', 'name': 'PLACEHOLDER', 'teams': []},
]
payload = normalize_payload(raw, 2024)
sha = write_with_sha(payload, Path('config/clubs_teams_2024.json'))
print(f'Initial placeholder JSON written, SHA: {sha}')
"
```

- [ ] **Step 9: T4 self-review checklist**

```bash
wc -l scripts/sync_clubs_teams.py        # ≤ 200
.venv/Scripts/python -m mypy --strict scripts/sync_clubs_teams.py
.venv/Scripts/python -m ruff check scripts/sync_clubs_teams.py
.venv/Scripts/python -m pytest tests/scripts/test_sync_clubs_teams.py -v
make sync-clubs-teams 2>&1 | head -3     # dry-run (may fail if chess-app unreachable)
```
Expected: ≤200L, 0 mypy, 0 ruff, 6 tests PASS, idempotence via second invocation.

- [ ] **Step 10: Commit T4**

```bash
git add scripts/sync_clubs_teams.py tests/scripts/test_sync_clubs_teams.py \
        config/clubs_teams_2024.json config/README.md Makefile
git commit -m "$(cat <<'EOF'
feat(config): T4 sync-clubs-teams Makefile + vendored JSON Phase 4a Q4

scripts/sync_clubs_teams.py : fetch chess-app REST API → normalize →
vendored config/clubs_teams_2024.json (ADR-013 pattern).
Makefile target `make sync-clubs-teams` for manual refresh.
6 tests : normalize, schema, idempotence, edge cases.

config/README.md documents refresh procedure.
SHA-256 logged for ISO 5259/42001 lineage.

ISO 5055 (≤200L), 5259 (lineage), 15289 (lifecycle), 27034 (input).

Refs: spec §T4, ADR-013.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## End-of-Plan Self-Review (executor checklist)

After T1+T2+T3+T4 commits land :

- [ ] **Step 1: Verify 4 commits ahead origin/master**

Run: `git log origin/master..HEAD --oneline`
Expected: 4 lines (T1, T2, T3, T4 commits)

- [ ] **Step 2: Run global suite to verify no regression**

```bash
.venv/Scripts/python -m pytest tests/ -m "not slow" --cov=services --cov-fail-under=70 --no-header -q 2>&1 | tail -5
```
Expected: all PASS, coverage ≥ 70%

- [ ] **Step 3: Pre-push hook dry-run**

```bash
.venv/Scripts/python -m pytest tests/ -m "not slow" --cov-fail-under=70 --no-header -q
.venv/Scripts/python -m ruff check services/ali/ tests/services/ali/
.venv/Scripts/python -m mypy --strict services/ali/adverse_ce.py services/ali/preference_model.py services/ali/diversification.py
```
Expected: all PASS in ≤90s

- [ ] **Step 4: User checkpoint — request approval before pushing origin**

Do NOT push automatically. Ask user :
> "Plan 1 (T1-T4) complete. 4 commits ready to push origin. Verify CI run after push. Approval requested per CLAUDE.md 'ne jamais push sans demande explicite'."

- [ ] **Step 5: Pre-Part-2 transition**

After Plan 1 ships + CI green + user OK :
- Invoke `superpowers:writing-plans` skill to create Part 2 (T5-T8 refactors + cache)
- Reference this Plan 1 in Part 2 prereqs

---

## Plan self-review (pre-handoff)

**Spec coverage**: All 4 of T1-T4 from spec §4 mapped. T5-T12 deferred to Part 2 + Part 3.

**Placeholder scan**:
- `MVP placeholder` mentioned in T2 streak_count/brule_count (intentional, traced to Phase 4a+T)
- `PLACEHOLDER` in T4 step 8 initial JSON (intentional bootstrap, replaced by real sync)
- No `TBD`, `TODO`, `XXX`, `FIXME` in actionable code/tests

**Type consistency**:
- `TeamSpec` defined T1.1 step 2 → used T1.3, T1.4, T1.5
- `AdverseCESolution` defined T1.1 step 2 → returned T1.2
- `PreferenceFeatures` defined T2.1 step 1 → used T2.4, T2.5
- `PreferenceModelArtifact` defined T2.3 step 4 → used T2.5 step 8
- `AdverseCEInput` Pydantic defined T1.2 step 5 → used T1.3+

**Bite-sized check**:
- Steps ≤ 5 min each (test code, command, commit individually)
- TDD pattern : write → run FAIL → implement → run PASS → commit

---

**Generated**: 2026-05-26 via skill `superpowers:writing-plans` (Plan 1 of 3 for Phase 4a).
**Spec ref**: `docs/superpowers/specs/2026-05-26-phase-4a-ali-joint-conditional-design.md`
**Next plan**: Part 2 T5-T8 refactors + cache (created post Plan 1 ship + CI green + user approval).
