# Phase 4a Part 2a — ALI Joint-Conditional Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the already-shipped CE-adverse solver (T1) into the ALI scenario generator so that, when an opponent club fields several teams the same weekend, the target team is sampled from the residual pool (after superior teams consume their players) — structurally fixing D-P3-19.

**Architecture:** Backward-compatible dispatch in `ScenarioGenerator.generate()`: `simultaneous_teams=None` keeps the unchanged Phase 3 path; a non-None list triggers a Phase 4a path that (1) solves the *superior* teams top-down via the existing `AdverseCESolver` (A02 §3.7.b mirror, max-Elo objective), (2) collects the players they consume, (3) loads the target pool with those players excluded (NEW `exclude_players` param on `PlayerPoolLoader.load_pool`), (4) runs the unchanged Phase 3 TopK+MC over the reduced pool. The Phase 4a orchestration lives in a NEW small module `services/ali/joint_conditional.py` to keep `generator.py` under the ISO 5055 300-line limit.

**Tech Stack:** Python 3.13, OR-Tools CP-SAT (via shipped `adverse_ce.py`), pandas, pytest, Pydantic v2, mypy --strict, ruff, radon.

---

## Scope

This is **Part 2a** of Phase 4a Part 2. It covers the two tasks on the **critical path for D-P3-19 acceptance** — the offline path consumed directly by the T9 local pilot and T10 Kaggle backtest (which call `generator.generate()`, not the HTTP API):

- **Task 1 (= spec T6)** — `PlayerPoolLoader.load_pool(exclude_players)`.
- **Task 2 (= spec T5)** — `generator.generate(simultaneous_teams, target_team)` BC dispatch + Phase 4a orchestration.

**Deferred to Part 2b (separate plan, after acceptance):** spec T7 (`/compose` API + Pydantic `simultaneous_teams`) and T8 (`ce_adverse_cache.py` SQLite). Rationale (writing-plans Scope Check): these serve the **production serving path**; D-P3-19 acceptance is proven **offline** (T9 pilot local + T10 Kaggle), both of which consume `generator.generate()` directly. The API + cache are required before Phase 5 deploy, not before acceptance. Splitting keeps each plan a working, testable unit.

### Design decision — adverse objective (wiring gap resolved)

Spec Q2 envisaged a *MAP-preference + diversification* objective for the adverse allocation. The shipped `AdverseCESolver` uses a **max-Elo** objective and returns **one** allocation per team; `preference_model` (T2) and `diversification` (T3) are not yet consumed.

**Decision for Part 2a:** ship the **MVP single-allocation top-down exclusion** (max-Elo). This captures the *dominant* D-P3-19 signal — the structural top-down conditioning that caused 11/13 D8 Phase A gate failures (acceptance report §2.3). This is exactly the approximation the spec itself sanctions (Q5: "Top-down ancestral sampling APPROXIMATION assumant top-down … @TODO post-MVP Option B if A.recall < 0.65 (Phase 4c contingency)").

The richer **mixture over K diverse adverse allocations** (preference-scored superior-team compositions → K exclusion sets → blended target scenarios), which is what wires `preference_model` + `diversification`, is deferred and **tracked** as debt `D-2026-06-16-adverse-allocation-mixture-preference-diversification` (target: Phase 4c contingency, triggered if T10 acceptance recall < 0.65). This is not silent: T2/T3 remain forward-built and tested; their integration point is documented.

### Noyau (A02 §3.7.f) for the adverse allocation

`AdverseCEInput.historical_noyau` is honoured by the solver only when non-empty. `services/ffe/checkers.py` holds noyau logic but `ALIDataCache` exposes no `get_noyau`. Part 2a passes **empty `historical_noyau`** (noyau constraint skipped) for the MVP and tracks the wiring as debt `D-2026-06-16-adverse-noyau-wiring` (Phase 4a fast-follow). The max-Elo top-down exclusion is valid without it; noyau only refines *which* near-Elo-tied player a superior team keeps.

---

## File Structure

**NEW files:**
- `services/ali/joint_conditional.py` — Phase 4a adverse-exclusion orchestration (pure, no I/O beyond the injected solver). Keeps `generator.py` ≤ 300 lines (ISO 5055). One responsibility: "given the club pool + simultaneous teams + target, return the set of players consumed by the superior teams."
- `tests/services/ali/test_joint_conditional.py` — fast unit tests using fake `PlayerCandidate` lists (no parquet).
- `tests/services/ali/test_generator_phase4a.py` — Phase 4a dispatch tests using a lightweight fake cache/pool (no 92 s parquet load).

**MODIFIED files:**
- `services/ali/pool_loader.py` — add `exclude_players: set[str] | None = None` to `load_pool`.
- `services/ali/generator.py` — add `simultaneous_teams` + `target_team` params + dispatch (delegates to `joint_conditional`).
- `tests/test_pool_loader.py` *(or NEW `tests/services/ali/test_pool_loader_exclude.py` if no existing file)* — exclusion tests.

**UNCHANGED (must stay green):**
- `tests/test_generator.py` — all 9 Phase 3 tests pass without modification (T9 no-regression gate).

---

## Task 1 (spec T6): `PlayerPoolLoader.load_pool(exclude_players)`

**Files:**
- Modify: `services/ali/pool_loader.py:30-53`
- Test: `tests/services/ali/test_pool_loader_exclude.py` (NEW — fast, fake cache)

- [ ] **Step 1: Write the failing tests**

Create `tests/services/ali/test_pool_loader_exclude.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/services/ali/test_pool_loader_exclude.py -v`
Expected: FAIL — `load_pool() got an unexpected keyword argument 'exclude_players'`.

- [ ] **Step 3: Add the `exclude_players` parameter**

In `services/ali/pool_loader.py`, change `load_pool` (lines 30-53) to:

```python
    def load_pool(
        self,
        club_id: str,
        round_date: str,  # noqa: ARG002 (reserve usage futur J02)
        overrides: list[dict[str, Any]] | None = None,
        exclude_players: set[str] | None = None,
    ) -> list[PlayerCandidate]:
        """Return eligible candidates. F7 survivor filter applied.

        `exclude_players` (Phase 4a D-P3-19): nr_ffe already consumed by the
        opponent club's superior teams (A02 §3.7.b top-down). Applied LAST,
        after overrides, so an injected override cannot re-introduce an
        excluded player.
        """
        df = self._cache.lookup_club(club_id)
        if df.empty and not overrides:
            return []

        candidates: dict[str, PlayerCandidate] = {}
        for _, row in df.iterrows():
            c = _row_to_candidate(row)
            if not c.licence_active:
                continue  # F7 survivor filter
            candidates[c.nr_ffe] = c

        if overrides:
            for raw in overrides:
                c = _override_to_candidate(raw)
                candidates[c.nr_ffe] = c

        exclude = exclude_players or set()
        return [c for c in candidates.values() if c.nr_ffe not in exclude]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/services/ali/test_pool_loader_exclude.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Verify no regression on existing pool_loader callers**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ -k "pool_loader" -v -m "not slow"`
Expected: all PASS (existing pool_loader tests unchanged — new param is optional).

- [ ] **Step 6: Quality gates**

Run:
```
.\.venv\Scripts\python.exe -m ruff check services/ali/pool_loader.py tests/services/ali/test_pool_loader_exclude.py
.\.venv\Scripts\python.exe -m mypy --strict services/ali/pool_loader.py
.\.venv\Scripts\python.exe -m radon cc services/ali/pool_loader.py -nb
```
Expected: ruff 0 errors, mypy 0 errors, radon no block above B.

- [ ] **Step 7: Commit**

```bash
git add services/ali/pool_loader.py tests/services/ali/test_pool_loader_exclude.py
git commit -m "feat(ali): add exclude_players to PlayerPoolLoader.load_pool (Phase 4a T6)"
```

---

## Task 2 (spec T5a): NEW `services/ali/joint_conditional.py` adverse exclusion

**Files:**
- Create: `services/ali/joint_conditional.py`
- Test: `tests/services/ali/test_joint_conditional.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/services/ali/test_joint_conditional.py`:

```python
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
    excl = compute_adverse_exclusions(
        pool=_pool(24), teams=_teams(), target_team="CLUB 1", seed=42
    )
    assert excl == set()


def test_compute_exclusions_excludes_one_superior_board_count() -> None:
    excl = compute_adverse_exclusions(
        pool=_pool(24), teams=_teams(), target_team="CLUB 2", seed=42
    )
    # exactly one superior team (CLUB 1), 8 boards -> 8 players consumed
    assert len(excl) == 8
    assert excl.issubset({f"P{i:03d}" for i in range(24)})


def test_compute_exclusions_two_superior_teams() -> None:
    teams = [
        TeamSpec(team_name="CLUB 1", division="N1", board_count=8),
        TeamSpec(team_name="CLUB 2", division="N2", board_count=8),
        TeamSpec(team_name="CLUB 3", division="N3", board_count=8, target_team=True),
    ]
    excl = compute_adverse_exclusions(
        pool=_pool(30), teams=teams, target_team="CLUB 3", seed=42
    )
    assert len(excl) == 16  # two superior teams x 8 boards, disjoint (top-down drain)


def test_compute_exclusions_infeasible_raises() -> None:
    # superior team needs 8 boards but only 4 players in pool -> INFEASIBLE
    teams = [
        TeamSpec(team_name="CLUB 1", division="N1", board_count=8),
        TeamSpec(team_name="CLUB 2", division="N3", board_count=4, target_team=True),
    ]
    with pytest.raises(RuntimeError, match="infeasible|INFEASIBLE"):
        compute_adverse_exclusions(
            pool=_pool(4), teams=teams, target_team="CLUB 2", seed=42
        )


def test_compute_exclusions_deterministic() -> None:
    a = compute_adverse_exclusions(
        pool=_pool(24), teams=_teams(), target_team="CLUB 2", seed=42
    )
    b = compute_adverse_exclusions(
        pool=_pool(24), teams=_teams(), target_team="CLUB 2", seed=42
    )
    assert a == b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/services/ali/test_joint_conditional.py -v`
Expected: FAIL — `No module named 'services.ali.joint_conditional'`.

- [ ] **Step 3: Write the module**

Create `services/ali/joint_conditional.py`:

```python
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
contingency). Noyau A02 §3.7.f wiring deferred (debt D-2026-06-16-adverse-
noyau-wiring); empty historical_noyau here.

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

_FEASIBLE = ("OPTIMAL", "FEASIBLE")


def superior_teams(teams: list[TeamSpec], target_team: str) -> list[TeamSpec]:
    """Teams ranked above `target_team` (caller provides top-down force order).

    `teams` is ordered team_1..team_N by descending force (A02 §3.7.b), matching
    AdverseCEInput's contract. Returns every team strictly before the target.

    Raises:
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

    Raises:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/services/ali/test_joint_conditional.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Quality gates + coverage**

Run:
```
.\.venv\Scripts\python.exe -m ruff check services/ali/joint_conditional.py tests/services/ali/test_joint_conditional.py
.\.venv\Scripts\python.exe -m mypy --strict services/ali/joint_conditional.py
.\.venv\Scripts\python.exe -m radon cc services/ali/joint_conditional.py -nb
.\.venv\Scripts\python.exe -m pytest tests/services/ali/test_joint_conditional.py --cov=services/ali/joint_conditional --cov-report=term-missing
```
Expected: ruff/mypy 0 errors; radon ≤ B; coverage ≥ 90 %.

- [ ] **Step 6: Commit**

```bash
git add services/ali/joint_conditional.py tests/services/ali/test_joint_conditional.py
git commit -m "feat(ali): joint-conditional adverse exclusion module (Phase 4a T5a)"
```

---

## Task 3 (spec T5b): `generator.generate(simultaneous_teams, target_team)` dispatch

**Files:**
- Modify: `services/ali/generator.py:58-73` (signature + dispatch)
- Test: `tests/services/ali/test_generator_phase4a.py` (NEW — fake cache, fast)
- Verify unchanged: `tests/test_generator.py` (Phase 3 no-regression)

- [ ] **Step 1: Write the failing tests**

Create `tests/services/ali/test_generator_phase4a.py`:

```python
"""Tests for ScenarioGenerator Phase 4a dispatch (T5b).

Uses a fake cache/pool so the dispatch is tested fast (no 92 s parquet load).
Asserts: (a) simultaneous_teams=None -> Phase 3 path unchanged (no exclusion),
(b) non-None -> joint_conditional exclusion applied to the target pool,
(c) BC default keeps the existing call sites working.

Document ID: ALICE-TEST-GENERATOR-PHASE4A
Version: 1.0.0
"""

from __future__ import annotations

from typing import Any

import pytest

from services.ali.generator import ScenarioGenerator
from services.ali.types import CompetitionContext, PlayerCandidate, TeamSpec


def _player(nr: str, elo: int) -> PlayerCandidate:
    return PlayerCandidate(
        nr_ffe=nr, nom=nr, prenom="X", elo=elo, club="CLUB", mute=False,
        genre="M", categorie="SE", licence_active=True,
    )


class _SpyPoolLoader:
    """Records the exclude_players passed to load_pool, returns a fixed pool."""

    def __init__(self, pool: list[PlayerCandidate]) -> None:
        self._pool = pool
        self.seen_exclude: set[str] | None = "UNSET"  # type: ignore[assignment]

    def load_pool(
        self,
        club_id: str,  # noqa: ARG002
        round_date: str,  # noqa: ARG002
        overrides: Any = None,  # noqa: ARG002
        exclude_players: set[str] | None = None,
    ) -> list[PlayerCandidate]:
        self.seen_exclude = exclude_players
        excl = exclude_players or set()
        return [p for p in self._pool if p.nr_ffe not in excl]


def _ctx() -> CompetitionContext:
    return CompetitionContext(
        competition_code="A02", niveau="N3", ronde=5, team_size=8,
        noyau_min=50, max_mutes=3, elo_max=None,
    )


def _teams() -> list[TeamSpec]:
    return [
        TeamSpec(team_name="CLUB 1", division="N1", board_count=8),
        TeamSpec(team_name="CLUB 2", division="N3", board_count=8, target_team=True),
    ]


def _gen_with_spy(monkeypatch: pytest.MonkeyPatch, pool: list[PlayerCandidate]):
    """Build a generator whose heavy collaborators are stubbed, with a spy pool."""
    spy = _SpyPoolLoader(pool)
    gen = ScenarioGenerator.__new__(ScenarioGenerator)  # bypass __init__ I/O
    gen._engine = None  # type: ignore[attr-defined]
    gen._classifier = None  # type: ignore[attr-defined]
    gen._cache = None  # type: ignore[attr-defined]
    gen._pool_loader = spy  # type: ignore[attr-defined]
    gen._history_enricher = None  # type: ignore[attr-defined]
    gen._decay_lambda = 0.9  # type: ignore[attr-defined]

    # Stub the Phase 3 pipeline after pool load: return a sentinel ScenarioSet.
    from services.ali import generator as gmod

    def _fake_phase3(self: Any, pool_arg: list[PlayerCandidate], **kw: Any) -> Any:
        return {"pool_size": len(pool_arg)}

    monkeypatch.setattr(gmod.ScenarioGenerator, "_run_phase3", _fake_phase3, raising=False)
    return gen, spy


def test_bc_none_passes_empty_exclusion(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = [_player(f"P{i:03d}", 2400 - i * 20) for i in range(20)]
    gen, spy = _gen_with_spy(monkeypatch, pool)
    out = gen.generate(
        opponent_club_id="CLUB", round_date="2024-11-15", context=_ctx(),
        saison=2024, current_round=5, nb_rondes_total=11,
        simultaneous_teams=None,
    )
    assert spy.seen_exclude in (None, set())
    assert out["pool_size"] == 20


def test_phase4a_excludes_superior_team_players(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = [_player(f"P{i:03d}", 2400 - i * 20) for i in range(24)]
    gen, spy = _gen_with_spy(monkeypatch, pool)
    out = gen.generate(
        opponent_club_id="CLUB", round_date="2024-11-15", context=_ctx(),
        saison=2024, current_round=5, nb_rondes_total=11,
        simultaneous_teams=_teams(), target_team="CLUB 2",
    )
    assert spy.seen_exclude is not None
    assert len(spy.seen_exclude) == 8  # CLUB 1 consumed 8 players
    assert out["pool_size"] == 16  # 24 - 8


def test_phase4a_target_first_excludes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = [_player(f"P{i:03d}", 2400 - i * 20) for i in range(24)]
    gen, spy = _gen_with_spy(monkeypatch, pool)
    out = gen.generate(
        opponent_club_id="CLUB", round_date="2024-11-15", context=_ctx(),
        saison=2024, current_round=5, nb_rondes_total=11,
        simultaneous_teams=_teams(), target_team="CLUB 1",
    )
    assert spy.seen_exclude == set()
    assert out["pool_size"] == 24
```

> **Note for implementer:** Task 3 introduces a private `_run_phase3(self, pool, *, context, saison, current_round, nb_rondes_total, n_topk, n_mc_pairs, seed, opponent_club_id, round_date)` method by extracting the *existing* body of `generate()` (steps 2-9, lines 79-144) verbatim into it. The fast tests stub `_run_phase3`; the unchanged `tests/test_generator.py` exercises the real path end-to-end. This extraction also keeps `generate()` itself short (ISO 5055 ≤ 50 lines).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/services/ali/test_generator_phase4a.py -v`
Expected: FAIL — `generate() got an unexpected keyword argument 'simultaneous_teams'`.

- [ ] **Step 3: Extract Phase 3 body into `_run_phase3` and add the dispatch**

In `services/ali/generator.py`:

(a) Add imports near the top (after existing imports):

```python
from services.ali.joint_conditional import compute_adverse_exclusions
```

and add `TeamSpec` to the `TYPE_CHECKING` block:

```python
    from services.ali.types import CompetitionContext, PlayerCandidate, TeamSpec
```

(b) Replace the current `generate()` (lines 58-144) with a thin dispatcher plus an extracted `_run_phase3`. The new `generate()`:

```python
    def generate(  # noqa: PLR0913
        self,
        opponent_club_id: str,
        round_date: str,
        context: CompetitionContext,
        saison: int,
        current_round: int,
        nb_rondes_total: int,
        overrides: list[dict[str, Any]] | None = None,
        n_topk: int = 10,
        n_mc_pairs: int = 5,
        seed: int = 42,
        simultaneous_teams: list[TeamSpec] | None = None,
        target_team: str | None = None,
    ) -> ScenarioSet:
        """Generate a ScenarioSet (20 scenarios).

        Phase 3 (`simultaneous_teams is None`): sample the target club's full
        pool. Phase 4a (`simultaneous_teams` provided): exclude players consumed
        by the club's superior teams (A02 §3.7.b top-down), then run the same
        Phase 3 pipeline over the residual pool (D-P3-19 fix).
        """
        exclude_players: set[str] = set()
        if simultaneous_teams is not None:
            if target_team is None:
                raise ValueError("target_team required when simultaneous_teams is set")
            full_pool = self._pool_loader.load_pool(
                opponent_club_id, round_date, overrides
            )
            exclude_players = compute_adverse_exclusions(
                pool=full_pool,
                teams=simultaneous_teams,
                target_team=target_team,
                seed=seed,
            )

        pool = self._pool_loader.load_pool(
            opponent_club_id, round_date, overrides, exclude_players=exclude_players
        )
        if len(pool) < context.team_size:
            raise ValueError(
                f"pool too small for {opponent_club_id}: "
                f"{len(pool)} < {context.team_size}",
            )
        return self._run_phase3(
            pool,
            context=context,
            saison=saison,
            current_round=current_round,
            nb_rondes_total=nb_rondes_total,
            n_topk=n_topk,
            n_mc_pairs=n_mc_pairs,
            seed=seed,
            opponent_club_id=opponent_club_id,
            round_date=round_date,
        )
```

(c) Create `_run_phase3` by moving the *existing* steps 2-9 (current lines 79-144) verbatim, renaming the local `pool` references as needed. It must keep the existing `_compute_lineage` call signature and `ScenarioSet` construction. Signature:

```python
    def _run_phase3(  # noqa: PLR0913
        self,
        pool: list[PlayerCandidate],
        *,
        context: CompetitionContext,
        saison: int,
        current_round: int,
        nb_rondes_total: int,
        n_topk: int,
        n_mc_pairs: int,
        seed: int,
        opponent_club_id: str,
        round_date: str,
    ) -> ScenarioSet:
        """Phase 3 pipeline: enrich -> copula -> TopK + MC -> ScenarioSet."""
        # (steps 2-9 from the original generate(), unchanged)
```

> The pool-size guard moved up into `generate()` (above) so it fires identically for both paths; remove the duplicate guard from `_run_phase3`'s body when extracting.

- [ ] **Step 4: Run the NEW Phase 4a tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/services/ali/test_generator_phase4a.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the Phase 3 regression suite UNCHANGED (T9 gate)**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_generator.py -v -m slow`
Expected: all 9 Phase 3 tests PASS **without any edit to `tests/test_generator.py`**. (This is the no-regression acceptance gate T9.)

- [ ] **Step 6: Quality gates + coverage + line count**

Run:
```
.\.venv\Scripts\python.exe -m ruff check services/ali/generator.py tests/services/ali/test_generator_phase4a.py
.\.venv\Scripts\python.exe -m mypy --strict services/ali/generator.py
.\.venv\Scripts\python.exe -m radon cc services/ali/generator.py -nb
python -c "print(sum(1 for _ in open('services/ali/generator.py')))"
.\.venv\Scripts\python.exe -m pytest tests/services/ali/test_generator_phase4a.py tests/services/ali/test_joint_conditional.py --cov=services/ali/generator --cov=services/ali/joint_conditional --cov-report=term-missing
```
Expected: ruff/mypy 0 errors; radon ≤ B; `generator.py` line count ≤ 300; combined module coverage ≥ 90 %.

- [ ] **Step 7: Commit**

```bash
git add services/ali/generator.py tests/services/ali/test_generator_phase4a.py
git commit -m "feat(ali): generate() Phase 4a BC dispatch + joint-conditional exclusion (Phase 4a T5b)"
```

---

## Self-Review

**1. Spec coverage (T5, T6):**
- T6 DoD — signature `exclude_players: set[str] | None = None` ✓ (Task 1); filter post-F7/overrides ✓; BC None ✓; ≥4 NEW tests ✓ (empty, subset, all, overrides).
- T5 DoD — signature `simultaneous_teams: list[TeamSpec] | None`, `target_team: str | None` ✓ (Task 3); dispatch None→Phase 3 ✓; Phase 4a top-down + exclude ✓ (via Task 2); Phase 3 tests 100 % PASS unchanged ✓ (Step 5); ≥8 NEW Phase 4a-path tests ✓ (Task 2: 8 + Task 3: 3 = 11); coverage ≥90 % ✓ (Steps).
- T5 self-review items — `pytest test_generator.py` unchanged ✓; NEW path PASS ✓; UNSAT propagation ✓ (`test_compute_exclusions_infeasible_raises`, Q7); determinism ✓ (`test_compute_exclusions_deterministic`).
- Deferred with tracking (not silent): T7/T8 → Part 2b; preference+diversification mixture → debt `D-2026-06-16-adverse-allocation-mixture-preference-diversification`; noyau wiring → debt `D-2026-06-16-adverse-noyau-wiring`.

**2. Placeholder scan:** No TBD/TODO/"handle edge cases" in steps — all code is concrete. The one prose note (Task 3 extraction of steps 2-9) refers to existing, readable code, not an unwritten dependency.

**3. Type consistency:** `compute_adverse_exclusions(pool, teams, target_team, seed, max_time_sec)` and `superior_teams(teams, target_team)` are used identically in tests and module. `AdverseCEInput(pool, teams, historical_noyau, max_time_sec, seed)` matches the shipped Pydantic model (`services/ali/adverse_ce.py:42-61`). `AdverseCESolution.solver_status` / `.assignments` match `services/ali/types.py:126-143`. `load_pool(..., exclude_players=...)` keyword matches Task 1.

**Open risk flagged for the executor:** `tests/test_generator.py` Step 5 uses the real `ali_data_cache` fixture (~92 s, `@pytest.mark.slow`); run it explicitly with `-m slow`. If the extraction of `_run_phase3` accidentally changes the lineage hash (it must not — same inputs to `_compute_lineage`), `test_generator_lineage_hash_propagated` will catch it.

---

## Debts created by this plan (to add to `memory/project_debt_current.md`)

- `D-2026-06-16-adverse-allocation-mixture-preference-diversification` — MVP ships single max-Elo top-down exclusion; the diverse-allocation mixture wiring `preference_model` (T2) + `diversification` (T3) is deferred to Phase 4c contingency, triggered if T10 acceptance recall < 0.65 (Q5).
- `D-2026-06-16-adverse-noyau-wiring` — `historical_noyau` passed empty in `joint_conditional`; wire `services/ffe/checkers.py` noyau per superior team. Phase 4a fast-follow (before T10 Kaggle if cheap).
- `D-2026-06-16-phase-4a-part-2b-api-cache` — spec T7 (`/compose` + Pydantic `simultaneous_teams`) and T8 (`ce_adverse_cache.py` SQLite) deferred to Part 2b, after D-P3-19 acceptance (prod-path, not on the offline acceptance critical path).
