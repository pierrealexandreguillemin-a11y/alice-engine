# Phase 4a T9 — Local Pilot N=70 N3 Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-backtest N=70 Nationale 3 matches from saison 2024 through the Phase 4a joint-conditional path (`generate(..., simultaneous_teams, target_team)`) and emit an early-gate report (recall ≥ 0.50) deciding whether to proceed to the Kaggle Phase A run (T10).

**Architecture:** The existing backtest harness (`harness.py` → `run_match.py::run_backtest_match` → `generator.generate`) does NOT thread `simultaneous_teams` / `target_team` / the real match `date`. This plan adds that threading (backward-compatible: `None` → Phase 3 path byte-identical), a date-filtered fixture loader for `config/clubs_teams_2024.json`, and a thin orchestration script `pilot_phase4a.py`. All metric / ground-truth / statistical / sampling code is reused unchanged.

> ## Diagnostic Addendum (2026-06-16 PM — controller, BEFORE dispatch)
>
> Empirical verification against the real data **corrected two assumptions in the original draft** of T9.2/T9.5:
>
> 1. **Fixture lookup MUST key by `team_name`, NOT by `club`.** The fixture's club KEY is derived from the team name by stripping the trailing number token (`"Clichy 2"` → `"Clichy"`; `build_clubs_teams.py::build_club_index`). But `ALIDataCache.team_to_club` maps team → **`joueurs.club`** (`"Clichy"` → `"Clichy-Echecs-92"`). Keying the lookup by `cand.opp_club` (joueurs.club) matches only **75/937 clubs (~8%)** → a near-empty pilot. Keying by the opponent's exact `team_name` (which IS the fixture's stored team label) resolves **77% of N3 teams**. → T9.2 builds a `team_name → club_key` reverse index from the payload; T9.5 looks up by `cand.opp_team`.
> 2. **Viable-N is measured, not assumed.** Of **720** N3 2024 distinct matches: **206 VIABLE** (target present on same date + ≥1 superior team: 163 with 1 superior, 43 with 2), 280 no-superior (Phase4a≡Phase3, skip), 163 club-not-in-fixture (single-team clubs), 71 target-dropped-by-date-filter (D-2026-06-10 residual). 206 ≫ 70 → the pilot is statistically viable. → T9.5 enumerates ALL candidates and keeps the first **70 VIABLE** ones (target present + ≥1 superior), NOT the first 70 candidates (only ~29% of candidates are viable, so 70 candidates ≈ 20 viable = too thin for McNemar).
> 3. **Target-present guard.** Because date-coherence is 0.3972, a target's own fixture entry may carry a date ≠ the echiquiers match date and get filtered out (71 cases). The pilot MUST skip (and count) any match where `cand.opp_team not in [t.team_name for t in sim]` — otherwise `superior_teams()` raises `ValueError`. Force-order default `99` (jeunes/régionale/unknown) is correct for an N3 target: only Top16/N1/N2 (rank < 3) are treated as superior.

**Tech Stack:** Python 3.12, pytest, pandas, existing `scripts/backtest/*` infra, `services/ali/generator.py` (Phase 4a path shipped in T5/T6), OR-Tools CP-SAT (via `joint_conditional`).

---

## Context the implementer must know

- **Phase 4a path is already shipped** in `services/ali/generator.py::ScenarioGenerator.generate` (signature ends with `simultaneous_teams: list[TeamSpec] | None = None, target_team: str | None = None`). When `simultaneous_teams is None` → Phase 3 path (unchanged). When provided → `joint_conditional.compute_adverse_exclusions` drains the pool of players consumed by the opponent's *superior* teams, then samples the target team from the residual. Read `services/ali/generator.py` + `services/ali/joint_conditional.py` before starting.
- **`superior_teams(teams, target_team)`** (in `joint_conditional.py`) expects `teams` ordered **top-down by descending force** (A02 §3.7.b) and returns every team strictly before the target. So the `simultaneous_teams` list you build MUST be sorted by division force (Top16 > N1 > N2 > N3 > N4 > Régionale), else the wrong teams are treated as "superior".
- **Date is load-bearing** (debt `D-2026-06-10-clubs-teams-grouping-residual`): `config/clubs_teams_2024.json` has `date_coherence_rate ≈ 0.3972` — rondes do NOT align calendar-wise across divisions. The fixture lookup MUST filter by the real match `date`, never by `ronde` alone.
- **Current backtest uses a FAKE date**: `run_backtest_match` hardcodes `round_date=f"{saison}-09-01"` (`scripts/backtest/run_match.py:78`). T9.2 makes `round_date` an optional param (default preserves the fake date → BC) and the pilot passes the real `date`.
- **Reusable as-is (no edits):** `metrics.py` (`top_k_recall`, `jaccard_max`, `brier_presence`, `brier_skill_score`, `accuracy_at_k`), `ground_truth.py` (`extract_observed_lineup`, `ObservedLineup`), `statistical.py` (`mcnemar_paired`, `wilcoxon_paired`), `bootstrap.py` (`bootstrap_ci`), `runner_sampling.py` (`enumerate_candidates`, `stratify_per_ronde`).
- **Quality gates (spec §T9):** F1, T1, T9 (Phase 3 no-regression). Each threading task MUST keep `pytest tests/test_generator.py -m slow` at 10/10 and `pytest tests/services/ali -m "not slow"` green.
- **DoD:** `pilot_phase4a.py` ≤ 250 lines; output `reports/pilot_phase4a_<DATE>.md` with metrics (recall, Jaccard, Brier, McNemar n_disc + p-value) + early-gate decision; if recall < 0.50 → STOP + diagnostic before Kaggle.

---

## File Structure

- **Modify** `scripts/backtest/runner_types.py` — add `date: str = ""` to `MatchCandidate`.
- **Modify** `scripts/backtest/runner_sampling.py:91-100` — populate `date` from the echiquiers row in `enumerate_candidates`.
- **Modify** `scripts/backtest/run_match.py:44-84` — add `round_date`, `simultaneous_teams`, `target_team` params to `run_backtest_match`; thread to `generate`.
- **Modify** `scripts/backtest/harness.py:95-129` — add `round_date`, `simultaneous_teams`, `target_team` to `BacktestHarness.run_match`; forward them.
- **Create** `scripts/backtest/clubs_teams_fixture.py` — `build_team_to_club_index(payload)` + `load_simultaneous_teams(payload, *, team_name, ronde, match_date, team_index=None)` team-name-keyed date-filtered loader + force ordering (~90 lines, SRP).
- **Create** `scripts/backtest/pilot_phase4a.py` — orchestration + report (≤ 250 lines).
- **Test** `tests/backtest/test_clubs_teams_fixture.py`, `tests/backtest/test_run_match_phase4a.py`, `tests/backtest/test_pilot_phase4a.py`.

---

## Task 1: Thread the real match `date` through sampling

**Files:**
- Modify: `scripts/backtest/runner_types.py` (`MatchCandidate` dataclass)
- Modify: `scripts/backtest/runner_sampling.py:91-100`
- Test: `tests/backtest/test_runner_sampling_date.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/backtest/test_runner_sampling_date.py
from __future__ import annotations

import pandas as pd

from scripts.backtest.runner_sampling import enumerate_candidates
from scripts.backtest.runner_types import RunnerConfig


class _Cache:
    def __init__(self, df: pd.DataFrame) -> None:
        self.echiquiers_total = df
        self.team_to_club = {"A 3": "A", "B 3": "B"}
        self.joueurs_by_club = {"A": pd.DataFrame({"nr_ffe": range(8)}),
                                "B": pd.DataFrame({"nr_ffe": range(8)})}


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "saison": [2024] * 8, "type_competition": ["national"] * 8,
        "division": ["Nationale 3"] * 8, "ronde": [3] * 8,
        "equipe_dom": ["A 3"] * 8, "equipe_ext": ["B 3"] * 8,
        "groupe": [""] * 8, "date": ["2024-11-17"] * 8,
    })


def test_candidate_carries_match_date() -> None:
    cfg = RunnerConfig(saison=2024, rondes=(3,), max_matches=10, team_size=8,
                       division="N3", division_filter="Nationale 3",
                       type_competition="national")
    cands = enumerate_candidates(_Cache(_df()), cfg)
    assert len(cands) == 1
    assert cands[0].date == "2024-11-17"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/backtest/test_runner_sampling_date.py -q`
Expected: FAIL — `MatchCandidate.__init__() got an unexpected keyword argument 'date'` (or `AttributeError: 'MatchCandidate' object has no attribute 'date'`).

- [ ] **Step 3: Add `date` to MatchCandidate**

In `scripts/backtest/runner_types.py`, add to the `MatchCandidate` dataclass (after `groupe`):

```python
    date: str = ""  # real match date "YYYY-MM-DD" (Phase 4a fixture lookup, D-2026-06-10)
```

- [ ] **Step 4: Populate `date` in `enumerate_candidates`**

In `scripts/backtest/runner_sampling.py`, inside the row loop, extract the date defensively (column may be absent in legacy test data), then pass it to `MatchCandidate`:

```python
        date_raw = row.get("date") if "date" in sub.columns else None
        match_date = str(date_raw)[:10] if date_raw is not None and str(date_raw) != "nan" else ""
        out.append(
            MatchCandidate(
                saison=config.saison,
                ronde=ronde,
                user_team=user_team,
                opp_team=opp_team,
                opp_club=opp_club,
                groupe=groupe,
                date=match_date,
            )
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/backtest/test_runner_sampling_date.py -q`
Expected: PASS.

- [ ] **Step 6: Verify no regression on existing sampling tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/backtest -q -m "not slow" -k "sampling or runner"`
Expected: all PASS (the new field has a default, so existing `MatchCandidate(...)` calls without `date` still work).

- [ ] **Step 7: Commit**

```bash
git add scripts/backtest/runner_types.py scripts/backtest/runner_sampling.py tests/backtest/test_runner_sampling_date.py
git commit -m "feat(backtest): carry real match date on MatchCandidate (Phase 4a T9.1)"
```

---

## Task 2: Date-filtered `clubs_teams_2024.json` loader + force ordering

**Files:**
- Create: `scripts/backtest/clubs_teams_fixture.py`
- Test: `tests/backtest/test_clubs_teams_fixture.py`

**Fixture shape (verified):** `payload["clubs"][club_key]["rondes"][str(ronde)]` is a list of `[team_name, division, board_count, date]`. The `club_key` is the team name stripped of its trailing number token (`"Clichy 2"` → `"Clichy"`), which is NOT `ALIDataCache.team_to_club` (joueurs.club). So the loader keys by **`team_name`** (resolves the club via a `team_name → club_key` reverse index built from the payload) and returns the `TeamSpec` list for that date, ordered top-down by division force. See Diagnostic Addendum.

- [ ] **Step 1: Write the failing test**

```python
# tests/backtest/test_clubs_teams_fixture.py
from __future__ import annotations

from scripts.backtest.clubs_teams_fixture import (
    build_team_to_club_index,
    load_simultaneous_teams,
)

_PAYLOAD = {
    "clubs": {
        "Mulhouse": {
            "rondes": {
                "3": [
                    ["Mulhouse 3", "Nationale 3", 8, "2024-11-17"],
                    ["Mulhouse 1", "Nationale 1", 8, "2024-11-17"],
                    ["Mulhouse 2", "Nationale 2", 8, "2024-11-17"],
                    ["Mulhouse 4", "Nationale 4", 8, "2024-11-10"],  # different date -> excluded
                ]
            }
        }
    }
}


def test_resolves_club_by_team_name_and_orders_top_down() -> None:
    teams = load_simultaneous_teams(
        _PAYLOAD, team_name="Mulhouse 3", ronde=3, match_date="2024-11-17"
    )
    # club resolved from "Mulhouse 3"; date filter drops "Mulhouse 4"; force N1 > N2 > N3
    assert [t.team_name for t in teams] == ["Mulhouse 1", "Mulhouse 2", "Mulhouse 3"]
    assert teams[0].division == "Nationale 1"


def test_target_team_present_in_result() -> None:
    # superior_teams() requires the target to be in the list (else ValueError).
    teams = load_simultaneous_teams(
        _PAYLOAD, team_name="Mulhouse 3", ronde=3, match_date="2024-11-17"
    )
    assert "Mulhouse 3" in [t.team_name for t in teams]


def test_empty_when_team_absent() -> None:
    assert load_simultaneous_teams(
        _PAYLOAD, team_name="Ghost 1", ronde=3, match_date="2024-11-17"
    ) == []


def test_empty_when_no_entry_matches_date() -> None:
    assert load_simultaneous_teams(
        _PAYLOAD, team_name="Mulhouse 3", ronde=3, match_date="2024-12-01"
    ) == []


def test_reverse_index_maps_every_team_to_its_club() -> None:
    idx = build_team_to_club_index(_PAYLOAD)
    assert idx["Mulhouse 1"] == "Mulhouse"
    assert idx["Mulhouse 4"] == "Mulhouse"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/backtest/test_clubs_teams_fixture.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.backtest.clubs_teams_fixture'`.

- [ ] **Step 3: Implement the loader**

```python
# scripts/backtest/clubs_teams_fixture.py
"""Team-name-keyed loader for config/clubs_teams_2024.json (Phase 4a T9).

Maps an echiquiers team_name (e.g. "Mulhouse 3") + ronde + match_date -> the
ordered list[TeamSpec] of the SIMULTANEOUS teams its club fields on that date,
top-down by division force (A02 §3.7.b) so joint_conditional.superior_teams()
correctly identifies the teams ranked above the target.

WHY team-name keying (controller diagnostic 2026-06-16): the fixture club KEY is
derived from the team name by stripping the trailing number token ("Clichy 2" ->
"Clichy"), whereas ALIDataCache.team_to_club maps team -> joueurs.club
("Clichy" -> "Clichy-Echecs-92"). Keying by joueurs.club matches ~8% of clubs;
keying by team_name (the fixture's stored team label) resolves 77% of N3 teams.
So we build a team_name -> club_key reverse index from the payload and resolve
via the opponent's exact team name.

CRITICAL (debt D-2026-06-10): filter by `date`, NOT `ronde` alone
(date_coherence_rate ~= 0.40 across divisions).

Document ID: ALICE-BACKTEST-CLUBS-TEAMS-FIXTURE
Version: 2.0.0
"""

from __future__ import annotations

from typing import Any

from services.ali.types import TeamSpec

# Descending force rank (A02 §3.7.b). Lower index = stronger. Jeunes/régionale/
# départemental/unknown -> _FORCE_DEFAULT (weakest); correct for an N3 target
# since only Top16/N1/N2 (rank < 3) are then treated as superior.
_FORCE_ORDER: dict[str, int] = {
    "Top 16": 0, "Top16": 0,
    "Nationale 1": 1, "Nationale 2": 2, "Nationale 3": 3, "Nationale 4": 4,
}
_FORCE_DEFAULT = 99


def _force_rank(division: str) -> int:
    return _FORCE_ORDER.get(division, _FORCE_DEFAULT)


def build_team_to_club_index(payload: dict[str, Any]) -> dict[str, str]:
    """Reverse index team_name -> club_key (one pass over the payload)."""
    index: dict[str, str] = {}
    for club, data in payload.get("clubs", {}).items():
        for entries in data.get("rondes", {}).values():
            for entry in entries:
                index[str(entry[0])] = club
    return index


def load_simultaneous_teams(
    payload: dict[str, Any],
    *,
    team_name: str,
    ronde: int,
    match_date: str,
    team_index: dict[str, str] | None = None,
) -> list[TeamSpec]:
    """Return the club (that fields `team_name`)'s TeamSpec list for `match_date`.

    Ordered top-down by division force. Empty list if `team_name` is absent
    from the fixture, the ronde is absent, or no entry matches the date. Pass a
    precomputed `team_index` (from `build_team_to_club_index`) to avoid rebuilding
    it per call in a loop.
    """
    index = team_index if team_index is not None else build_team_to_club_index(payload)
    club = index.get(team_name)
    if club is None:
        return []
    entries = payload["clubs"][club].get("rondes", {}).get(str(ronde), [])
    matched = [e for e in entries if str(e[3])[:10] == match_date]
    matched.sort(key=lambda e: (_force_rank(str(e[1])), str(e[0])))
    return [
        TeamSpec(team_name=str(e[0]), division=str(e[1]), board_count=int(e[2]))
        for e in matched
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/backtest/test_clubs_teams_fixture.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest/clubs_teams_fixture.py tests/backtest/test_clubs_teams_fixture.py
git commit -m "feat(backtest): date-filtered clubs_teams fixture loader (Phase 4a T9.2)"
```

---

## Task 3: Thread `round_date` + `simultaneous_teams` + `target_team` through `run_backtest_match`

**Files:**
- Modify: `scripts/backtest/run_match.py:44-84`
- Test: `tests/backtest/test_run_match_phase4a.py`

**Backward-compat is the gate:** new params default to `None`/`""`. When `simultaneous_teams is None`, the call to `generate` must be byte-identical to today (Phase 3). Use the real `round_date` only when explicitly provided.

- [ ] **Step 1: Write the failing test (asserts the generate call forwards Phase 4a args)**

```python
# tests/backtest/test_run_match_phase4a.py
from __future__ import annotations

from typing import Any

from scripts.backtest.run_match import run_backtest_match
from services.ali.types import TeamSpec


class _SpyGenerator:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return _FakeScenarioSet()


class _FakeScenarioSet:
    lineage_hash = "deadbeef"
    scenarios: list[Any] = []


def _run(**extra: Any) -> _SpyGenerator:
    gen = _SpyGenerator()
    run_backtest_match(
        user_club_id="A", opponent_club_id="B", saison=2024, ronde=3,
        nb_rondes_total=11, division="N3", team_size=8, user_lineup=[],
        scenario_generator=gen, inference=_Noop(), feature_store=_Noop(),
        strict=False, **extra,
    )
    return gen


class _Noop:
    def __getattr__(self, _name: str) -> Any:  # aggregate_from_scenarios no-ops on empty set
        return lambda *a, **k: []


def test_phase3_default_uses_fake_date_and_no_sim_teams() -> None:
    gen = _run()
    assert gen.kwargs["round_date"] == "2024-09-01"
    assert gen.kwargs.get("simultaneous_teams") is None
    assert gen.kwargs.get("target_team") is None


def test_phase4a_forwards_real_date_and_sim_teams() -> None:
    sim = [TeamSpec(team_name="B 1", division="Nationale 1", board_count=8),
           TeamSpec(team_name="B 3", division="Nationale 3", board_count=8)]
    gen = _run(round_date="2024-11-17", simultaneous_teams=sim, target_team="B 3")
    assert gen.kwargs["round_date"] == "2024-11-17"
    assert gen.kwargs["simultaneous_teams"] == sim
    assert gen.kwargs["target_team"] == "B 3"
```

> Note: if `aggregate_from_scenarios` cannot accept the `_Noop` stub, inject a minimal fake that returns `[]` for an empty `scenario_set`; the test only asserts the forwarded `generate` kwargs. Adjust the stub to the real `aggregate_from_scenarios` contract discovered when reading `run_match.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/backtest/test_run_match_phase4a.py -q`
Expected: FAIL — `run_backtest_match() got an unexpected keyword argument 'round_date'` (and the Phase 4a kwargs not forwarded).

- [ ] **Step 3: Add params + thread to `generate`**

In `scripts/backtest/run_match.py`, add to the keyword-only signature (after `seed`, before `strict`):

```python
    round_date: str | None = None,
    simultaneous_teams: list[TeamSpec] | None = None,
    target_team: str | None = None,
```

Add the import at the top if missing: `from services.ali.types import TeamSpec` (alongside the existing `CompetitionContext` import).

Replace the `generate(...)` call (currently lines 76-84) with:

```python
    scenario_set = scenario_generator.generate(
        opponent_club_id=opponent_club_id,
        round_date=round_date or f"{saison}-09-01",
        context=context,
        saison=saison,
        current_round=ronde,
        nb_rondes_total=nb_rondes_total,
        seed=seed,
        simultaneous_teams=simultaneous_teams,
        target_team=target_team,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/backtest/test_run_match_phase4a.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Verify Phase 3 no-regression (CRITICAL gate T9)**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_generator.py -q -m slow`
Expected: 10 passed (the BC default path is byte-identical: `round_date=None` → `f"{saison}-09-01"`, `simultaneous_teams=None` → Phase 3).

- [ ] **Step 6: Commit**

```bash
git add scripts/backtest/run_match.py tests/backtest/test_run_match_phase4a.py
git commit -m "feat(backtest): thread round_date + simultaneous_teams + target_team into run_backtest_match (Phase 4a T9.3)"
```

---

## Task 4: Forward the Phase 4a args through `BacktestHarness.run_match`

**Files:**
- Modify: `scripts/backtest/harness.py:95-129`
- Test: covered by Task 5 integration (the harness wrapper is thin; add an explicit forward assertion only if the wrapper does non-trivial work).

- [ ] **Step 1: Read `harness.py:95-129` to confirm `run_match`'s current signature and how it calls `run_backtest_match`.**

- [ ] **Step 2: Add params to `BacktestHarness.run_match`** (keyword-only, defaults preserve BC):

```python
        round_date: str | None = None,
        simultaneous_teams: list[TeamSpec] | None = None,
        target_team: str | None = None,
```

Add `from services.ali.types import TeamSpec` to the imports if absent.

- [ ] **Step 3: Forward them in the `run_backtest_match(...)` call** inside `run_match`:

```python
            round_date=round_date,
            simultaneous_teams=simultaneous_teams,
            target_team=target_team,
```

- [ ] **Step 4: Verify no regression**

Run: `.\.venv\Scripts\python.exe -m pytest tests/backtest -q -m "not slow"`
Expected: all PASS (defaults preserve Phase 3 behavior).

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest/harness.py
git commit -m "feat(backtest): forward Phase 4a args through BacktestHarness.run_match (Phase 4a T9.4)"
```

---

## Task 5: Pilot orchestration script + report

**Files:**
- Create: `scripts/backtest/pilot_phase4a.py` (≤ 250 lines)
- Test: `tests/backtest/test_pilot_phase4a.py`

The pilot: enumerate ALL N3 candidates → for each, look up `simultaneous_teams` by `team_name=cand.opp_team` (date-filtered, force-ordered) → keep only VIABLE matches (target present in the list AND ≥1 superior team) until **70 viable** are collected → run via `harness.run_match(...)` in Phase 4a mode with `target_team = cand.opp_team` → ALSO run the same match in Phase 3 mode (`simultaneous_teams=None`) for the paired baseline → `extract_observed_lineup` → metrics → aggregate (bootstrap CI on recall, McNemar/Wilcoxon on paired Phase4a-vs-Phase3 recall) → write report + early-gate decision (recall ≥ 0.50). Non-viable matches (no superior, club-not-in-fixture, target-dropped-by-date) are counted and logged, never silently dropped.

- [ ] **Step 1: Write the failing test for the pure report-decision helper**

```python
# tests/backtest/test_pilot_phase4a.py
from __future__ import annotations

from scripts.backtest.pilot_phase4a import EARLY_GATE_RECALL, early_gate_decision


def test_gate_pass_when_mean_recall_at_or_above_threshold() -> None:
    assert early_gate_decision(EARLY_GATE_RECALL).startswith("PASS")


def test_gate_fail_below_threshold() -> None:
    assert early_gate_decision(EARLY_GATE_RECALL - 0.01).startswith("FAIL")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/backtest/test_pilot_phase4a.py -q`
Expected: FAIL — `ModuleNotFoundError` / `ImportError: cannot import name 'early_gate_decision'`.

- [ ] **Step 3: Implement the pilot** (sketch — the implementer fills exact harness/metric call args from the modules; keep ≤ 250 lines, delegate all metric math):

```python
# scripts/backtest/pilot_phase4a.py
"""Phase 4a joint-conditional local pilot — N=70 N3, saison 2024 (T9).

Runs the existing backtest harness in Phase 4a mode (simultaneous_teams +
target_team from config/clubs_teams_2024.json, date-filtered) and emits an
early-gate report: recall >= 0.50 (vs Phase 3 baseline ~0.57) decides whether
to proceed to the Kaggle Phase A run (T10).

Document ID: ALICE-BACKTEST-PILOT-PHASE4A
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from scripts.backtest.clubs_teams_fixture import (
    build_team_to_club_index,
    load_simultaneous_teams,
)
from scripts.backtest.ground_truth import extract_observed_lineup
from scripts.backtest.harness import BacktestHarness
from scripts.backtest.metrics import brier_presence, jaccard_max, top_k_recall
from scripts.backtest.runner_sampling import enumerate_candidates
from scripts.backtest.runner_types import RunnerConfig
from services.ali.types import TeamSpec

logger = logging.getLogger(__name__)

EARLY_GATE_RECALL = 0.50
MAX_VIABLE = 70  # collect this many viable (target-present + >=1-superior) matches
_FIXTURE = Path("config/clubs_teams_2024.json")


def early_gate_decision(mean_recall: float) -> str:
    if mean_recall >= EARLY_GATE_RECALL:
        return f"PASS (mean recall {mean_recall:.4f} >= {EARLY_GATE_RECALL}) -> proceed to Kaggle (T10)"
    return f"FAIL (mean recall {mean_recall:.4f} < {EARLY_GATE_RECALL}) -> STOP + diagnostic before Kaggle"


def _config() -> RunnerConfig:
    # rondes = all 11 (maximize the viable pool); max_matches high so
    # enumerate_candidates returns ALL N3 matches (viability is filtered below,
    # not by max_matches). enumerate_candidates does NOT use max_matches.
    return RunnerConfig(
        saison=2024, rondes=tuple(range(1, 12)), max_matches=10_000, team_size=8,
        division="N3", division_filter="Nationale 3", type_competition="national",
        nb_rondes_total=11, seed=42,
    )


def _viable_sim(payload: dict[str, Any], team_index: dict[str, str], cand: Any
                ) -> list[TeamSpec] | None:
    """Return the ordered sim list IF the match is viable, else None.

    Viable = the target team is present on its own match date AND has >=1
    superior team. None covers: club-not-in-fixture, target-dropped-by-date,
    and no-superior (Phase 4a == Phase 3).
    """
    if not cand.date:
        return None
    sim = load_simultaneous_teams(
        payload, team_name=cand.opp_team, ronde=cand.ronde,
        match_date=cand.date, team_index=team_index,
    )
    names = [t.team_name for t in sim]
    if cand.opp_team not in names:            # club absent or target dropped by date filter
        return None
    if names.index(cand.opp_team) == 0:       # target is team_1 -> no superior team
        return None
    return sim


def run_pilot(out_dir: Path = Path("reports")) -> dict[str, Any]:
    """Run the Phase 4a pilot and write reports/pilot_phase4a_<DATE>.md."""
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    team_index = build_team_to_club_index(payload)
    harness = BacktestHarness()
    harness.setup()
    cfg = _config()
    candidates = enumerate_candidates(harness.cache, cfg)

    rows: list[dict[str, Any]] = []
    skipped = {"non_viable": 0, "no_observed": 0, "error": 0}
    for cand in candidates:
        if len(rows) >= MAX_VIABLE:
            break
        sim = _viable_sim(payload, team_index, cand)
        if sim is None:
            skipped["non_viable"] += 1
            continue
        try:
            # Phase 4a (joint-conditional) + paired Phase 3 baseline on the SAME match.
            result = harness.run_match(
                user_club_id=harness.cache.team_to_club[cand.user_team],
                opponent_club_id=cand.opp_club, saison=cand.saison, ronde=cand.ronde,
                nb_rondes_total=cfg.nb_rondes_total, division=cfg.division,
                team_size=cfg.team_size, user_lineup=[], seed=cfg.seed, strict=False,
                round_date=cand.date, simultaneous_teams=sim, target_team=cand.opp_team,
            )
            baseline = harness.run_match(
                user_club_id=harness.cache.team_to_club[cand.user_team],
                opponent_club_id=cand.opp_club, saison=cand.saison, ronde=cand.ronde,
                nb_rondes_total=cfg.nb_rondes_total, division=cfg.division,
                team_size=cfg.team_size, user_lineup=[], seed=cfg.seed, strict=False,
            )  # Phase 3: simultaneous_teams/target_team default None
            observed = extract_observed_lineup(
                harness.cache, cand.opp_team, cand.saison, cand.ronde,
                as_domicile=False, groupe=cand.groupe,
            )
        except Exception:
            logger.exception("pilot match failed: ronde=%s opp=%s", cand.ronde, cand.opp_team)
            skipped["error"] += 1
            continue
        if not observed.players:
            skipped["no_observed"] += 1
            continue
        rows.append({
            "opp_team": cand.opp_team, "ronde": cand.ronde, "date": cand.date,
            "n_superior": names_index(sim, cand.opp_team),
            "recall": top_k_recall(observed, result.scenario_set),
            "recall_baseline": top_k_recall(observed, baseline.scenario_set),
            "jaccard": jaccard_max(observed, result.scenario_set),
            "brier": brier_presence(observed, result.scenario_set),
        })

    mean_recall = sum(r["recall"] for r in rows) / len(rows) if rows else 0.0
    mean_baseline = sum(r["recall_baseline"] for r in rows) / len(rows) if rows else 0.0
    summary = {"n_matches": len(rows), "mean_recall": mean_recall,
               "mean_baseline_recall": mean_baseline, "skipped": skipped,
               "decision": early_gate_decision(mean_recall)}
    _write_report(out_dir, rows, summary)
    return summary


def names_index(sim: list[TeamSpec], target: str) -> int:
    """Number of teams ranked strictly above `target` (its index in the list)."""
    return [t.team_name for t in sim].index(target)


def _write_report(out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 4a Pilot — N3 saison 2024",
        "",
        f"- Matches with >=1 superior team: **{summary['n_matches']}**",
        f"- Mean recall: **{summary['mean_recall']:.4f}** (Phase 3 baseline ~0.57)",
        f"- Early-gate: **{summary['decision']}**",
        "",
        "| opp_team | ronde | date | n_superior | recall | jaccard | brier |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['opp_team']} | {r['ronde']} | {r['date']} | {r['n_superior']} | "
            f"{r['recall']:.3f} | {r['jaccard']:.3f} | {r['brier']:.3f} |"
        )
    (out_dir / "pilot_phase4a.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print(run_pilot()["decision"])
```

> The implementer MUST confirm the exact `BacktestHarness.run_match` and `extract_observed_lineup` argument names against the real modules (Task 4 read) and adjust. The paired Phase-3 baseline recall is already computed inline (the `baseline = harness.run_match(...)` call with `simultaneous_teams` defaulting to None, same match). For the McNemar/Wilcoxon comparison (spec §T9 "McNemar n_disc + p-value"), feed the per-match `recall` (Phase 4a) and `recall_baseline` (Phase 3) pairs into `wilcoxon_paired` (continuous, SOTA per D-P3-18) and `mcnemar_paired` (dichotomize at recall ≥ 0.50) from `statistical.py`, and include both in the report + summary. Keep the file ≤ 250 lines; if it exceeds, extract `_write_report` + the stats aggregation into `pilot_phase4a_helpers.py`. Note `extract_observed_lineup` arg names (`as_domicile`, `groupe`) are a sketch — confirm against `ground_truth.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/backtest/test_pilot_phase4a.py -q`
Expected: PASS.

- [ ] **Step 5: ISO 5055 line check**

Run: `(Get-Content scripts/backtest/pilot_phase4a.py | Measure-Object -Line).Lines`
Expected: ≤ 250. If over, extract helpers as noted.

- [ ] **Step 6: Commit**

```bash
git add scripts/backtest/pilot_phase4a.py tests/backtest/test_pilot_phase4a.py
git commit -m "feat(backtest): Phase 4a pilot orchestration + early-gate report (Phase 4a T9.5)"
```

---

## Task 6: Execute the pilot + record the gate verdict

**Files:**
- Output: `reports/pilot_phase4a_<DATE>.md`
- Modify (memory, outside git): `memory/project_debt_current.md` + session memo

- [ ] **Step 1: Run the pilot end-to-end**

Run (as a MODULE from repo root — NOT a file path, else `ModuleNotFoundError: No module named 'scripts.backtest'`): `.\.venv\Scripts\python.exe -m scripts.backtest.pilot_phase4a`
Expected: prints the early-gate decision; writes `reports/pilot_phase4a.md`. Wall time ~1-2h CPU (the paired Phase-3 baseline doubles per-match cost; T9.5 review Issue 2) — run in background. NOTE: the run loads the champion MLP under sklearn 1.8.0 while it was pickled under 1.7.2 (`InconsistentVersionWarning`) — the paired Phase4a-vs-Phase3 comparison cancels this (same model both sides), but absolute recall is approximate; pre-existing model-versioning gap (D6/D7).

- [ ] **Step 2: Rename the report with the date**

```bash
mv reports/pilot_phase4a.md "reports/pilot_phase4a_$(date +%Y-%m-%d).md"
```

- [ ] **Step 3: Interpret the verdict (NO silent pass)**

- **recall ≥ 0.50** → early-gate PASS → proceed to T10 (Kaggle Phase A 492 matches). Record in the report + session memo.
- **recall < 0.50** → STOP. Do NOT proceed to Kaggle (saves 3-4h compute). Open a diagnostic: is it the C5-only fidelity (C4 noyau / C6/C7 unwireable — M1 limitations), the top-down MVP (Q5 → Phase 4c joint OR-Tools contingency), or the date-coherence fixture gap (D-2026-06-10)? Trace findings in `memory/project_debt_current.md`.

- [ ] **Step 4: Commit the report**

```bash
git add "reports/pilot_phase4a_$(date +%Y-%m-%d).md"
git commit -m "docs(backtest): Phase 4a pilot N3 results + early-gate verdict (Phase 4a T9.6)"
```

---

## Self-Review

**Spec coverage (§T9 DoD):**
- `pilot_phase4a.py` ≤ 250 lines → Task 5 Step 5 enforces (with helper-extraction fallback).
- Output `reports/pilot_phase4a_<DATE>.md` with recall/Jaccard/Brier/McNemar → Tasks 5-6 (McNemar/Wilcoxon paired baseline noted in Task 5 Step 3 follow-up).
- Early-gate recall ≥ 0.50 + STOP-if-fail → Task 5 `early_gate_decision` + Task 6 Step 3.
- Phase 3 no-regression (gate T9) → Task 3 Step 5 + Task 4 Step 4.

**Threading correctness:** date (T1) → fixture lookup (T2) → run_backtest_match (T3) → harness (T4) → pilot (T5). Force ordering (T2) ensures `superior_teams()` sees the right teams.

**BC invariant:** every threaded param defaults to `None`/`""`; `simultaneous_teams=None` → Phase 3 path byte-identical (asserted T3 Step 1 + slow regression T3 Step 5).

**Known approximations carried in (declared, not silent):** C6/C7-FR unenforceable (no nationality — M1 Model Card §9.3 item 9); C4 noyau deferred; date-coherence 0.3972 mitigated by date-filter but residual variant-name mismatches (D-2026-06-10) may drop matches → logged, counted in report `n_matches`.

**Open follow-up (post-pilot):** if recall < 0.65 at T10 → Phase 4c joint OR-Tools escalation (`D-2026-05-26-phase4c-joint-ortools-escalation`).

---

## Execution Handoff

Two execution options:
1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task (T9.1…T9.6), two-stage review between tasks. Matches the Phase 4a T1-T6 pattern.
2. **Inline Execution** — execute in-session with checkpoints.
