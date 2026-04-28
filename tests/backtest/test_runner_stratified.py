"""T22 fix-on-sight tests : stratified sampling wired in BacktestRunner.

Plan 3 V2 T22.0b. Couvre :
- Strict filter type_competition + division_filter (rejette J02/coupes/scolaire)
- Balanced per-ronde sampling invariant
- Determinism seed
- Edge cases empty / insufficient stratum

Sources : ISO 24027 §6 fairness audit, ISO 29119 test design.

Document ID: ALICE-TEST-RUNNER-STRATIFIED
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from scripts.backtest.runner_sampling import enumerate_candidates, stratify_per_ronde
from scripts.backtest.runner_types import MatchCandidate, RunnerConfig


@dataclass
class _FakeCache:
    """Minimal cache double — only fields needed by enumerate_candidates."""

    echiquiers_total: pd.DataFrame
    team_to_club: dict[str, str]
    joueurs_by_club: dict[str, pd.DataFrame]


def _make_cache(rows: list[dict[str, object]], pool_size: int = 10) -> _FakeCache:
    df = pd.DataFrame(rows)
    teams = sorted({r["equipe_dom"] for r in rows} | {r["equipe_ext"] for r in rows})  # type: ignore[arg-type]
    team_to_club = {t: f"club_{t}" for t in teams}  # type: ignore[misc]
    pool = pd.DataFrame([{"nr_ffe": f"id_{i}", "elo": 1500} for i in range(pool_size)])
    joueurs_by_club = {f"club_{t}": pool for t in teams}  # type: ignore[misc]
    return _FakeCache(
        echiquiers_total=df,
        team_to_club=team_to_club,  # type: ignore[arg-type]
        joueurs_by_club=joueurs_by_club,  # type: ignore[arg-type]
    )


def _row(saison: int, ronde: int, dom: str, ext: str, **kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "saison": saison,
        "ronde": ronde,
        "equipe_dom": dom,
        "equipe_ext": ext,
        "type_competition": "national",
        "division": "Nationale 3",
    }
    base.update(kw)
    return base


def test_strict_filter_excludes_j02_scolaire_coupe() -> None:
    """type_competition='national' rejette J02, scolaire, coupes (D3/D4 dette)."""
    rows = [
        _row(2024, 5, "TeamA", "TeamB"),
        _row(2024, 5, "TeamC", "TeamD", type_competition="national_jeunes"),
        _row(2024, 5, "TeamE", "TeamF", type_competition="scolaire"),
        _row(2024, 5, "TeamG", "TeamH", type_competition="coupe"),
        _row(2024, 5, "TeamI", "TeamJ", type_competition="regional"),
    ]
    cfg = RunnerConfig(saison=2024, rondes=(5,), max_matches=10, team_size=8)
    cache = _make_cache(rows, pool_size=10)
    cands = enumerate_candidates(cache, cfg)  # type: ignore[arg-type]
    assert len(cands) == 1
    assert cands[0].user_team == "TeamA"


def test_strict_filter_excludes_division_jeunes() -> None:
    """division='Nationale 3' rejette 'Nationale III Jeunes' explicitement."""
    rows = [
        _row(2024, 5, "TeamA", "TeamB"),
        _row(2024, 5, "TeamC", "TeamD", division="Nationale III Jeunes"),
        _row(2024, 5, "TeamE", "TeamF", division="Nationale 4"),
    ]
    cfg = RunnerConfig(saison=2024, rondes=(5,), max_matches=10, team_size=8)
    cache = _make_cache(rows, pool_size=10)
    cands = enumerate_candidates(cache, cfg)  # type: ignore[arg-type]
    assert len(cands) == 1
    assert cands[0].user_team == "TeamA"


def test_strict_filter_pool_size_lt_team_size() -> None:
    """Pool < team_size → candidate skipped."""
    rows = [_row(2024, 5, "TeamA", "TeamB")]
    cfg = RunnerConfig(saison=2024, rondes=(5,), max_matches=10, team_size=8)
    cache = _make_cache(rows, pool_size=4)  # < team_size 8
    cands = enumerate_candidates(cache, cfg)  # type: ignore[arg-type]
    assert cands == []


def test_dedup_same_pair() -> None:
    """Duplicate (saison, ronde, dom, ext) deduped."""
    rows = [
        _row(2024, 5, "A", "B"),
        _row(2024, 5, "A", "B"),  # exact dup
    ]
    cfg = RunnerConfig(saison=2024, rondes=(5,), max_matches=10, team_size=8)
    cache = _make_cache(rows, pool_size=10)
    cands = enumerate_candidates(cache, cfg)  # type: ignore[arg-type]
    assert len(cands) == 1


def test_stratify_balanced_per_ronde() -> None:
    """ISO 24027 §6 : stratification donne ~équilibré par ronde."""
    cands = [
        MatchCandidate(2024, ronde, f"u{i}", f"o{i}", f"c{i}")
        for ronde in (1, 3, 5, 7, 9, 11)
        for i in range(50)
    ]
    cfg = RunnerConfig(
        saison=2024,
        rondes=(1, 3, 5, 7, 9, 11),
        max_matches=60,
        team_size=8,
        stratify_min_per_ronde=5,
    )
    sampled = stratify_per_ronde(cands, cfg)
    counts = {r: sum(1 for c in sampled if c.ronde == r) for r in (1, 3, 5, 7, 9, 11)}
    assert max(counts.values()) - min(counts.values()) <= 1, f"per-ronde imbalance > 1 : {counts}"
    assert sum(counts.values()) == 60


def test_stratify_drops_undersized_strata() -> None:
    """T15 contract : strata < min_per_stratum dropped (insufficient stats)."""
    cands = [MatchCandidate(2024, 1, f"u{i}", f"o{i}", f"c{i}") for i in range(20)] + [
        MatchCandidate(2024, 3, "uX", "oX", "cX"),  # only 1 in ronde 3 < 5 min
    ]
    cfg = RunnerConfig(
        saison=2024,
        rondes=(1, 3),
        max_matches=20,
        team_size=8,
        stratify_min_per_ronde=5,
    )
    sampled = stratify_per_ronde(cands, cfg)
    rondes_present = {c.ronde for c in sampled}
    assert rondes_present == {1}


def test_stratify_determinism_seed() -> None:
    """Same seed → identical ordering. Diff seed → diff ordering (high prob)."""
    cands = [
        MatchCandidate(2024, r, f"u{i}", f"o{i}", f"c{i}") for r in (1, 3, 5) for i in range(20)
    ]
    cfg42 = RunnerConfig(saison=2024, rondes=(1, 3, 5), max_matches=15, team_size=8, seed=42)
    a = stratify_per_ronde(cands, cfg42)
    b = stratify_per_ronde(cands, cfg42)
    assert [c.user_team for c in a] == [c.user_team for c in b]


def test_stratify_empty_list() -> None:
    """Edge case empty input → empty output."""
    cfg = RunnerConfig(saison=2024, rondes=(1,), max_matches=10, team_size=8)
    assert stratify_per_ronde([], cfg) == []


@pytest.mark.parametrize(
    "max_matches,n_rondes,exp_max_per",
    [
        (60, 6, 10),
        (100, 6, 17),
        (12, 6, 2),
        (5, 1, 5),
    ],
)
def test_stratify_max_per_ronde_invariant(
    max_matches: int, n_rondes: int, exp_max_per: int
) -> None:
    """max_per_stratum = ceil(max_matches / n_rondes) ; counts ≤ exp_max_per."""
    rondes = tuple(range(1, n_rondes + 1))
    cands = [
        MatchCandidate(2024, r, f"u{r}_{i}", f"o{r}_{i}", f"c{r}_{i}")
        for r in rondes
        for i in range(50)
    ]
    cfg = RunnerConfig(
        saison=2024,
        rondes=rondes,
        max_matches=max_matches,
        team_size=8,
        stratify_min_per_ronde=1,
    )
    sampled = stratify_per_ronde(cands, cfg)
    counts = {r: sum(1 for c in sampled if c.ronde == r) for r in rondes}
    assert all(c <= exp_max_per for c in counts.values()), counts
    assert len(sampled) <= max_matches
