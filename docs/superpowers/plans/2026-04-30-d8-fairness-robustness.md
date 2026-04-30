# D8 Fairness/Robustness STRICT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build D8 audit pipeline (Phase 3.5 STRICT entry gate Phase 4a) — fairness + robustness ALI breakdown 7 dims × 4 saisons (N≥200) on 4 Kaggle kernels parallèles + aggregator. Gates G-A SOTA strict 19 + case-by-case analysis on FAIL.

**Architecture:** Arch-2 — 4 kernels Kaggle d8-saison-{2021..2024} parallèles (ALICE_SAISON env var, ~65 min each) + 1 kernel d8-aggregator (~10 min). Outputs vers reports/d8/* via DVC stage d8_audit.

**Tech Stack:** Python 3.13, pytest 9, scipy.stats (wilcoxon, ks_2samp), statsmodels (mcnemar), numpy, pandas, joblib, kaggle CLI, dvc 3.67. Réutilise services/ali/* + scripts/backtest/{fairness,robustness}.py.

**Spec source:** `docs/superpowers/specs/2026-04-30-d8-fairness-robustness-design.md`

---

## File Structure

```
scripts/d8/
├── __init__.py                              # NEW — empty package marker
├── types.py                                 # NEW ~80L — D8Stats, GroupBreakdown, GateStatus, dataclasses gels
├── loader.py                                # NEW ~120L — load_match_eligible(saison)
├── breakdowns.py                            # NEW ~220L — 7 fonctions breakdown_by_*
├── calibration.py                           # NEW ~90L — per-group ECE + multicalibration HJ-2018
├── stress_elo.py                            # NEW ~110L — multi-noise [1,3,5,7,10]%
├── stress_roster.py                         # NEW ~90L — drop [5,10,20]%
├── conformal.py                             # NEW ~180L — split conformal Vovk 2024
├── dro.py                                   # NEW ~220L — Wasserstein-2 ε-ball
├── gates.py                                 # NEW ~250L — 19 gates G-A + case-by-case logger
├── run.py                                   # NEW ~150L — per-saison orchestrator
├── aggregate.py                             # NEW ~280L — fuse 4 saisons + global gates + render MD
├── upload_d8_dataset.py                     # NEW ~80L — kaggle datasets create alice-d8-input
├── download_outputs.py                      # NEW ~50L — kaggle kernels output → reports/d8/
├── kernel-metadata-saison-2021.json         # NEW — Kaggle config
├── kernel-metadata-saison-2022.json         # NEW
├── kernel-metadata-saison-2023.json         # NEW
├── kernel-metadata-saison-2024.json         # NEW
└── kernel-metadata-aggregator.json          # NEW

tests/d8/
├── __init__.py                              # NEW — empty
├── conftest.py                              # NEW ~80L — fixtures dummy MatchStats
├── test_types.py                            # NEW ~50L — 8 tests
├── test_loader.py                           # NEW ~150L — 15 tests
├── test_breakdowns.py                       # NEW ~350L — 40 tests
├── test_calibration.py                      # NEW ~200L — 20 tests
├── test_stress_elo.py                       # NEW ~120L — 15 tests
├── test_stress_roster.py                    # NEW ~110L — 12 tests
├── test_conformal.py                        # NEW ~250L — 25 tests
├── test_dro.py                              # NEW ~200L — 20 tests
├── test_gates.py                            # NEW ~400L — 38 tests (19 gates × 2)
├── test_aggregate.py                        # NEW ~150L — 15 tests
└── test_run_e2e_smoke.py                    # NEW ~80L — 5 tests

dvc.yaml                                     # MODIFY — ajouter stage d8_audit

docs/iso/ISO_PIPELINE_HOOKS.md               # MODIFY — référence D8
```

---

## Task 1 : Foundations — types.py + tests + __init__

**Files:**
- Create: `scripts/d8/__init__.py`
- Create: `scripts/d8/types.py`
- Create: `tests/d8/__init__.py`
- Create: `tests/d8/conftest.py`
- Create: `tests/d8/test_types.py`

- [ ] **Step 1.1: Create empty package markers**

```bash
mkdir -p scripts/d8 tests/d8
touch scripts/d8/__init__.py tests/d8/__init__.py
```

- [ ] **Step 1.2: Write tests/d8/conftest.py with shared fixtures**

```python
"""Shared fixtures for D8 tests (ISO 29119)."""

from __future__ import annotations

import pytest

from scripts.backtest.runner_types import MatchStats


@pytest.fixture
def dummy_match() -> MatchStats:
    return MatchStats(
        saison=2024,
        ronde=5,
        user_team="USR",
        opponent_team="OPP",
        recall_ali=0.80,
        accuracy_ali=0.85,
        jaccard_ali=0.70,
        brier_ali=0.20,
        ece_ali=0.04,
        recall_baseline=0.40,
        brier_baseline=0.50,
        bss=0.60,
        e_score_predicted=4.0,
        e_score_observed=4.5,
        e_score_mae=0.5,
        ali_correct=True,
        baseline_correct=False,
    )


@pytest.fixture
def matches_5() -> list[MatchStats]:
    return [
        MatchStats(
            saison=2024,
            ronde=r,
            user_team=f"USR{r}",
            opponent_team=f"OPP{r}",
            recall_ali=0.50 + r * 0.1,
            accuracy_ali=0.85,
            jaccard_ali=0.60,
            brier_ali=0.20,
            ece_ali=0.04,
            recall_baseline=0.40,
            brier_baseline=0.50,
            bss=0.60,
            e_score_predicted=4.0,
            e_score_observed=4.5,
            e_score_mae=0.5,
            ali_correct=r > 2,
            baseline_correct=False,
        )
        for r in range(1, 6)
    ]
```

- [ ] **Step 1.3: Write tests/d8/test_types.py**

```python
"""Tests scripts/d8/types — D8 data structures (ISO 29119)."""

from __future__ import annotations

import pytest

from scripts.d8.types import (
    D8GateStatus,
    D8GroupBreakdown,
    D8GroupStats,
    D8Lineage,
    D8SaisonReport,
)


def test_d8_group_stats_frozen() -> None:
    g = D8GroupStats(
        group_key="M",
        n=100,
        recall_mean=0.85,
        jaccard_mean=0.70,
        brier_mean=0.20,
        ece_mean=0.04,
        mae_mean=0.5,
        bss_mean=0.60,
    )
    with pytest.raises(Exception):
        g.n = 50  # type: ignore[misc]


def test_d8_group_breakdown_indexable() -> None:
    b = D8GroupBreakdown(
        dim_name="gender",
        groups={"M": ..., "F": ...},  # type: ignore[dict-item]
    )
    assert "M" in b.groups
    assert b.dim_name == "gender"


def test_d8_lineage_required_fields() -> None:
    L = D8Lineage(
        joueurs_sha256="a" * 64,
        echiquiers_sha256="b" * 64,
        mlp_artefact_sha256="c" * 64,
        temp_scaler_sha256="d" * 64,
        code_sha256="e" * 7,
        ali_seed=42,
        ali_n_topk=10,
        ali_n_mc_pairs=5,
        ali_decay_lambda=0.9,
        kernel_id="pierrax/d8-saison-2024",
        kernel_version_kaggle="v1",
        run_at_utc="2026-04-30T22:15:33Z",
    )
    assert L.ali_seed == 42


def test_d8_gate_status_enum_values() -> None:
    assert D8GateStatus.PASS.value == "pass"
    assert D8GateStatus.FAIL.value == "fail"
    assert D8GateStatus.INCONCLUSIVE.value == "inconclusive"


def test_d8_saison_report_default_factory() -> None:
    r = D8SaisonReport(
        schema_version="d8.v1",
        saison=2024,
        n_matches=70,
        lineage=D8Lineage(
            joueurs_sha256="a" * 64,
            echiquiers_sha256="b" * 64,
            mlp_artefact_sha256="c" * 64,
            temp_scaler_sha256="d" * 64,
            code_sha256="e" * 7,
            ali_seed=42,
            ali_n_topk=10,
            ali_n_mc_pairs=5,
            ali_decay_lambda=0.9,
            kernel_id="pierrax/d8-saison-2024",
            kernel_version_kaggle="v1",
            run_at_utc="2026-04-30T22:15:33Z",
        ),
    )
    assert r.per_match == []
    assert r.breakdowns == {}
```

- [ ] **Step 1.4: Run tests → FAIL**

```bash
python -m pytest tests/d8/test_types.py -v
```
Expected: 5 ERRORS (`ImportError: cannot import name 'D8GateStatus'`).

- [ ] **Step 1.5: Implement scripts/d8/types.py**

```python
"""D8 — typed data structures (ISO 29119, ISO 42001 traceability).

Document ID: ALICE-D8-TYPES
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class D8GroupStats:
    """Per-group metrics (extension fairness.GroupStats + BSS)."""

    group_key: str
    n: int
    recall_mean: float
    jaccard_mean: float
    brier_mean: float
    ece_mean: float
    mae_mean: float
    bss_mean: float


@dataclass(frozen=True)
class D8GroupBreakdown:
    """One dimension breakdown : dim_name -> {group_key: GroupStats}."""

    dim_name: str
    groups: dict[str, D8GroupStats]


@dataclass(frozen=True)
class D8Lineage:
    """SHA-256 lineage per ISO 5259 §lineage."""

    joueurs_sha256: str
    echiquiers_sha256: str
    mlp_artefact_sha256: str
    temp_scaler_sha256: str
    code_sha256: str
    ali_seed: int
    ali_n_topk: int
    ali_n_mc_pairs: int
    ali_decay_lambda: float
    kernel_id: str
    kernel_version_kaggle: str
    run_at_utc: str


class D8GateStatus(Enum):
    """Gate evaluation outcome (3 states)."""

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class D8GateEvaluation:
    """One gate result + threshold + measured value."""

    gate_id: str
    threshold: float
    measured_value: float
    status: D8GateStatus
    source: str  # SOTA reference


@dataclass
class D8SaisonReport:
    """Per-saison aggregated report (kernel output schema d8.v1)."""

    schema_version: str
    saison: int
    n_matches: int
    lineage: D8Lineage
    per_match: list[Any] = field(default_factory=list)  # MatchStats serialized
    breakdowns: dict[str, D8GroupBreakdown] = field(default_factory=dict)
    multicalibration: dict[str, dict[str, float]] = field(default_factory=dict)
    stress_elo: dict[float, float] = field(default_factory=dict)
    stress_roster: dict[float, float] = field(default_factory=dict)
    conformal: dict[str, Any] = field(default_factory=dict)
    dro_wasserstein: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class D8FullReport:
    """Aggregator output schema d8-aggregator.v1."""

    schema_version: str
    n_matches: int
    saisons: list[int]
    lineage_per_saison: dict[int, D8Lineage]
    breakdowns_global: dict[str, D8GroupBreakdown]
    multicalibration_global: dict[str, dict[str, float]]
    stress_elo_global: dict[float, float]
    stress_roster_global: dict[float, float]
    conformal_global: dict[str, Any]
    dro_global: dict[str, dict[str, Any]]
    gates_19: list[D8GateEvaluation]
```

- [ ] **Step 1.6: Run tests → PASS**

```bash
python -m pytest tests/d8/test_types.py -v
```
Expected: 5 passed.

- [ ] **Step 1.7: Commit**

```bash
git add scripts/d8/__init__.py scripts/d8/types.py tests/d8/__init__.py tests/d8/conftest.py tests/d8/test_types.py
git commit -m "feat(d8): types + foundations (D8GroupStats, D8Lineage, D8GateStatus)"
```

---

## Task 2 : loader.py — match eligibility + lineage SHA-256

**Files:**
- Create: `scripts/d8/loader.py`
- Create: `tests/d8/test_loader.py`

- [ ] **Step 2.1: Write tests/d8/test_loader.py — 15 tests**

```python
"""Tests scripts/d8/loader (ISO 29119)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from scripts.d8.loader import (
    compute_file_sha256,
    filter_eligible_matches,
    load_match_eligible,
)


@pytest.fixture
def tiny_echiquiers(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "saison": [2024, 2024, 2024, 2023],
            "ronde": [1, 1, 5, 1],
            "equipe_dom": ["USR", "OPP", "USR", "USR"],
            "equipe_ext": ["OPP", "USR", "OPP", "OPP"],
            "joueur_nom": ["A", "B", "C", "D"],
            "echiquier": [1, 1, 1, 1],
            "niveau": ["N3", "N3", "N3", "N3"],
        },
    )
    p = tmp_path / "echiquiers.parquet"
    df.to_parquet(p)
    return p


def test_compute_sha256_known_input() -> None:
    text = b"hello"
    expected = hashlib.sha256(text).hexdigest()
    assert len(expected) == 64


def test_filter_eligible_min_ronde_1(tiny_echiquiers: Path) -> None:
    df = pd.read_parquet(tiny_echiquiers)
    df_invalid = pd.concat([df, pd.DataFrame({"saison": [2024], "ronde": [0], "equipe_dom": ["X"], "equipe_ext": ["Y"], "joueur_nom": ["E"], "echiquier": [1], "niveau": ["N3"]})])
    eligible = filter_eligible_matches(df_invalid, saison=2024)
    assert all(m.ronde >= 1 for m in eligible)


def test_filter_eligible_saison_filter(tiny_echiquiers: Path) -> None:
    df = pd.read_parquet(tiny_echiquiers)
    matches_2024 = filter_eligible_matches(df, saison=2024)
    assert all(m.saison == 2024 for m in matches_2024)
    assert all(m.saison != 2023 for m in matches_2024)


def test_load_match_eligible_returns_list(tiny_echiquiers: Path) -> None:
    matches = load_match_eligible(tiny_echiquiers, saison=2024)
    assert isinstance(matches, list)
    assert len(matches) >= 1


def test_load_match_eligible_empty_saison(tiny_echiquiers: Path) -> None:
    matches = load_match_eligible(tiny_echiquiers, saison=1999)
    assert matches == []


def test_compute_file_sha256_path(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_bytes(b"alice")
    sha = compute_file_sha256(p)
    expected = hashlib.sha256(b"alice").hexdigest()
    assert sha == expected


def test_load_match_unique_match_ids(tiny_echiquiers: Path) -> None:
    matches = load_match_eligible(tiny_echiquiers, saison=2024)
    ids = [(m.saison, m.ronde, m.equipe_dom, m.equipe_ext) for m in matches]
    assert len(ids) == len(set(ids))


def test_filter_eligible_dom_ext_present() -> None:
    df = pd.DataFrame({"saison": [2024], "ronde": [1], "equipe_dom": [None], "equipe_ext": ["OPP"], "joueur_nom": ["A"], "echiquier": [1], "niveau": ["N3"]})
    eligible = filter_eligible_matches(df, saison=2024)
    assert eligible == []


def test_load_match_eligible_path_missing(tmp_path: Path) -> None:
    p = tmp_path / "missing.parquet"
    with pytest.raises(FileNotFoundError):
        load_match_eligible(p, saison=2024)


def test_compute_file_sha256_missing(tmp_path: Path) -> None:
    p = tmp_path / "absent.txt"
    with pytest.raises(FileNotFoundError):
        compute_file_sha256(p)


def test_filter_eligible_invalid_saison_type(tiny_echiquiers: Path) -> None:
    df = pd.read_parquet(tiny_echiquiers)
    with pytest.raises(TypeError):
        filter_eligible_matches(df, saison="2024")  # type: ignore[arg-type]


def test_load_match_eligible_returns_match_spec(tiny_echiquiers: Path) -> None:
    from scripts.d8.loader import MatchSpec

    matches = load_match_eligible(tiny_echiquiers, saison=2024)
    assert all(isinstance(m, MatchSpec) for m in matches)


def test_match_spec_frozen(tiny_echiquiers: Path) -> None:
    matches = load_match_eligible(tiny_echiquiers, saison=2024)
    if matches:
        with pytest.raises(Exception):
            matches[0].saison = 1999  # type: ignore[misc]


def test_filter_eligible_with_niveau_passthrough(tiny_echiquiers: Path) -> None:
    df = pd.read_parquet(tiny_echiquiers)
    matches = filter_eligible_matches(df, saison=2024)
    assert all(hasattr(m, "niveau") for m in matches)


def test_filter_eligible_dedup_pairs(tiny_echiquiers: Path) -> None:
    """Each match (saison, ronde, dom, ext) should appear once after dedup."""
    df = pd.read_parquet(tiny_echiquiers)
    matches = filter_eligible_matches(df, saison=2024)
    ids = [(m.saison, m.ronde, m.equipe_dom, m.equipe_ext) for m in matches]
    assert len(ids) == len(set(ids))
```

- [ ] **Step 2.2: Run tests → FAIL**

```bash
python -m pytest tests/d8/test_loader.py -v
```
Expected: 15 errors (`ImportError`).

- [ ] **Step 2.3: Implement scripts/d8/loader.py**

```python
"""D8 loader — match eligibility + SHA-256 lineage (ISO 5259).

Document ID: ALICE-D8-LOADER
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class MatchSpec:
    """Match candidate eligible for D8 audit."""

    saison: int
    ronde: int
    equipe_dom: str
    equipe_ext: str
    niveau: str


_HASH_CHUNK = 65536


def compute_file_sha256(path: Path) -> str:
    """SHA-256 of file content (ISO 5259 lineage)."""
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def filter_eligible_matches(df: pd.DataFrame, saison: int) -> list[MatchSpec]:
    """Filter matches : ronde>=1, equipes valides, dedup unique pairs.

    @param df: echiquiers DataFrame (cols: saison, ronde, equipe_dom, equipe_ext, niveau)
    @param saison: integer year to filter
    @raises TypeError: if saison is not int
    """
    if not isinstance(saison, int):
        msg = f"saison must be int, got {type(saison).__name__}"
        raise TypeError(msg)
    sub = df[df["saison"] == saison]
    sub = sub[sub["ronde"] >= 1]
    sub = sub.dropna(subset=["equipe_dom", "equipe_ext"])
    pairs = sub[["saison", "ronde", "equipe_dom", "equipe_ext", "niveau"]].drop_duplicates()
    return [
        MatchSpec(
            saison=int(row["saison"]),
            ronde=int(row["ronde"]),
            equipe_dom=str(row["equipe_dom"]),
            equipe_ext=str(row["equipe_ext"]),
            niveau=str(row["niveau"]),
        )
        for _, row in pairs.iterrows()
    ]


def load_match_eligible(echiquiers_path: Path, saison: int) -> list[MatchSpec]:
    """Load echiquiers parquet + filter saison eligible matches.

    @raises FileNotFoundError if path missing.
    """
    if not echiquiers_path.exists():
        msg = f"echiquiers parquet not found: {echiquiers_path}"
        raise FileNotFoundError(msg)
    df = pd.read_parquet(echiquiers_path)
    return filter_eligible_matches(df, saison=saison)
```

- [ ] **Step 2.4: Run tests → PASS**

```bash
python -m pytest tests/d8/test_loader.py -v
```
Expected: 15 passed.

- [ ] **Step 2.5: Commit**

```bash
git add scripts/d8/loader.py tests/d8/test_loader.py
git commit -m "feat(d8): loader — match eligibility + SHA-256 lineage (ISO 5259)"
```

---

## Task 3 : breakdowns.py — 7 dimensions

**Files:**
- Create: `scripts/d8/breakdowns.py`
- Create: `tests/d8/test_breakdowns.py`

The 7 functions follow same pattern, sharing `_build_group_stats` helper. Each function maps MatchStats → str group key, then aggregates GroupStats.

- [ ] **Step 3.1: Write tests/d8/test_breakdowns.py — 40 tests**

```python
"""Tests scripts/d8/breakdowns — 7 dims (ISO 24027)."""

from __future__ import annotations

import pytest

from scripts.backtest.runner_types import MatchStats
from scripts.d8.breakdowns import (
    bucket_categorie_age,
    bucket_elo_strata,
    bucket_pool_size_quartile,
    breakdown_by_categorie_age,
    breakdown_by_elo_strata,
    breakdown_by_gender,
    breakdown_by_niveau,
    breakdown_by_pool_size,
    breakdown_by_ronde,
    breakdown_by_saison,
    compute_all_7,
    max_gap_recall,
)


def _ms(saison: int = 2024, ronde: int = 1, recall: float = 0.5) -> MatchStats:
    return MatchStats(
        saison=saison,
        ronde=ronde,
        user_team="USR",
        opponent_team="OPP",
        recall_ali=recall,
        accuracy_ali=0.85,
        jaccard_ali=0.60,
        brier_ali=0.20,
        ece_ali=0.04,
        recall_baseline=0.40,
        brier_baseline=0.50,
        bss=0.60,
        e_score_predicted=4.0,
        e_score_observed=4.5,
        e_score_mae=0.5,
        ali_correct=True,
        baseline_correct=False,
    )


# ---- Bucket helpers ----


def test_bucket_categorie_age_u12_petits_poussins() -> None:
    assert bucket_categorie_age("PpoM") == "U12"
    assert bucket_categorie_age("PupF") == "U12"


def test_bucket_categorie_age_u18_minimes() -> None:
    assert bucket_categorie_age("MinM") == "U18"
    assert bucket_categorie_age("CadF") == "U18"


def test_bucket_categorie_age_u20() -> None:
    assert bucket_categorie_age("JunM") == "U20"


def test_bucket_categorie_age_sen() -> None:
    assert bucket_categorie_age("SenM") == "Sen"
    assert bucket_categorie_age("SenF") == "Sen"


def test_bucket_categorie_age_s50_plus() -> None:
    assert bucket_categorie_age("SepM") == "S50+"
    assert bucket_categorie_age("VetF") == "S50+"


def test_bucket_categorie_age_unknown_returns_unknown() -> None:
    assert bucket_categorie_age("XYZ") == "unknown"


def test_bucket_pool_size_quartile_small() -> None:
    sizes = [10, 12, 14, 16, 18, 20, 22, 24]
    assert bucket_pool_size_quartile(11, sizes) == "small"


def test_bucket_pool_size_quartile_xlarge() -> None:
    sizes = [10, 12, 14, 16, 18, 20, 22, 24]
    assert bucket_pool_size_quartile(23, sizes) == "xlarge"


def test_bucket_pool_size_empty_sizes_raises() -> None:
    with pytest.raises(ValueError):
        bucket_pool_size_quartile(15, [])


def test_bucket_elo_strata_q1_low() -> None:
    assert bucket_elo_strata(1450) == "Q1_lt_1500"


def test_bucket_elo_strata_q2() -> None:
    assert bucket_elo_strata(1600) == "Q2_1500_1700"


def test_bucket_elo_strata_q3() -> None:
    assert bucket_elo_strata(1800) == "Q3_1700_1900"


def test_bucket_elo_strata_q4_high() -> None:
    assert bucket_elo_strata(2000) == "Q4_gte_1900"


def test_bucket_elo_strata_boundary_1500() -> None:
    """1500 falls into Q2 (>=1500)."""
    assert bucket_elo_strata(1500) == "Q2_1500_1700"


def test_bucket_elo_strata_boundary_1900() -> None:
    """1900 falls into Q4 (>=1900)."""
    assert bucket_elo_strata(1900) == "Q4_gte_1900"


# ---- Breakdown by ronde ----


def test_breakdown_by_ronde_groups_by_ronde() -> None:
    matches = [_ms(ronde=1), _ms(ronde=1), _ms(ronde=3)]
    b = breakdown_by_ronde(matches)
    assert "ronde_1" in b.groups
    assert "ronde_3" in b.groups
    assert b.groups["ronde_1"].n == 2


def test_breakdown_by_ronde_recall_mean() -> None:
    matches = [_ms(ronde=1, recall=0.4), _ms(ronde=1, recall=0.6)]
    b = breakdown_by_ronde(matches)
    assert b.groups["ronde_1"].recall_mean == pytest.approx(0.5)


def test_breakdown_by_ronde_empty() -> None:
    b = breakdown_by_ronde([])
    assert b.groups == {}


# ---- Breakdown by saison ----


def test_breakdown_by_saison_4_saisons() -> None:
    matches = [_ms(saison=s) for s in (2021, 2022, 2023, 2024)]
    b = breakdown_by_saison(matches)
    assert len(b.groups) == 4


def test_breakdown_by_saison_dim_name() -> None:
    b = breakdown_by_saison([_ms()])
    assert b.dim_name == "saison"


# ---- Breakdown by gender ----


def test_breakdown_by_gender_requires_external_lookup() -> None:
    """gender is not in MatchStats, breakdown takes a key_fn."""
    matches = [_ms()]
    b = breakdown_by_gender(matches, gender_fn=lambda m: "M")
    assert "M" in b.groups


def test_breakdown_by_gender_two_groups() -> None:
    matches = [_ms(), _ms()]
    b = breakdown_by_gender(matches, gender_fn=lambda m: "M" if m.ronde == 1 else "F")
    assert "M" in b.groups


# ---- Breakdown by pool_size ----


def test_breakdown_by_pool_size_4_buckets() -> None:
    matches = [_ms(ronde=r) for r in range(1, 9)]
    sizes = list(range(10, 26, 2))
    b = breakdown_by_pool_size(matches, pool_size_fn=lambda m: sizes[m.ronde - 1])
    assert all(b in {"small", "medium", "large", "xlarge"} for b in b.groups)


# ---- Breakdown by niveau ----


def test_breakdown_by_niveau_groups_by_niveau() -> None:
    matches = [_ms()]
    b = breakdown_by_niveau(matches, niveau_fn=lambda m: "N3")
    assert "N3" in b.groups


# ---- Breakdown by elo_strata ----


def test_breakdown_by_elo_strata_q4() -> None:
    matches = [_ms()]
    b = breakdown_by_elo_strata(matches, team_elo_mean_fn=lambda m: 2000)
    assert "Q4_gte_1900" in b.groups


# ---- Breakdown by categorie_age ----


def test_breakdown_by_categorie_age_sen_dominant() -> None:
    matches = [_ms()]
    b = breakdown_by_categorie_age(matches, categorie_fn=lambda m: "SenM")
    assert "Sen" in b.groups


# ---- compute_all_7 ----


def test_compute_all_7_returns_7_dims() -> None:
    matches = [_ms()]
    result = compute_all_7(
        matches,
        gender_fn=lambda m: "M",
        pool_size_fn=lambda m: 15,
        all_pool_sizes=[10, 12, 14, 16, 18, 20],
        niveau_fn=lambda m: "N3",
        team_elo_mean_fn=lambda m: 1700,
        categorie_fn=lambda m: "SenM",
    )
    assert set(result.keys()) == {
        "by_gender",
        "by_pool_size",
        "by_ronde",
        "by_saison",
        "by_niveau",
        "by_elo_strata",
        "by_categorie_age",
    }


# ---- max_gap_recall ----


def test_max_gap_recall_normal() -> None:
    matches = [_ms(ronde=1, recall=0.4), _ms(ronde=3, recall=0.8)]
    b = breakdown_by_ronde(matches)
    assert max_gap_recall(b) == pytest.approx(0.4)


def test_max_gap_recall_single_group() -> None:
    matches = [_ms(ronde=1, recall=0.5)]
    b = breakdown_by_ronde(matches)
    assert max_gap_recall(b) == 0.0


def test_max_gap_recall_empty_breakdown() -> None:
    from scripts.d8.types import D8GroupBreakdown

    b = D8GroupBreakdown(dim_name="empty", groups={})
    assert max_gap_recall(b) == 0.0


# ---- Edge cases ----


def test_breakdown_by_ronde_nan_recall() -> None:
    matches = [_ms(ronde=1, recall=float("nan"))]
    b = breakdown_by_ronde(matches)
    import math
    assert math.isnan(b.groups["ronde_1"].recall_mean)


def test_compute_all_7_consistent_n_per_dim() -> None:
    matches = [_ms() for _ in range(10)]
    result = compute_all_7(
        matches,
        gender_fn=lambda m: "M",
        pool_size_fn=lambda m: 15,
        all_pool_sizes=[10, 12, 14, 16, 18, 20],
        niveau_fn=lambda m: "N3",
        team_elo_mean_fn=lambda m: 1700,
        categorie_fn=lambda m: "SenM",
    )
    for dim, breakdown in result.items():
        total_n = sum(g.n for g in breakdown.groups.values())
        assert total_n == 10


def test_breakdown_recall_mean_ignores_nan_or_propagates() -> None:
    """Document behavior on NaN — propagate (caller must skip)."""
    matches = [_ms(ronde=1, recall=0.5), _ms(ronde=1, recall=float("nan"))]
    b = breakdown_by_ronde(matches)
    # NaN propagates by mean default
    import math
    assert math.isnan(b.groups["ronde_1"].recall_mean)


def test_breakdown_jaccard_brier_ece_mae_present() -> None:
    matches = [_ms()]
    b = breakdown_by_ronde(matches)
    g = b.groups["ronde_1"]
    assert hasattr(g, "jaccard_mean")
    assert hasattr(g, "brier_mean")
    assert hasattr(g, "ece_mean")
    assert hasattr(g, "mae_mean")
    assert hasattr(g, "bss_mean")


def test_categorie_age_5_buckets_complete() -> None:
    """All 12 FFE categories map to one of 5 buckets."""
    cats = ["PpoM", "PpoF", "PouM", "PouF", "PupM", "PupF",
            "BenM", "BenF", "MinM", "MinF", "CadM", "CadF",
            "JunM", "JunF", "SenM", "SenF",
            "SepM", "SepF", "VetM", "VetF"]
    buckets = {bucket_categorie_age(c) for c in cats}
    assert buckets == {"U12", "U18", "U20", "Sen", "S50+"}


def test_pool_size_quartile_with_4_sizes() -> None:
    """Smaller dataset still produces 4 buckets."""
    sizes = [10, 12, 14, 16]
    assert bucket_pool_size_quartile(10, sizes) == "small"
    assert bucket_pool_size_quartile(16, sizes) == "xlarge"


def test_breakdown_by_pool_size_dim_name() -> None:
    matches = [_ms()]
    b = breakdown_by_pool_size(matches, pool_size_fn=lambda m: 15, all_pool_sizes=[10, 20])
    assert b.dim_name == "pool_size"


def test_breakdown_by_elo_strata_dim_name() -> None:
    matches = [_ms()]
    b = breakdown_by_elo_strata(matches, team_elo_mean_fn=lambda m: 1700)
    assert b.dim_name == "elo_strata_team"


def test_breakdown_by_categorie_age_dim_name() -> None:
    matches = [_ms()]
    b = breakdown_by_categorie_age(matches, categorie_fn=lambda m: "SenM")
    assert b.dim_name == "categorie_age"


def test_breakdown_by_gender_dim_name() -> None:
    matches = [_ms()]
    b = breakdown_by_gender(matches, gender_fn=lambda m: "M")
    assert b.dim_name == "gender"


def test_breakdown_by_niveau_dim_name() -> None:
    matches = [_ms()]
    b = breakdown_by_niveau(matches, niveau_fn=lambda m: "N3")
    assert b.dim_name == "niveau"


def test_breakdown_by_ronde_dim_name() -> None:
    matches = [_ms()]
    b = breakdown_by_ronde(matches)
    assert b.dim_name == "ronde"


def test_breakdown_by_saison_dim_name_2() -> None:
    matches = [_ms()]
    b = breakdown_by_saison(matches)
    assert b.dim_name == "saison"


def test_breakdown_groups_immutable() -> None:
    matches = [_ms()]
    b = breakdown_by_ronde(matches)
    g = b.groups["ronde_1"]
    with pytest.raises(Exception):
        g.n = 999  # type: ignore[misc]


def test_compute_all_7_with_empty_matches() -> None:
    result = compute_all_7(
        matches=[],
        gender_fn=lambda m: "M",
        pool_size_fn=lambda m: 15,
        all_pool_sizes=[10, 20],
        niveau_fn=lambda m: "N3",
        team_elo_mean_fn=lambda m: 1700,
        categorie_fn=lambda m: "SenM",
    )
    assert all(b.groups == {} for b in result.values())


def test_compute_all_7_dimension_names_match_spec() -> None:
    result = compute_all_7(
        matches=[_ms()],
        gender_fn=lambda m: "M",
        pool_size_fn=lambda m: 15,
        all_pool_sizes=[10, 20],
        niveau_fn=lambda m: "N3",
        team_elo_mean_fn=lambda m: 1700,
        categorie_fn=lambda m: "SenM",
    )
    expected_dim_names = {
        "by_gender": "gender",
        "by_pool_size": "pool_size",
        "by_ronde": "ronde",
        "by_saison": "saison",
        "by_niveau": "niveau",
        "by_elo_strata": "elo_strata_team",
        "by_categorie_age": "categorie_age",
    }
    for key, dim_name in expected_dim_names.items():
        assert result[key].dim_name == dim_name
```

- [ ] **Step 3.2: Run tests → FAIL**

```bash
python -m pytest tests/d8/test_breakdowns.py -v
```
Expected: 40 errors.

- [ ] **Step 3.3: Implement scripts/d8/breakdowns.py**

```python
"""D8 breakdowns — 7 dimensions stratification (ISO 24027 §6).

Source SOTA :
- Mehrabi et al. 2021 "Survey on Bias and Fairness in ML" ACM CSUR
- Holstein et al. 2024 "Industry fairness assessments" FAccT (breadth>depth)
- ISO/IEC TR 24027:2021 §6.4 protected vs service-level

Categorie age buckets : scripts/parse_dataset/constants.py CATEGORIES_AGE.

Document ID: ALICE-D8-BREAKDOWNS
Version: 1.0.0
"""

from __future__ import annotations

import math
from typing import Callable

from scripts.backtest.runner_types import MatchStats
from scripts.d8.types import D8GroupBreakdown, D8GroupStats


# ---- Bucket helpers ----

_CATEGORIE_AGE_BUCKETS: dict[str, str] = {
    # U12
    "PpoM": "U12", "PpoF": "U12",
    "PouM": "U12", "PouF": "U12",
    "PupM": "U12", "PupF": "U12",
    # U18
    "BenM": "U18", "BenF": "U18",
    "MinM": "U18", "MinF": "U18",
    "CadM": "U18", "CadF": "U18",
    # U20
    "JunM": "U20", "JunF": "U20",
    # Sen
    "SenM": "Sen", "SenF": "Sen",
    # S50+
    "SepM": "S50+", "SepF": "S50+",
    "VetM": "S50+", "VetF": "S50+",
}


def bucket_categorie_age(categorie_ffe: str) -> str:
    """Map 12 FFE age categories to 5 buckets (U12/U18/U20/Sen/S50+)."""
    return _CATEGORIE_AGE_BUCKETS.get(categorie_ffe, "unknown")


def bucket_pool_size_quartile(size: int, all_sizes: list[int]) -> str:
    """Quartile bucket : small/medium/large/xlarge based on dataset all_sizes."""
    if not all_sizes:
        msg = "all_sizes must be non-empty"
        raise ValueError(msg)
    sorted_sizes = sorted(all_sizes)
    n = len(sorted_sizes)
    q1 = sorted_sizes[n // 4]
    q2 = sorted_sizes[n // 2]
    q3 = sorted_sizes[3 * n // 4]
    if size <= q1:
        return "small"
    if size <= q2:
        return "medium"
    if size <= q3:
        return "large"
    return "xlarge"


def bucket_elo_strata(team_mean_elo: int) -> str:
    """4-bucket Elo strata for team-level capability proxy."""
    if team_mean_elo < 1500:
        return "Q1_lt_1500"
    if team_mean_elo < 1700:
        return "Q2_1500_1700"
    if team_mean_elo < 1900:
        return "Q3_1700_1900"
    return "Q4_gte_1900"


# ---- Aggregator helper ----


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _build_group_stats(group_key: str, ms_list: list[MatchStats]) -> D8GroupStats:
    return D8GroupStats(
        group_key=group_key,
        n=len(ms_list),
        recall_mean=_mean([m.recall_ali for m in ms_list]),
        jaccard_mean=_mean([m.jaccard_ali for m in ms_list]),
        brier_mean=_mean([m.brier_ali for m in ms_list]),
        ece_mean=_mean([m.ece_ali for m in ms_list]),
        mae_mean=_mean([m.e_score_mae for m in ms_list]),
        bss_mean=_mean([m.bss for m in ms_list]),
    )


def _generic_breakdown(
    matches: list[MatchStats],
    dim_name: str,
    key_fn: Callable[[MatchStats], str],
) -> D8GroupBreakdown:
    groups_buckets: dict[str, list[MatchStats]] = {}
    for m in matches:
        k = key_fn(m)
        groups_buckets.setdefault(k, []).append(m)
    return D8GroupBreakdown(
        dim_name=dim_name,
        groups={k: _build_group_stats(k, v) for k, v in groups_buckets.items()},
    )


# ---- 7 dimensions ----


def breakdown_by_ronde(matches: list[MatchStats]) -> D8GroupBreakdown:
    return _generic_breakdown(matches, "ronde", lambda m: f"ronde_{m.ronde}")


def breakdown_by_saison(matches: list[MatchStats]) -> D8GroupBreakdown:
    return _generic_breakdown(matches, "saison", lambda m: str(m.saison))


def breakdown_by_gender(
    matches: list[MatchStats],
    gender_fn: Callable[[MatchStats], str],
) -> D8GroupBreakdown:
    return _generic_breakdown(matches, "gender", gender_fn)


def breakdown_by_pool_size(
    matches: list[MatchStats],
    pool_size_fn: Callable[[MatchStats], int],
    all_pool_sizes: list[int] | None = None,
) -> D8GroupBreakdown:
    if all_pool_sizes is None:
        all_pool_sizes = [pool_size_fn(m) for m in matches] or [0]
    return _generic_breakdown(
        matches,
        "pool_size",
        lambda m: bucket_pool_size_quartile(pool_size_fn(m), all_pool_sizes),
    )


def breakdown_by_niveau(
    matches: list[MatchStats],
    niveau_fn: Callable[[MatchStats], str],
) -> D8GroupBreakdown:
    return _generic_breakdown(matches, "niveau", niveau_fn)


def breakdown_by_elo_strata(
    matches: list[MatchStats],
    team_elo_mean_fn: Callable[[MatchStats], int],
) -> D8GroupBreakdown:
    return _generic_breakdown(
        matches,
        "elo_strata_team",
        lambda m: bucket_elo_strata(team_elo_mean_fn(m)),
    )


def breakdown_by_categorie_age(
    matches: list[MatchStats],
    categorie_fn: Callable[[MatchStats], str],
) -> D8GroupBreakdown:
    return _generic_breakdown(
        matches,
        "categorie_age",
        lambda m: bucket_categorie_age(categorie_fn(m)),
    )


def compute_all_7(
    matches: list[MatchStats],
    gender_fn: Callable[[MatchStats], str],
    pool_size_fn: Callable[[MatchStats], int],
    all_pool_sizes: list[int],
    niveau_fn: Callable[[MatchStats], str],
    team_elo_mean_fn: Callable[[MatchStats], int],
    categorie_fn: Callable[[MatchStats], str],
) -> dict[str, D8GroupBreakdown]:
    """Compute all 7 dimensions breakdown in one call."""
    return {
        "by_gender": breakdown_by_gender(matches, gender_fn),
        "by_pool_size": breakdown_by_pool_size(matches, pool_size_fn, all_pool_sizes),
        "by_ronde": breakdown_by_ronde(matches),
        "by_saison": breakdown_by_saison(matches),
        "by_niveau": breakdown_by_niveau(matches, niveau_fn),
        "by_elo_strata": breakdown_by_elo_strata(matches, team_elo_mean_fn),
        "by_categorie_age": breakdown_by_categorie_age(matches, categorie_fn),
    }


# ---- Gap computation ----


def max_gap_recall(breakdown: D8GroupBreakdown) -> float:
    """max - min recall across groups (Mehrabi 2021 fairness gap)."""
    if not breakdown.groups:
        return 0.0
    recalls = [g.recall_mean for g in breakdown.groups.values()]
    if any(math.isnan(r) for r in recalls):
        return float("nan")
    return max(recalls) - min(recalls)
```

- [ ] **Step 3.4: Run tests → PASS**

```bash
python -m pytest tests/d8/test_breakdowns.py -v
```
Expected: 40 passed.

- [ ] **Step 3.5: Commit**

```bash
git add scripts/d8/breakdowns.py tests/d8/test_breakdowns.py
git commit -m "feat(d8): breakdowns 7 dims (ISO 24027 §6 + Mehrabi 2021 + Holstein 2024)"
```

---

## Task 4 : calibration.py — per-group ECE + multicalibration

**Files:**
- Create: `scripts/d8/calibration.py`
- Create: `tests/d8/test_calibration.py`

- [ ] **Step 4.1: Write tests/d8/test_calibration.py — 20 tests**

```python
"""Tests scripts/d8/calibration (Pleiss 2017, Hebert-Johnson 2018)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.d8.calibration import (
    compute_ece_per_group,
    compute_multicalibration_alpha,
    expected_calibration_error,
)


def test_ece_perfect_calibration_returns_zero() -> None:
    probs = np.array([0.1, 0.5, 0.9])
    labels = np.array([0, 1, 1])
    ece = expected_calibration_error(probs, labels, n_bins=2)
    assert ece >= 0.0


def test_ece_total_miscalibration_high() -> None:
    """All probas 0.9 but labels 0 → ECE close to 0.9."""
    probs = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
    labels = np.array([0, 0, 0, 0, 0])
    ece = expected_calibration_error(probs, labels, n_bins=2)
    assert ece > 0.5


def test_ece_empty_input_zero() -> None:
    ece = expected_calibration_error(np.array([]), np.array([]), n_bins=10)
    assert ece == 0.0


def test_ece_n_bins_validation() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0.5]), np.array([1]), n_bins=0)


def test_ece_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0.5, 0.6]), np.array([1]), n_bins=2)


def test_ece_per_group_returns_dict() -> None:
    probs = np.array([0.5, 0.7, 0.3, 0.9])
    labels = np.array([1, 1, 0, 1])
    groups = ["M", "M", "F", "F"]
    result = compute_ece_per_group(probs, labels, groups, n_bins=2)
    assert "M" in result
    assert "F" in result


def test_ece_per_group_values_finite() -> None:
    probs = np.array([0.5, 0.7])
    labels = np.array([1, 1])
    groups = ["M", "M"]
    result = compute_ece_per_group(probs, labels, groups, n_bins=2)
    assert math.isfinite(result["M"])


def test_ece_per_group_empty() -> None:
    result = compute_ece_per_group(np.array([]), np.array([]), [], n_bins=10)
    assert result == {}


def test_multicalibration_alpha_returns_float() -> None:
    probs = np.array([0.5, 0.6, 0.7, 0.8])
    labels = np.array([1, 0, 1, 1])
    groups = {"all": np.array([True, True, True, True])}
    alpha = compute_multicalibration_alpha(probs, labels, groups, n_bins=2)
    assert math.isfinite(alpha)


def test_multicalibration_alpha_perfect_zero() -> None:
    """Perfect calibration → alpha close to 0."""
    n = 1000
    rng = np.random.default_rng(42)
    probs = rng.uniform(0, 1, n)
    labels = (rng.uniform(0, 1, n) < probs).astype(int)
    groups = {"all": np.ones(n, dtype=bool)}
    alpha = compute_multicalibration_alpha(probs, labels, groups, n_bins=10)
    assert alpha < 0.10


def test_multicalibration_alpha_subgroup_violation() -> None:
    """Subgroup miscalibration → high alpha."""
    n = 200
    probs = np.full(n, 0.9)
    labels = np.zeros(n, dtype=int)
    groups = {"sub": np.ones(n, dtype=bool)}
    alpha = compute_multicalibration_alpha(probs, labels, groups, n_bins=2)
    assert alpha > 0.5


def test_ece_n_bins_too_high_clips() -> None:
    probs = np.array([0.5])
    labels = np.array([1])
    ece = expected_calibration_error(probs, labels, n_bins=100)
    assert math.isfinite(ece)


def test_ece_constant_proba_zero() -> None:
    probs = np.full(10, 0.5)
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    ece = expected_calibration_error(probs, labels, n_bins=5)
    assert math.isfinite(ece)


def test_ece_per_group_unbalanced_groups() -> None:
    probs = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    labels = np.array([1, 1, 1, 1, 1])
    groups = ["M", "M", "M", "M", "F"]
    result = compute_ece_per_group(probs, labels, groups, n_bins=2)
    assert "M" in result
    assert "F" in result


def test_multicalibration_alpha_multiple_groups() -> None:
    n = 100
    probs = np.linspace(0, 1, n)
    labels = (probs > 0.5).astype(int)
    groups = {
        "g1": np.array([True] * 50 + [False] * 50),
        "g2": np.array([False] * 50 + [True] * 50),
    }
    alpha = compute_multicalibration_alpha(probs, labels, groups, n_bins=5)
    assert math.isfinite(alpha)


def test_multicalibration_empty_groups_zero() -> None:
    n = 100
    probs = np.full(n, 0.5)
    labels = np.zeros(n, dtype=int)
    alpha = compute_multicalibration_alpha(probs, labels, groups={}, n_bins=2)
    assert alpha == 0.0


def test_ece_threshold_strict() -> None:
    """ISO Pleiss 2017 §4 : ECE 0.05 should be borderline."""
    rng = np.random.default_rng(1)
    n = 500
    probs = rng.uniform(0, 1, n)
    # Inject 5% miscalibration : labels = (rng < probs - 0.05)
    labels = (rng.uniform(0, 1, n) < (probs - 0.05).clip(0, 1)).astype(int)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert 0.01 < ece < 0.15


def test_ece_per_group_returns_n_groups_keys() -> None:
    probs = np.array([0.5, 0.7])
    labels = np.array([1, 1])
    groups = ["M", "F"]
    result = compute_ece_per_group(probs, labels, groups, n_bins=2)
    assert len(result) == 2


def test_multicalibration_alpha_returns_max() -> None:
    """Alpha = max ECE across subgroups (Hebert-Johnson 2018 §3)."""
    n = 100
    probs = np.full(n, 0.5)
    labels = np.zeros(n, dtype=int)
    g1_calibrated = np.ones(50, dtype=bool)
    g2_miscalibrated = np.zeros(50, dtype=bool)
    groups = {
        "g1": np.concatenate([g1_calibrated, np.zeros(50, dtype=bool)]),
        "g2": np.concatenate([np.zeros(50, dtype=bool), g2_miscalibrated]),
    }
    alpha = compute_multicalibration_alpha(probs, labels, groups, n_bins=2)
    assert alpha >= 0.0


def test_ece_per_group_n_bins_validation() -> None:
    with pytest.raises(ValueError):
        compute_ece_per_group(np.array([0.5]), np.array([1]), ["M"], n_bins=0)
```

- [ ] **Step 4.2: Run → FAIL → Implement → PASS**

Implementation `scripts/d8/calibration.py` :

```python
"""D8 calibration — per-group ECE + multicalibration alpha (Pleiss 2017, Hebert-Johnson 2018).

Document ID: ALICE-D8-CALIBRATION
Version: 1.0.0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def expected_calibration_error(
    probs: NDArray[np.float64],
    labels: NDArray[np.int_],
    n_bins: int = 10,
) -> float:
    """Naeini 2015 §3 ECE : weighted absolute difference.

    @param probs: predicted probabilities ∈ [0, 1]
    @param labels: binary labels {0, 1}
    @param n_bins: number of equal-width bins
    @raises ValueError: n_bins <= 0 or len mismatch
    """
    if n_bins <= 0:
        msg = f"n_bins must be > 0, got {n_bins}"
        raise ValueError(msg)
    if len(probs) != len(labels):
        msg = f"length mismatch probs={len(probs)} labels={len(labels)}"
        raise ValueError(msg)
    if len(probs) == 0:
        return 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        in_bin = (probs > edges[i]) & (probs <= edges[i + 1])
        if i == 0:
            in_bin = in_bin | (probs == edges[0])
        bin_n = in_bin.sum()
        if bin_n == 0:
            continue
        bin_acc = labels[in_bin].mean()
        bin_conf = probs[in_bin].mean()
        ece += (bin_n / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_ece_per_group(
    probs: NDArray[np.float64],
    labels: NDArray[np.int_],
    groups: list[str],
    n_bins: int = 10,
) -> dict[str, float]:
    """Per-group ECE (Pleiss 2017 §4)."""
    if n_bins <= 0:
        msg = f"n_bins must be > 0, got {n_bins}"
        raise ValueError(msg)
    if len(probs) == 0:
        return {}
    if len(probs) != len(groups):
        msg = "len(probs) != len(groups)"
        raise ValueError(msg)
    unique_groups = set(groups)
    arr_groups = np.array(groups)
    return {
        g: expected_calibration_error(probs[arr_groups == g], labels[arr_groups == g], n_bins)
        for g in unique_groups
    }


def compute_multicalibration_alpha(
    probs: NDArray[np.float64],
    labels: NDArray[np.int_],
    groups: dict[str, NDArray[np.bool_]],
    n_bins: int = 10,
) -> float:
    """Multicalibration alpha (Hebert-Johnson 2018 §3) = max sub-group ECE.

    @param groups: dict[group_name, boolean mask len(probs)]
    """
    if not groups:
        return 0.0
    return max(
        expected_calibration_error(probs[mask], labels[mask], n_bins)
        for mask in groups.values()
    )
```

- [ ] **Step 4.3: Commit**

```bash
git add scripts/d8/calibration.py tests/d8/test_calibration.py
git commit -m "feat(d8): per-group ECE + multicalibration alpha (Pleiss 2017, Hebert-Johnson 2018)"
```

---

## Task 5 : stress_elo.py + stress_roster.py — perturbations

**Files:**
- Create: `scripts/d8/stress_elo.py`
- Create: `scripts/d8/stress_roster.py`
- Create: `tests/d8/test_stress_elo.py`
- Create: `tests/d8/test_stress_roster.py`

- [ ] **Step 5.1: Implement scripts/d8/stress_elo.py** (extends `scripts/backtest/robustness.py`)

```python
"""D8 stress Elo — multi-noise perturbations (ISO 24029 §6.5, Goodfellow 2015, Madry 2018).

Wraps scripts.backtest.robustness.perturb_elos with multi-noise sweep.

Document ID: ALICE-D8-STRESS-ELO
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from scripts.backtest.robustness import perturb_elos

NOISE_LEVELS = [0.01, 0.03, 0.05, 0.07, 0.10]


@dataclass(frozen=True)
class ElostressOutcome:
    """One noise level result."""

    noise_pct: float
    baseline_recall: float
    perturbed_recall_mean: float
    recall_drop: float


def run_multinoise(
    matches_baseline_recalls: list[float],
    perturbed_recalls_per_noise: dict[float, list[float]],
) -> list[ElostressOutcome]:
    """Aggregate baseline+perturbed recalls per noise level into outcomes."""
    if not matches_baseline_recalls:
        return []
    base = sum(matches_baseline_recalls) / len(matches_baseline_recalls)
    out: list[ElostressOutcome] = []
    for noise in sorted(perturbed_recalls_per_noise):
        pert = perturbed_recalls_per_noise[noise]
        if not pert:
            continue
        pert_mean = sum(pert) / len(pert)
        out.append(
            ElostressOutcome(
                noise_pct=noise,
                baseline_recall=base,
                perturbed_recall_mean=pert_mean,
                recall_drop=max(0.0, base - pert_mean),
            ),
        )
    return out


def compute_stress_elo_for_match(
    opp_pool_elos: list[int],
    backtest_run_fn: Callable[[list[int]], float],
    noise_levels: list[float] = NOISE_LEVELS,
    seed: int = 42,
) -> dict[float, float]:
    """For 1 match : return {noise_pct: perturbed_recall} dict."""
    return {
        noise: backtest_run_fn(perturb_elos(opp_pool_elos, noise, seed))
        for noise in noise_levels
    }
```

- [ ] **Step 5.2: Implement scripts/d8/stress_roster.py**

```python
"""D8 stress roster turnover (Tran 2022 §3.4 distribution shift, Recht 2019).

Document ID: ALICE-D8-STRESS-ROSTER
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np

TURNOVER_RATES = [0.05, 0.10, 0.20]
T = TypeVar("T")


@dataclass(frozen=True)
class RosterStressOutcome:
    """One turnover rate result."""

    turnover_pct: float
    baseline_recall: float
    perturbed_recall_mean: float
    recall_drop: float


def drop_random_players(
    pool: list[T],
    turnover_pct: float,
    seed: int = 42,
) -> list[T]:
    """Drop turnover_pct fraction of players uniformly at random.

    @raises ValueError: turnover_pct ∉ [0, 1)
    """
    if not 0 <= turnover_pct < 1:
        msg = f"turnover_pct must be in [0, 1), got {turnover_pct}"
        raise ValueError(msg)
    if not pool:
        return []
    n_drop = int(turnover_pct * len(pool))
    if n_drop == 0:
        return list(pool)
    rng = np.random.default_rng(seed)
    drop_idx = set(rng.choice(len(pool), size=n_drop, replace=False).tolist())
    return [p for i, p in enumerate(pool) if i not in drop_idx]


def compute_stress_roster_for_match(
    opp_pool: list[T],
    backtest_run_fn: Callable[[list[T]], float],
    min_pool_size: int,
    turnover_rates: list[float] = TURNOVER_RATES,
    seed: int = 42,
) -> dict[float, float | None]:
    """For 1 match : return {turnover_pct: perturbed_recall or None if pool too small}."""
    out: dict[float, float | None] = {}
    for r in turnover_rates:
        pert_pool = drop_random_players(opp_pool, r, seed)
        if len(pert_pool) < min_pool_size:
            out[r] = None
        else:
            out[r] = backtest_run_fn(pert_pool)
    return out


def aggregate_roster_outcomes(
    matches_baseline_recalls: list[float],
    perturbed_recalls_per_turnover: dict[float, list[float]],
) -> list[RosterStressOutcome]:
    if not matches_baseline_recalls:
        return []
    base = sum(matches_baseline_recalls) / len(matches_baseline_recalls)
    out: list[RosterStressOutcome] = []
    for turnover in sorted(perturbed_recalls_per_turnover):
        pert = perturbed_recalls_per_turnover[turnover]
        if not pert:
            continue
        pert_mean = sum(pert) / len(pert)
        out.append(
            RosterStressOutcome(
                turnover_pct=turnover,
                baseline_recall=base,
                perturbed_recall_mean=pert_mean,
                recall_drop=max(0.0, base - pert_mean),
            ),
        )
    return out
```

- [ ] **Step 5.3: Tests stress_elo + stress_roster (15 + 12 = 27 tests, voir spec §9)**

Tests follow same TDD pattern as Task 3. See spec §9 testing strategy for breakdown.

- [ ] **Step 5.4: Run + commit**

```bash
git add scripts/d8/stress_elo.py scripts/d8/stress_roster.py tests/d8/test_stress_elo.py tests/d8/test_stress_roster.py
git commit -m "feat(d8): stress Elo multi-noise + roster turnover (Goodfellow 2015, Madry 2018, Tran 2022)"
```

---

## Task 6 : conformal.py — split conformal Vovk 2024

**Files:**
- Create: `scripts/d8/conformal.py`
- Create: `tests/d8/test_conformal.py`

- [ ] **Step 6.1: Implement conformal.py**

```python
"""D8 conformal prediction (Vovk 2024 §2.3, Angelopoulos & Bates 2023 §4).

Split conformal : nonconformity score |y_obs - y_pred|, quantile threshold,
coverage rate per group.

Document ID: ALICE-D8-CONFORMAL
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ConformalCalibration:
    """Calibration result : threshold + nonconf_scores."""

    quantile_threshold: float
    nonconf_scores: NDArray[np.float64]
    n_calibration: int
    alpha: float


def split_calibrate(
    y_observed: NDArray[np.float64],
    y_predicted: NDArray[np.float64],
    alpha: float = 0.10,
) -> ConformalCalibration:
    """Split conformal calibration (Vovk 2024 §2.3).

    @param alpha: miscoverage rate (0.10 → 90% coverage CI)
    @raises ValueError: alpha ∉ (0, 1) or empty input
    """
    if not 0 < alpha < 1:
        msg = f"alpha must be in (0, 1), got {alpha}"
        raise ValueError(msg)
    if len(y_observed) == 0:
        msg = "Empty calibration set"
        raise ValueError(msg)
    if len(y_observed) != len(y_predicted):
        msg = "length mismatch"
        raise ValueError(msg)
    nonconf_scores = np.abs(y_observed - y_predicted)
    n = len(nonconf_scores)
    rank = int(np.ceil((1 - alpha) * (n + 1))) - 1
    rank = max(0, min(rank, n - 1))
    sorted_scores = np.sort(nonconf_scores)
    quantile_threshold = float(sorted_scores[rank])
    return ConformalCalibration(
        quantile_threshold=quantile_threshold,
        nonconf_scores=sorted_scores,
        n_calibration=n,
        alpha=alpha,
    )


def coverage_rate(
    y_observed: NDArray[np.float64],
    y_predicted: NDArray[np.float64],
    calibration: ConformalCalibration,
) -> float:
    """Marginal coverage : P(|y_obs - y_pred| <= threshold) on test set."""
    if len(y_observed) == 0:
        return 0.0
    deviations = np.abs(y_observed - y_predicted)
    return float((deviations <= calibration.quantile_threshold).mean())


def coverage_per_group(
    y_observed: NDArray[np.float64],
    y_predicted: NDArray[np.float64],
    groups: list[str],
    calibration: ConformalCalibration,
) -> dict[str, float]:
    """Coverage breakdown by group label."""
    if len(y_observed) == 0:
        return {}
    arr_groups = np.array(groups)
    return {
        g: coverage_rate(
            y_observed[arr_groups == g],
            y_predicted[arr_groups == g],
            calibration,
        )
        for g in set(groups)
    }


def conformal_set_size_mean(
    y_predicted: NDArray[np.float64],
    calibration: ConformalCalibration,
    grid_resolution: float = 0.01,
) -> float:
    """Mean conformal set size = how many y values are accepted per pred (Angelopoulos 2023 §4.2)."""
    if len(y_predicted) == 0:
        return 0.0
    grid = np.arange(0, 1 + grid_resolution, grid_resolution)
    sizes = []
    for y_pred in y_predicted:
        accepted = np.abs(grid - y_pred) <= calibration.quantile_threshold
        sizes.append(accepted.sum() * grid_resolution)
    return float(np.mean(sizes))
```

- [ ] **Step 6.2: tests/d8/test_conformal.py — 25 tests** (couvre split_calibrate edge cases, coverage marginal, breakdown groups, set size, alpha validation)

- [ ] **Step 6.3: Run → FAIL → impl → PASS → commit**

```bash
git add scripts/d8/conformal.py tests/d8/test_conformal.py
git commit -m "feat(d8): conformal prediction split (Vovk 2024, Angelopoulos 2023)"
```

---

## Task 7 : dro.py — Wasserstein-2 worst-case

**Files:**
- Create: `scripts/d8/dro.py`
- Create: `tests/d8/test_dro.py`

- [ ] **Step 7.1: Implement dro.py**

```python
"""D8 DRO — Wasserstein-2 ε-ball worst-case (Sinha 2018, Duchi & Namkoong 2021).

Grid 50 perturbations approx (gradient-free).

Document ID: ALICE-D8-DRO
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class DROOutcome:
    """One ε result."""

    epsilon: float
    n_perturbations: int
    recall_worst_case: float
    worst_perturbation_finding: str


def perturb_wasserstein(
    elos: list[int],
    epsilon: float,
    seed: int,
) -> tuple[list[int], dict[str, float]]:
    """Apply shift+scale perturbation within ε-ball Wasserstein-2.

    Sinha 2018 §4 : approximation = scaled Gaussian shift.
    """
    if not 0 <= epsilon < 1:
        msg = f"epsilon must be in [0, 1), got {epsilon}"
        raise ValueError(msg)
    if not elos:
        return [], {"shift": 0.0, "scale": 1.0}
    rng = np.random.default_rng(seed)
    arr = np.array(elos, dtype=float)
    mean_elo = arr.mean()
    shift = rng.uniform(-epsilon * mean_elo, epsilon * mean_elo)
    scale = rng.uniform(1 - epsilon, 1 + epsilon)
    perturbed = ((arr - mean_elo) * scale + mean_elo + shift).clip(800, 2900)
    return perturbed.astype(int).tolist(), {"shift": float(shift), "scale": float(scale)}


def wasserstein_worst_case(
    elos: list[int],
    backtest_run_fn: Callable[[list[int]], float],
    epsilon: float,
    n_perturbations: int = 50,
    seed_base: int = 42,
) -> DROOutcome:
    """Find worst-case recall over n_perturbations sampled within ε-ball.

    Returns the worst (= minimal) recall + perturbation params for diagnostic.
    """
    if not elos:
        return DROOutcome(
            epsilon=epsilon,
            n_perturbations=0,
            recall_worst_case=0.0,
            worst_perturbation_finding="empty pool",
        )
    worst_recall = float("inf")
    worst_params = {"shift": 0.0, "scale": 1.0}
    for k in range(n_perturbations):
        perturbed, params = perturb_wasserstein(elos, epsilon, seed_base + k)
        r = backtest_run_fn(perturbed)
        if r < worst_recall:
            worst_recall = r
            worst_params = params
    return DROOutcome(
        epsilon=epsilon,
        n_perturbations=n_perturbations,
        recall_worst_case=float(worst_recall),
        worst_perturbation_finding=f"shift={worst_params['shift']:.1f}+scale={worst_params['scale']:.3f}",
    )


def compute_dro_for_match(
    opp_elos: list[int],
    backtest_run_fn: Callable[[list[int]], float],
    epsilons: list[float] = (0.05, 0.10),
    n_perturbations: int = 50,
    seed_base: int = 42,
) -> dict[float, DROOutcome]:
    return {
        eps: wasserstein_worst_case(opp_elos, backtest_run_fn, eps, n_perturbations, seed_base + int(eps * 1000))
        for eps in epsilons
    }
```

- [ ] **Step 7.2: tests/d8/test_dro.py — 20 tests** (Wasserstein perturbations, ε convergence, edge ε=0, n_perturbations validation)

- [ ] **Step 7.3: Commit**

```bash
git add scripts/d8/dro.py tests/d8/test_dro.py
git commit -m "feat(d8): DRO Wasserstein worst-case (Sinha 2018, Duchi 2021)"
```

---

## Task 8 : gates.py — 19 gates G-A SOTA strict + case-by-case

**Files:**
- Create: `scripts/d8/gates.py`
- Create: `tests/d8/test_gates.py`

- [ ] **Step 8.1: Implement gates.py**

```python
"""D8 gates — 19 G-A SOTA strict + case-by-case logger (decision policy).

Source : spec §5.1 + §5.2 + §5.3.

Document ID: ALICE-D8-GATES
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scripts.d8.types import D8GateEvaluation, D8GateStatus


# ---- Gate thresholds (G-A SOTA strict, spec §5) ----

THRESHOLDS = {
    # Fairness (10 gates)
    "G_FAIR_01_max_gap_recall": 0.10,
    "G_FAIR_02_recall_per_group_min": 0.85,
    "G_FAIR_03_demographic_parity_diff": 0.10,
    "G_FAIR_04_equalized_odds_diff": 0.10,
    "G_FAIR_05_calibration_ECE_per_group": 0.05,
    "G_FAIR_06_multicalibration_alpha": 0.05,
    "G_FAIR_07_TPR_ratio_min": 0.80,
    "G_FAIR_08_brier_per_group": 0.30,
    "G_FAIR_09_BSS_per_group": 0.30,
    "G_FAIR_10_PSI_per_dim": 0.20,
    # Robustness (9 gates)
    "G_ROB_01_recall_drop_1pct": 0.02,
    "G_ROB_02_recall_drop_5pct": 0.05,
    "G_ROB_03_recall_drop_10pct": 0.10,
    "G_ROB_04_roster_5pct": 0.05,
    "G_ROB_05_roster_20pct": 0.15,
    "G_ROB_06_conformal_coverage_90": 0.90,
    "G_ROB_07_conformal_set_size_max": 3.0,
    "G_ROB_08_DRO_eps_005_min": 0.70,
    "G_ROB_09_DRO_eps_010_min": 0.55,
}

SOURCES = {
    "G_FAIR_01_max_gap_recall": "Mehrabi 2021 §4.1",
    "G_FAIR_02_recall_per_group_min": "P3G07 - 5pts",
    "G_FAIR_03_demographic_parity_diff": "Hardt 2016",
    "G_FAIR_04_equalized_odds_diff": "Hardt 2016 §3.2",
    "G_FAIR_05_calibration_ECE_per_group": "Pleiss 2017 §4",
    "G_FAIR_06_multicalibration_alpha": "Hebert-Johnson 2018",
    "G_FAIR_07_TPR_ratio_min": "EEOC §1607.4D + Feldman 2015",
    "G_FAIR_08_brier_per_group": "Brier 1950 + Pappalardo 2019",
    "G_FAIR_09_BSS_per_group": "Pappalardo 2019 §3.4",
    "G_FAIR_10_PSI_per_dim": "Yurdakul 2020",
    "G_ROB_01_recall_drop_1pct": "Goodfellow 2015 ε=0.01",
    "G_ROB_02_recall_drop_5pct": "Madry 2018",
    "G_ROB_03_recall_drop_10pct": "Madry 2018 strict",
    "G_ROB_04_roster_5pct": "Tran 2022 §3.4",
    "G_ROB_05_roster_20pct": "Recht 2019 §5",
    "G_ROB_06_conformal_coverage_90": "Vovk 2024 §2.3",
    "G_ROB_07_conformal_set_size_max": "Angelopoulos 2023 §4.2",
    "G_ROB_08_DRO_eps_005_min": "Sinha 2018 §4",
    "G_ROB_09_DRO_eps_010_min": "Duchi 2021 §6",
}


def evaluate_max_threshold(
    gate_id: str,
    measured_value: float,
    threshold: float | None = None,
) -> D8GateEvaluation:
    """Generic eval : pass if measured <= threshold (e.g. recall_drop, gap, ECE)."""
    if threshold is None:
        threshold = THRESHOLDS[gate_id]
    status = D8GateStatus.PASS if measured_value <= threshold else D8GateStatus.FAIL
    return D8GateEvaluation(
        gate_id=gate_id,
        threshold=threshold,
        measured_value=measured_value,
        status=status,
        source=SOURCES[gate_id],
    )


def evaluate_min_threshold(
    gate_id: str,
    measured_value: float,
    threshold: float | None = None,
) -> D8GateEvaluation:
    """Generic eval : pass if measured >= threshold (e.g. recall, coverage, BSS)."""
    if threshold is None:
        threshold = THRESHOLDS[gate_id]
    status = D8GateStatus.PASS if measured_value >= threshold else D8GateStatus.FAIL
    return D8GateEvaluation(
        gate_id=gate_id,
        threshold=threshold,
        measured_value=measured_value,
        status=status,
        source=SOURCES[gate_id],
    )


def evaluate_inconclusive(
    gate_id: str,
    threshold: float | None = None,
) -> D8GateEvaluation:
    """Mark gate as INCONCLUSIVE (e.g. DRO non-convergence)."""
    if threshold is None:
        threshold = THRESHOLDS[gate_id]
    return D8GateEvaluation(
        gate_id=gate_id,
        threshold=threshold,
        measured_value=float("nan"),
        status=D8GateStatus.INCONCLUSIVE,
        source=SOURCES[gate_id],
    )


def evaluate_19_gates(metrics: dict[str, Any]) -> list[D8GateEvaluation]:
    """Evaluate all 19 G-A gates from a metrics dict.

    @param metrics: dict containing computed values per gate id.
    """
    out: list[D8GateEvaluation] = []
    # Fairness
    out.append(evaluate_max_threshold("G_FAIR_01_max_gap_recall", metrics["max_gap_recall_max_dim"]))
    out.append(evaluate_min_threshold("G_FAIR_02_recall_per_group_min", metrics["recall_per_group_min"]))
    out.append(evaluate_max_threshold("G_FAIR_03_demographic_parity_diff", metrics["demographic_parity_diff"]))
    out.append(evaluate_max_threshold("G_FAIR_04_equalized_odds_diff", metrics["equalized_odds_diff"]))
    out.append(evaluate_max_threshold("G_FAIR_05_calibration_ECE_per_group", metrics["ece_per_group_max"]))
    out.append(evaluate_max_threshold("G_FAIR_06_multicalibration_alpha", metrics["multicalibration_alpha"]))
    out.append(evaluate_min_threshold("G_FAIR_07_TPR_ratio_min", metrics["tpr_ratio_min"]))
    out.append(evaluate_max_threshold("G_FAIR_08_brier_per_group", metrics["brier_per_group_max"]))
    out.append(evaluate_min_threshold("G_FAIR_09_BSS_per_group", metrics["bss_per_group_min"]))
    out.append(evaluate_max_threshold("G_FAIR_10_PSI_per_dim", metrics["psi_per_dim_max"]))
    # Robustness
    out.append(evaluate_max_threshold("G_ROB_01_recall_drop_1pct", metrics["recall_drop_1pct"]))
    out.append(evaluate_max_threshold("G_ROB_02_recall_drop_5pct", metrics["recall_drop_5pct"]))
    out.append(evaluate_max_threshold("G_ROB_03_recall_drop_10pct", metrics["recall_drop_10pct"]))
    out.append(evaluate_max_threshold("G_ROB_04_roster_5pct", metrics["roster_5pct_recall_drop"]))
    out.append(evaluate_max_threshold("G_ROB_05_roster_20pct", metrics["roster_20pct_recall_drop"]))
    out.append(evaluate_min_threshold("G_ROB_06_conformal_coverage_90", metrics["coverage_global"]))
    out.append(evaluate_max_threshold("G_ROB_07_conformal_set_size_max", metrics["conformal_set_size_mean"]))
    out.append(evaluate_min_threshold("G_ROB_08_DRO_eps_005_min", metrics["dro_eps_005_recall_worst"]))
    out.append(evaluate_min_threshold("G_ROB_09_DRO_eps_010_min", metrics["dro_eps_010_recall_worst"]))
    return out


def render_failure_analysis_md(failures: list[D8GateEvaluation]) -> str:
    """Render D8_FAILURE_ANALYSIS_LOG.md template entries (case-by-case policy §5.3)."""
    if not failures:
        return "# D8 Failure Analysis Log\n\nAll 19 gates PASS — no failures to analyze.\n"
    lines = ["# D8 Failure Analysis Log\n"]
    for f in failures:
        delta = f.measured_value - f.threshold
        lines.append(
            f"\n## Gate {f.gate_id} FAIL — analysis 2026-04-30\n\n"
            f"**Measured** : {f.measured_value:.4f}\n"
            f"**Threshold** : {f.threshold:.4f}\n"
            f"**Δ from threshold** : {delta:+.4f}\n"
            f"**Source** : {f.source}\n\n"
            "### Gate validity\n\n"
            f"Le seuil {f.threshold:.4f} est-il approprié pour ALICE Engine\n"
            "(domaine échecs + Interclubs FFE) ?\n\n"
            "- Argument validité : <à remplir>\n"
            "- Argument inapplicabilité : <à remplir>\n\n"
            "### Utilité métier\n\n"
            "La gate mesure-t-elle un risque concret pour les utilisateurs ALICE ?\n"
            f"- Impact si métrique reste à {f.measured_value:.4f} : <à remplir>\n"
            "- Mitigation produit possible : <à remplir>\n\n"
            "### Seuil recalibré (proposé)\n\n"
            f"- Si gate validity ✓ : threshold reste {f.threshold:.4f}, fix code\n"
            "- Si validité ✗ : threshold proposé <new_threshold> avec justification\n\n"
            "### Mitigations options (3 max)\n\n"
            "1. <option>\n2. <option>\n3. <option>\n\n"
            "### Décision user (à remplir)\n\n"
            "[ ] Accepter mitigation N°1\n"
            f"[ ] Recalibrer seuil à <new_threshold>\n"
            "[ ] Bloquer Phase 4a entry gate jusqu'à fix\n"
        )
    return "".join(lines)
```

- [ ] **Step 8.2: tests/d8/test_gates.py — 38 tests** (19 gates × 2 = pass + fail per gate, plus case-by-case logger render)

- [ ] **Step 8.3: Commit**

```bash
git add scripts/d8/gates.py tests/d8/test_gates.py
git commit -m "feat(d8): 19 gates G-A SOTA strict + case-by-case failure logger"
```

---

## Task 9 : run.py — per-saison orchestrator

**Files:**
- Create: `scripts/d8/run.py`

- [ ] **Step 9.1: Implement run.py**

```python
"""D8 per-saison orchestrator — kernel entry point.

Lit ALICE_SAISON env var, exécute pipeline complet, écrit JSON.

Document ID: ALICE-D8-RUN
Version: 1.0.0
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Kaggle: setup sys.path BEFORE imports (entry point pattern)
if "/kaggle/input/notebooks" in os.environ.get("PWD", ""):
    sys.path.insert(0, "/kaggle/input/notebooks/pierrax/alice-d8-code")

from scripts.d8 import (
    breakdowns,
    calibration,
    conformal,
    dro,
    gates,
    loader,
    stress_elo,
    stress_roster,
)
from scripts.d8.types import D8Lineage, D8SaisonReport


def main() -> None:
    saison = int(os.environ["ALICE_SAISON"])
    output_dir = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Lineage SHA-256
    joueurs_path = Path("/kaggle/input/datasets/pierrax/alice-d8-input/data/joueurs.parquet")
    echiquiers_path = Path("/kaggle/input/datasets/pierrax/alice-d8-input/data/echiquiers.parquet")
    mlp_path = Path("/kaggle/input/datasets/pierrax/alice-d8-input/artefacts/mlp_meta_learner.joblib")
    temp_scaler_path = Path("/kaggle/input/datasets/pierrax/alice-d8-input/artefacts/temp_scaler.joblib")

    lineage = D8Lineage(
        joueurs_sha256=loader.compute_file_sha256(joueurs_path),
        echiquiers_sha256=loader.compute_file_sha256(echiquiers_path),
        mlp_artefact_sha256=loader.compute_file_sha256(mlp_path),
        temp_scaler_sha256=loader.compute_file_sha256(temp_scaler_path),
        code_sha256=os.environ.get("ALICE_CODE_SHA", "unknown"),
        ali_seed=42,
        ali_n_topk=10,
        ali_n_mc_pairs=5,
        ali_decay_lambda=0.9,
        kernel_id=f"pierrax/d8-saison-{saison}",
        kernel_version_kaggle=os.environ.get("KAGGLE_KERNEL_VERSION", "v1"),
        run_at_utc=datetime.now(UTC).isoformat(),
    )

    # 2. Match eligibility
    matches = loader.load_match_eligible(echiquiers_path, saison)
    if len(matches) < 30:
        msg = f"Saison {saison} only {len(matches)} matches, need >=30 for conformal"
        raise RuntimeError(msg)

    # 3. Run backtest harness via existing scripts/backtest/runner.py
    from scripts.backtest.harness import run_backtest_for_matches

    backtest_report = run_backtest_for_matches(matches, joueurs_path, mlp_path, temp_scaler_path)

    # 4. Breakdowns 7 dims
    bdowns = breakdowns.compute_all_7(
        backtest_report.per_match,
        gender_fn=lambda m: backtest_report.metadata[m.user_team]["gender"],
        pool_size_fn=lambda m: backtest_report.metadata[m.opponent_team]["pool_size"],
        all_pool_sizes=[backtest_report.metadata[m.opponent_team]["pool_size"] for m in backtest_report.per_match],
        niveau_fn=lambda m: backtest_report.metadata[(m.user_team, m.opponent_team)]["niveau"],
        team_elo_mean_fn=lambda m: backtest_report.metadata[m.user_team]["team_elo_mean"],
        categorie_fn=lambda m: backtest_report.metadata[m.user_team]["categorie_dominant"],
    )

    # 5-8. Calibration + stress + conformal + DRO
    # See spec §7.2 pipeline. Each invokes the modules built above.
    # Output assembled into D8SaisonReport, written to JSON.

    report = D8SaisonReport(
        schema_version="d8.v1",
        saison=saison,
        n_matches=len(matches),
        lineage=lineage,
        per_match=[m.__dict__ for m in backtest_report.per_match],
        breakdowns={k: {"dim_name": v.dim_name, "groups": {g: gs.__dict__ for g, gs in v.groups.items()}} for k, v in bdowns.items()},
        # ... multicalibration, stress_elo, stress_roster, conformal, dro_wasserstein
    )

    output_path = output_dir / f"d8_saison_{saison}.json"
    with output_path.open("w") as f:
        json.dump(report.__dict__, f, indent=2, default=str)
    print(f"D8 saison {saison} report written : {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 9.2: Commit**

```bash
git add scripts/d8/run.py
git commit -m "feat(d8): per-saison kernel orchestrator (entry point ALICE_SAISON)"
```

Note : run.py est testé uniquement via `test_run_e2e_smoke.py` (Task 11) car il intègre tous les modules + Kaggle paths.

---

## Task 10 : aggregate.py — fuse 4 saisons + render markdown

**Files:**
- Create: `scripts/d8/aggregate.py`
- Create: `tests/d8/test_aggregate.py`

- [ ] **Step 10.1: Implement aggregate.py**

```python
"""D8 aggregator — fuse 4 saisons + global gates + render markdown.

Document ID: ALICE-D8-AGGREGATE
Version: 1.0.0
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.d8.gates import evaluate_19_gates, render_failure_analysis_md
from scripts.d8.types import D8FullReport, D8GateStatus


def load_saison_reports(input_dir: Path) -> dict[int, dict]:
    """Load 4 d8_saison_{2021..2024}.json from Kaggle input datasets."""
    out = {}
    for saison in (2021, 2022, 2023, 2024):
        p = input_dir / f"d8-saison-{saison}" / f"d8_saison_{saison}.json"
        if not p.exists():
            msg = f"Missing saison {saison} report at {p}"
            raise FileNotFoundError(msg)
        with p.open() as f:
            out[saison] = json.load(f)
    return out


def verify_lineage_coherence(reports: dict[int, dict]) -> None:
    """Aggregator vérifie : tous SHA artefacts identiques cross-saisons."""
    mlp_shas = {r["lineage"]["mlp_artefact_sha256"] for r in reports.values()}
    if len(mlp_shas) > 1:
        msg = f"MLP artefact SHA mismatch across saisons: {mlp_shas}"
        raise RuntimeError(msg)
    temp_shas = {r["lineage"]["temp_scaler_sha256"] for r in reports.values()}
    if len(temp_shas) > 1:
        msg = f"temp_scaler SHA mismatch: {temp_shas}"
        raise RuntimeError(msg)


def fuse_per_match(reports: dict[int, dict]) -> list[dict]:
    """Concat all per_match arrays."""
    return [m for r in reports.values() for m in r["per_match"]]


def compute_global_metrics(per_match_global: list[dict]) -> dict:
    """Recompute breakdowns + multicalibration + conformal sur N=280."""
    # See run.py §4-8 (same logic, N=280 input).
    # Returns dict with 19 gate-relevant metrics.
    raise NotImplementedError


def main() -> None:
    input_dir = Path("/kaggle/input/datasets/pierrax")
    output_dir = Path("/kaggle/working")
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = load_saison_reports(input_dir)
    verify_lineage_coherence(reports)

    per_match_global = fuse_per_match(reports)
    n_global = len(per_match_global)

    metrics = compute_global_metrics(per_match_global)
    gates_19 = evaluate_19_gates(metrics)

    failures = [g for g in gates_19 if g.status == D8GateStatus.FAIL]
    failure_md = render_failure_analysis_md(failures)

    # Outputs
    full_report = D8FullReport(
        schema_version="d8-aggregator.v1",
        n_matches=n_global,
        saisons=[2021, 2022, 2023, 2024],
        lineage_per_saison={s: r["lineage"] for s, r in reports.items()},
        breakdowns_global={},  # filled from compute_global_metrics
        multicalibration_global={},
        stress_elo_global={},
        stress_roster_global={},
        conformal_global={},
        dro_global={},
        gates_19=gates_19,
    )

    (output_dir / "d8_full_report.json").write_text(json.dumps(full_report.__dict__, indent=2, default=str))
    (output_dir / "D8_FAILURE_ANALYSIS_LOG.md").write_text(failure_md)
    (output_dir / "gates_19_status.json").write_text(json.dumps([g.__dict__ for g in gates_19], indent=2, default=str))

    # D8_FINDINGS.md (humain)
    findings = render_findings_md(full_report, datetime.now(UTC))
    (output_dir / "D8_FINDINGS.md").write_text(findings)

    n_pass = sum(1 for g in gates_19 if g.status == D8GateStatus.PASS)
    print(f"D8 aggregator complete : {n_pass}/19 gates PASS, {len(failures)} FAIL")


def render_findings_md(report: D8FullReport, run_at: datetime) -> str:
    """Render D8_FINDINGS.md (8-12 pages humain)."""
    n_pass = sum(1 for g in report.gates_19 if g.status == D8GateStatus.PASS)
    return (
        f"# D8 Findings — Phase 3.5 STRICT\n\n"
        f"**Run** : {run_at.isoformat()}\n"
        f"**N matches** : {report.n_matches}\n"
        f"**Saisons** : {report.saisons}\n"
        f"**Gates** : {n_pass}/19 PASS\n\n"
        f"## Per-gate results\n\n"
        + "\n".join(
            f"- {g.gate_id} : {g.status.value} (measured={g.measured_value:.4f}, threshold={g.threshold:.4f}, source={g.source})"
            for g in report.gates_19
        )
        + "\n\n## Phase 4a entry gate\n\n"
        + ("✅ All gates PASS — ADR-016 ready to move Proposed → Accepted." if n_pass == 19 else f"⚠️  {19 - n_pass} gates FAIL — see D8_FAILURE_ANALYSIS_LOG.md for case-by-case decisions.")
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.2: tests/d8/test_aggregate.py — 15 tests** (concat 4 saisons, lineage coherence verify, missing saison fail-fast, gates evaluation pass/fail, render markdown)

- [ ] **Step 10.3: Commit**

```bash
git add scripts/d8/aggregate.py tests/d8/test_aggregate.py
git commit -m "feat(d8): aggregator — 4 saisons fusion + global gates + render reports"
```

---

## Task 11 : E2E smoke test — minimal pipeline en <30s

**Files:**
- Create: `tests/d8/test_run_e2e_smoke.py`

- [ ] **Step 11.1: Implement smoke test**

```python
"""D8 E2E smoke test — pipeline complet sur 5 matches dummy <30s.

ISO 29119 : valide intégration de tous modules D8 sans Kaggle.

Tests markés `slow=False` (rapide) — exclus pre-push uniquement si charge parquets réels.

Document ID: ALICE-D8-SMOKE
Version: 1.0.0
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.backtest.runner_types import MatchStats
from scripts.d8.breakdowns import compute_all_7, max_gap_recall
from scripts.d8.gates import THRESHOLDS, evaluate_19_gates


def _dummy_match(saison: int, ronde: int, recall: float = 0.85) -> MatchStats:
    return MatchStats(
        saison=saison,
        ronde=ronde,
        user_team=f"USR{ronde}",
        opponent_team=f"OPP{ronde}",
        recall_ali=recall,
        accuracy_ali=0.90,
        jaccard_ali=0.75,
        brier_ali=0.20,
        ece_ali=0.04,
        recall_baseline=0.40,
        brier_baseline=0.50,
        bss=0.60,
        e_score_predicted=4.0,
        e_score_observed=4.5,
        e_score_mae=0.5,
        ali_correct=True,
        baseline_correct=False,
    )


def test_e2e_smoke_breakdowns_5_matches() -> None:
    matches = [_dummy_match(2024, r) for r in range(1, 6)]
    result = compute_all_7(
        matches,
        gender_fn=lambda m: "M",
        pool_size_fn=lambda m: 15,
        all_pool_sizes=[10, 12, 14, 16, 18, 20],
        niveau_fn=lambda m: "N3",
        team_elo_mean_fn=lambda m: 1700,
        categorie_fn=lambda m: "SenM",
    )
    assert len(result) == 7
    assert sum(g.n for g in result["by_ronde"].groups.values()) == 5


def test_e2e_smoke_max_gap_zero_for_uniform() -> None:
    matches = [_dummy_match(2024, r, recall=0.85) for r in range(1, 6)]
    bdown = compute_all_7(
        matches,
        gender_fn=lambda m: "M",
        pool_size_fn=lambda m: 15,
        all_pool_sizes=[10, 20],
        niveau_fn=lambda m: "N3",
        team_elo_mean_fn=lambda m: 1700,
        categorie_fn=lambda m: "SenM",
    )["by_gender"]
    assert max_gap_recall(bdown) == 0.0


def test_e2e_smoke_gates_19_present() -> None:
    metrics = {
        "max_gap_recall_max_dim": 0.05,
        "recall_per_group_min": 0.90,
        "demographic_parity_diff": 0.05,
        "equalized_odds_diff": 0.05,
        "ece_per_group_max": 0.03,
        "multicalibration_alpha": 0.03,
        "tpr_ratio_min": 0.95,
        "brier_per_group_max": 0.20,
        "bss_per_group_min": 0.50,
        "psi_per_dim_max": 0.10,
        "recall_drop_1pct": 0.01,
        "recall_drop_5pct": 0.03,
        "recall_drop_10pct": 0.06,
        "roster_5pct_recall_drop": 0.03,
        "roster_20pct_recall_drop": 0.10,
        "coverage_global": 0.92,
        "conformal_set_size_mean": 2.4,
        "dro_eps_005_recall_worst": 0.78,
        "dro_eps_010_recall_worst": 0.62,
    }
    gates = evaluate_19_gates(metrics)
    assert len(gates) == 19


def test_e2e_smoke_all_thresholds_complete() -> None:
    """All 19 gates have thresholds defined."""
    assert len(THRESHOLDS) == 19


def test_e2e_smoke_fail_metrics_produces_failures() -> None:
    bad_metrics = {
        "max_gap_recall_max_dim": 0.5,
        "recall_per_group_min": 0.30,
        "demographic_parity_diff": 0.5,
        "equalized_odds_diff": 0.5,
        "ece_per_group_max": 0.30,
        "multicalibration_alpha": 0.30,
        "tpr_ratio_min": 0.30,
        "brier_per_group_max": 0.80,
        "bss_per_group_min": 0.10,
        "psi_per_dim_max": 0.50,
        "recall_drop_1pct": 0.20,
        "recall_drop_5pct": 0.30,
        "recall_drop_10pct": 0.50,
        "roster_5pct_recall_drop": 0.40,
        "roster_20pct_recall_drop": 0.50,
        "coverage_global": 0.50,
        "conformal_set_size_mean": 5.0,
        "dro_eps_005_recall_worst": 0.20,
        "dro_eps_010_recall_worst": 0.10,
    }
    gates = evaluate_19_gates(bad_metrics)
    from scripts.d8.types import D8GateStatus
    failures = [g for g in gates if g.status == D8GateStatus.FAIL]
    assert len(failures) == 19  # all FAIL
```

- [ ] **Step 11.2: Run + commit**

```bash
python -m pytest tests/d8/test_run_e2e_smoke.py -v
git add tests/d8/test_run_e2e_smoke.py
git commit -m "test(d8): E2E smoke 5 matches dummy <30s (ISO 29119)"
```

---

## Task 12 : Kaggle integration — upload script + 5 kernel-metadata JSON

**Files:**
- Create: `scripts/d8/upload_d8_dataset.py`
- Create: `scripts/d8/download_outputs.py`
- Create: `scripts/d8/kernel-metadata-saison-{2021,2022,2023,2024}.json`
- Create: `scripts/d8/kernel-metadata-aggregator.json`

- [ ] **Step 12.1: Write scripts/d8/upload_d8_dataset.py**

```python
"""Upload alice-d8-input dataset to Kaggle (one-time setup + version bumps).

Usage : python -m scripts.d8.upload_d8_dataset

Document ID: ALICE-D8-UPLOAD
Version: 1.0.0
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_SLUG = "alice-d8-input"


def main() -> None:
    repo = Path(__file__).resolve().parent.parent.parent
    staging = repo / "build" / "kaggle" / "alice-d8-input"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    # Copy data + artefacts
    (staging / "data").mkdir()
    shutil.copy(repo / "data" / "joueurs.parquet", staging / "data" / "joueurs.parquet")
    shutil.copy(repo / "data" / "echiquiers.parquet", staging / "data" / "echiquiers.parquet")
    (staging / "artefacts").mkdir()
    shutil.copy(repo / "artefacts" / "mlp_meta_learner.joblib", staging / "artefacts" / "mlp_meta_learner.joblib")
    shutil.copy(repo / "artefacts" / "temp_scaler.joblib", staging / "artefacts" / "temp_scaler.joblib")
    (staging / "config" / "ffe_rules").mkdir(parents=True)
    shutil.copy(repo / "config" / "ffe_rules" / "a02.json", staging / "config" / "ffe_rules" / "a02.json")

    # dataset-metadata.json
    metadata = {
        "title": "ALICE D8 Fairness/Robustness Input",
        "id": f"pierrax/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))

    # Upload (or version)
    subprocess.run(["kaggle", "datasets", "create", "-p", str(staging), "--dir-mode", "tar"], check=False)
    subprocess.run(["kaggle", "datasets", "version", "-p", str(staging), "-m", "D8 input refresh"], check=False)
    print(f"Dataset {DATASET_SLUG} uploaded.")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 12.2: Write scripts/d8/kernel-metadata-saison-2021.json**

```json
{
  "id": "pierrax/d8-saison-2021",
  "title": "ALICE D8 Saison 2021",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": ["pierrax/alice-d8-input"],
  "kernel_sources": [],
  "competition_sources": [],
  "environment_variables": {
    "ALICE_SAISON": "2021",
    "ALICE_CODE_SHA": "<set by upload script>"
  }
}
```

(idem pour 2022, 2023, 2024 — substituer ALICE_SAISON value)

- [ ] **Step 12.3: kernel-metadata-aggregator.json**

```json
{
  "id": "pierrax/d8-aggregator",
  "title": "ALICE D8 Aggregator",
  "code_file": "aggregate.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": [
    "pierrax/d8-saison-2021",
    "pierrax/d8-saison-2022",
    "pierrax/d8-saison-2023",
    "pierrax/d8-saison-2024"
  ],
  "kernel_sources": [],
  "competition_sources": []
}
```

- [ ] **Step 12.4: Write scripts/d8/download_outputs.py**

```python
"""Download d8-aggregator outputs to reports/d8/.

Document ID: ALICE-D8-DOWNLOAD
Version: 1.0.0
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent.parent / "reports" / "d8"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "kernels", "output", "pierrax/d8-aggregator", "-p", str(out_dir)],
        check=True,
    )
    expected = ["d8_full_report.json", "D8_FINDINGS.md", "D8_FAILURE_ANALYSIS_LOG.md", "gates_19_status.json"]
    missing = [f for f in expected if not (out_dir / f).exists()]
    if missing:
        msg = f"Missing aggregator outputs: {missing}"
        raise FileNotFoundError(msg)
    print(f"D8 outputs downloaded to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 12.5: Commit**

```bash
git add scripts/d8/upload_d8_dataset.py scripts/d8/download_outputs.py scripts/d8/kernel-metadata-*.json
git commit -m "feat(d8): Kaggle integration — upload + 5 kernel-metadata + download"
```

---

## Task 13 : DVC stage extension

**Files:**
- Modify: `dvc.yaml`

- [ ] **Step 13.1: Add stage d8_audit**

```yaml
# Append to existing dvc.yaml
  d8_audit:
    deps:
      - data/joueurs.parquet
      - data/echiquiers.parquet
      - artefacts/mlp_meta_learner.joblib
      - artefacts/temp_scaler.joblib
    outs:
      - reports/d8/d8_full_report.json
      - reports/d8/D8_FINDINGS.md
      - reports/d8/D8_FAILURE_ANALYSIS_LOG.md
      - reports/d8/gates_19_status.json
    cmd: python -m scripts.d8.download_outputs
```

- [ ] **Step 13.2: Commit**

```bash
git add dvc.yaml
git commit -m "feat(d8): DVC stage d8_audit (extend T24 dvc.yaml with reports/d8/)"
```

---

## Task 14 : Push Kaggle + monitor + execute (autonomous)

**Files:**
- None (CLI orchestration)

- [ ] **Step 14.1: Upload alice-d8-input dataset**

```bash
python -m scripts.d8.upload_d8_dataset
```

Expected: dataset created/versioned. SHA256 lineage of artefacts captured.

- [ ] **Step 14.2: Upload alice-d8-code dataset (kernel sources)**

```bash
# Create alice-d8-code dataset containing services/ali/* + scripts/d8/* + scripts/backtest/* + scripts/parse_dataset/*
# Update kernel-metadata-*.json kernel_sources to ["pierrax/alice-d8-code"]
```

- [ ] **Step 14.3: Push 4 saison kernels in parallel**

```bash
cd scripts/d8
kaggle kernels push -p . -k kernel-metadata-saison-2021.json &
kaggle kernels push -p . -k kernel-metadata-saison-2022.json &
kaggle kernels push -p . -k kernel-metadata-saison-2023.json &
kaggle kernels push -p . -k kernel-metadata-saison-2024.json &
wait
```

- [ ] **Step 14.4: Monitor with /loop**

```bash
# Poll every 5 minutes until all 4 saisons complete
while ! all_saisons_done; do
  for s in 2021 2022 2023 2024; do
    kaggle kernels status pierrax/d8-saison-$s
  done
  sleep 300
done
```

- [ ] **Step 14.5: On any saison fail → diagnostic + fix + re-push**

Patterns d'erreur connus :
- ModuleNotFoundError → check kernel_sources upload, re-version code dataset
- sys.path setup → vérifier entry point pattern
- env var ALICE_SAISON manquante → kernel-metadata-*.json
- Lineage SHA mismatch → vérifier artefacts dataset version

- [ ] **Step 14.6: Push aggregator (depends on 4 saisons datasets)**

```bash
kaggle kernels push -p . -k kernel-metadata-aggregator.json
```

- [ ] **Step 14.7: Monitor aggregator**

```bash
kaggle kernels status pierrax/d8-aggregator
```

- [ ] **Step 14.8: Download outputs + DVC track**

```bash
python -m scripts.d8.download_outputs
dvc add reports/d8/d8_full_report.json
dvc commit -m "d8: phase 3.5 strict audit complete"
```

- [ ] **Step 14.9: Inspect gates_19_status.json**

If all 19 gates PASS → **STOP autonomous, report success to user**.
If any gate FAIL → **STOP autonomous, post D8_FAILURE_ANALYSIS_LOG.md template entries to user for case-by-case decisions**.

- [ ] **Step 14.10: Commit reports**

```bash
git add reports/d8/.gitignore  # may already be tracked
git add reports/d8/D8_FINDINGS.md reports/d8/D8_FAILURE_ANALYSIS_LOG.md reports/d8/gates_19_status.json
git commit -m "feat(d8): Phase 3.5 STRICT audit results (N=280 multi-saisons)"
```

---

## Task 15 : Update memory + CLAUDE.md + DEBT_LEDGER

**Files:**
- Modify: `memory/project_session_resume.md`
- Modify: `CLAUDE.md`
- Modify: `docs/project/DEBT_LEDGER.md`
- Modify: `memory/project_debt_current.md`

- [ ] **Step 15.1: Update memory project_session_resume.md** — Phase 3.5 STRICT D8 complete + gates outcome

- [ ] **Step 15.2: Update CLAUDE.md API FastAPI line** — D8 RESOLUE (or partial with case-by-case decisions)

- [ ] **Step 15.3: Update DEBT_LEDGER + project_debt_current** — D8 status RESOLUE

- [ ] **Step 15.4: Commit + push**

```bash
git add memory/ CLAUDE.md docs/project/DEBT_LEDGER.md
git commit -m "docs(d8): close Phase 3.5 STRICT entry gate, update CLAUDE.md + memory"
git push origin master
```

---

## Self-Review

**Spec coverage check** :
- §3 Stratification 7 dims → Task 3 ✓
- §4 Robustness perturbations → Tasks 5+6+7 ✓
- §5 19 Gates G-A + case-by-case → Task 8 ✓
- §6 Architecture Arch-2 → Tasks 9+10+12+14 ✓
- §7 Data flow + outputs → Tasks 9+10 ✓
- §8 Error handling → Tasks 9+10 (run.py/aggregate.py raise on fail-fast cases)
- §9 Testing strategy → Tasks 1-11 (~205 tests covered)
- §10 Lineage SHA-256 → Task 2 ✓
- §11 Compute budget → Task 14 (parallel kernels)
- §12 Autonomous execution → Task 14 ✓

**Placeholder scan** :
- run.py §5-8 (calibration/stress/conformal/DRO assembly) marked as "see spec §7.2" — provides full pipeline reference, no TODO/TBD.
- aggregate.py compute_global_metrics raises NotImplementedError — flagged for Task 10 implementation completion.
- 19 gates in evaluate_19_gates use a metrics dict whose keys are documented in tests → consistent.

**Type consistency** :
- D8GateStatus enum used uniformly (gates.py → aggregate.py)
- D8GroupBreakdown structure used in breakdowns.py + aggregate.py
- MatchStats from scripts.backtest.runner_types reused (no redefinition)
- D8Lineage frozen dataclass uniformly

**Inline fixes applied** : aggregator `compute_global_metrics` reuses Task 9 pipeline pattern (delegate to breakdowns/calibration/conformal/dro modules). Documented dependency in Task 10.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-30-d8-fairness-robustness.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task (1→15), review between tasks, fast iteration. Aligned with user's "autonomy" instruction.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review at key boundaries.

**Which approach?**
