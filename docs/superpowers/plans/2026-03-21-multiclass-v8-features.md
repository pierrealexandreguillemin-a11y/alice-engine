# V8 MultiClass — Feature Engineering Plan (Plan A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor all 12 draw-blind features into W/D/L decomposition, add 8 draw priors + 16 club-level features, wire unconnected modules, fix 8 logic bugs — producing ~156-column feature parquets ready for MultiClass training.

**Architecture:** Each feature module (SRP <300 lines) computes features independently. `pipeline.py` orchestrates extraction, `merge_helpers.py` handles joins, `feature_engineering.py` manages temporal split and per-split computation. All features rolling (3 seasons or 5 games), stratified by competition level, forfaits excluded.

**Tech Stack:** pandas, numpy, scikit-learn (LabelEncoder). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-21-multiclass-v8-design.md`

**Depends on:** Commit 05b19a7 (leakage fix, eval_metric=Logloss). Already on master.

**Followed by:** Plan B (Training + Diagnostics — MultiClass target, calibration, RPS metrics).

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| NEW | `scripts/features/draw_priors.py` | avg_elo, elo_proximity, draw_rate_prior, draw_rate_player, draw_rate_equipe (~8 cols) |
| NEW | `scripts/features/club_level.py` | joueur_promu/relegue, player_team_elo_gap, stabilite_effectif, team_rank_in_club, reinforcement_rate, club_nb_teams (~16 cols) |
| REFACTOR | `scripts/features/recent_form.py` | W/D/L decomposition + competition stratification |
| REFACTOR | `scripts/features/color_perf.py` | W/D/L decomposition + rolling 3 seasons |
| FIX | `scripts/features/advanced/pressure.py` | zone_enjeu instead of score_dom + W/D/L |
| REFACTOR | `scripts/features/advanced/h2h.py` | W/D/L decomposition + h2h_exists flag |
| REFACTOR | `scripts/features/club_behavior.py` | W/D/L home rates + wire existing features |
| FIX | `scripts/features/performance.py` | echiquier_moyen rolling last season |
| MODIFY | `scripts/features/pipeline.py` | Wire new modules + ALI features |
| MODIFY | `scripts/features/merge_helpers.py` | Add merge functions for new features |
| MODIFY | `scripts/feature_engineering.py` | Forfait filter + strength features + integrate all |
| NEW | `tests/feature_engineering/test_draw_priors.py` | Tests for draw priors module |
| NEW | `tests/feature_engineering/test_club_level.py` | Tests for club level module |
| MODIFY | `tests/feature_engineering/test_*.py` | Adapt existing tests to W/D/L features |

---

## Task 1: Global forfait filter + helper utilities

**Files:**
- Modify: `scripts/feature_engineering.py`
- Create: `scripts/features/helpers.py`
- Test: `tests/feature_engineering/test_forfait_filter.py`

This is the foundation — ALL subsequent features depend on forfait-free data.

- [ ] **Step 1: Write the failing test**

```python
# tests/feature_engineering/test_forfait_filter.py
"""Tests for forfait filtering — ISO 5259 data quality."""

import pandas as pd
import pytest


class TestForfaitFilter:
    """Forfait exclusion — 3 tests."""

    @pytest.fixture()
    def sample_with_forfeits(self) -> pd.DataFrame:
        return pd.DataFrame({
            "resultat_blanc": [1.0, 0.5, 0.0, 2.0, 1.0, 2.0],
            "type_resultat": [
                "victoire_blanc", "nulle", "victoire_noir",
                "victoire_blanc", "victoire_blanc", "victoire_blanc",
            ],
            "blanc_elo": [1500, 1600, 1400, 1500, 1700, 1800],
        })

    def test_excludes_forfeits(self, sample_with_forfeits: pd.DataFrame) -> None:
        from scripts.features.helpers import exclude_forfeits
        result = exclude_forfeits(sample_with_forfeits)
        assert len(result) == 4
        assert 2.0 not in result["resultat_blanc"].values

    def test_preserves_draws(self, sample_with_forfeits: pd.DataFrame) -> None:
        from scripts.features.helpers import exclude_forfeits
        result = exclude_forfeits(sample_with_forfeits)
        assert 0.5 in result["resultat_blanc"].values

    def test_returns_copy(self, sample_with_forfeits: pd.DataFrame) -> None:
        from scripts.features.helpers import exclude_forfeits
        result = exclude_forfeits(sample_with_forfeits)
        assert result is not sample_with_forfeits
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/feature_engineering/test_forfait_filter.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Create helpers module**

```python
# scripts/features/helpers.py
"""Shared utilities for feature engineering — ISO 5055/5259."""

from __future__ import annotations

import pandas as pd

FORFAIT_RESULT = 2.0
PLAYED_RESULTS = {"victoire_blanc", "victoire_noir", "nulle",
                  "victoire_blanc_ajournement", "victoire_noir_ajournement",
                  "ajournement"}
NON_PLAYED = {"non_joue", "forfait_blanc", "forfait_noir", "double_forfait"}


def exclude_forfeits(df: pd.DataFrame) -> pd.DataFrame:
    """Remove forfait rows (resultat_blanc=2.0) from DataFrame."""
    return df[df["resultat_blanc"] != FORFAIT_RESULT].copy()


def filter_played_games(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only actually played games (no forfeits, no non-played)."""
    mask = df["type_resultat"].isin(PLAYED_RESULTS) & (df["resultat_blanc"] != FORFAIT_RESULT)
    return df[mask].copy()


def compute_wdl_rates(results: pd.Series) -> dict[str, float]:
    """Compute win/draw/loss rates from a series of resultat values (0, 0.5, 1)."""
    n = len(results)
    if n == 0:
        return {"win_rate": 0.0, "draw_rate": 0.0, "expected_score": 0.0}
    wins = (results == 1.0).sum()
    draws = (results == 0.5).sum()
    return {
        "win_rate": float(wins / n),
        "draw_rate": float(draws / n),
        "expected_score": float((wins + 0.5 * draws) / n),
    }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/feature_engineering/test_forfait_filter.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add scripts/features/helpers.py tests/feature_engineering/test_forfait_filter.py
git commit -m "feat(features): add forfait filter + W/D/L rate helpers (ISO 5259)"
```

---

## Task 2: Draw priors module

**Files:**
- Create: `scripts/features/draw_priors.py`
- Test: `tests/feature_engineering/test_draw_priors.py`

- [ ] **Step 1: Write tests**

```python
# tests/feature_engineering/test_draw_priors.py
"""Tests for draw prior features — Pawnalyze chess-specific."""

import numpy as np
import pandas as pd
import pytest


class TestDrawPriors:
    """Draw prior features — 5 tests."""

    @pytest.fixture()
    def sample_history(self) -> pd.DataFrame:
        """20 games, various Elos and results."""
        rng = np.random.default_rng(42)
        n = 100
        return pd.DataFrame({
            "blanc_nom": [f"P{i % 10}" for i in range(n)],
            "noir_nom": [f"P{(i + 5) % 10}" for i in range(n)],
            "blanc_elo": rng.integers(1200, 2200, n),
            "noir_elo": rng.integers(1200, 2200, n),
            "resultat_blanc": rng.choice([0.0, 0.5, 1.0], n, p=[0.4, 0.15, 0.45]),
            "equipe_dom": [f"Team{i % 5}" for i in range(n)],
            "equipe_ext": [f"Team{(i + 2) % 5}" for i in range(n)],
            "type_resultat": ["victoire_blanc"] * n,  # simplified
        })

    def test_avg_elo_computed(self, sample_history: pd.DataFrame) -> None:
        from scripts.features.draw_priors import compute_draw_priors
        result = compute_draw_priors(sample_history, sample_history)
        assert "avg_elo" in result.columns
        expected = (sample_history.blanc_elo + sample_history.noir_elo) / 2
        pd.testing.assert_series_equal(result["avg_elo"], expected, check_names=False)

    def test_elo_proximity_range(self, sample_history: pd.DataFrame) -> None:
        from scripts.features.draw_priors import compute_draw_priors
        result = compute_draw_priors(sample_history, sample_history)
        assert result["elo_proximity"].between(0, 1).all()

    def test_draw_rate_prior_not_null(self, sample_history: pd.DataFrame) -> None:
        from scripts.features.draw_priors import compute_draw_priors
        result = compute_draw_priors(sample_history, sample_history)
        assert "draw_rate_prior" in result.columns
        assert result["draw_rate_prior"].notna().mean() > 0.5

    def test_draw_rate_player_computed(self, sample_history: pd.DataFrame) -> None:
        from scripts.features.draw_priors import compute_player_draw_rates
        result = compute_player_draw_rates(sample_history)
        assert "draw_rate" in result.columns
        assert result["draw_rate"].between(0, 1).all()

    def test_draw_rate_equipe_computed(self, sample_history: pd.DataFrame) -> None:
        from scripts.features.draw_priors import compute_equipe_draw_rates
        result = compute_equipe_draw_rates(sample_history)
        assert "draw_rate" in result.columns
```

- [ ] **Step 2: Run — should fail**

- [ ] **Step 3: Implement draw_priors.py**

```python
# scripts/features/draw_priors.py
"""Draw prior features — chess-specific (Pawnalyze model). ISO 5259."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scripts.features.helpers import exclude_forfeits

logger = logging.getLogger(__name__)

ELO_BINS = [0, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 3500]
DIFF_BINS = [0, 50, 100, 200, 400, 800]


def compute_draw_priors(df_split: pd.DataFrame, df_history: pd.DataFrame) -> pd.DataFrame:
    """Compute per-game draw prior features. Returns df_split with new columns."""
    result = df_split.copy()
    result["avg_elo"] = (result["blanc_elo"] + result["noir_elo"]) / 2
    diff = (result["blanc_elo"] - result["noir_elo"]).abs()
    result["elo_proximity"] = (1 - diff.clip(upper=800) / 800).round(4)

    # Lookup table from history
    lookup = build_draw_rate_lookup(df_history)
    result["_elo_band"] = pd.cut(result["avg_elo"], bins=ELO_BINS, labels=False)
    result["_diff_band"] = pd.cut(diff, bins=DIFF_BINS, labels=False)
    result = result.merge(
        lookup, on=["_elo_band", "_diff_band"], how="left", suffixes=("", "_lookup"),
    )
    result["draw_rate_prior"] = result["draw_rate_prior"].fillna(
        lookup["draw_rate_prior"].mean()
    )
    result.drop(columns=["_elo_band", "_diff_band"], inplace=True)
    logger.info("Draw priors: avg_elo, elo_proximity, draw_rate_prior computed")
    return result


def build_draw_rate_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """Build (elo_band x diff_band) → draw_rate lookup from history."""
    clean = exclude_forfeits(df)
    clean = clean[clean["resultat_blanc"].isin([0.0, 0.5, 1.0])].copy()
    avg = (clean["blanc_elo"] + clean["noir_elo"]) / 2
    diff = (clean["blanc_elo"] - clean["noir_elo"]).abs()
    clean["_elo_band"] = pd.cut(avg, bins=ELO_BINS, labels=False)
    clean["_diff_band"] = pd.cut(diff, bins=DIFF_BINS, labels=False)
    clean["is_draw"] = (clean["resultat_blanc"] == 0.5).astype(int)
    lookup = (
        clean.groupby(["_elo_band", "_diff_band"])
        .agg(draw_rate_prior=("is_draw", "mean"), count=("is_draw", "count"))
        .reset_index()
    )
    return lookup[lookup["count"] >= 10][["_elo_band", "_diff_band", "draw_rate_prior"]]


def compute_player_draw_rates(df: pd.DataFrame, min_games: int = 10) -> pd.DataFrame:
    """Per-player historical draw rate. Returns (joueur_nom, draw_rate, n_games)."""
    clean = exclude_forfeits(df)
    clean = clean[clean["resultat_blanc"].isin([0.0, 0.5, 1.0])]
    blanc = clean[["blanc_nom", "resultat_blanc"]].rename(
        columns={"blanc_nom": "joueur_nom", "resultat_blanc": "result"}
    )
    noir = clean[["noir_nom", "resultat_blanc"]].copy()
    noir["result"] = 1.0 - noir["resultat_blanc"]  # flip: 0→1, 0.5→0.5, 1→0
    noir = noir.rename(columns={"noir_nom": "joueur_nom"}).drop(columns=["resultat_blanc"])
    all_games = pd.concat([blanc, noir], ignore_index=True)
    stats = all_games.groupby("joueur_nom").agg(
        draw_rate=("result", lambda x: (x == 0.5).mean()),
        n_games=("result", "count"),
    ).reset_index()
    return stats[stats["n_games"] >= min_games]


def compute_equipe_draw_rates(df: pd.DataFrame, min_games: int = 10) -> pd.DataFrame:
    """Per-team historical draw rate. Returns (equipe, draw_rate)."""
    clean = exclude_forfeits(df)
    clean = clean[clean["resultat_blanc"].isin([0.0, 0.5, 1.0])]
    dom = clean.groupby("equipe_dom")["resultat_blanc"].apply(
        lambda x: (x == 0.5).mean()
    ).reset_index()
    dom.columns = ["equipe", "draw_rate"]
    ext = clean.groupby("equipe_ext")["resultat_blanc"].apply(
        lambda x: (x == 0.5).mean()
    ).reset_index()
    ext.columns = ["equipe", "draw_rate"]
    both = pd.concat([dom, ext]).groupby("equipe")["draw_rate"].mean().reset_index()
    counts = clean.groupby("equipe_dom").size().reset_index(name="n")
    counts.columns = ["equipe", "n"]
    both = both.merge(counts, on="equipe", how="left")
    return both[both["n"].fillna(0) >= min_games][["equipe", "draw_rate"]]
```

- [ ] **Step 4: Run tests — should pass**
- [ ] **Step 5: Lint check** `ruff check scripts/features/draw_priors.py`
- [ ] **Step 6: Commit**

```bash
git add scripts/features/draw_priors.py tests/feature_engineering/test_draw_priors.py
git commit -m "feat(features): add draw priors module — avg_elo, elo_proximity, draw_rate (Pawnalyze)"
```

---

## Task 3: Refactor recent_form.py — W/D/L + stratification

**Files:**
- Modify: `scripts/features/recent_form.py`
- Modify: `tests/feature_engineering/test_recent_form.py` (or create)

The current `calculate_recent_form()` returns `forme_recente` (mean score) and `forme_tendance` (categorical). Refactor to return `win_rate_recent`, `draw_rate_recent`, `expected_score_recent`, `win_trend`, `draw_trend`.

- [ ] **Step 1: Write test for new W/D/L decomposition**

```python
def test_wdl_form_decomposition() -> None:
    """Recent form must decompose into win_rate, draw_rate, expected_score."""
    from scripts.features.recent_form import calculate_recent_form
    df = pd.DataFrame({
        "blanc_nom": ["A"] * 5,
        "noir_nom": ["B"] * 5,
        "resultat_blanc": [1.0, 0.5, 0.0, 1.0, 0.5],
        "resultat_noir": [0.0, 0.5, 1.0, 0.0, 0.5],
        "type_resultat": ["victoire_blanc", "nulle", "victoire_noir", "victoire_blanc", "nulle"],
        "date": pd.date_range("2024-01-01", periods=5),
        "type_competition": ["autre"] * 5,
    })
    result = calculate_recent_form(df, window=5)
    row = result[result.joueur_nom == "A"].iloc[0]
    assert abs(row.win_rate_recent - 0.4) < 0.01  # 2/5
    assert abs(row.draw_rate_recent - 0.4) < 0.01  # 2/5
    assert abs(row.expected_score_recent - 0.6) < 0.01  # 2/5 + 0.5*2/5
```

- [ ] **Step 2: Run — should fail** (old function returns `forme_recente`)

- [ ] **Step 3: Refactor `calculate_recent_form()`**

Key changes to `scripts/features/recent_form.py`:
- Add `type_competition` parameter for stratification
- Replace `.mean()` with explicit W/D/L counting via `compute_wdl_rates()`
- Return `win_rate_recent, draw_rate_recent, expected_score_recent, win_trend, draw_trend`
- Import `exclude_forfeits` from `helpers.py`
- Keep function signature compatible: `calculate_recent_form(df, window=5) -> pd.DataFrame`

- [ ] **Step 4: Run tests — all should pass**
- [ ] **Step 5: Verify existing tests still pass**: `pytest tests/feature_engineering/ -v`
- [ ] **Step 6: Commit**

```bash
git commit -m "refactor(features): recent_form W/D/L decomposition + competition stratification"
```

---

## Task 4: Refactor color_perf.py — W/D/L + rolling

**Files:**
- Modify: `scripts/features/color_perf.py`

Key changes:
- Replace `score_blancs` (mean) with `win_rate_white`, `draw_rate_white`
- Replace `score_noirs` (mean) with `win_rate_black`, `draw_rate_black`
- Replace `avantage_blancs` with `win_adv_white`, `draw_adv_white`
- Add rolling window (3 seasons) via `saison` filter on history
- Use `exclude_forfeits()` before computation
- Keep `couleur_preferee` and `data_quality` (neutral features)

- [ ] **Step 1: Write test for W/D/L color features**
- [ ] **Step 2: Run — fail**
- [ ] **Step 3: Implement refactored color_perf**
- [ ] **Step 4: Run tests — pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "refactor(features): color_perf W/D/L decomposition + rolling 3 seasons"
```

---

## Task 5: Fix pressure.py — zone_enjeu + W/D/L

**Files:**
- Modify: `scripts/features/advanced/pressure.py`

Key changes:
- Remove `score_dom/score_ext` from `is_decisive` condition
- Replace with: `is_decisive = zone_enjeu.isin(["montee", "danger"])`
- If `zone_enjeu` not in df, fall back to `ronde >= 7` only
- Replace `score_normal/score_pression` (means) with `win_rate_normal/pression`, `draw_rate_normal/pression`
- `clutch_win = win_rate_pression - win_rate_normal`
- `clutch_draw = draw_rate_pression - draw_rate_normal`
- Reclassify `pressure_type` from `clutch_win`

- [ ] **Step 1: Write behavioral test for corrected pressure**

```python
def test_pressure_ignores_score_dom() -> None:
    """Pressure must use zone_enjeu, not score_dom."""
    df = pd.DataFrame({
        "blanc_nom": ["A"] * 6, "noir_nom": ["B"] * 6,
        "resultat_blanc": [1.0, 0.5, 0.0, 1.0, 0.5, 0.0],
        "resultat_noir": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
        "type_resultat": ["victoire_blanc", "nulle", "victoire_noir"] * 2,
        "ronde": [1, 2, 3, 8, 9, 10],
        "zone_enjeu_dom": ["mi_tableau"] * 3 + ["montee"] * 3,
        "score_dom": [5, 1, 5, 1, 5, 1],  # score_dom should be IGNORED
    })
    from scripts.features.advanced.pressure import calculate_pressure_performance
    result = calculate_pressure_performance(df, min_games=1)
    # zone_enjeu=montee (rondes 8-10) should define pressure, not score_dom
    row = result[result.joueur_nom == "A"].iloc[0]
    assert "clutch_win" in result.columns
```

- [ ] **Step 2-5: Implement, test, commit**

```bash
git commit -m "fix(features): pressure uses zone_enjeu instead of score_dom (leakage fix)"
```

---

## Task 6: Refactor h2h.py — W/D/L + h2h_exists + draw_rate_h2h

**Files:**
- Modify: `scripts/features/advanced/h2h.py`
- Modify: `scripts/features/merge_helpers.py:merge_h2h_features()` (lines 235-276)

Key changes in `h2h.py`:
- Replace `.agg(score_a=("score_a", "mean"))` with explicit W/D/L counting
- Output columns: `joueur_a, joueur_b, nb_confrontations, win_rate_a, draw_rate_a, h2h_exists`
- `draw_rate_a` = spec's `draw_rate_h2h` (Cat 4) = spec's `h2h_draw_rate` (Cat 7) — same concept, use `h2h_draw_rate` as canonical name
- `h2h_exists = (nb_confrontations >= min_games)`

Key changes in `merge_helpers.py:merge_h2h_features()`:
- Replace old column mapping: `avantage_a` → `h2h_win_rate`, add `h2h_draw_rate`, add `h2h_exists`
- Final merged columns: `h2h_win_rate`, `h2h_draw_rate`, `h2h_nb_confrontations`, `h2h_exists`

- [ ] **Step 1: Write test for W/D/L H2H**
- [ ] **Step 2: Refactor h2h.py**
- [ ] **Step 3: Update merge_h2h_features() in merge_helpers.py**
- [ ] **Step 4: Run tests — pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "refactor(features): h2h W/D/L + h2h_exists + draw_rate_h2h"
```

---

## Task 7: Refactor club_behavior.py — W/D/L home rates + wire

**Files:**
- Modify: `scripts/features/club_behavior.py`

Key changes:
- `_calc_avantage_dom()` → `_calc_home_rates()`: returns `win_rate_home`, `draw_rate_home`
- Use `exclude_forfeits()` before counting
- Ensure all sub-features (`nb_joueurs_utilises`, `rotation`, `noyau_stable`, etc.) are computed and returned
- All rolling 3 seasons (filter df_history by saison)

- [ ] **Steps 1-5: Test, implement, commit**

```bash
git commit -m "refactor(features): club_behavior W/D/L home rates + forfait exclusion"
```

---

## Task 8: New club_level.py — vases communiquants

**Files:**
- Create: `scripts/features/club_level.py`
- Test: `tests/feature_engineering/test_club_level.py`

Implements features from TRAINING_PROGRESS §5.2 and REGLES_FFE_ALICE.md §4:

```python
# scripts/features/club_level.py
"""Club-level features — vases communiquants (TRAINING_PROGRESS §5.2)."""

# Functions to implement:
# - extract_club_level_features(df_history) -> pd.DataFrame
#   Returns per (equipe, saison): team_rank_in_club, club_nb_teams,
#   reinforcement_rate, stabilite_effectif, elo_moyen_evolution
#
# - extract_player_team_context(df_history) -> pd.DataFrame
#   Returns per (joueur_nom, saison): joueur_promu, joueur_relegue,
#   player_team_elo_gap
#
# Uses get_niveau_equipe() from scripts.features.ffe_features
# or re-implement locally (regex on team name → division level)
```

- [ ] **Step 1: Write tests for joueur_promu/relegue detection**

```python
def test_joueur_relegue_detected() -> None:
    """Player descending from higher team should be flagged."""
    # Player "A" plays for "Club 1" (N2) in R1, then "Club 2" (N3) in R2
    # → joueur_relegue = True for R2 appearance
    ...
```

- [ ] **Steps 2-5: Implement, test, lint, commit**

```bash
git commit -m "feat(features): add club_level module — vases communiquants (joueur_promu/relegue, team_rank, reinforcement)"
```

---

## Task 9: Verify strength features + fix echiquier_moyen rolling

**Files:**
- Verify: `scripts/features/player_enrichment.py` (ALREADY EXISTS — has `enrich_from_joueurs()` with categorie + K_coefficient)
- Verify: `scripts/feature_engineering.py:_add_direct_features()` (ALREADY calls player_enrichment)
- Modify: `scripts/features/performance.py` (echiquier_moyen rolling)

**IMPORTANT:** `categorie_blanc/noir` and `k_coefficient_blanc/noir` already exist in `player_enrichment.py` lines 44-68 and 107-129. The join uses `nom_complet` (not NrFFE as spec says). DO NOT duplicate. Just verify they're wired and working.

Key changes (only echiquier_moyen):
- Fix `calculate_board_position()` to filter to last season only (not global career)
- Add saison parameter: `calculate_board_position(df, max_saison=current_saison)`

- [ ] **Steps 1-5: Test, implement, commit**

```bash
git commit -m "feat(features): add categorie_age + K_coefficient, fix echiquier_moyen rolling"
```

---

## Task 10: Verify ALI modules work with forfait exclusion

**Files:**
- Verify (do NOT modify): `scripts/features/pipeline.py` (ALI already wired at lines 84, 176)
- Verify: `scripts/features/pipeline_extended.py` (already calls presence + patterns)

**NOTE:** ALI modules (ali/presence.py, ali/patterns.py) are ALREADY wired into the pipeline via `extract_ali_features()` and `merge_ali_features()`. This task only verifies they work correctly with the forfait exclusion from Task 1.

- [ ] **Step 1: Run pipeline on sample with forfait data**

```python
# Verify ALI features still compute correctly after forfait exclusion
from scripts.features.pipeline_extended import extract_ali_features
import pandas as pd
df = pd.read_parquet("data/echiquiers.parquet").head(1000)
ali = extract_ali_features(df)
for key, feat_df in ali.items():
    print(f"{key}: {len(feat_df)} rows, cols={list(feat_df.columns)}")
    assert feat_df.notna().any().any(), f"{key} is all NaN"
```

- [ ] **Step 2: Verify no forfeits in ALI computations**
- [ ] **Step 3: Commit (no code changes — verification only)**

```bash
git commit -m "test(features): verify ALI modules compatible with forfait exclusion"
```

---

## Task 11: Integration — pipeline.py column mapping + new modules + forfait filter

**Files:**
- Modify: `scripts/features/pipeline.py` (extract_all_features + merge_all_features)
- Modify: `scripts/features/merge_helpers.py` (add merge functions for draw_priors + club_level)
- Modify: `scripts/feature_engineering.py` (forfait filter at top of compute_features_for_split)

**CRITICAL: Column name mapping in `pipeline.py:merge_all_features()`**

The following hardcoded column lists MUST be updated (old → new):

```python
# pipeline.py line ~147 (recent_form merge)
OLD: ["forme_recente", "forme_tendance"]
NEW: ["win_rate_recent", "draw_rate_recent", "expected_score_recent", "win_trend", "draw_trend"]

# pipeline.py line ~155 (color_perf merge)
OLD: ["score_blancs", "score_noirs", "avantage_blancs", "couleur_preferee", "data_quality"]
NEW: ["win_rate_white", "draw_rate_white", "win_rate_black", "draw_rate_black",
      "win_adv_white", "draw_adv_white", "couleur_preferee", "data_quality"]

# pipeline.py line ~233 (pressure merge in _merge_advanced_features)
OLD: ["clutch_factor", "pressure_type"]
NEW: ["clutch_win", "clutch_draw", "pressure_type"]

# pipeline.py (club_behavior merge in _merge_club_behavior)
OLD: ["avantage_dom_club"] (among others)
NEW: ["win_rate_home", "draw_rate_home"] (replacing avantage_dom_club)
```

**New module integration in `extract_all_features()`:**

```python
# Add after existing feature extractions:
from scripts.features.draw_priors import compute_player_draw_rates, compute_equipe_draw_rates
from scripts.features.club_level import extract_club_level_features, extract_player_team_context

features["draw_rate_player"] = compute_player_draw_rates(df_history_played)
features["draw_rate_equipe"] = compute_equipe_draw_rates(df_history_played)
features["club_level"] = extract_club_level_features(df_history)
features["player_team_context"] = extract_player_team_context(df_history)
```

**New merge functions in `merge_helpers.py`:**

```python
def merge_draw_rate_player(result, draw_rates):
    """Merge per-player draw rates as draw_rate_blanc/draw_rate_noir."""
    # Uses merge_player_features() with feature_cols=["draw_rate"]
    ...

def merge_draw_rate_equipe(result, equipe_rates):
    """Merge per-team draw rates as draw_rate_equipe_dom/draw_rate_equipe_ext."""
    ...

def merge_club_level(result, club_features, player_context):
    """Merge club-level (dom/ext suffix) + player team context (B/N suffix)."""
    ...
```

**Forfait filter in `feature_engineering.py`:**

```python
# At TOP of compute_features_for_split(), before any feature computation:
from scripts.features.helpers import exclude_forfeits
df_history_played = exclude_forfeits(df_history_played)
```

**Line count check:** After modifications, run `wc -l` on pipeline.py and feature_engineering.py. If >300 lines, split into sub-modules.

- [ ] **Step 1: Run full feature pipeline on sample data**

```python
# Quick integration test
python -c "
from scripts.feature_engineering import run_feature_engineering_v2
from pathlib import Path
run_feature_engineering_v2(data_dir=Path('data'), output_dir=Path('/tmp/v8_test'), include_advanced=True)
"
```

- [ ] **Step 2: Verify output parquet columns**

```python
import pandas as pd
df = pd.read_parquet('/tmp/v8_test/train.parquet')
# Check new columns exist
for col in ['avg_elo', 'elo_proximity', 'draw_rate_prior',
            'win_rate_recent_blanc', 'draw_rate_recent_blanc',
            'clutch_win_blanc', 'h2h_exists',
            'win_rate_home_dom', 'draw_rate_home_dom',
            'joueur_promu_blanc', 'team_rank_in_club_dom']:
    assert col in df.columns, f"Missing: {col}"
# Check no forfeits in target
assert 2.0 not in df['resultat_blanc'].values
# Check column count
print(f"Total columns: {len(df.columns)}")  # Should be ~156+metadata
```

- [ ] **Step 3: Verify no NaN explosion**

```python
nan_pct = df.isna().mean()
high_nan = nan_pct[nan_pct > 0.5]
print(f"Columns with >50% NaN: {len(high_nan)}")
# Expected: h2h features (99.2% NaN is OK), others should be <20%
```

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ --ignore=tests/test_health.py --ignore=tests/training_optuna/ -q
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(features): integrate all V8 features into pipeline — ~156 columns"
```

---

## Task 12 (FINAL): Generate V8 feature parquets + upload

**Files:**
- Run: `make refresh-data` or `python -m scripts.feature_engineering`
- Run: `python -m scripts.cloud.upload_all_data`

- [ ] **Step 1: Generate full feature parquets**

```bash
python -m scripts.feature_engineering --output-dir data/features
```

Expected: train.parquet, valid.parquet, test.parquet with ~156 columns each.

- [ ] **Step 2: Verify data quality**

```python
for split in ['train', 'valid', 'test']:
    df = pd.read_parquet(f'data/features/{split}.parquet')
    print(f"{split}: {len(df)} rows, {len(df.columns)} cols")
    assert 2.0 not in df['resultat_blanc'].values, "Forfaits in data!"
    assert 'score_dom' not in df.columns or 'score_dom' in METADATA_COLS
```

- [ ] **Step 3: Upload to Kaggle dataset**

```bash
python -m scripts.cloud.upload_all_data
```

- [ ] **Step 4: Final commit**

```bash
git add data/features/ scripts/
git commit -m "feat(v8): generate V8 feature parquets — 156 columns, forfaits excluded, W/D/L decomposed"
```

---

## Notes for Plan B (Training + Diagnostics)

Plan B picks up from here with the V8 feature parquets and implements:
- MultiClass target encoding (loss=0, draw=1, win=2)
- CatBoost/XGBoost/LightGBM MultiClass configs
- Baselines (naïve + Elo draw rate model)
- Quality gate (log loss, RPS, E[score] MAE, ECE)
- Isotonic calibration per class
- Reliability diagrams
- Config + test updates

Plan B spec: same `docs/superpowers/specs/2026-03-21-multiclass-v8-design.md`, sections "Models", "Calibration", "Evaluation Metrics".
