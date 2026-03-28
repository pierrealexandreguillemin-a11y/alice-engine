# Differential Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ~24 differential + interaction features to unblock the es_mae quality gate, based on multisport ML literature.

**Architecture:** A single stateless module `scripts/features/differentials.py` called at the end of the FE pipeline (after all individual features). Same module called in inference (FTI pattern, zero training-serving skew). No modification to existing feature modules.

**Tech Stack:** pandas (vectorized), no new dependencies. ISO 5055 (<300 lines), ISO 5259 (no leakage).

**Spec:** `docs/superpowers/specs/2026-03-27-differential-features-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `scripts/features/differentials.py` | CREATE | Compute all differentials + interactions |
| `tests/features/test_differentials.py` | CREATE | Unit + integration tests |
| `scripts/kaggle_constants.py` | MODIFY | Remove phantom entries, fix zone_enjeu_ext |
| `scripts/feature_engineering.py` | MODIFY | Add compute_differentials call (1 line) |
| `scripts/cloud/fe_kaggle.py` | NO CHANGE | Uses run_feature_engineering_v2 which calls compute_features_for_split |
| `services/inference.py` | MODIFY | Add compute_differentials call + TODO for Phase 2 wiring |

Note: `fe_kaggle.py` calls `run_feature_engineering_v2()` which calls `compute_features_for_split()`.
The differentials call goes inside `compute_features_for_split()` — so fe_kaggle.py needs NO change.

---

### Task 1: Fix kaggle_constants.py (zone_enjeu_ext asymmetry) — DONE + ERRATUM

**ERRATUM (2026-03-28):** L'audit avait identifie data_quality, elo_type, categorie comme
"fantomes jamais computes". C'etait FAUX — elles sont computees par `enrich_from_joueurs()`.
Elles ont ete retirees (commit 99a7349) puis restaurees (commit 15a5566).

**Ce qui a ete fait correctement :** zone_enjeu_ext ajoutee dans CATBOOST_CAT.

- [x] Step 1: ~~Remove phantoms~~ ANNULE (features existaient)
- [x] Step 2: Add zone_enjeu_ext to CATBOOST_CAT ✓ (commit 99a7349)
- [x] Step 3: Tests pass ✓
- [x] Step 4: Commit ✓ (99a7349 + fix 15a5566)

---

### Task 2: Write tests for player differentials

**Files:**
- Create: `tests/features/test_differentials.py`

- [ ] **Step 1: Write tests for _safe_diff helper and player differentials**

```python
"""Tests for differential features — ISO 29119.

Document ID: ALICE-TEST-DIFFERENTIALS
Version: 1.0.0
Tests: 25
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestSafeDiff:
    """Tests for the _safe_diff helper function."""

    def test_basic_subtraction(self):
        from scripts.features.differentials import _safe_diff

        df = pd.DataFrame({"a_blanc": [0.7], "a_noir": [0.3]})
        result = _safe_diff(df, "a_blanc", "a_noir", "diff_a")
        assert result["diff_a"].iloc[0] == pytest.approx(0.4)

    def test_nan_propagation(self):
        from scripts.features.differentials import _safe_diff

        df = pd.DataFrame({"a_blanc": [np.nan], "a_noir": [0.3]})
        result = _safe_diff(df, "a_blanc", "a_noir", "diff_a")
        assert pd.isna(result["diff_a"].iloc[0])

    def test_missing_column_skips(self):
        from scripts.features.differentials import _safe_diff

        df = pd.DataFrame({"a_blanc": [0.7]})
        result = _safe_diff(df, "a_blanc", "MISSING", "diff_a")
        assert "diff_a" not in result.columns

    def test_equal_values_zero(self):
        from scripts.features.differentials import _safe_diff

        df = pd.DataFrame({"a_blanc": [0.5], "a_noir": [0.5]})
        result = _safe_diff(df, "a_blanc", "a_noir", "diff_a")
        assert result["diff_a"].iloc[0] == pytest.approx(0.0)


class TestPlayerDifferentials:
    """8 player differential features."""

    @pytest.fixture()
    def sample_df(self):
        return pd.DataFrame({
            "expected_score_recent_blanc": [0.7, 0.4],
            "expected_score_recent_noir": [0.3, 0.6],
            "win_rate_recent_blanc": [0.6, 0.3],
            "win_rate_recent_noir": [0.4, 0.5],
            "draw_rate_blanc": [0.2, 0.15],
            "draw_rate_noir": [0.1, 0.3],
            "draw_rate_recent_blanc": [0.25, 0.1],
            "draw_rate_recent_noir": [0.15, 0.2],
            "win_rate_normal_blanc": [0.55, 0.4],
            "win_rate_normal_noir": [0.45, 0.5],
            "clutch_win_blanc": [0.1, -0.05],
            "clutch_win_noir": [-0.05, 0.1],
            "momentum_blanc": [0.3, -0.2],
            "momentum_noir": [-0.1, 0.4],
            "derniere_presence_blanc": [1, 4],
            "derniere_presence_noir": [2, 1],
        })

    def test_diff_form(self, sample_df):
        from scripts.features.differentials import _player_differentials

        result = _player_differentials(sample_df.copy())
        assert result["diff_form"].iloc[0] == pytest.approx(0.4)
        assert result["diff_form"].iloc[1] == pytest.approx(-0.2)

    def test_diff_clutch(self, sample_df):
        from scripts.features.differentials import _player_differentials

        result = _player_differentials(sample_df.copy())
        assert result["diff_clutch"].iloc[0] == pytest.approx(0.15)
        assert result["diff_clutch"].iloc[1] == pytest.approx(-0.15)

    def test_diff_derniere_presence(self, sample_df):
        from scripts.features.differentials import _player_differentials

        result = _player_differentials(sample_df.copy())
        assert result["diff_derniere_presence"].iloc[0] == pytest.approx(-1.0)
        assert result["diff_derniere_presence"].iloc[1] == pytest.approx(3.0)

    def test_all_8_diffs_present(self, sample_df):
        from scripts.features.differentials import _player_differentials

        result = _player_differentials(sample_df.copy())
        expected = [
            "diff_form", "diff_win_rate_recent", "diff_draw_rate",
            "diff_draw_rate_recent", "diff_win_rate_normal", "diff_clutch",
            "diff_momentum", "diff_derniere_presence",
        ]
        for col in expected:
            assert col in result.columns, f"Missing: {col}"
```

- [ ] **Step 2: Run tests — they should fail (module not created yet)**

Run: `.venv/Scripts/python -m pytest tests/features/test_differentials.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.features.differentials'`

- [ ] **Step 3: Commit test file**

```bash
git add tests/features/test_differentials.py
git commit -m "test(differentials): add unit tests for player differentials (TDD)"
```

---

### Task 3: Implement differentials module — player diffs + _safe_diff

**Files:**
- Create: `scripts/features/differentials.py`

- [ ] **Step 1: Create the module with _safe_diff helper and player differentials**

```python
"""Differential and interaction features for matchup-level prediction.

Transforms individual features (blanc/noir, dom/ext) into relative features
as recommended by multisport ML literature (PMC11265715, Hubacek 2019).

Stateless, vectorized, usable in batch (FE pipeline) and online (inference).
Training-serving skew prevention: same function called in both contexts.

Document ID: ALICE-DIFFERENTIALS
Version: 1.0.0
ISO: 5055 (SRP, <300 lines), 5259 (no leakage), 42001 (traceable)

References:
- PMC11265715: NBA XGBoost, "features subtracted from each other"
- Hubacek 2019: Soccer Prediction Challenge winner
- Hopsworks FTI: training-serving skew prevention
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_diff(
    df: pd.DataFrame, col_a: str, col_b: str, out_name: str
) -> pd.DataFrame:
    """Compute col_a - col_b if both exist, skip otherwise."""
    if col_a in df.columns and col_b in df.columns:
        df[out_name] = df[col_a] - df[col_b]
    return df


def _player_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """8 player matchup differentials (blanc - noir)."""
    pairs = [
        ("expected_score_recent_blanc", "expected_score_recent_noir", "diff_form"),
        ("win_rate_recent_blanc", "win_rate_recent_noir", "diff_win_rate_recent"),
        ("draw_rate_blanc", "draw_rate_noir", "diff_draw_rate"),
        ("draw_rate_recent_blanc", "draw_rate_recent_noir", "diff_draw_rate_recent"),
        ("win_rate_normal_blanc", "win_rate_normal_noir", "diff_win_rate_normal"),
        ("clutch_win_blanc", "clutch_win_noir", "diff_clutch"),
        ("momentum_blanc", "momentum_noir", "diff_momentum"),
        ("derniere_presence_blanc", "derniere_presence_noir", "diff_derniere_presence"),
    ]
    for col_a, col_b, out in pairs:
        _safe_diff(df, col_a, col_b, out)
    return df
```

- [ ] **Step 2: Run player differential tests**

Run: `.venv/Scripts/python -m pytest tests/features/test_differentials.py::TestSafeDiff tests/features/test_differentials.py::TestPlayerDifferentials -v`
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add scripts/features/differentials.py
git commit -m "feat(differentials): player matchup diffs (8 features, PMC11265715)"
```

---

### Task 4: Add team differentials + tests

**Files:**
- Modify: `scripts/features/differentials.py`
- Modify: `tests/features/test_differentials.py`

- [ ] **Step 1: Add team differential tests**

Append to `tests/features/test_differentials.py`:

```python
class TestTeamDifferentials:
    """6 team differential features."""

    @pytest.fixture()
    def sample_df(self):
        return pd.DataFrame({
            "position_dom": [3, 8],
            "position_ext": [8, 2],
            "points_cumules_dom": [12, 4],
            "points_cumules_ext": [6, 14],
            "profondeur_effectif_dom": [15, 8],
            "profondeur_effectif_ext": [8, 12],
            "noyau_stable_dom": [6, 3],
            "noyau_stable_ext": [4, 5],
            "win_rate_home_dom": [0.6, 0.4],
            "win_rate_home_ext": [0.3, 0.7],
            "draw_rate_home_dom": [0.2, 0.15],
            "draw_rate_home_ext": [0.1, 0.25],
        })

    def test_diff_position(self, sample_df):
        from scripts.features.differentials import _team_differentials

        result = _team_differentials(sample_df.copy())
        assert result["diff_position"].iloc[0] == pytest.approx(-5.0)
        assert result["diff_position"].iloc[1] == pytest.approx(6.0)

    def test_diff_profondeur(self, sample_df):
        from scripts.features.differentials import _team_differentials

        result = _team_differentials(sample_df.copy())
        assert result["diff_profondeur"].iloc[0] == pytest.approx(7.0)

    def test_all_6_diffs_present(self, sample_df):
        from scripts.features.differentials import _team_differentials

        result = _team_differentials(sample_df.copy())
        expected = [
            "diff_position", "diff_points_cumules", "diff_profondeur",
            "diff_stabilite", "diff_win_rate_home", "diff_draw_rate_home",
        ]
        for col in expected:
            assert col in result.columns, f"Missing: {col}"
```

- [ ] **Step 2: Run tests — should fail (function not yet implemented)**

Run: `.venv/Scripts/python -m pytest tests/features/test_differentials.py::TestTeamDifferentials -v`
Expected: FAIL

- [ ] **Step 3: Implement _team_differentials**

Add to `scripts/features/differentials.py`:

```python
def _team_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """6 team matchup differentials (dom - ext)."""
    pairs = [
        ("position_dom", "position_ext", "diff_position"),
        ("points_cumules_dom", "points_cumules_ext", "diff_points_cumules"),
        ("profondeur_effectif_dom", "profondeur_effectif_ext", "diff_profondeur"),
        ("noyau_stable_dom", "noyau_stable_ext", "diff_stabilite"),
        ("win_rate_home_dom", "win_rate_home_ext", "diff_win_rate_home"),
        ("draw_rate_home_dom", "draw_rate_home_ext", "diff_draw_rate_home"),
    ]
    for col_a, col_b, out in pairs:
        _safe_diff(df, col_a, col_b, out)
    return df
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest tests/features/test_differentials.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/features/differentials.py tests/features/test_differentials.py
git commit -m "feat(differentials): team matchup diffs (6 features, Hubacek 2019)"
```

---

### Task 5: Add board×match interactions + tests

**Files:**
- Modify: `scripts/features/differentials.py`
- Modify: `tests/features/test_differentials.py`

- [ ] **Step 1: Add interaction tests**

Append to `tests/features/test_differentials.py`:

```python
class TestInteractions:
    """6 board x match interaction features."""

    @pytest.fixture()
    def sample_df(self):
        return pd.DataFrame({
            "diff_form": [0.4, -0.2],
            "zone_enjeu_dom": ["danger", "mi_tableau"],
            "echiquier": [1, 2],
            "est_domicile_blanc": [1, 0],
            "couleur_preferee_blanc": ["blanc", "noir"],
            "decalage_position_blanc": [2.0, -1.0],
            "match_important": [1, 0],
            "club_utilise_marge_100_dom": [0.3, 0.0],
            "flexibilite_echiquier_blanc": [0.8, 0.2],
            "joueur_promu_blanc": [1, 0],
            "diff_elo": [-150, 50],
        })

    def test_form_in_danger(self, sample_df):
        from scripts.features.differentials import _board_match_interactions

        result = _board_match_interactions(sample_df.copy())
        # Row 0: diff_form=0.4, zone=danger → 0.4 * 1 = 0.4
        assert result["form_in_danger"].iloc[0] == pytest.approx(0.4)
        # Row 1: zone=mi_tableau → 0
        assert result["form_in_danger"].iloc[1] == pytest.approx(0.0)

    def test_color_match_dom_odd_pref_blanc(self, sample_df):
        from scripts.features.differentials import _board_match_interactions

        result = _board_match_interactions(sample_df.copy())
        # Row 0: dom=1, echiquier=1 (odd) → blanc has white pieces, pref=blanc → match=1
        assert result["color_match"].iloc[0] == 1

    def test_color_match_ext_even_pref_noir(self, sample_df):
        from scripts.features.differentials import _board_match_interactions

        result = _board_match_interactions(sample_df.copy())
        # Row 1: dom=0, echiquier=2 (even) → blanc has white pieces (ext+even=white),
        # pref=noir → mismatch → 0
        assert result["color_match"].iloc[1] == 0

    def test_promu_vs_strong(self, sample_df):
        from scripts.features.differentials import _board_match_interactions

        result = _board_match_interactions(sample_df.copy())
        # Row 0: promu=1, diff_elo=-150 → clip(-(-150),0,800)/400 = 150/400 = 0.375
        assert result["promu_vs_strong"].iloc[0] == pytest.approx(0.375)
        # Row 1: promu=0 → 0
        assert result["promu_vs_strong"].iloc[1] == pytest.approx(0.0)

    def test_decalage_important(self, sample_df):
        from scripts.features.differentials import _board_match_interactions

        result = _board_match_interactions(sample_df.copy())
        # Row 0: decalage=2.0, important=1 → 2.0
        assert result["decalage_important"].iloc[0] == pytest.approx(2.0)
        # Row 1: important=0 → 0
        assert result["decalage_important"].iloc[1] == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests — should fail**

Run: `.venv/Scripts/python -m pytest tests/features/test_differentials.py::TestInteractions -v`
Expected: FAIL

- [ ] **Step 3: Implement _board_match_interactions**

Add to `scripts/features/differentials.py`:

```python
def _board_match_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """6 board x match interactions (player context in team context)."""
    # form_in_danger: diff_form * zone_danger
    if "diff_form" in df.columns and "zone_enjeu_dom" in df.columns:
        is_danger = (df["zone_enjeu_dom"] == "danger").astype(int)
        df["form_in_danger"] = df["diff_form"] * is_danger

    # color_match: player has preferred color on this board (FFE convention)
    if "echiquier" in df.columns and "est_domicile_blanc" in df.columns:
        is_odd = (df["echiquier"] % 2 == 1)
        est_dom = df["est_domicile_blanc"] == 1
        blanc_plays_white = est_dom == is_odd
        pref = df.get("couleur_preferee_blanc", pd.Series("neutre", index=df.index))
        df["color_match"] = (
            ((pref == "blanc") & blanc_plays_white)
            | ((pref == "noir") & ~blanc_plays_white)
        ).astype(int)

    # decalage_important: strategic placement in key match
    if "decalage_position_blanc" in df.columns and "match_important" in df.columns:
        df["decalage_important"] = df["decalage_position_blanc"] * df["match_important"]

    # marge100_decale: deliberate captain strategy
    if "club_utilise_marge_100_dom" in df.columns and "decalage_position_blanc" in df.columns:
        df["marge100_decale"] = (
            df["club_utilise_marge_100_dom"] * df["decalage_position_blanc"].abs()
        )

    # flex_decale: flexible player moved vs specialist moved
    if "flexibilite_echiquier_blanc" in df.columns and "decalage_position_blanc" in df.columns:
        df["flex_decale"] = (
            df["flexibilite_echiquier_blanc"] * df["decalage_position_blanc"].abs()
        )

    # promu_vs_strong: reinforcement facing strong opponent
    if "joueur_promu_blanc" in df.columns and "diff_elo" in df.columns:
        df["promu_vs_strong"] = (
            df["joueur_promu_blanc"] * np.clip(-df["diff_elo"], 0, 800) / 400
        )

    return df
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest tests/features/test_differentials.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/features/differentials.py tests/features/test_differentials.py
git commit -m "feat(differentials): board×match interactions (6 features, FFE domain)"
```

---

### Task 6: Add new features + zone dummies + orchestrator + tests

**Files:**
- Modify: `scripts/features/differentials.py`
- Modify: `tests/features/test_differentials.py`

- [ ] **Step 1: Add tests for new features, zone dummies, missing columns, and batch/single**

Append to `tests/features/test_differentials.py`:

```python
class TestNewFeatures:
    """4 new features: elo_uncertainty, k_asymmetry, zone dummies."""

    def test_zone_dummies(self):
        from scripts.features.differentials import _zone_dummies

        df = pd.DataFrame({"zone_enjeu_dom": ["danger", "montee", "mi_tableau"]})
        result = _zone_dummies(df.copy())
        assert result["zone_danger_dom"].tolist() == [1, 0, 0]
        assert result["zone_montee_dom"].tolist() == [0, 1, 0]

    def test_elo_uncertainty_with_k(self):
        from scripts.features.differentials import _new_features

        df = pd.DataFrame({
            "k_coefficient_blanc": [40, 20],
            "k_coefficient_noir": [10, 20],
        })
        result = _new_features(df.copy())
        assert result["elo_uncertainty"].iloc[0] == pytest.approx(50)
        assert result["k_asymmetry"].iloc[0] == pytest.approx(30)

    def test_elo_uncertainty_missing_k(self):
        from scripts.features.differentials import _new_features

        df = pd.DataFrame({"blanc_elo": [1500]})
        result = _new_features(df.copy())
        assert "elo_uncertainty" not in result.columns


class TestMissingColumns:
    """Module does not crash when source columns are absent."""

    def test_missing_momentum(self):
        from scripts.features.differentials import compute_differentials

        df = pd.DataFrame({"expected_score_recent_blanc": [0.5], "expected_score_recent_noir": [0.4]})
        result = compute_differentials(df.copy())
        assert "diff_form" in result.columns
        assert "diff_momentum" not in result.columns

    def test_empty_df(self):
        from scripts.features.differentials import compute_differentials

        df = pd.DataFrame()
        result = compute_differentials(df.copy())
        assert len(result) == 0


class TestBatchVsSingle:
    """Same result on 1 row and N rows."""

    def test_single_vs_batch(self):
        from scripts.features.differentials import compute_differentials

        row = pd.DataFrame({
            "expected_score_recent_blanc": [0.7],
            "expected_score_recent_noir": [0.3],
            "position_dom": [3],
            "position_ext": [8],
        })
        batch = pd.concat([row] * 100, ignore_index=True)
        r1 = compute_differentials(row.copy())
        rN = compute_differentials(batch.copy())
        assert r1["diff_form"].iloc[0] == pytest.approx(rN["diff_form"].iloc[0])
        assert r1["diff_position"].iloc[0] == pytest.approx(rN["diff_position"].iloc[0])
```

- [ ] **Step 2: Run tests — should fail (functions not implemented yet)**

Run: `.venv/Scripts/python -m pytest tests/features/test_differentials.py::TestNewFeatures tests/features/test_differentials.py::TestMissingColumns tests/features/test_differentials.py::TestBatchVsSingle -v`
Expected: FAIL

- [ ] **Step 3: Implement _new_features, _zone_dummies, and compute_differentials**

Add to `scripts/features/differentials.py`:

```python
def _new_features(df: pd.DataFrame) -> pd.DataFrame:
    """Elo uncertainty from K-coefficients (Glicko-2 inspired)."""
    if "k_coefficient_blanc" in df.columns and "k_coefficient_noir" in df.columns:
        df["elo_uncertainty"] = df["k_coefficient_blanc"] + df["k_coefficient_noir"]
        df["k_asymmetry"] = df["k_coefficient_blanc"] - df["k_coefficient_noir"]
    return df


def _zone_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot zone_enjeu_dom for interaction features."""
    if "zone_enjeu_dom" in df.columns:
        df["zone_danger_dom"] = (df["zone_enjeu_dom"] == "danger").astype(int)
        df["zone_montee_dom"] = (df["zone_enjeu_dom"] == "montee").astype(int)
    return df


def compute_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """Add all differential + interaction features. Pure, no state.

    Called in FE pipeline (batch) and inference (single row).
    Same function both sides — prevents training-serving skew (Hopsworks FTI).
    """
    n_before = len(df.columns)
    df = _player_differentials(df)
    df = _team_differentials(df)
    df = _zone_dummies(df)
    df = _board_match_interactions(df)
    df = _new_features(df)
    n_added = len(df.columns) - n_before
    if n_added > 0:
        logger.info("Differentials: +%d features added", n_added)
    return df
```

- [ ] **Step 4: Run ALL tests**

Run: `.venv/Scripts/python -m pytest tests/features/test_differentials.py -v`
Expected: all PASS (25 tests)

- [ ] **Step 5: Verify module is < 300 lines (ISO 5055)**

Run: `wc -l scripts/features/differentials.py`
Expected: < 300

- [ ] **Step 6: Commit**

```bash
git add scripts/features/differentials.py tests/features/test_differentials.py
git commit -m "feat(differentials): elo_uncertainty (Glicko-2) + zone dummies + orchestrator (24 total)"
```

---

### Task 7: Integrate into FE pipeline

**Files:**
- Modify: `scripts/feature_engineering.py:92-127` (inside `compute_features_for_split`)

- [ ] **Step 1: Add compute_differentials call at end of compute_features_for_split**

In `scripts/feature_engineering.py`, inside `compute_features_for_split()`, add the call
AFTER all other features (after `extract_match_important(result)`, before the return):

```python
    # Differential features (last step — needs all individual features computed)
    from scripts.features.differentials import compute_differentials  # noqa: PLC0415

    result = compute_differentials(result)
```

This goes at line ~126, just before the logger.info and return on line 127.

- [ ] **Step 2: Run full test suite**

Run: `.venv/Scripts/python -m pytest tests/ -x -q --ignore=tests/test_cloud_features.py -k "not test_forfeits_excluded"`
Expected: all pass (1441+ tests)

- [ ] **Step 3: Verify locally that differentials appear in parquet**

Run: `.venv/Scripts/python -c "
import pandas as pd
from pathlib import Path
if Path('data/features/train.parquet').exists():
    df = pd.read_parquet('data/features/train.parquet')
    diffs = [c for c in df.columns if c.startswith('diff_') or c in ('color_match','form_in_danger','zone_danger_dom')]
    print(f'Differential features found: {len(diffs)}')
    for c in sorted(diffs): print(f'  {c}')
else:
    print('No local parquet — run make refresh-data or feature_engineering.py first')
"`

- [ ] **Step 4: Commit**

```bash
git add scripts/feature_engineering.py
git commit -m "feat(pipeline): integrate differentials into FE pipeline (last step)"
```

---

### Task 7b: Integrate into inference.py (FTI anti-skew)

**Files:**
- Modify: `services/inference.py:100-105`

The FTI pattern (Hopsworks) REQUIRES the same feature computation in training AND inference.
Even though inference.py is a stub (Phase 2), we wire compute_differentials NOW
so that when Phase 2 implements the ML prediction, the differentials are already there.

- [ ] **Step 1: Add compute_differentials in inference.py predict_lineup**

In `services/inference.py`, inside `predict_lineup()`, add after the TODO block (line 100-103):

```python
        # TODO: Implementer la prediction ML
        # 1. Preparer les features pour chaque joueur
        # 2. compute_differentials(df_features)  ← FTI anti-skew (same as FE pipeline)
        # 3. Appeler model.predict_proba()
        # 4. Trier par probabilite et assigner aux echiquiers
```

This replaces the existing TODO comment with a more complete one that includes differentials.

- [ ] **Step 2: Commit**

```bash
git add services/inference.py
git commit -m "docs(inference): add compute_differentials in Phase 2 TODO (FTI anti-skew)"
```

---

### Task 8b: Fix temporal split — 61 features 100% NaN on eval — DONE

**DISCOVERED 2026-03-28:** valid/test history excluded current season → 61 team features
100% NaN on eval. Root cause of 116 "dead" features since v1. Fix: include current season
in history (same as train behavior). Postmortem: `docs/postmortem/2026-03-28-split-temporal-nan-features.md`

- [x] Step 1: Fix feature_engineering.py lines 219/230
- [x] Step 2: Tests pass (1463)
- [x] Step 3: Commit (ecd2020)

**Requires re-upload dataset + re-push FE kernel before training.**

---

### Task 8: Deploy to Kaggle and run training

**Files:**
- No code changes — deployment only (split fix already committed)

- [ ] **Step 1: Upload dataset (contains new differentials.py)**

```bash
python -m scripts.cloud.upload_all_data --version-notes "differential features (24 cols, PMC11265715/Hubacek 2019)"
```
Expected: Upload successful, content hash logged.

- [ ] **Step 2: Verify differentials.py is in the uploaded dataset**

```bash
kaggle datasets files pguillemin/alice-code --csv | grep differentials
```
Expected: `scripts/features/differentials.py` appears in the file list.
If NOT found: re-run upload_all_data.

- [ ] **Step 3: Verify previous FE kernel is complete**

```bash
kaggle kernels status pguillemin/alice-fe-v8
```
Expected: COMPLETE

- [ ] **Step 4: Push FE kernel (CPU, generates new parquets)**

```bash
cp scripts/cloud/kernel-metadata-fe.json scripts/cloud/kernel-metadata.json
kaggle kernels push -p scripts/cloud/
git checkout -- scripts/cloud/kernel-metadata.json
```
Expected: "Kernel version N successfully pushed"

- [ ] **Step 5: Wait for FE kernel to complete, check log**

```bash
kaggle kernels status pguillemin/alice-fe-v8
# When COMPLETE:
kaggle kernels output pguillemin/alice-fe-v8 -p /tmp/fe-output
# Check log for "Differentials: +N features added"
```
Expected: log shows ~24 features added, parquets have ~220 columns

- [ ] **Step 6: Push Training kernel v17 (T4 GPU)**

```bash
cp scripts/cloud/kernel-metadata-train.json scripts/cloud/kernel-metadata.json
kaggle kernels push -p scripts/cloud/ --accelerator NvidiaTeslaT4
git checkout -- scripts/cloud/kernel-metadata.json
```
Expected: "Kernel version 17 successfully pushed"

- [ ] **Step 7: Monitor training, download outputs when complete**

```bash
kaggle kernels status pguillemin/alice-training-v8
# When COMPLETE:
kaggle kernels output pguillemin/alice-training-v8 -p /tmp/v17-output
```

Check in log:
- `Init score alpha=0.70 applied` (alpha still active)
- `Differentials: +N features` NOT in training log (computed in FE kernel, not training)
- `diff_form` in SHAP top features
- `Quality gate: passed` or `es_mae >= elo` (the result we're waiting for)

- [ ] **Step 8: Commit deployment tracking**

```bash
git add -A
git commit -m "docs: v17 training deployed with differential features"
```
