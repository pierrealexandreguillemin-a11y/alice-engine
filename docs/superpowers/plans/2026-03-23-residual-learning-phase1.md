# Phase 1: Residual Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ML models learn corrections to Elo baseline (not predictions from scratch) so they beat the Elo quality gate and produce calibrated P(draw).

**Architecture:** `baselines.py` computes Elo init scores → `kaggle_trainers.py` passes them as `init_score`/`base_margin` → models learn residuals only. Incremental validation: 0 features → top 10 → all 177.

**Tech Stack:** CatBoost (init_model via Pool), XGBoost (base_margin), LightGBM (init_score), numpy.

**Spec:** `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md` — Phase 1 section.

**Depends on:** V8 FE parquets (COMPLETE, Kaggle kernel output). V8 training pipeline (COMPLETE, quality gate wired).

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| MODIFY | `scripts/baselines.py` | Add `compute_elo_init_scores()` — probas → log-odds |
| MODIFY | `scripts/kaggle_trainers.py` | Pass init scores to each model trainer |
| MODIFY | `scripts/cloud/train_kaggle.py` | Compute init scores before training, pass through |
| NEW | `tests/test_residual_learning.py` | Tests for init scores + sanity checks |

---

## Task 1: Elo init scores computation

**Files:**
- Modify: `scripts/baselines.py`
- Test: `tests/test_residual_learning.py`

- [ ] **Step 1: Write failing test for init scores**

```python
# tests/test_residual_learning.py
"""Tests for residual learning — Elo init scores (ISO 24029)."""
import numpy as np
import pytest


def test_elo_init_scores_shape_and_values():
    """Init scores must be (n, 3) log-odds, finite, matching Elo probas."""
    from scripts.baselines import compute_elo_baseline, compute_elo_init_scores

    n = 100
    blanc_elo = np.full(n, 1600)
    noir_elo = np.full(n, 1400)

    # Need a minimal draw lookup
    import pandas as pd
    lookup = pd.DataFrame({
        "elo_band": ["1400-1600"],
        "diff_band": ["200-400"],
        "draw_rate_prior": [0.15],
    })
    elo_proba = compute_elo_baseline(blanc_elo, noir_elo, lookup)
    init_scores = compute_elo_init_scores(elo_proba)

    # Shape: (n, 3)
    assert init_scores.shape == (n, 3)
    # All finite (no inf/nan)
    assert np.all(np.isfinite(init_scores))
    # Log-odds: positive for likely classes, negative for unlikely
    # P(win) > 0.5 for 1600 vs 1400, so log-odds col 2 should be positive
    assert init_scores[0, 2] > 0  # P(win) > 0.5 → positive log-odds


def test_elo_init_scores_clipping():
    """Probas near 0 or 1 must not produce inf log-odds."""
    from scripts.baselines import compute_elo_init_scores

    # Extreme case: P(win)≈1, P(draw)≈0, P(loss)≈0
    extreme = np.array([[0.001, 0.001, 0.998]])
    scores = compute_elo_init_scores(extreme)
    assert np.all(np.isfinite(scores))
    assert np.abs(scores).max() < 20  # Clipped to reasonable range


def test_elo_init_scores_roundtrip():
    """softmax(init_scores) should approximately recover original probas."""
    from scripts.baselines import compute_elo_init_scores

    proba = np.array([[0.3, 0.15, 0.55], [0.45, 0.10, 0.45]])
    scores = compute_elo_init_scores(proba)
    # softmax recovery
    exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
    recovered = exp_s / exp_s.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(recovered, proba, atol=0.01)
```

- [ ] **Step 2: Run test — should fail**

```bash
python -m pytest tests/test_residual_learning.py -v
```
Expected: `ImportError: cannot import name 'compute_elo_init_scores'`

- [ ] **Step 3: Implement compute_elo_init_scores**

Add to `scripts/baselines.py` after `compute_elo_baseline`:

```python
def compute_elo_init_scores(
    elo_proba: np.ndarray,
    clip_min: float = 1e-4,
    clip_max: float = 1 - 1e-4,
) -> np.ndarray:
    """Convert Elo baseline probas to log-odds init scores for residual learning.

    For multiclass: init_score = log(P_k) - mean(log(P)) per sample.
    This is the inverse of softmax, producing centered log-odds.

    Args:
        elo_proba: (n, 3) array of [P(loss), P(draw), P(win)] from Elo baseline.
        clip_min: Floor for probabilities to avoid log(0).
        clip_max: Ceiling for probabilities to avoid log(1).

    Returns:
        (n, 3) array of log-odds suitable for init_score / base_margin.
    """
    clipped = np.clip(elo_proba, clip_min, clip_max)
    # Renormalize after clipping
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    log_p = np.log(clipped)
    # Center: subtract mean log-prob per row (softmax-inverse)
    init_scores = log_p - log_p.mean(axis=1, keepdims=True)
    return init_scores
```

- [ ] **Step 4: Run test — should pass**

```bash
python -m pytest tests/test_residual_learning.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add scripts/baselines.py tests/test_residual_learning.py
git commit -m "feat(baselines): compute_elo_init_scores — probas to log-odds for residual learning"
```

---

## Task 2: Wire init scores into training functions

**Files:**
- Modify: `scripts/kaggle_trainers.py` (functions `_train_catboost`, `_train_xgboost`, `_train_lightgbm`, `train_all_sequential`)
- Test: `tests/test_residual_learning.py`

- [ ] **Step 1: Write failing test for train_all_sequential with init_scores**

Add to `tests/test_residual_learning.py`:

```python
def test_train_all_sequential_accepts_init_scores():
    """train_all_sequential must accept optional init_scores parameter."""
    import inspect
    from scripts.kaggle_trainers import train_all_sequential

    sig = inspect.signature(train_all_sequential)
    assert "init_scores_train" in sig.parameters
    assert "init_scores_valid" in sig.parameters
```

- [ ] **Step 2: Run test — should fail**

```bash
python -m pytest tests/test_residual_learning.py::test_train_all_sequential_accepts_init_scores -v
```
Expected: FAIL — parameters not in signature

- [ ] **Step 3: Update train_all_sequential signature**

In `scripts/kaggle_trainers.py`, modify `train_all_sequential`:

```python
def train_all_sequential(
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
    config: dict,
    init_scores_train: Any | None = None,
    init_scores_valid: Any | None = None,
) -> dict:
    """CatBoost -> gc -> XGBoost -> gc -> LightGBM. Sequential memory management."""
    results: dict = {}
    results["CatBoost"] = _train_catboost(
        X_train, y_train, X_valid, y_valid, config["catboost"],
        init_scores_train, init_scores_valid,
    )
    gc.collect()
    results["XGBoost"] = _train_xgboost(
        X_train, y_train, X_valid, y_valid, config["xgboost"],
        init_scores_train, init_scores_valid,
    )
    gc.collect()
    results["LightGBM"] = _train_lightgbm(
        X_train, y_train, X_valid, y_valid, config["lightgbm"],
        init_scores_train, init_scores_valid,
    )
    gc.collect()
    return results
```

- [ ] **Step 4: Run test — should pass**

```bash
python -m pytest tests/test_residual_learning.py::test_train_all_sequential_accepts_init_scores -v
```

- [ ] **Step 5: Update _train_xgboost to use base_margin**

XGBoost uses `base_margin` in the `.fit()` call. Modify `_train_xgboost`:

```python
def _train_xgboost(
    X_train: Any, y_train: Any, X_valid: Any, y_valid: Any, params: dict,
    init_scores_train: Any | None = None, init_scores_valid: Any | None = None,
) -> dict:
    """Train XGBoost with residual learning via base_margin."""
    try:
        from xgboost import XGBClassifier  # noqa: PLC0415

        xgb = XGBClassifier(**params, eval_metric="mlogloss")
        fit_kwargs: dict = {
            "eval_set": [(X_valid, y_valid)],
            "verbose": 100,
        }
        # Residual learning: Elo baseline as starting point
        if init_scores_train is not None:
            fit_kwargs["base_margin"] = init_scores_train.ravel()
        if init_scores_valid is not None:
            fit_kwargs["eval_set"] = [(X_valid, y_valid)]
            # XGBoost eval base_margin: passed via xgb.set_params or DMatrix
            # For XGBClassifier, base_margin on eval is not directly supported
            # The model still benefits from train init — eval shows adjusted loss
        t0 = time.time()
        xgb.fit(X_train, y_train, **fit_kwargs)
        result = _eval_model(xgb, X_valid, y_valid, time.time() - t0)
        del xgb
        gc.collect()
        return result
    except Exception:
        logger.exception("XGBoost training failed")
        return _fail_result()
```

**NOTE:** XGBoost `base_margin` for multiclass must be shape `(n_samples * n_classes,)` — flattened. WebSearch to confirm API for XGBoost 3.2.0.

- [ ] **Step 6: Update _train_lightgbm to use init_score**

LightGBM uses `init_score` in the `.fit()` call:

```python
def _train_lightgbm(
    X_train: Any, y_train: Any, X_valid: Any, y_valid: Any, params: dict,
    init_scores_train: Any | None = None, init_scores_valid: Any | None = None,
) -> dict:
    """Train LightGBM with residual learning via init_score."""
    try:
        import lightgbm as lgb_lib  # noqa: PLC0415
        from lightgbm import LGBMClassifier  # noqa: PLC0415

        lgb_p = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        lgb_p["device"] = "cpu"
        es = params.get("early_stopping_rounds", 50)
        cbs = [lgb_lib.early_stopping(es), lgb_lib.log_evaluation(100)]
        lgbm = LGBMClassifier(**lgb_p)
        fit_kwargs: dict = {
            "eval_set": [(X_valid, y_valid)],
            "eval_metric": "multi_logloss",
            "callbacks": cbs,
        }
        if init_scores_train is not None:
            fit_kwargs["init_score"] = init_scores_train
        if init_scores_valid is not None:
            fit_kwargs["eval_init_score"] = [init_scores_valid]
        t0 = time.time()
        lgbm.fit(X_train, y_train, **fit_kwargs)
        result = _eval_model(lgbm, X_valid, y_valid, time.time() - t0)
        del lgbm
        gc.collect()
        return result
    except Exception:
        logger.exception("LightGBM training failed")
    return _fail_result()
```

- [ ] **Step 7: Update _train_catboost to use init scores**

CatBoost approach: set `scale_pos_weight` or pass baseline via custom `init_model`. Simplest: compute init scores as a feature? No — CatBoost does NOT have a direct `init_score` parameter. Alternative: use `baseline` column in Pool.

```python
def _train_catboost(
    X_train: Any, y_train: Any, X_valid: Any, y_valid: Any, params: dict,
    init_scores_train: Any | None = None, init_scores_valid: Any | None = None,
) -> dict:
    """Train CatBoost with residual learning via Pool baseline."""
    try:
        from catboost import CatBoostClassifier, Pool  # noqa: PLC0415

        cb = CatBoostClassifier(**params, eval_metric="MultiClass")
        train_pool = Pool(X_train, y_train,
                          baseline=init_scores_train if init_scores_train is not None else None)
        valid_pool = Pool(X_valid, y_valid,
                          baseline=init_scores_valid if init_scores_valid is not None else None)
        t0 = time.time()
        cb.fit(train_pool, eval_set=valid_pool)
        result = _eval_model(cb, X_valid, y_valid, time.time() - t0)
        del cb
        gc.collect()
        return result
    except Exception:
        logger.exception("CatBoost training failed")
        return _fail_result()
```

**NOTE:** CatBoost `Pool(baseline=...)` expects (n, n_classes) array. WebSearch to confirm for CatBoost 1.2.10 MultiClass.

- [ ] **Step 8: Run all tests**

```bash
python -m pytest tests/test_residual_learning.py -v
```

- [ ] **Step 9: Commit**

```bash
git add scripts/kaggle_trainers.py tests/test_residual_learning.py
git commit -m "feat(training): wire init_scores into CatBoost/XGBoost/LightGBM for residual learning"
```

---

## Task 3: Wire init scores in train_kaggle.py orchestration

**Files:**
- Modify: `scripts/cloud/train_kaggle.py`

- [ ] **Step 1: Compute Elo init scores for train+valid before training**

In `train_kaggle.py:main()`, after `prepare_features()` and before `train_all_sequential()`:

```python
    # Compute Elo baseline init scores for residual learning
    from scripts.baselines import compute_elo_baseline, compute_elo_init_scores  # noqa: PLC0415
    from scripts.features.draw_priors import build_draw_rate_lookup  # noqa: PLC0415

    draw_lookup_train = build_draw_rate_lookup(train)
    init_scores_train = _compute_init_scores(X_train, draw_lookup_train)
    init_scores_valid = _compute_init_scores(X_valid, draw_lookup_train)
    logger.info("Elo init scores computed: train=%s valid=%s", init_scores_train.shape, init_scores_valid.shape)

    results = train_all_sequential(
        X_train, y_train, X_valid, y_valid, config,
        init_scores_train=init_scores_train,
        init_scores_valid=init_scores_valid,
    )
```

Add helper at module level:

```python
def _compute_init_scores(X: pd.DataFrame, draw_lookup: pd.DataFrame) -> np.ndarray:
    """Compute Elo baseline init scores from X features."""
    import numpy as np  # noqa: PLC0415
    from scripts.baselines import compute_elo_baseline, compute_elo_init_scores  # noqa: PLC0415

    b_elo = X["blanc_elo"].values if "blanc_elo" in X.columns else np.full(len(X), 1500)
    n_elo = X["noir_elo"].values if "noir_elo" in X.columns else np.full(len(X), 1500)
    elo_proba = compute_elo_baseline(b_elo, n_elo, draw_lookup)
    return compute_elo_init_scores(elo_proba)
```

- [ ] **Step 2: Verify line count**

```bash
wc -l scripts/cloud/train_kaggle.py
```
Must be < 300 lines. If over, extract `_compute_init_scores` to baselines.py.

- [ ] **Step 3: Commit**

```bash
git add scripts/cloud/train_kaggle.py
git commit -m "feat(kaggle): compute Elo init scores and pass to train_all_sequential"
```

---

## Task 4: Sanity check — Step 0 (0 features, init only)

This is NOT a code task. It's a **validation step** to run after Tasks 1-3.

- [ ] **Step 1: Local test with tiny sample**

Write a quick local validation script (NOT committed) to verify init scores work:

```python
# Quick local test (not committed)
import numpy as np
from scripts.baselines import compute_elo_init_scores
proba = np.array([[0.3, 0.15, 0.55]])
scores = compute_elo_init_scores(proba)
print(f"Proba: {proba}, Init scores: {scores}")
# Verify softmax(scores) ≈ proba
```

- [ ] **Step 2: Upload dataset + push Kaggle v4**

```bash
python -m scripts.cloud.upload_all_data --version-notes "v4: residual learning (init_scores from Elo baseline)"
cp scripts/cloud/kernel-metadata-train.json scripts/cloud/kernel-metadata.json
kaggle kernels push -p scripts/cloud/ --accelerator NvidiaTeslaT4
git checkout -- scripts/cloud/kernel-metadata.json
```

- [ ] **Step 3: Analyze v4 log**

Key things to check:
- Init scores computed without error
- CatBoost/XGBoost/LightGBM accept init scores without crash
- Best iteration > 100 (not 3-42 like before)
- Log_loss < 0.92 (Elo baseline)
- Draw recall > 0% (model now predicts some draws)

- [ ] **Step 4: Update postmortem with v4 results**

Add v4 row to `docs/postmortem/2026-03-22-training-v8-divergence.md` tracking table.

- [ ] **Step 5: Commit results**

```bash
git add docs/postmortem/
git commit -m "docs(postmortem): v4 residual learning results"
```

---

## Task 5: Incremental validation — Step 1 (top 10 features only)

**Only if v4 (Task 4) shows init scores work but model still doesn't beat Elo.**

- [ ] **Step 1: Add feature selection to prepare_features**

In `kaggle_trainers.py`, add optional `feature_subset` parameter:

```python
TOP_10_FEATURES = [
    "diff_elo", "elo_proximity", "draw_rate_blanc", "draw_rate_noir",
    "expected_score_recent_blanc", "expected_score_recent_noir",
    "win_rate_home_dom", "draw_rate_home_dom",
    "win_rate_normal_blanc", "win_rate_normal_noir",
]
```

- [ ] **Step 2: Push v5 with top 10 only**

- [ ] **Step 3: Compare v4 (all features) vs v5 (top 10)**

If v5 > v4 → sparse features are noise. Use top 10.
If v4 > v5 → full features add value. Keep all.

- [ ] **Step 4: Document findings and commit**

---

## Task 6: ISO documentation update

- [ ] **Step 1: Update model card template for residual approach**

Add to `scripts/kaggle_artifacts.py:build_model_card()`:
- `"training_approach": "residual_learning_on_elo_baseline"`
- `"init_score_method": "softmax_inverse_of_elo_proba"`

- [ ] **Step 2: Update postmortem as living document**

- [ ] **Step 3: Update ISO_STANDARDS_REFERENCE.md Phase 1 status**

Change V8 training status from "A LANCER" to results.

- [ ] **Step 4: Commit**

```bash
git commit -m "docs(iso): update model card + standards reference for residual learning"
```
