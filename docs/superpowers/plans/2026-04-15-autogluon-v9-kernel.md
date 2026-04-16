# AutoGluon V9 Benchmark Kernel — Implementation Plan

> **⚠ ELIMINE — ADR-011 (2026-04-16)** : AutoGluon elimine du pipeline ALICE.
> Pas de residual learning, calibration incompatible CE, test logloss 0.5716 > V9 LGB 0.5619.
> Voir `docs/architecture/DECISIONS.md` §ADR-011 et `docs/postmortem/2026-04-16-autogluon-v9-time-allocation-failure.md`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run AutoGluon 1.5 `best_quality` on V9 features (204 cols = 201 + 3 Elo probas) as benchmark against V9 single models (LGB 0.5619, XGB 0.5622).

**Architecture:** Single Kaggle kernel (CPU, 10h). AG trains multiclass (W/D/L) with `log_loss`, `calibrate=True`, 5-fold bagging, 1 stack level. Outputs: test predictions, leaderboard, feature importance. Quality gates T1-T12 on AG best model and AG stacked ensemble.

**Tech Stack:** AutoGluon >= 1.5 (pip install at runtime), pandas, numpy. Reuses `_load_features` from `train_kaggle.py`, `prepare_features` from `kaggle_trainers.py`, `check_quality_gates` from `kaggle_quality_gates.py`.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/cloud/train_autogluon_v9.py` | Create | AG pipeline: fit, evaluate, gates, save |
| `scripts/cloud/train_final_autogluon.py` | Create | Entry point (sys.path + pip install + main) |
| `scripts/cloud/kernel-metadata-autogluon-v9.json` | Create | Kaggle kernel metadata |
| `tests/test_autogluon_v9.py` | Create | Unit tests for Elo proba features + output structure |

---

### Task 1: Entry point + kernel metadata

**Files:**
- Create: `scripts/cloud/train_final_autogluon.py`
- Create: `scripts/cloud/kernel-metadata-autogluon-v9.json`

- [ ] **Step 1: Create entry point**

```python
"""V9 AutoGluon Benchmark — best_quality, multiclass, 204 features."""

import os
import subprocess
import sys
from pathlib import Path

# Install AutoGluon (not pre-installed on Kaggle)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "autogluon>=1.5"],
    stdout=subprocess.DEVNULL,
)

for p in [
    Path("/kaggle/input/alice-code"),
    Path("/kaggle/input/datasets/pguillemin/alice-code"),
]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

from scripts.cloud.train_autogluon_v9 import main  # noqa: E402

main()
```

- [ ] **Step 2: Create kernel metadata**

```json
{
  "id": "pguillemin/alice-autogluon-v9",
  "title": "alice-autogluon-v9",
  "code_file": "train_final_autogluon.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": ["pguillemin/alice-code"],
  "competition_sources": [],
  "kernel_sources": ["pguillemin/alice-fe-v8"]
}
```

Note: `enable_internet: true` for `pip install autogluon`.

- [ ] **Step 3: Commit**

```bash
git add scripts/cloud/train_final_autogluon.py scripts/cloud/kernel-metadata-autogluon-v9.json
git commit -m "feat(ag): entry point + kernel metadata for AutoGluon V9 benchmark"
```

---

### Task 2: Write failing tests

**Files:**
- Create: `tests/test_autogluon_v9.py`

- [ ] **Step 1: Write tests**

```python
"""Tests AutoGluon V9 Benchmark — ISO 29119.

Document ID: ALICE-TEST-AG-V9
Version: 1.0.0
Tests count: 4
Classes: TestEloProbaFeatures, TestAgOutputs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestEloProbaFeatures:
    """Tests Elo proba feature computation — 2 tests."""

    def test_elo_probas_sum_to_one(self) -> None:
        """P_elo(W) + P_elo(D) + P_elo(L) must sum to 1."""
        from scripts.cloud.train_autogluon_v9 import _compute_elo_proba_features

        df = pd.DataFrame({"blanc_elo": [1500, 1800, 2000], "noir_elo": [1500, 1200, 1900]})
        result = _compute_elo_proba_features(df)
        sums = result[["p_elo_win", "p_elo_draw", "p_elo_loss"]].sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_elo_probas_range_valid(self) -> None:
        """All Elo probas in [0, 1]."""
        from scripts.cloud.train_autogluon_v9 import _compute_elo_proba_features

        df = pd.DataFrame({"blanc_elo": [800, 1500, 2800], "noir_elo": [2800, 1500, 800]})
        result = _compute_elo_proba_features(df)
        for col in ["p_elo_win", "p_elo_draw", "p_elo_loss"]:
            assert (result[col] >= 0).all()
            assert (result[col] <= 1).all()


class TestAgOutputs:
    """Tests output structure — 2 tests."""

    def test_predictions_parquet_columns(self) -> None:
        """Predictions parquet must have y_true + 3 proba columns."""
        required = ["y_true", "p_loss", "p_draw", "p_win"]
        df = pd.DataFrame({col: [0.0] for col in required})
        for col in required:
            assert col in df.columns

    def test_leaderboard_has_score_column(self) -> None:
        """Leaderboard CSV must have model + score_val columns."""
        df = pd.DataFrame({"model": ["LGB"], "score_val": [-0.56]})
        assert "model" in df.columns
        assert "score_val" in df.columns
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_autogluon_v9.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Commit**

```bash
git add tests/test_autogluon_v9.py
git commit -m "test(ag): failing tests for AutoGluon V9 Elo probas + output structure"
```

---

### Task 3: Implement AutoGluon V9 pipeline

**Files:**
- Create: `scripts/cloud/train_autogluon_v9.py`

- [ ] **Step 1: Create the AG pipeline**

```python
"""AutoGluon V9 Benchmark — best_quality multiclass (ISO 42001/5259).

Trains AutoGluon on V9 features + 3 Elo proba features. No init_scores
(AG doesn't support them). Benchmark against V9 single models.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))
AG_TIME_LIMIT = 36000  # 10h (2h margin for post-processing)
AG_PRESETS = "best_quality"
AG_BAG_FOLDS = 5
AG_STACK_LEVELS = 1  # V8 postmortem: L2/L3 overfit


def _setup_kaggle_imports() -> None:
    """Find alice-code dataset and add to sys.path."""
    candidates = [
        Path("/kaggle/input/alice-code"),
        Path("/kaggle/input/datasets/pguillemin/alice-code"),
    ]
    kaggle_input = next((c for c in candidates if c.exists()), None)
    if kaggle_input:
        sys.path.insert(0, str(kaggle_input))
        logger.info("sys.path += %s", kaggle_input)


def _compute_elo_proba_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute P_elo(W/D/L) from blanc_elo/noir_elo as explicit features.

    Gives AutoGluon the Elo baseline signal that V9 models get via init_scores.
    Uses the same logistic model as scripts/baselines.py + draw_rate_lookup.
    """
    blanc_elo = df["blanc_elo"].fillna(1500).values
    noir_elo = df["noir_elo"].fillna(1500).values
    # White advantage: dynamic +20 Elo (simplified, full lookup in production)
    blanc_adj = blanc_elo + 20.0
    expected_w = 1.0 / (1.0 + 10.0 ** ((noir_elo - blanc_adj) / 400.0))
    expected_l = 1.0 / (1.0 + 10.0 ** ((blanc_adj - noir_elo) / 400.0))
    # Draw = 1 - W - L (simplified; full version uses draw_rate_lookup.parquet)
    p_draw = np.clip(1.0 - expected_w - expected_l, 0.01, 0.99)
    # Renormalize
    total = expected_w + p_draw + expected_l
    result = pd.DataFrame({
        "p_elo_win": expected_w / total,
        "p_elo_draw": p_draw / total,
        "p_elo_loss": expected_l / total,
    }, index=df.index)
    return result


def main() -> None:
    """AutoGluon V9 benchmark pipeline."""
    logger.info("ALICE Engine — AutoGluon V9 Benchmark")
    _setup_kaggle_imports()

    from autogluon.tabular import TabularPredictor  # noqa: PLC0415

    from scripts.kaggle_trainers import LABEL_COLUMN, prepare_features  # noqa: PLC0415

    # Load features (same as V9 Training Final)
    from scripts.cloud.train_kaggle import _load_features  # noqa: PLC0415

    train_raw, valid_raw, test_raw, features_dir = _load_features()
    logger.info("Loaded: train=%d valid=%d test=%d", len(train_raw), len(valid_raw), len(test_raw))

    # Prepare features (encoding, target mapping)
    X_train, y_train, X_valid, y_valid, X_test, y_test, encoders = prepare_features(
        train_raw, valid_raw, test_raw,
    )

    # NaN audit
    for name, df in [("train", X_train), ("valid", X_valid), ("test", X_test)]:
        dead = [c for c in df.columns if df[c].isna().mean() > 0.99]
        if dead:
            raise ValueError(f"{len(dead)} features >99% NaN on {name}")

    # Add 3 Elo proba features (compensates for missing init_scores)
    for X, raw in [(X_train, train_raw), (X_valid, valid_raw), (X_test, test_raw)]:
        elo_feats = _compute_elo_proba_features(raw)
        for col in elo_feats.columns:
            X[col] = elo_feats[col].values

    logger.info("Features: %d (201 V9 + 3 Elo probas)", X_train.shape[1])

    # Build AG training DataFrame (AG needs label column in the DataFrame)
    train_ag = X_train.copy()
    train_ag["target"] = y_train.values
    valid_ag = X_valid.copy()
    valid_ag["target"] = y_valid.values

    version = datetime.now(tz=UTC).strftime("v%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- FIT ---
    predictor = TabularPredictor(
        label="target",
        eval_metric="log_loss",
        problem_type="multiclass",
        path=str(out_dir / "ag_models"),
    )
    predictor.fit(
        train_data=train_ag,
        tuning_data=valid_ag,
        presets=AG_PRESETS,
        time_limit=AG_TIME_LIMIT,
        num_bag_folds=AG_BAG_FOLDS,
        num_stack_levels=AG_STACK_LEVELS,
        calibrate=True,
        num_gpus=0,
        ag_args_fit={"ag.max_memory_usage_ratio": 1.5},
        verbosity=2,
    )

    # --- LEADERBOARD ---
    leaderboard = predictor.leaderboard(test_raw if "target" in test_raw else None, silent=True)
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    logger.info("Leaderboard:\n%s", leaderboard.to_string())

    # --- PREDICTIONS ---
    # Best model (stacked ensemble)
    probas_test_best = predictor.predict_proba(X_test)
    probas_valid_best = predictor.predict_proba(X_valid)

    # Best single model
    info = predictor.info()
    best_single = None
    for m in leaderboard["model"].values:
        if "Ensemble" not in m and "Stack" not in m:
            best_single = m
            break
    if best_single:
        probas_test_single = predictor.predict_proba(X_test, model=best_single)
        logger.info("Best single model: %s", best_single)

    # Save predictions
    _save_predictions(probas_test_best, y_test, out_dir / "predictions_test_ensemble.parquet")
    _save_predictions(probas_valid_best, y_valid, out_dir / "predictions_valid_ensemble.parquet")
    if best_single:
        _save_predictions(probas_test_single, y_test, out_dir / "predictions_test_single.parquet")

    # --- QUALITY GATES T1-T12 ---
    from scripts.kaggle_quality_gates import check_quality_gates  # noqa: PLC0415
    from scripts.kaggle_metrics import (  # noqa: PLC0415
        compute_ece,
        compute_expected_score_mae,
        compute_multiclass_brier,
        compute_rps,
    )
    from sklearn.metrics import log_loss  # noqa: PLC0415

    y_arr = y_test.values
    probas_arr = probas_test_best.values if hasattr(probas_test_best, "values") else np.asarray(probas_test_best)
    # Ensure column order is [loss, draw, win] = [0, 1, 2]
    if hasattr(probas_test_best, "columns"):
        probas_arr = probas_test_best[[0, 1, 2]].values

    test_ll = float(log_loss(y_arr, probas_arr))
    test_rps = float(compute_rps(y_arr, probas_arr))
    test_brier = float(compute_multiclass_brier(y_arr, probas_arr))
    test_es_mae = float(compute_expected_score_mae(y_arr, probas_arr))
    mean_p_draw = float(probas_arr[:, 1].mean())
    observed_draw = float((y_arr == 1).mean())
    draw_bias = mean_p_draw - observed_draw

    logger.info("AG ENSEMBLE: ll=%.6f rps=%.6f es_mae=%.6f draw_bias=%.6f",
                test_ll, test_rps, test_es_mae, draw_bias)

    for c, cls in enumerate(["loss", "draw", "win"]):
        ece = float(compute_ece((y_arr == c).astype(float), probas_arr[:, c]))
        logger.info("  ECE %s: %.4f", cls, ece)

    # --- METADATA ---
    metadata = {
        "version": version,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "pipeline": "AutoGluon V9 benchmark",
        "preset": AG_PRESETS,
        "features": int(X_train.shape[1]),
        "train_rows": len(train_ag),
        "test_rows": len(X_test),
        "time_limit": AG_TIME_LIMIT,
        "num_bag_folds": AG_BAG_FOLDS,
        "num_stack_levels": AG_STACK_LEVELS,
        "best_model_ensemble": predictor.model_best,
        "best_model_single": best_single,
        "metrics_ensemble": {
            "test_log_loss": test_ll,
            "test_rps": test_rps,
            "test_brier": test_brier,
            "test_es_mae": test_es_mae,
            "mean_p_draw": mean_p_draw,
            "draw_calibration_bias": round(draw_bias, 6),
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Feature importance
    try:
        imp = predictor.feature_importance(valid_ag, silent=True)
        imp.to_csv(out_dir / "feature_importance.csv")
    except Exception:
        logger.warning("Feature importance failed (non-blocking)")

    logger.info("Done. AG V9 benchmark complete.")


def _save_predictions(probas: pd.DataFrame, y: pd.Series, path: Path) -> None:
    """Save predictions parquet with y_true + 3 proba columns."""
    df = pd.DataFrame({
        "y_true": y.values,
        "p_loss": probas.iloc[:, 0].values if hasattr(probas, "iloc") else probas[:, 0],
        "p_draw": probas.iloc[:, 1].values if hasattr(probas, "iloc") else probas[:, 1],
        "p_win": probas.iloc[:, 2].values if hasattr(probas, "iloc") else probas[:, 2],
    })
    df.to_parquet(path, index=False)
    logger.info("Saved predictions: %s (%d rows)", path.name, len(df))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run tests — expect PASS**

```bash
pytest tests/test_autogluon_v9.py -v
```

- [ ] **Step 3: Lint**

```bash
ruff check scripts/cloud/train_autogluon_v9.py
```

- [ ] **Step 4: Commit**

```bash
git add scripts/cloud/train_autogluon_v9.py
git commit -m "feat(ag): AutoGluon V9 benchmark pipeline with Elo proba features"
```

---

### Task 4: Upload whitelist + integration

**Files:**
- Modify: `scripts/cloud/upload_all_data.py`

- [ ] **Step 1: Add train_autogluon_v9.py to cloud_modules whitelist**

Same edit as OOF plan — add to `cloud_modules` list:

```python
    cloud_modules = [
        "autogluon_diagnostics.py",
        "autogluon_model_card.py",
        "train_kaggle.py",
        "train_oof_stack.py",
        "train_autogluon_v9.py",  # NEW: AG V9 benchmark
        "optuna_kaggle.py",
        "grid_search.py",
        "grid_gaps.py",
        "grid_gaps2.py",
        "grid_tier2.py",
    ]
```

- [ ] **Step 2: Run all tests**

```bash
pytest tests/test_autogluon_v9.py tests/test_oof_stack.py tests/test_cloud_training.py -v
```

- [ ] **Step 3: Commit**

```bash
git add scripts/cloud/upload_all_data.py
git commit -m "fix(upload): add train_autogluon_v9.py + train_oof_stack.py to cloud_modules"
```

- [ ] **Step 4: Upload dataset + push kernel (9-step process)**

Follow `kernel-push` skill:
1. Upload: `python scripts/cloud/upload_all_data.py`
2. Wait propagation (120s)
3. Copy metadata: `cp scripts/cloud/kernel-metadata-autogluon-v9.json scripts/cloud/kernel-metadata.json`
4. Push: `kaggle kernels push -p scripts/cloud`

---

## Time Budget Verification

| Component | Estimate | Notes |
|-----------|----------|-------|
| pip install autogluon | ~3 min | 900 MB deps |
| Feature loading + prep | ~5 min | 1.1M train + 232K test |
| Elo proba computation | ~10 sec | Vectorized numpy |
| AG fit (best_quality) | **~8-9h** | 1.21M rows, 204 features, 5-fold bag, 1 stack |
| AG predict (test) | ~5 min | Stacked ensemble |
| Quality gates + save | ~10 min | |
| **Total** | **~9-10h** | 2h margin on 12h |

AG fit is the bottleneck. `time_limit=36000` (10h) controls this.
AG will train as many models as it can within budget.
