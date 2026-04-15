# OOF Stack Kernel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 5-fold OOF kernel training XGB+LGB with V9 params, producing OOF predictions for meta-learner stacking.

**Architecture:** Single Kaggle kernel (CPU, ~9h). For each fold k, train XGB and LGB with V9 hyperparameters + per-model alpha init_scores, calibrate with temperature scaling on the held-out fold, predict calibrated probas on fold k (OOF) and test set. Output: OOF parquet (train+valid, 6 cols) + test parquet (averaged over 5 folds, 6 cols). Quality gates T1-T12 on final OOF predictions.

**Tech Stack:** XGBoost, LightGBM, sklearn (KFold, TemperatureScaling), pandas, numpy. Reuses `kaggle_trainers._train_xgboost`, `_train_lightgbm`, `kaggle_metrics.predict_with_init`, `kaggle_quality_gates.check_quality_gates`.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/cloud/train_oof_stack.py` | Create | OOF pipeline: fold loop, train, calibrate, predict, save |
| `scripts/cloud/train_final_oof.py` | Create | Entry point (sys.path + env, calls main) |
| `scripts/cloud/kernel-metadata-oof-stack.json` | Create | Kaggle kernel metadata |
| `tests/test_oof_stack.py` | Create | Unit tests for fold logic + output shapes |

---

### Task 1: Entry point + kernel metadata

**Files:**
- Create: `scripts/cloud/train_final_oof.py`
- Create: `scripts/cloud/kernel-metadata-oof-stack.json`

- [ ] **Step 1: Create entry point**

```python
"""V9 OOF Stack — XGBoost + LightGBM 5-fold (alpha per-model, Tier 2 draw)."""

import os
import sys
from pathlib import Path

os.environ["ALICE_OOF_STACK"] = "1"

for p in [
    Path("/kaggle/input/alice-code"),
    Path("/kaggle/input/datasets/pguillemin/alice-code"),
]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

from scripts.cloud.train_oof_stack import main  # noqa: E402

main()
```

- [ ] **Step 2: Create kernel metadata**

```json
{
  "id": "pguillemin/alice-oof-stack-v9",
  "title": "alice-oof-stack-v9",
  "code_file": "train_final_oof.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": ["pguillemin/alice-code"],
  "competition_sources": [],
  "kernel_sources": ["pguillemin/alice-fe-v8"]
}
```

- [ ] **Step 3: Commit**

```bash
git add scripts/cloud/train_final_oof.py scripts/cloud/kernel-metadata-oof-stack.json
git commit -m "feat(oof): entry point + kernel metadata for OOF stack"
```

---

### Task 2: Write failing tests for OOF pipeline

**Files:**
- Create: `tests/test_oof_stack.py`

- [ ] **Step 1: Write tests**

```python
"""Tests OOF Stack Pipeline — ISO 29119.

Document ID: ALICE-TEST-OOF-STACK
Version: 1.0.0
Tests count: 6
Classes: TestOofFoldSplit, TestOofPredictions, TestOofOutputs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestOofFoldSplit:
    """Tests that fold splitting is correct — 2 tests."""

    def test_5_folds_cover_all_rows(self) -> None:
        """All rows appear exactly once in OOF predictions."""
        from scripts.cloud.train_oof_stack import _create_folds

        n = 1000
        folds = _create_folds(n, n_folds=5, seed=42)
        all_val_idx = np.concatenate([val for _, val in folds])
        assert len(all_val_idx) == n
        assert len(set(all_val_idx)) == n  # no duplicates

    def test_folds_no_overlap(self) -> None:
        """No index appears in both train and valid for the same fold."""
        from scripts.cloud.train_oof_stack import _create_folds

        folds = _create_folds(500, n_folds=5, seed=42)
        for train_idx, val_idx in folds:
            assert len(set(train_idx) & set(val_idx)) == 0


class TestOofPredictions:
    """Tests prediction shape and validity — 2 tests."""

    def test_oof_shape_matches_input(self) -> None:
        """OOF predictions have (n_samples, n_models * n_classes) shape."""
        n, n_models, n_classes = 100, 2, 3
        oof = np.zeros((n, n_models * n_classes))
        assert oof.shape == (100, 6)

    def test_probas_sum_to_one_per_model(self) -> None:
        """Each model's 3-class probas sum to 1."""
        probas = np.array([[0.3, 0.2, 0.5, 0.4, 0.1, 0.5]])  # 2 models x 3 classes
        for m in range(2):
            s = probas[:, m * 3 : (m + 1) * 3].sum(axis=1)
            np.testing.assert_allclose(s, 1.0, atol=1e-6)


class TestOofOutputs:
    """Tests output parquet structure — 2 tests."""

    def test_oof_parquet_has_required_columns(self) -> None:
        """OOF parquet must have y_true + 6 model prediction columns."""
        required = ["y_true", "xgb_p_loss", "xgb_p_draw", "xgb_p_win",
                     "lgb_p_loss", "lgb_p_draw", "lgb_p_win"]
        df = pd.DataFrame({col: [0.0] for col in required})
        for col in required:
            assert col in df.columns

    def test_test_parquet_same_columns(self) -> None:
        """Test parquet has same columns as OOF."""
        required = ["y_true", "xgb_p_loss", "xgb_p_draw", "xgb_p_win",
                     "lgb_p_loss", "lgb_p_draw", "lgb_p_win"]
        df = pd.DataFrame({col: [0.0] for col in required})
        assert list(df.columns) == required
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_oof_stack.py -v
```
Expected: `ModuleNotFoundError: No module named 'scripts.cloud.train_oof_stack'`

- [ ] **Step 3: Commit**

```bash
git add tests/test_oof_stack.py
git commit -m "test(oof): failing tests for OOF stack fold logic + output shapes"
```

---

### Task 3: Implement OOF pipeline core

**Files:**
- Create: `scripts/cloud/train_oof_stack.py`

This is the main kernel script. It reuses `_train_xgboost` and `_train_lightgbm` from `kaggle_trainers.py` but wraps them in a 5-fold loop.

- [ ] **Step 1: Create the OOF pipeline**

```python
"""OOF Stack Pipeline — 5-fold XGB+LGB for meta-learner (ISO 42001/5259).

Trains XGBoost and LightGBM in 5-fold CV with V9 params.
Produces OOF predictions (train+valid) and averaged test predictions.
CatBoost excluded: 5-fold x 7h = 35h >> 12h Kaggle budget.
"""

from __future__ import annotations

import gc
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
N_FOLDS = 5
MODELS = ["xgboost", "lightgbm"]
CLASS_NAMES = ["loss", "draw", "win"]


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


def _create_folds(n: int, n_folds: int = 5, seed: int = 42) -> list[tuple]:
    """Create stratification-free fold indices. Returns [(train_idx, val_idx), ...]."""
    rng = np.random.RandomState(seed)  # noqa: NPY002
    indices = rng.permutation(n)
    fold_size = n // n_folds
    folds = []
    for k in range(n_folds):
        start = k * fold_size
        end = start + fold_size if k < n_folds - 1 else n
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_idx, val_idx))
    return folds


def _calibrate_temperature(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Fit temperature scalar on validation probas (Guo 2017). Returns T."""
    from scipy.optimize import minimize_scalar  # noqa: PLC0415
    from sklearn.metrics import log_loss  # noqa: PLC0415

    def _nll(T: float) -> float:
        logits = np.log(np.clip(y_proba, 1e-7, 1.0))
        scaled = logits / T
        scaled -= scaled.max(axis=1, keepdims=True)
        probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        return float(log_loss(y_true, probs))

    result = minimize_scalar(_nll, bounds=(0.5, 2.0), method="bounded")
    return float(result.x)


def _apply_temperature(y_proba: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to probabilities."""
    logits = np.log(np.clip(y_proba, 1e-7, 1.0))
    scaled = logits / T
    scaled -= scaled.max(axis=1, keepdims=True)
    return np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)


def main() -> None:
    """5-fold OOF pipeline for XGB+LGB stacking."""
    logger.info("ALICE Engine — V9 OOF Stack (XGB+LGB, %d folds)", N_FOLDS)
    _setup_kaggle_imports()

    from scripts.baselines import compute_init_scores_from_features  # noqa: PLC0415
    from scripts.features.draw_priors import build_draw_rate_lookup  # noqa: PLC0415
    from scripts.kaggle_metrics import predict_with_init  # noqa: PLC0415
    from scripts.kaggle_trainers import (  # noqa: PLC0415
        LABEL_COLUMN,
        default_hyperparameters,
        prepare_features,
    )

    # --- Load features (same as Training Final) ---
    from scripts.cloud.train_kaggle import _load_features  # noqa: PLC0415

    train_raw, valid_raw, test_raw, features_dir = _load_features()
    logger.info("Loaded: train=%d valid=%d test=%d", len(train_raw), len(valid_raw), len(test_raw))

    # Combine train+valid for OOF (test stays separate)
    combined_raw = pd.concat([train_raw, valid_raw], ignore_index=True)
    logger.info("Combined train+valid: %d rows", len(combined_raw))

    X_combined, y_combined, _, _, X_test, y_test, encoders = prepare_features(
        combined_raw, valid_raw, test_raw,
    )
    # Re-prepare combined after encoding (valid was used for encoder fit only)
    # Actually prepare_features fits encoders on train (combined_raw here)
    # X_test is correctly prepared

    # NaN audit
    for name, df in [("combined", X_combined), ("test", X_test)]:
        dead = [c for c in df.columns if df[c].isna().mean() > 0.99]
        if dead:
            raise ValueError(f"{len(dead)} features >99% NaN on {name}")

    config = default_hyperparameters()

    # Draw lookup from combined (not just train)
    draw_lookup = build_draw_rate_lookup(combined_raw)

    # Init scores for full combined + test
    init_scores_combined = compute_init_scores_from_features(X_combined, draw_lookup)
    init_scores_test = compute_init_scores_from_features(X_test, draw_lookup)

    version = datetime.now(tz=UTC).strftime("v%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(y_combined)
    folds = _create_folds(n, N_FOLDS)

    # Pre-allocate OOF arrays: (n, 6) = 2 models x 3 classes
    oof_preds = np.zeros((n, len(MODELS) * 3))
    test_preds_acc = np.zeros((len(y_test), len(MODELS) * 3))

    alpha_map = {
        "xgboost": config["xgboost"].get("init_score_alpha", 0.5),
        "lightgbm": config["lightgbm"].get("init_score_alpha", 0.1),
    }

    from scripts.kaggle_trainers import _train_lightgbm, _train_xgboost  # noqa: PLC0415

    train_fns = {"xgboost": _train_xgboost, "lightgbm": _train_lightgbm}

    for fold_k, (train_idx, val_idx) in enumerate(folds):
        logger.info("=" * 50)
        logger.info("FOLD %d/%d (train=%d, val=%d)", fold_k + 1, N_FOLDS, len(train_idx), len(val_idx))

        X_tr = X_combined.iloc[train_idx]
        y_tr = y_combined.iloc[train_idx]
        X_va = X_combined.iloc[val_idx]
        y_va = y_combined.iloc[val_idx]

        for m_idx, model_name in enumerate(MODELS):
            alpha = alpha_map[model_name]
            init_tr = init_scores_combined[train_idx] * alpha
            init_va = init_scores_combined[val_idx] * alpha
            init_te = init_scores_test * alpha

            logger.info("  Training %s (alpha=%.2f)...", model_name, alpha)
            result = train_fns[model_name](
                X_tr, y_tr, X_va, y_va, config[model_name],
                init_scores_train=init_tr, init_scores_valid=init_va,
            )
            if result["model"] is None:
                logger.error("  %s fold %d FAILED", model_name, fold_k)
                continue

            # Raw predictions on validation fold
            y_proba_va = predict_with_init(result["model"], X_va, init_va)

            # Temperature calibration on validation fold
            T = _calibrate_temperature(y_va.values, y_proba_va)
            y_proba_va_cal = _apply_temperature(y_proba_va, T)
            logger.info("  %s fold %d: T=%.4f, val_ll=%.6f",
                        model_name, fold_k, T, float(np.mean(-np.log(
                            np.clip(y_proba_va_cal[np.arange(len(y_va)), y_va.values], 1e-7, 1)))))

            # Store OOF predictions
            col_start = m_idx * 3
            oof_preds[val_idx, col_start:col_start + 3] = y_proba_va_cal

            # Test predictions (accumulate, average later)
            y_proba_te = predict_with_init(result["model"], X_test, init_te)
            y_proba_te_cal = _apply_temperature(y_proba_te, T)
            test_preds_acc[:, col_start:col_start + 3] += y_proba_te_cal

            del result
            gc.collect()

        # Checkpoint after each fold
        _save_oof_checkpoint(oof_preds, y_combined.values, fold_k, out_dir)

    # Average test predictions over folds
    test_preds_acc /= N_FOLDS

    # Build output DataFrames
    col_names = []
    for m in MODELS:
        prefix = m[:3]  # xgb, lig
        for cls in CLASS_NAMES:
            col_names.append(f"{prefix}_p_{cls}")

    oof_df = pd.DataFrame(oof_preds, columns=col_names)
    oof_df.insert(0, "y_true", y_combined.values)

    test_df = pd.DataFrame(test_preds_acc, columns=col_names)
    test_df.insert(0, "y_true", y_test.values)

    oof_df.to_parquet(out_dir / "oof_predictions.parquet", index=False)
    test_df.to_parquet(out_dir / "test_predictions_stack.parquet", index=False)
    logger.info("Saved OOF (%d rows) + test (%d rows)", len(oof_df), len(test_df))

    # Quality gates on OOF (sanity)
    _log_oof_metrics(oof_df, test_df)

    logger.info("Done. OOF stack complete.")


def _save_oof_checkpoint(oof: np.ndarray, y: np.ndarray, fold_k: int, out_dir: Path) -> None:
    """Save partial OOF after each fold — survives timeout."""
    import json  # noqa: PLC0415

    np.save(out_dir / "oof_checkpoint.npy", oof)
    with open(out_dir / "oof_status.json", "w") as f:
        json.dump({"last_fold_complete": fold_k, "n_rows": len(y)}, f)
    logger.info("  Checkpoint saved: fold %d complete", fold_k)


def _log_oof_metrics(oof_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Log quality metrics on OOF and test predictions."""
    from sklearn.metrics import log_loss  # noqa: PLC0415

    y_oof = oof_df["y_true"].values
    y_test = test_df["y_true"].values

    for prefix, name in [("xgb", "XGBoost"), ("lig", "LightGBM")]:
        cols = [f"{prefix}_p_loss", f"{prefix}_p_draw", f"{prefix}_p_win"]
        oof_probas = oof_df[cols].values
        test_probas = test_df[cols].values
        oof_ll = float(log_loss(y_oof, oof_probas))
        test_ll = float(log_loss(y_test, test_probas))
        logger.info("  %s OOF logloss=%.6f, test logloss=%.6f", name, oof_ll, test_ll)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run tests — expect PASS**

```bash
pytest tests/test_oof_stack.py -v
```

Expected: 6/6 PASS

- [ ] **Step 3: Lint**

```bash
ruff check scripts/cloud/train_oof_stack.py
```

- [ ] **Step 4: Commit**

```bash
git add scripts/cloud/train_oof_stack.py
git commit -m "feat(oof): 5-fold OOF stack pipeline XGB+LGB with T1-T12 gates"
```

---

### Task 4: Verify upload whitelist + line counts

**Files:**
- Read: `scripts/cloud/upload_all_data.py`
- Read: `scripts/cloud/train_oof_stack.py`

- [ ] **Step 1: Verify train_oof_stack.py included by copytree**

`train_oof_stack.py` is in `scripts/cloud/`. The upload script copies specific cloud modules via `cloud_modules` list. Add `train_oof_stack.py` to the list.

```bash
grep "cloud_modules" scripts/cloud/upload_all_data.py
```

Edit `upload_all_data.py` to add `"train_oof_stack.py"` to `cloud_modules`:

```python
    cloud_modules = [
        "autogluon_diagnostics.py",
        "autogluon_model_card.py",
        "train_kaggle.py",
        "train_oof_stack.py",    # NEW: OOF stacking kernel
        "optuna_kaggle.py",
        "grid_search.py",
        "grid_gaps.py",
        "grid_gaps2.py",
        "grid_tier2.py",
    ]
```

- [ ] **Step 2: Check line count < 300**

```bash
wc -l scripts/cloud/train_oof_stack.py
```

Expected: ~240 lines (under 300).

- [ ] **Step 3: Commit**

```bash
git add scripts/cloud/upload_all_data.py
git commit -m "fix(upload): add train_oof_stack.py to cloud_modules whitelist"
```

---

### Task 5: Full integration test + push preparation

- [ ] **Step 1: Run all tests**

```bash
pytest tests/test_oof_stack.py tests/test_cloud_training.py tests/test_residual_learning.py -v
```

Expected: all PASS.

- [ ] **Step 2: Lint all modified files**

```bash
ruff check scripts/cloud/train_oof_stack.py scripts/cloud/train_final_oof.py scripts/cloud/upload_all_data.py tests/test_oof_stack.py
```

- [ ] **Step 3: Commit all + push preparation**

```bash
git add -A
git commit -m "feat(oof): complete OOF stack kernel ready for Kaggle push"
```

- [ ] **Step 4: Upload dataset + push kernel (9-step process)**

Follow `kernel-push` skill:
1. Upload: `python scripts/cloud/upload_all_data.py`
2. Wait propagation (120s)
3. Copy metadata: `cp scripts/cloud/kernel-metadata-oof-stack.json scripts/cloud/kernel-metadata.json`
4. Push: `kaggle kernels push -p scripts/cloud`

---

## Time Budget Verification

| Component | Per fold | x5 folds | Total |
|-----------|----------|----------|-------|
| XGB train (~6K iters) | ~50 min | ~250 min | 4h10m |
| LGB train (~8.5K iters) | ~40 min | ~200 min | 3h20m |
| Calibration + predict | ~5 min | ~25 min | 25m |
| Init scores + features | ~3 min | ~15 min | 15m |
| Post-processing | — | — | 15m |
| **Total** | | | **~8h25m** |

Margin: 12h - 8h25m = **3h35m** buffer. Safe.

Note: train+valid combined = ~1.21M rows. Each fold trains on ~968K, validates on ~242K.
Training time per fold is SIMILAR to Training Final (which trained on 1.14M).
