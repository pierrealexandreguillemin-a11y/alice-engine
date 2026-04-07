# Stacking Ensemble Evaluation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate stacking meta-learner on 3 converged V8 models before finalizing the serving champion. Data-driven decision with full metrics.

**Architecture:** Load existing prediction parquets (no retraining), fit LogisticRegression/MLP meta-learners on valid set, evaluate on test set with same quality gates as V8. Script in `scripts/evaluate_stacking.py`.

**Tech Stack:** scikit-learn (StackingClassifier concepts, LogisticRegression, MLPClassifier, CalibratedClassifierCV), pandas, numpy. Reuse `scripts/kaggle_metrics.py` for RPS/ECE/E[score] MAE.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/evaluate_stacking.py` | Create | Stacking evaluation script (< 200 lines) |
| `tests/test_stacking_evaluation.py` | Create | Unit tests for stacking helpers |
| `docs/project/V8_MODEL_COMPARISON.md` | Modify (Section 11) | Update with stacking results |

## Artifact Locations

```
XGBoost v5 (champion):
  valid: reports/v8_xgboost_v5_resume/XGBoost_valid_predictions.parquet
  test:  reports/v8_xgboost_v5_resume/XGBoost_test_predictions.parquet

LightGBM v7:
  valid: C:/Users/pierr/Downloads/lightgbm-v7-output/v20260405_172602/LightGBM_valid_predictions.parquet
  test:  C:/Users/pierr/Downloads/lightgbm-v7-output/v20260405_172602/LightGBM_test_predictions.parquet

CatBoost v6:
  valid: C:/Users/pierr/Downloads/catboost-v6-output/v20260405_081528/CatBoost_valid_predictions.parquet
  test:  C:/Users/pierr/Downloads/catboost-v6-output/v20260405_081528/CatBoost_test_predictions.parquet
```

**Parquet schema (all 6 files, identical):**
```
y_true:            int64   (0=loss, 1=draw, 2=win)
y_proba_loss:      float32 (raw model probability)
y_proba_draw:      float32
y_proba_win:       float32
y_pred:            int64   (argmax of raw)
y_proba_cal_loss:  float32 (calibrated probability)
y_proba_cal_draw:  float32
y_proba_cal_win:   float32
y_pred_calibrated: int64   (argmax of calibrated)
```

**Rows:** valid=70,647, test=231,532. `y_true` aligned across all 3 models.

---

### Task 1: Stacking helper functions + tests

**Files:**
- Create: `scripts/evaluate_stacking.py`
- Create: `tests/test_stacking_evaluation.py`

- [ ] **Step 1: Write failing tests for data loading and meta-feature assembly**

Create `tests/test_stacking_evaluation.py`:

```python
"""Tests for stacking evaluation — ISO 29119.

Document ID: ALICE-TEST-STACKING
Version: 1.0.0
Tests count: 8
"""

import numpy as np
import pandas as pd
import pytest


def _make_pred_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic prediction parquet with same schema as V8 outputs."""
    rng = np.random.RandomState(seed)
    raw = rng.dirichlet([1, 1, 1], size=n).astype(np.float32)
    y_true = rng.choice([0, 1, 2], size=n, p=[0.42, 0.13, 0.45])
    return pd.DataFrame({
        "y_true": y_true.astype(np.int64),
        "y_proba_loss": raw[:, 0],
        "y_proba_draw": raw[:, 1],
        "y_proba_win": raw[:, 2],
        "y_pred": raw.argmax(axis=1).astype(np.int64),
        "y_proba_cal_loss": raw[:, 0],
        "y_proba_cal_draw": raw[:, 1],
        "y_proba_cal_win": raw[:, 2],
        "y_pred_calibrated": raw.argmax(axis=1).astype(np.int64),
    })


class TestAssembleMetaFeatures:
    """Tests for assembling 9-column meta-feature matrix."""

    def test_output_shape_9_columns(self, tmp_path: object) -> None:
        from scripts.evaluate_stacking import assemble_meta_features

        dfs = [_make_pred_df(50, seed=i) for i in range(3)]
        X, y = assemble_meta_features(dfs, calibrated=True)
        assert X.shape == (50, 9)
        assert y.shape == (50,)

    def test_uses_raw_probas_when_calibrated_false(self) -> None:
        from scripts.evaluate_stacking import assemble_meta_features

        dfs = [_make_pred_df(30, seed=i) for i in range(3)]
        X_raw, _ = assemble_meta_features(dfs, calibrated=False)
        X_cal, _ = assemble_meta_features(dfs, calibrated=True)
        assert X_raw.shape == X_cal.shape == (30, 9)

    def test_y_true_consistent_across_models(self) -> None:
        from scripts.evaluate_stacking import assemble_meta_features

        df1 = _make_pred_df(20, seed=0)
        df2 = _make_pred_df(20, seed=0)  # same y_true
        df3 = _make_pred_df(20, seed=0)
        _, y = assemble_meta_features([df1, df2, df3], calibrated=True)
        assert np.array_equal(y, df1["y_true"].values)

    def test_raises_on_mismatched_y_true(self) -> None:
        from scripts.evaluate_stacking import assemble_meta_features

        df1 = _make_pred_df(20, seed=0)
        df2 = _make_pred_df(20, seed=99)  # different y_true
        df3 = _make_pred_df(20, seed=0)
        with pytest.raises(ValueError, match="y_true mismatch"):
            assemble_meta_features([df1, df2, df3], calibrated=True)


class TestComputeAllMetrics:
    """Tests for metrics computation wrapper."""

    def test_returns_required_keys(self) -> None:
        from scripts.evaluate_stacking import compute_all_metrics

        rng = np.random.RandomState(42)
        y_true = rng.choice([0, 1, 2], size=200, p=[0.42, 0.13, 0.45])
        y_proba = rng.dirichlet([2, 1, 2], size=200)
        metrics = compute_all_metrics(y_true, y_proba)
        required = {
            "log_loss", "rps", "es_mae", "brier",
            "ece_loss", "ece_draw", "ece_win",
            "draw_calibration_bias", "accuracy", "f1_macro",
        }
        assert required.issubset(set(metrics.keys()))

    def test_perfect_predictions_low_loss(self) -> None:
        from scripts.evaluate_stacking import compute_all_metrics

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_proba = np.eye(3)[y_true] * 0.98 + 0.01 / 3
        y_proba /= y_proba.sum(axis=1, keepdims=True)
        metrics = compute_all_metrics(y_true, y_proba)
        assert metrics["log_loss"] < 0.1
        assert metrics["accuracy"] == 1.0


class TestStackingEvaluation:
    """Tests for the full stacking pipeline on synthetic data."""

    def test_lr_meta_learner_produces_valid_probas(self) -> None:
        from scripts.evaluate_stacking import fit_meta_learner

        rng = np.random.RandomState(42)
        X_train = rng.dirichlet([2, 1, 2], size=(300, 3)).reshape(300, 9)
        y_train = rng.choice([0, 1, 2], size=300)
        X_test = rng.dirichlet([2, 1, 2], size=(100, 3)).reshape(100, 9)

        meta, probas = fit_meta_learner(X_train, y_train, X_test, kind="lr")
        assert probas.shape == (100, 3)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)
        assert (probas >= 0).all()

    def test_mlp_meta_learner_produces_valid_probas(self) -> None:
        from scripts.evaluate_stacking import fit_meta_learner

        rng = np.random.RandomState(42)
        X_train = rng.dirichlet([2, 1, 2], size=(300, 3)).reshape(300, 9)
        y_train = rng.choice([0, 1, 2], size=300)
        X_test = rng.dirichlet([2, 1, 2], size=(100, 3)).reshape(100, 9)

        meta, probas = fit_meta_learner(X_train, y_train, X_test, kind="mlp")
        assert probas.shape == (100, 3)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stacking_evaluation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.evaluate_stacking'`

- [ ] **Step 3: Implement helper functions**

Create `scripts/evaluate_stacking.py`:

```python
"""Stacking ensemble evaluation — ISO 25059/24029.

Document ID: ALICE-STACKING-EVAL
Version: 1.0.0

Evaluates LogisticRegression and MLP meta-learners on 3 converged V8
models' predictions. No retraining — uses existing prediction parquets.

References:
- scikit-learn StackingClassifier concepts
- Karaaslan & Erbay (2025), MDPI Electronics 15(1):1
- Phase 2 serving design: docs/superpowers/specs/2026-04-07-phase2-serving-design.md
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.neural_network import MLPClassifier

from scripts.kaggle_metrics import (
    compute_ece,
    compute_expected_score_mae,
    compute_multiclass_brier,
    compute_rps,
)

logger = logging.getLogger(__name__)

# Column name constants matching V8 prediction parquets
_RAW_COLS = ["y_proba_loss", "y_proba_draw", "y_proba_win"]
_CAL_COLS = ["y_proba_cal_loss", "y_proba_cal_draw", "y_proba_cal_win"]


def assemble_meta_features(
    pred_dfs: list[Any],
    calibrated: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (n, 9) meta-feature matrix from 3 models' predictions.

    Args:
        pred_dfs: List of 3 DataFrames with prediction columns.
        calibrated: Use calibrated probas (True) or raw (False).

    Returns:
        (X, y): X is (n, 9) float64, y is (n,) int64.

    Raises:
        ValueError: If y_true differs across models.
    """
    cols = _CAL_COLS if calibrated else _RAW_COLS
    y_ref = pred_dfs[0]["y_true"].values

    arrays = []
    for i, df in enumerate(pred_dfs):
        if not np.array_equal(df["y_true"].values, y_ref):
            msg = f"y_true mismatch between model 0 and model {i}"
            raise ValueError(msg)
        arrays.append(df[cols].values.astype(np.float64))

    X = np.hstack(arrays)  # (n, 9)
    return X, y_ref.astype(np.int64)


def compute_all_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Compute all quality gate metrics for a probability matrix.

    Args:
        y_true: (n,) true labels (0=loss, 1=draw, 2=win).
        y_proba: (n, 3) probability matrix [P(loss), P(draw), P(win)].

    Returns:
        Dict with: log_loss, rps, es_mae, brier, ece_loss, ece_draw,
        ece_win, draw_calibration_bias, accuracy, f1_macro.
    """
    y_pred = y_proba.argmax(axis=1)
    is_draw = (y_true == 1).astype(float)

    return {
        "log_loss": log_loss(y_true, y_proba, labels=[0, 1, 2]),
        "rps": compute_rps(y_true, y_proba),
        "es_mae": compute_expected_score_mae(y_true, y_proba),
        "brier": compute_multiclass_brier(y_true, y_proba),
        "ece_loss": compute_ece(is_draw=(y_true == 0).astype(float), y_proba_class=y_proba[:, 0]),
        "ece_draw": compute_ece(is_draw, y_proba_class=y_proba[:, 1]),
        "ece_win": compute_ece((y_true == 2).astype(float), y_proba_class=y_proba[:, 2]),
        "draw_calibration_bias": float(y_proba[:, 1].mean() - is_draw.mean()),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def fit_meta_learner(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    kind: str = "lr",
) -> tuple[Any, np.ndarray]:
    """Fit a meta-learner on stacked predictions, return model + test probas.

    Args:
        X_train: (n_train, 9) meta-features from valid set.
        y_train: (n_train,) true labels.
        X_test: (n_test, 9) meta-features from test set.
        kind: "lr" for LogisticRegression, "mlp" for MLPClassifier.

    Returns:
        (model, probas): fitted model and (n_test, 3) probability matrix.
    """
    if kind == "lr":
        model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            C=1.0,
        )
    elif kind == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(16,),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        )
    else:
        msg = f"Unknown meta-learner kind: {kind}"
        raise ValueError(msg)

    model.fit(X_train, y_train)
    probas = model.predict_proba(X_test)
    return model, probas


def calibrate_probas(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Post-hoc isotonic calibration of a fitted meta-learner.

    Uses CalibratedClassifierCV with cv="prefit" on the training data,
    then predicts on test data.

    Returns:
        (n_test, 3) recalibrated probability matrix.
    """
    calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrator.fit(X_train, y_train)
    return calibrator.predict_proba(X_test)
```

- [ ] **Step 4: Fix ECE call signature**

The `compute_ece` function in `kaggle_metrics.py` takes `(y_true, y_proba_class)` where `y_true` is a binary indicator. The call in `compute_all_metrics` above uses wrong kwarg names. Fix:

```python
        "ece_loss": compute_ece((y_true == 0).astype(float), y_proba[:, 0]),
        "ece_draw": compute_ece((y_true == 1).astype(float), y_proba[:, 1]),
        "ece_win": compute_ece((y_true == 2).astype(float), y_proba[:, 2]),
```

(Remove `is_draw=` and `y_proba_class=` kwargs — `compute_ece` takes positional args.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_stacking_evaluation.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add scripts/evaluate_stacking.py tests/test_stacking_evaluation.py
git commit -m "feat(ml): add stacking evaluation helpers with tests"
```

---

### Task 2: Main evaluation script + run

**Files:**
- Modify: `scripts/evaluate_stacking.py` (add `main()`)

- [ ] **Step 1: Add main() function to evaluate_stacking.py**

Append to `scripts/evaluate_stacking.py`:

```python
def main() -> None:
    """Run full stacking evaluation on V8 prediction artifacts."""
    import json
    from pathlib import Path

    import pandas as pd

    # --- Load predictions ---
    xgb_dir = Path("reports/v8_xgboost_v5_resume")
    lgb_dir = Path("C:/Users/pierr/Downloads/lightgbm-v7-output/v20260405_172602")
    cb_dir = Path("C:/Users/pierr/Downloads/catboost-v6-output/v20260405_081528")

    splits: dict[str, list[pd.DataFrame]] = {}
    for split in ("valid", "test"):
        splits[split] = [
            pd.read_parquet(xgb_dir / f"XGBoost_{split}_predictions.parquet"),
            pd.read_parquet(lgb_dir / f"LightGBM_{split}_predictions.parquet"),
            pd.read_parquet(cb_dir / f"CatBoost_{split}_predictions.parquet"),
        ]
        logger.info("Loaded %s: %d rows", split, len(splits[split][0]))

    # --- Assemble meta-features ---
    X_valid, y_valid = assemble_meta_features(splits["valid"], calibrated=True)
    X_test, y_test = assemble_meta_features(splits["test"], calibrated=True)
    logger.info("Meta-features: valid=%s, test=%s", X_valid.shape, X_test.shape)

    # --- Baselines: individual models on test (calibrated) ---
    model_names = ["XGBoost_v5", "LightGBM_v7", "CatBoost_v6"]
    results: dict[str, dict[str, float]] = {}

    for i, name in enumerate(model_names):
        probas_test = splits["test"][i][_CAL_COLS].values.astype(np.float64)
        results[name] = compute_all_metrics(y_test, probas_test)
        logger.info("%s: log_loss=%.5f, es_mae=%.5f", name, results[name]["log_loss"], results[name]["es_mae"])

    # --- Weighted average (reproduce existing blend test) ---
    xgb_test = splits["test"][0][_CAL_COLS].values.astype(np.float64)
    lgb_test = splits["test"][1][_CAL_COLS].values.astype(np.float64)
    cb_test = splits["test"][2][_CAL_COLS].values.astype(np.float64)
    blend_90_5_5 = 0.90 * xgb_test + 0.05 * lgb_test + 0.05 * cb_test
    results["Blend_90_5_5"] = compute_all_metrics(y_test, blend_90_5_5)

    # --- Stacking: LR meta-learner ---
    lr_model, lr_probas = fit_meta_learner(X_valid, y_valid, X_test, kind="lr")
    results["Stack_LR"] = compute_all_metrics(y_test, lr_probas)
    logger.info("Stack_LR: log_loss=%.5f, es_mae=%.5f", results["Stack_LR"]["log_loss"], results["Stack_LR"]["es_mae"])

    # --- Stacking: LR + isotonic recalibration ---
    lr_cal_probas = calibrate_probas(lr_model, X_valid, y_valid, X_test)
    results["Stack_LR_cal"] = compute_all_metrics(y_test, lr_cal_probas)

    # --- Stacking: MLP meta-learner ---
    mlp_model, mlp_probas = fit_meta_learner(X_valid, y_valid, X_test, kind="mlp")
    results["Stack_MLP"] = compute_all_metrics(y_test, mlp_probas)
    logger.info("Stack_MLP: log_loss=%.5f, es_mae=%.5f", results["Stack_MLP"]["log_loss"], results["Stack_MLP"]["es_mae"])

    # --- Stacking: MLP + isotonic recalibration ---
    mlp_cal_probas = calibrate_probas(mlp_model, X_valid, y_valid, X_test)
    results["Stack_MLP_cal"] = compute_all_metrics(y_test, mlp_cal_probas)

    # --- Also test on raw (uncalibrated) base model probas ---
    X_valid_raw, _ = assemble_meta_features(splits["valid"], calibrated=False)
    X_test_raw, _ = assemble_meta_features(splits["test"], calibrated=False)
    _, lr_raw_probas = fit_meta_learner(X_valid_raw, y_valid, X_test_raw, kind="lr")
    results["Stack_LR_raw"] = compute_all_metrics(y_test, lr_raw_probas)

    # --- Print comparison table ---
    print("\n" + "=" * 100)
    print("STACKING EVALUATION RESULTS — V8 MultiClass (test set, 231,532 samples)")
    print("=" * 100)
    header = f"{'Method':<20} {'log_loss':>10} {'rps':>10} {'es_mae':>10} {'brier':>10} {'ece_draw':>10} {'draw_bias':>10} {'accuracy':>10} {'f1_macro':>10}"
    print(header)
    print("-" * 100)
    for name, m in results.items():
        row = (
            f"{name:<20} "
            f"{m['log_loss']:>10.5f} "
            f"{m['rps']:>10.5f} "
            f"{m['es_mae']:>10.5f} "
            f"{m['brier']:>10.5f} "
            f"{m['ece_draw']:>10.5f} "
            f"{m['draw_calibration_bias']:>+10.5f} "
            f"{m['accuracy']:>10.4f} "
            f"{m['f1_macro']:>10.4f}"
        )
        print(row)

    # --- Decision gate ---
    xgb_es_mae = results["XGBoost_v5"]["es_mae"]
    best_stack_name = min(
        [k for k in results if k.startswith("Stack")],
        key=lambda k: results[k]["es_mae"],
    )
    best_stack_es_mae = results[best_stack_name]["es_mae"]
    gain = xgb_es_mae - best_stack_es_mae

    print(f"\n--- DECISION GATE ---")
    print(f"XGBoost v5 E[score] MAE: {xgb_es_mae:.5f}")
    print(f"Best stacking ({best_stack_name}) E[score] MAE: {best_stack_es_mae:.5f}")
    print(f"Gain: {gain:+.5f} ({'SIGNIFICANT' if gain > 0.001 else 'NOT SIGNIFICANT'})")

    if gain > 0.001:
        print(f"\n>>> RECOMMENDATION: Use {best_stack_name} — serve 3 models + meta-learner")
    else:
        print(f"\n>>> RECOMMENDATION: Use XGBoost v5 alone — stacking adds no meaningful gain")

    # --- Save results ---
    out_path = Path("reports/stacking_evaluation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: {mk: round(mv, 6) for mk, mv in v.items()} for k, v in results.items()}
    serializable["_decision"] = {
        "xgb_es_mae": round(xgb_es_mae, 6),
        "best_stack": best_stack_name,
        "best_stack_es_mae": round(best_stack_es_mae, 6),
        "gain": round(gain, 6),
        "significant": gain > 0.001,
    }
    out_path.write_text(json.dumps(serializable, indent=2))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    main()
```

- [ ] **Step 2: Run the evaluation**

Run: `python -m scripts.evaluate_stacking`
Expected: Table of 8 methods × 8 metrics, decision gate printed.

- [ ] **Step 3: Commit results**

```bash
git add scripts/evaluate_stacking.py reports/stacking_evaluation.json
git commit -m "feat(ml): stacking evaluation — LR/MLP meta-learners on V8 predictions"
```

---

### Task 3: Update V8 comparison doc with stacking results

**Files:**
- Modify: `docs/project/V8_MODEL_COMPARISON.md` (Section 11)

- [ ] **Step 1: Read stacking results from reports/stacking_evaluation.json**

Examine the output table. The key comparison is:

| Metric | XGBoost v5 (current champion) | Best stacking | Gain |
|--------|-------------------------------|---------------|------|
| E[score] MAE | ? | ? | ? |
| log_loss | ? | ? | ? |
| draw calibration bias | ? | ? | ? |

- [ ] **Step 2: Update Section 11 of V8_MODEL_COMPARISON.md**

Replace the existing Section 11 content with the full results table,
including stacking methods. Add the decision gate conclusion.

The section should contain:
1. The full comparison table (all 8 methods)
2. The decision gate (E[score] MAE threshold > 0.001)
3. The final champion recommendation with evidence
4. Note on methodology: meta-learner trained on valid, evaluated on test

- [ ] **Step 3: Commit**

```bash
git add docs/project/V8_MODEL_COMPARISON.md
git commit -m "docs: update V8 comparison with stacking evaluation results"
```

---

### Task 4: Update spec and memory with decision

**Files:**
- Modify: `docs/superpowers/specs/2026-04-07-phase2-serving-design.md`
- Modify: `memory/project_session_resume.md`

- [ ] **Step 1: Update Phase 2 spec Section 2 with actual results**

In `docs/superpowers/specs/2026-04-07-phase2-serving-design.md`, update
Section 2.2 "Stacking Protocol" to add "Results" subsection with the
actual numbers from the evaluation.

Update Section 2.3 "Serving Implications" to mark the chosen row.

- [ ] **Step 2: Update session resume memory**

Update `memory/project_session_resume.md` to reflect the stacking
evaluation outcome and the final champion decision.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-04-07-phase2-serving-design.md
git commit -m "docs: finalize champion decision after stacking evaluation"
```
