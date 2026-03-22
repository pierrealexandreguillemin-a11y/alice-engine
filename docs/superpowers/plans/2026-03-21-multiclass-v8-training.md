# V8 MultiClass — Training + Diagnostics Plan (Plan B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch training pipeline from binary classification to MultiClass 3-way (win/draw/loss), add calibration per-class, implement baselines (naive + Elo), add RPS/ECE/E[score] metrics, update quality gate — producing CANDIDATE models with calibrated P(win), P(draw), P(loss).

**Architecture:** `kaggle_trainers.py` handles target encoding + model configs. `kaggle_diagnostics.py` handles calibration + metrics. `train_kaggle.py` orchestrates baselines + quality gate. Config in `hyperparameters.yaml`.

**Tech Stack:** CatBoost (MultiClass), XGBoost (multi:softprob), LightGBM (multiclass), scikit-learn (IsotonicRegression, calibration metrics), numpy.

**Spec:** `docs/superpowers/specs/2026-03-21-multiclass-v8-design.md` — sections "Models", "Calibration", "Evaluation Metrics".

**Depends on:** Plan A completed (V8 feature parquets with ~156 columns, forfaits excluded).

**Followed by:** Kaggle push (new slug `alice-training-v8`, T4x2, upload_all_data first).

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| MODIFY | `scripts/kaggle_trainers.py` | Target 3-class, MultiClass configs, predict_proba 3-col |
| MODIFY | `scripts/kaggle_diagnostics.py` | Calibration 3-class, RPS, ECE, reliability diagrams |
| MODIFY | `scripts/cloud/train_kaggle.py` | Baselines, system metrics, quality gate |
| MODIFY | `scripts/cloud/train_autogluon_kaggle.py` | MultiClass target + METADATA_COLS |
| MODIFY | `config/hyperparameters.yaml` | MultiClass params, quality gate thresholds |
| MODIFY | `tests/test_cloud_features.py` | Adapt metrics tests |
| MODIFY | `tests/test_cloud_training.py` | Adapt quality gate + model card tests |
| NEW | `scripts/baselines.py` | Naive + Elo draw rate baselines |

---

## Task 1: Target encoding — binary to 3-class

**Files:**
- Modify: `scripts/kaggle_trainers.py:_split_xy()` (line 64-72)
- Modify: `scripts/kaggle_trainers.py:LABEL_COLUMN` area
- Test: `tests/test_cloud_features.py`

- [ ] **Step 1: Write failing test**

```python
# In tests/test_cloud_features.py or new file
def test_target_3class_encoding() -> None:
    """Target must be 3-class: loss=0, draw=1, win=2. Forfaits excluded."""
    import pandas as pd
    from scripts.kaggle_trainers import _split_xy

    df = pd.DataFrame({
        "resultat_blanc": [0.0, 0.5, 1.0, 2.0, 0.0, 1.0],
        "type_resultat": ["victoire_noir", "nulle", "victoire_blanc",
                          "victoire_blanc", "victoire_noir", "victoire_blanc"],
        # Need numeric columns for select_dtypes
        "blanc_elo": [1500, 1600, 1400, 1500, 1700, 1800],
        "noir_elo": [1400, 1500, 1600, 1400, 1500, 1600],
    })
    X, y = _split_xy(df.copy())
    # Forfaits (2.0) should be excluded
    assert len(y) == 5  # 6 - 1 forfait
    assert 2.0 not in df.loc[y.index, "resultat_blanc"].values
    # 3-class encoding: loss=0, draw=1, win=2
    assert set(y.unique()) <= {0, 1, 2}
    assert y[df.loc[y.index, "resultat_blanc"] == 0.0].iloc[0] == 0  # loss
    assert y[df.loc[y.index, "resultat_blanc"] == 0.5].iloc[0] == 1  # draw
    assert y[df.loc[y.index, "resultat_blanc"] == 1.0].iloc[0] == 2  # win
```

- [ ] **Step 2: Run — should fail** (current code: `y = (df[LABEL_COLUMN] == 1.0).astype(int)`)

- [ ] **Step 3: Implement 3-class target encoding**

Replace in `_split_xy()`:
```python
# OLD (binary):
y = (df[LABEL_COLUMN] == 1.0).astype(int)

# NEW (3-class, forfaits excluded):
from scripts.features.helpers import FORFAIT_RESULT
df = df[df[LABEL_COLUMN] != FORFAIT_RESULT].copy()
TARGET_MAP = {0.0: 0, 0.5: 1, 1.0: 2}
y = df[LABEL_COLUMN].map(TARGET_MAP).astype(int)
```

- [ ] **Step 4: Run test — should pass**
- [ ] **Step 5: Verify line count** `wc -l scripts/kaggle_trainers.py` (must be <300)
- [ ] **Step 6: Commit**

```bash
git commit -m "feat(training): 3-class target encoding (loss=0, draw=1, win=2), exclude forfaits"
```

---

## Task 1.5: Update categorical feature lists for Plan A column renames

**Files:**
- Modify: `scripts/kaggle_trainers.py` lines 15-28 (CATEGORICAL_FEATURES, CATBOOST_CAT_FEATURES, ADVANCED_CAT_FEATURES)

**CRITICAL: Plan A renames columns. These lists MUST match.**

- [ ] **Step 1: Update ADVANCED_CAT_FEATURES**

```python
# OLD entries to REMOVE:
"forme_tendance_blanc", "forme_tendance_noir"  # → replaced by win_trend_*/draw_trend_*

# NEW entries to ADD:
"win_trend_blanc", "win_trend_noir", "draw_trend_blanc", "draw_trend_noir",
"regularite_blanc", "regularite_noir",  # wired from ali/presence
"role_type_blanc", "role_type_noir",    # wired from ali/patterns
```

- [ ] **Step 2: Verify all categorical columns exist in sample parquet**

```python
import pandas as pd
df = pd.read_parquet("data/features/train.parquet")
from scripts.kaggle_trainers import CATEGORICAL_FEATURES, ADVANCED_CAT_FEATURES, CATBOOST_CAT_FEATURES
all_cats = set(CATEGORICAL_FEATURES) | set(ADVANCED_CAT_FEATURES) | set(CATBOOST_CAT_FEATURES)
missing = [c for c in all_cats if c not in df.columns]
assert not missing, f"Missing categoricals: {missing}"
```

- [ ] **Step 3: Commit**

```bash
git commit -m "fix(training): update categorical feature lists for Plan A column renames"
```

---

## Task 2: MultiClass model configs (in default_hyperparameters + train functions)

**Files:**
- Modify: `scripts/kaggle_trainers.py:default_hyperparameters()` (line ~116)
- Modify: `scripts/kaggle_trainers.py:_train_catboost()` (line ~175)
- Modify: `scripts/kaggle_trainers.py:_train_xgboost()` (line ~192)
- Modify: `scripts/kaggle_trainers.py:_train_lightgbm()` (line ~210)

**NOTE: Tasks 1-6 form an ATOMIC BATCH.** The codebase is currently in "V7.5" state (hyperparams updated but target/objective still binary). Do NOT run training between Task 1 and Task 6 — the code will be inconsistent.

- [ ] **Step 1: Write test**

```python
def test_default_hyperparameters_multiclass() -> None:
    """default_hyperparameters must include MultiClass objectives."""
    from scripts.kaggle_trainers import default_hyperparameters
    config = default_hyperparameters()
    assert config["catboost"]["loss_function"] == "MultiClass"
    assert config["xgboost"]["objective"] == "multi:softprob"
    assert config["xgboost"]["num_class"] == 3
    assert config["lightgbm"]["objective"] == "multiclass"
    assert config["lightgbm"]["num_class"] == 3
```

- [ ] **Step 2: Update default_hyperparameters()**

```python
# In default_hyperparameters(), add to each model dict:
"catboost": {
    "loss_function": "MultiClass",  # NEW — was implicit Logloss
    "iterations": 3000, "depth": 8, "border_count": 254,
    ...
},
"xgboost": {
    "objective": "multi:softprob",  # NEW — was implicit binary:logistic
    "num_class": 3,                 # NEW
    "n_estimators": 3000, "max_depth": 8,
    ...
},
"lightgbm": {
    "objective": "multiclass",      # NEW — was implicit binary
    "num_class": 3,                 # NEW
    "n_estimators": 3000, "num_leaves": 255,
    ...
},
```

- [ ] **Step 3: Update _train_catboost()**

```python
# Remove hardcoded eval_metric — it's now in params via loss_function
cb = CatBoostClassifier(**params, eval_metric="MultiClass")
```

- [ ] **Step 4: Update _train_xgboost()**

```python
# objective + num_class now come from params, just set eval_metric
xgb = XGBClassifier(**params, eval_metric="mlogloss")
```

- [ ] **Step 5: Update _train_lightgbm()**

```python
# objective + num_class now come from params
# Change eval_metric in callbacks:
eval_metric="multi_logloss"
```

- [ ] **Step 6: Update hyperparameters.yaml**

```yaml
# global section: update eval_metric from binary to multiclass
global:
  eval_metric: "multi_logloss"  # WAS "logloss" (binary). Now multiclass.

# catboost section: add loss_function
catboost:
  loss_function: "MultiClass"   # NEW — softmax 3-class

# xgboost section: add objective + num_class
xgboost:
  objective: "multi:softprob"   # NEW — softmax probabilities
  num_class: 3                  # NEW — loss/draw/win

# lightgbm section: add objective + num_class
lightgbm:
  objective: "multiclass"       # NEW — softmax 3-class
  num_class: 3                  # NEW — loss/draw/win
```

Also update `default_hyperparameters()` global section:
```python
"global": {"random_seed": 42, "early_stopping_rounds": 100, "eval_metric": "multi_logloss"},
```

**TestHyperparamsSync**: will pass because new YAML keys match new code keys. No skip_keys needed.

- [ ] **Step 7: Run TestHyperparamsSync to verify**

```bash
pytest tests/test_cloud_training.py::TestHyperparamsSync -v
```

- [ ] **Step 8: Commit**

```bash
git commit -m "feat(training): MultiClass configs in default_hyperparameters + train functions"
```

---

## Task 3: predict_proba 3 columns + eval functions

**Files:**
- Modify: `scripts/kaggle_trainers.py:_eval_model()` (line ~147)
- Modify: `scripts/kaggle_trainers.py:evaluate_on_test()` (line ~256)

The current code takes `predict_proba(X)[:, 1]` (binary). Must change to 3-column output.

- [ ] **Step 1: Write test**

```python
def test_eval_model_returns_3_probas() -> None:
    """_eval_model must return probas for 3 classes."""
    # Mock model with predict_proba returning (n, 3)
    ...
```

- [ ] **Step 2: Refactor _eval_model()**

```python
def _eval_model(model, X_valid, y_valid, train_time):
    import numpy as np
    y_proba = model.predict_proba(X_valid)  # (n, 3) matrix
    y_pred = np.argmax(y_proba, axis=1)
    metrics = compute_validation_metrics(y_valid.values, y_pred, y_proba)
    metrics["train_time_s"] = train_time
    # Feature importance unchanged
    importance = (
        dict(zip(X_valid.columns, model.feature_importances_, strict=False))
        if hasattr(model, "feature_importances_")
        else {}
    )
    return {"model": model, "metrics": metrics, "importance": importance}
```

- [ ] **Step 3: Refactor evaluate_on_test()**

```python
def evaluate_on_test(results, X_test, y_test):
    import numpy as np
    for _name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = r["model"].predict_proba(X_test)  # (n, 3)
        y_pred = np.argmax(y_proba, axis=1)
        # Compute all multiclass metrics
        r["metrics"]["test_log_loss"] = float(log_loss(y_test, y_proba))
        r["metrics"]["test_rps"] = float(_compute_rps(y_test, y_proba))
        r["metrics"]["test_expected_score_mae"] = float(
            _compute_expected_score_mae(y_test, y_proba)
        )
```

- [ ] **Step 4: Run tests — pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(training): predict_proba 3-col + multiclass eval metrics"
```

---

## Task 4: Baselines — naive + Elo draw rate model

**Files:**
- Create: `scripts/baselines.py`
- Test: `tests/test_baselines.py`

- [ ] **Step 1: Write tests**

```python
def test_naive_baseline_probas_sum_to_1() -> None:
    """Naive baseline must predict marginal distribution."""
    from scripts.baselines import compute_naive_baseline
    import numpy as np
    y_train = np.array([0, 0, 1, 2, 2, 0, 1, 2, 2, 0])
    probas = compute_naive_baseline(y_train, n_test=5)
    assert probas.shape == (5, 3)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0)

def test_elo_baseline_draw_rate_varies() -> None:
    """Elo baseline must predict different draw rates by Elo level."""
    from scripts.baselines import compute_elo_baseline
    # Low Elo pair → low draw rate
    # High Elo pair → high draw rate
    ...
```

- [ ] **Step 2: Implement baselines.py**

```python
# scripts/baselines.py
"""Baseline models for quality gate comparison — ISO 25059."""

from __future__ import annotations
import numpy as np
import pandas as pd

def compute_naive_baseline(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Always predict marginal class distribution. Returns (n_test, 3)."""
    counts = np.bincount(y_train, minlength=3)
    probs = counts / counts.sum()
    return np.tile(probs, (n_test, 1))

def compute_elo_baseline(
    blanc_elo: np.ndarray, noir_elo: np.ndarray,
    draw_rate_lookup: pd.DataFrame,
) -> np.ndarray:
    """Elo formula + Pawnalyze draw rate model. Returns (n, 3)."""
    diff = noir_elo - blanc_elo
    # Elo expected score (white perspective, +35 for white advantage)
    expected = 1 / (1 + 10 ** ((diff - 35) / 400))
    # Draw rate from lookup (avg_elo, abs_diff)
    avg = (blanc_elo + noir_elo) / 2
    abs_diff = np.abs(blanc_elo - noir_elo)
    # Lookup draw_rate_prior (simplified: linear interpolation)
    draw_rate = _lookup_draw_rate(avg, abs_diff, draw_rate_lookup)
    # Decompose: P(win) = expected - 0.5*draw_rate, P(loss) = 1 - P(win) - draw_rate
    p_win = np.clip(expected - 0.5 * draw_rate, 0, 1)
    p_draw = np.clip(draw_rate, 0, 1)
    p_loss = np.clip(1 - p_win - p_draw, 0, 1)
    # Normalize
    total = p_win + p_draw + p_loss
    return np.column_stack([p_loss / total, p_draw / total, p_win / total])

def _lookup_draw_rate(avg_elo, abs_diff, lookup):
    """Map (avg_elo, abs_diff) to draw rate via lookup table."""
    import pandas as pd
    from scripts.features.draw_priors import ELO_BINS, DIFF_BINS
    temp = pd.DataFrame({"avg": avg_elo, "diff": abs_diff})
    temp["_elo_band"] = pd.cut(temp["avg"], bins=ELO_BINS, labels=False)
    temp["_diff_band"] = pd.cut(temp["diff"], bins=DIFF_BINS, labels=False)
    merged = temp.merge(lookup, on=["_elo_band", "_diff_band"], how="left")
    return merged["draw_rate_prior"].fillna(lookup["draw_rate_prior"].mean()).values
```

- [ ] **Step 3: Run tests — pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat(baselines): naive + Elo draw rate baselines for quality gate"
```

---

## Task 5: Multiclass metrics — RPS, ECE, E[score] MAE

**Files:**
- Modify: `scripts/kaggle_diagnostics.py:_compute_metrics()` (line ~206)

- [ ] **Step 1: Write tests**

```python
def test_rps_perfect_prediction() -> None:
    """RPS = 0 for perfect predictions."""
    import numpy as np
    y_true = np.array([0, 1, 2])
    y_proba = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    from scripts.kaggle_diagnostics import _compute_rps
    assert _compute_rps(y_true, y_proba) == 0.0

def test_expected_score_mae() -> None:
    """E[score] MAE measures error on CE input."""
    import numpy as np
    y_true = np.array([2, 1, 0])  # win, draw, loss
    # Perfect probas
    y_proba = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
    from scripts.kaggle_diagnostics import _compute_expected_score_mae
    # E[score] for win=1.0, draw=0.5, loss=0.0. Predicted same. MAE=0.
    assert _compute_expected_score_mae(y_true, y_proba) == 0.0
```

- [ ] **Step 2: Implement multiclass _compute_metrics()**

Replace the current binary metrics with:

```python
def _compute_metrics(y_true, y_pred, y_proba):
    """Multiclass validation metrics with calibration quality (ISO 25059)."""
    import numpy as np
    from sklearn.metrics import accuracy_score, log_loss

    n_classes = 3
    # Multiclass log loss
    ml_log_loss = float(log_loss(y_true, y_proba, labels=[0, 1, 2]))
    # Baseline log loss (predict marginal)
    counts = np.bincount(y_true, minlength=n_classes)
    baseline_probs = counts / counts.sum()
    baseline_ll = float(log_loss(y_true, np.tile(baseline_probs, (len(y_true), 1))))
    # RPS
    rps = float(_compute_rps(y_true, y_proba))
    # E[score] MAE
    es_mae = float(_compute_expected_score_mae(y_true, y_proba))
    # Brier multiclass
    brier = float(_compute_multiclass_brier(y_true, y_proba, n_classes))
    # ECE per class
    ece = {f"ece_class_{c}": float(_compute_ece(y_true == c, y_proba[:, c]))
           for c in range(n_classes)}
    # Calibration check: mean P(draw) vs observed draw rate
    mean_p_draw = float(y_proba[:, 1].mean())
    observed_draw = float((y_true == 1).mean())
    accuracy = float(accuracy_score(y_true, y_pred))

    return {
        "log_loss": ml_log_loss,
        "log_loss_baseline": baseline_ll,
        "log_loss_ratio": round(ml_log_loss / baseline_ll, 4) if baseline_ll > 0 else 0,
        "rps": rps,
        "expected_score_mae": es_mae,
        "brier_multiclass": brier,
        **ece,
        "mean_p_draw": mean_p_draw,
        "observed_draw_rate": observed_draw,
        "draw_calibration_bias": round(mean_p_draw - observed_draw, 4),
        "accuracy_3class": accuracy,
    }


def _compute_rps(y_true, y_proba):
    """Ranked Probability Score — ordinal metric."""
    import numpy as np
    n_classes = y_proba.shape[1]
    one_hot = np.eye(n_classes)[y_true]
    cum_pred = np.cumsum(y_proba, axis=1)
    cum_true = np.cumsum(one_hot, axis=1)
    return float(np.mean(np.sum((cum_pred - cum_true) ** 2, axis=1) / (n_classes - 1)))


def _compute_expected_score_mae(y_true, y_proba):
    """MAE between predicted E[score] and actual score."""
    import numpy as np
    # Actual scores: loss=0, draw=0.5, win=1.0
    actual_scores = np.where(y_true == 2, 1.0, np.where(y_true == 1, 0.5, 0.0))
    # Predicted E[score] = P(win) + 0.5*P(draw)
    predicted_scores = y_proba[:, 2] + 0.5 * y_proba[:, 1]
    return float(np.mean(np.abs(predicted_scores - actual_scores)))


def _compute_multiclass_brier(y_true, y_proba, n_classes):
    """Multiclass Brier score."""
    import numpy as np
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def _compute_ece(y_true_binary, y_proba_class, n_bins=10):
    """Expected Calibration Error for one class."""
    import numpy as np
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_proba_class >= bins[i]) & (y_proba_class < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_pred = y_proba_class[mask].mean()
        avg_true = y_true_binary[mask].mean()
        ece += mask.sum() * abs(avg_pred - avg_true)
    return ece / len(y_true_binary)
```

- [ ] **Step 3: Run tests — pass**
- [ ] **Step 4: Line count check** `wc -l scripts/kaggle_diagnostics.py`

If >300 lines: extract `_compute_rps`, `_compute_expected_score_mae`, `_compute_multiclass_brier`, `_compute_ece` into `scripts/kaggle_metrics.py` and import them. This is expected — the 4 new functions add ~50 lines to an already 243-line file.

- [ ] **Step 5: Line count check** `wc -l scripts/kaggle_trainers.py`

If >300 lines: extract `check_quality_gates()` into `scripts/kaggle_quality_gate.py`. Current 290 + ~30 additions → ~320. Likely needs extraction.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(diagnostics): multiclass metrics — RPS, ECE, E[score] MAE, Brier"
```

---

## Task 6: Calibration 3-class (isotonic per class + renormalize)

**Files:**
- Modify: `scripts/kaggle_diagnostics.py:calibrate_models()` (line ~42)
- Modify: `scripts/kaggle_diagnostics.py:_save_predictions()` (line ~68)

- [ ] **Step 1: Write test**

```python
def test_calibration_3class_sums_to_1() -> None:
    """Calibrated probas must sum to 1 per row."""
    import numpy as np
    # After isotonic per class + renormalization, each row sums to 1
    ...
```

- [ ] **Step 2: Refactor calibrate_models() for 3 classes**

```python
def calibrate_models(results, X_valid, y_valid, out_dir):
    """Fit isotonic calibration PER CLASS. Save calibrators."""
    import joblib
    from sklearn.isotonic import IsotonicRegression
    import numpy as np

    calibrators = {}
    for name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = r["model"].predict_proba(X_valid)  # (n, 3)
        y_true = y_valid.values if hasattr(y_valid, "values") else y_valid

        # One isotonic regressor per class
        class_calibrators = []
        for c in range(3):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(y_proba[:, c], (y_true == c).astype(float))
            class_calibrators.append(iso)
        calibrators[name] = class_calibrators
        logger.info("  %s: 3-class isotonic calibration fitted", name)

    if calibrators:
        joblib.dump(calibrators, out_dir / "calibrators.joblib")
    return calibrators


def _apply_calibration(y_proba, class_calibrators):
    """Apply per-class isotonic + renormalize to sum=1."""
    import numpy as np
    calibrated = np.column_stack([
        cal.predict(y_proba[:, c]) for c, cal in enumerate(class_calibrators)
    ])
    # Renormalize
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return calibrated / row_sums
```

- [ ] **Step 3: Update _save_predictions() for 3-class calibrated output**

```python
# In _save_predictions():
if calibrators and name in calibrators:
    y_cal = _apply_calibration(y_proba, calibrators[name])
    data["y_proba_cal_loss"] = y_cal[:, 0]
    data["y_proba_cal_draw"] = y_cal[:, 1]
    data["y_proba_cal_win"] = y_cal[:, 2]
    data["y_pred_calibrated"] = np.argmax(y_cal, axis=1)
```

- [ ] **Step 4: Run tests — pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(diagnostics): 3-class isotonic calibration + renormalization"
```

---

## Task 7: Quality gate + baselines in train_kaggle.py

**Files:**
- Modify: `scripts/cloud/train_kaggle.py:main()` (line ~102)
- Modify: `scripts/kaggle_trainers.py:check_quality_gates()`
- Modify: `config/hyperparameters.yaml` (quality gate thresholds)

- [ ] **Step 1: Add baseline computation to train_kaggle.py**

```python
# In main(), after training:
from scripts.baselines import compute_naive_baseline, compute_elo_baseline
from scripts.features.draw_priors import build_draw_rate_lookup

naive_proba = compute_naive_baseline(y_train.values, len(y_test))
draw_lookup = build_draw_rate_lookup(train)
elo_proba = compute_elo_baseline(X_test["blanc_elo"].values, X_test["noir_elo"].values, draw_lookup)

baseline_metrics = {
    "naive": {"log_loss": log_loss(y_test, naive_proba), "rps": _compute_rps(y_test, naive_proba)},
    "elo": {"log_loss": log_loss(y_test, elo_proba), "rps": _compute_rps(y_test, elo_proba)},
}
```

- [ ] **Step 2: Rewrite check_quality_gates()**

```python
def check_quality_gates(results, baseline_metrics=None):
    """ISO 42001: MultiClass quality gate — calibration-first."""
    candidates = [(n, r) for n, r in results.items() if r["model"] is not None]
    if not candidates:
        return {"passed": False, "reason": "All models failed"}

    best_name, best_r = min(candidates, key=lambda x: x[1]["metrics"].get("log_loss", 999))
    m = best_r["metrics"]

    # Gate conditions (ALL must pass — spec line 106-114)
    gates = []
    if baseline_metrics:
        naive = baseline_metrics.get("naive", {})
        elo = baseline_metrics.get("elo", {})
        gates.append(("log_loss < naive", m["log_loss"] < naive.get("log_loss", 999)))
        gates.append(("log_loss < elo", m["log_loss"] < elo.get("log_loss", 999)))
        gates.append(("rps < naive", m["rps"] < naive.get("rps", 999)))
        gates.append(("rps < elo", m["rps"] < elo.get("rps", 999)))
        gates.append(("brier < naive", m.get("brier_multiclass", 1) < naive.get("brier", 999)))
        gates.append(("es_mae < elo", m.get("expected_score_mae", 1) < elo.get("es_mae", 999)))
    gates.append(("ece_draw < 0.05", m.get("ece_class_1", 1) < 0.05))
    gates.append(("draw_cal_bias < 0.02", abs(m.get("draw_calibration_bias", 1)) < 0.02))

    failed = [(name, ok) for name, ok in gates if not ok]
    if failed:
        reasons = [name for name, _ in failed]
        return {"passed": False, "reason": f"Failed: {', '.join(reasons)}", "best_model": best_name}
    return {"passed": True, "best_model": best_name, "best_log_loss": m["log_loss"]}
```

- [ ] **Step 3: Update hyperparameters.yaml quality thresholds**

```yaml
metrics_thresholds:
  log_loss:
    maximum: 1.10  # Must be < baseline (ratio < 1.0)
  rps:
    maximum: 0.25
    target: 0.18
  expected_score_mae:
    maximum: 0.40
    target: 0.30
  ece:
    maximum: 0.05  # V8 gate
    production_target: 0.015
  draw_calibration_bias:
    maximum: 0.02
```

- [ ] **Step 4: Run tests — pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(training): multiclass quality gate — log loss + RPS vs baselines + ECE"
```

---

## Task 8: Diagnostics — reliability diagrams + classification reports 3-class

**Files:**
- Modify: `scripts/kaggle_diagnostics.py:_save_calibration_curves()`
- Modify: `scripts/kaggle_diagnostics.py:_save_classification_reports()`
- Modify: `scripts/kaggle_diagnostics.py:_save_roc_curves()`

- [ ] **Step 1: Update _save_calibration_curves() for 3 classes**

```python
def _save_calibration_curves(results, X_test, y_test, out_dir):
    """Per-class calibration curves (reliability diagrams)."""
    from sklearn.calibration import calibration_curve
    for name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = r["model"].predict_proba(X_test)
        for c, label in enumerate(["loss", "draw", "win"]):
            prob_true, prob_pred = calibration_curve(
                (y_test == c).astype(int), y_proba[:, c], n_bins=10, strategy="uniform"
            )
            pd.DataFrame({"mean_predicted": prob_pred, "fraction_positive": prob_true}).to_csv(
                out_dir / f"{name}_calibration_{label}.csv", index=False
            )
```

- [ ] **Step 2: Update classification reports**

```python
# target_names: ["loss", "draw", "win"] instead of ["loss/draw", "win"]
reports[name] = classification_report(
    y_test, y_pred, target_names=["loss", "draw", "win"], output_dict=True
)
```

- [ ] **Step 3: Update ROC curves for multiclass (one-vs-rest)**

```python
def _save_roc_curves(results, X_test, y_test, out_dir):
    """Per-class ROC curves (one-vs-rest). ISO 25059 explicability."""
    from sklearn.metrics import roc_curve
    for name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = r["model"].predict_proba(X_test)
        for c, label in enumerate(["loss", "draw", "win"]):
            y_binary = (y_test == c).astype(int)
            fpr, tpr, thresholds = roc_curve(y_binary, y_proba[:, c])
            pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
                out_dir / f"{name}_roc_{label}.csv", index=False
            )
    logger.info("  ROC curves saved (one-vs-rest, 3 classes)")
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(diagnostics): 3-class reliability diagrams, classification reports, ROC OVR"
```

---

## Task 9: Config + AutoGluon + test suite

**Files:**
- Modify: `config/hyperparameters.yaml`
- Modify: `scripts/cloud/train_autogluon_kaggle.py`
- Modify: `tests/test_cloud_features.py`
- Modify: `tests/test_cloud_training.py`

- [ ] **Step 1: Update hyperparameters.yaml**

Replace binary metrics_thresholds with multiclass ones. Add RPS/ECE thresholds.

- [ ] **Step 2: Update AutoGluon script**

```python
# train_autogluon_kaggle.py: change target encoding
df[LABEL] = df[LABEL_SOURCE].map({0.0: 0, 0.5: 1, 1.0: 2}).astype(int)
# Exclude forfaits:
df = df[df[LABEL_SOURCE] != 2.0]
# eval_metric:
eval_metric = "log_loss"
```

- [ ] **Step 3: Update test_cloud_features.py**

Adapt `test_metrics_has_required_fields` for new multiclass metric names.

- [ ] **Step 4: Update test_cloud_training.py**

Adapt quality gate tests for new conditions.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ --ignore=tests/test_health.py --ignore=tests/training_optuna/ -q
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(config): multiclass params + AutoGluon 3-class + test suite adapted"
```

---

## Task 10: Local end-to-end integration test (5K SAMPLE)

**IMPORTANT:** Full feature parquets are NOT available locally (OOM 15GB).
This test generates features on a SMALL SAMPLE locally, then trains on that sample.
Full-scale execution happens on Kaggle (Plan B Task 11).

**Files:** None modified — validation only.

- [ ] **Step 1: Generate features on 5K sample + train**

```python
import pandas as pd
from pathlib import Path
from scripts.feature_engineering import compute_features_for_split, temporal_split

# Generate features on small sample (fits in 15GB RAM)
ech = pd.read_parquet("data/echiquiers.parquet")
train_raw, valid_raw, test_raw = temporal_split(ech)
history = ech[ech["saison"] <= train_raw["saison"].max()]

train = compute_features_for_split(train_raw.head(5000), history.head(20000), "train_sample", True, Path("data"))
valid = compute_features_for_split(valid_raw.head(1000), history.head(20000), "valid_sample", True, Path("data"))
test = compute_features_for_split(test_raw.head(1000), history.head(20000), "test_sample", True, Path("data"))

from scripts.kaggle_trainers import prepare_features, default_hyperparameters, train_all_sequential, evaluate_on_test

X_train, y_train, X_valid, y_valid, X_test, y_test, encoders = prepare_features(train, valid, test)

# Verify 3-class target
assert set(y_train.unique()) <= {0, 1, 2}, f"Target not 3-class: {y_train.unique()}"
assert 2.0 not in train["resultat_blanc"].values, "Forfaits in data!"

# Quick training (50 iterations)
config = default_hyperparameters()
for k in ["catboost", "xgboost", "lightgbm"]:
    config[k]["iterations" if k == "catboost" else "n_estimators"] = 50
    config[k]["early_stopping_rounds"] = 10
    if k == "catboost":
        config[k]["task_type"] = "CPU"

results = train_all_sequential(X_train, y_train, X_valid, y_valid, config)
evaluate_on_test(results, X_test, y_test)

# Verify output is 3-class
for name, r in results.items():
    if r["model"] is None:
        continue
    proba = r["model"].predict_proba(X_test.head(5))
    assert proba.shape[1] == 3, f"{name}: expected 3 cols, got {proba.shape[1]}"
    print(f"{name}: log_loss={r['metrics'].get('log_loss', 'N/A'):.4f}, "
          f"rps={r['metrics'].get('rps', 'N/A')}, "
          f"draw_cal_bias={r['metrics'].get('draw_calibration_bias', 'N/A')}")
```

- [ ] **Step 2: Verify calibration**

```python
from scripts.kaggle_diagnostics import calibrate_models, save_diagnostics
from pathlib import Path
import tempfile

out = Path(tempfile.mkdtemp())
save_diagnostics(results, X_test, y_test, X_valid, y_valid, X_train, out)

# Check calibrators saved
assert (out / "calibrators.joblib").exists()
# Check 3-class calibration curves
for model in ["CatBoost"]:
    for label in ["loss", "draw", "win"]:
        assert (out / f"{model}_calibration_{label}.csv").exists()
```

- [ ] **Step 3: Verify quality gate with baselines**

```python
from scripts.baselines import compute_naive_baseline
naive = compute_naive_baseline(y_train.values, len(y_test))
from sklearn.metrics import log_loss as sk_log_loss
naive_ll = sk_log_loss(y_test, naive)
print(f"Naive log_loss: {naive_ll:.4f}")
for name, r in results.items():
    m = r.get("metrics", {})
    ll = m.get("log_loss", 999)
    print(f"{name}: log_loss={ll:.4f} {'< naive' if ll < naive_ll else '> naive (FAIL)'}")
```

- [ ] **Step 4: Commit**

```bash
git commit -m "test(v8): local end-to-end MultiClass validation passed"
```

---

## Task 11: Kaggle pre-flight + push (feature engineering + training IN ONE KERNEL)

**CRITICAL:** Kaggle kernel does BOTH:
1. `run_feature_engineering_v2()` → generates V8 feature parquets (needs 30GB RAM)
2. Training → CatBoost/XGBoost/LightGBM MultiClass on those features
This is how V7 already worked (`train_kaggle.py` calls feature engineering internally).

**Files:**
- Modify: `scripts/cloud/kernel-metadata.json` (new slug)

**Pre-flight checklist (skill kaggle-deployment) :**

- [ ] **Step 1: Upload data + code (MUST include V8 modules)**

```bash
python -m scripts.cloud.upload_all_data
```

Verify the upload zip contains: `scripts/features/helpers.py`, `scripts/features/draw_priors.py`,
`scripts/features/club_level.py`, `scripts/features/merge_v8.py`, `scripts/features/player_enrichment.py` (with FFE K-coefficient fix).

- [ ] **Step 2: Version kernel slug**

```json
{"id": "pguillemin/alice-training-v8", "slug": "alice-training-v8"}
```

- [ ] **Step 3: Verify Kaggle UI settings**

Open `https://www.kaggle.com/code/pguillemin/alice-training-v8`:
- Session options → GPU T4 x2 ✓
- Add-ons → Secrets → HF_TOKEN ✓

- [ ] **Step 4: Verify no running kernel**

```bash
kaggle kernels status pguillemin/alice-training-v7
# Must be COMPLETE or ERROR, not RUNNING
```

- [ ] **Step 5: Push**

```bash
kaggle kernels push -p scripts/cloud/
kaggle kernels status pguillemin/alice-training-v8
```

- [ ] **Step 6: Monitor** — check logs for MultiClass training, 3-class probas, quality gate

---

## Cross-Plan Consistency: Plan A ↔ Plan B

**CRITICAL: Feature parquets are generated ON KAGGLE, not locally.**
`train_kaggle.py` calls `run_feature_engineering_v2()` → Plan A code runs on Kaggle.
Plan B's training code runs immediately after in the same kernel.
Local tests use small samples (5K rows) to verify logic, not full data.

| Interface | Plan A produces | Plan B consumes |
|-----------|----------------|-----------------|
| Feature code | V8 modules uploaded to Kaggle dataset | `train_kaggle.py` calls `run_feature_engineering_v2()` |
| Feature parquets | Generated ON KAGGLE (~198 cols, forfaits excluded) | `prepare_features()` loads from Kaggle working dir |
| Target column | `resultat_blanc` ∈ {0.0, 0.5, 1.0} (no 2.0) | `_split_xy()` maps to {0, 1, 2} |
| Column names | `win_rate_recent_blanc`, `h2h_draw_rate`, etc. | Categorical lists in `kaggle_trainers.py` must match |
| draw_rate_prior | In parquet (computed by Plan A draw_priors.py) | Used by baselines.py Elo baseline |
| Forfait helper | `scripts/features/helpers.py:FORFAIT_RESULT` | Imported in `_split_xy()` |
| RAM constraint | Local: 15GB (OOM on full data). Kaggle: 30GB (OK) | All full-scale runs on Kaggle only |

**CATEGORICAL_FEATURES and ADVANCED_CAT_FEATURES lists in kaggle_trainers.py must be updated to match new column names from Plan A.** Specifically:
- OLD: `forme_tendance_blanc/noir` → NEW: `win_trend_blanc/noir`, `draw_trend_blanc/noir`
- OLD: `pressure_type_blanc/noir` → stays same name (recalculated, same semantic)
- NEW additions: `regularite_blanc/noir`, `role_type_blanc/noir` (wired from ALI)
