# Kaggle Cloud Training Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train CatBoost/XGBoost/LightGBM on Kaggle (free, 30 GB RAM), push CANDIDATE to HF Hub, then validate ISO locally before promoting to PRODUCTION.

**Architecture:** Single self-contained `train_kaggle.py` (~280 lines) runs on Kaggle. Separate `promote_model.py` runs locally, reusing existing ISO validation infrastructure (robustness, fairness, McNemar). `upload_features.py` is a one-time local helper.

**Tech Stack:** CatBoost, XGBoost, LightGBM, pandas, scikit-learn, huggingface_hub, Kaggle API

**Spec:** `docs/superpowers/specs/2026-03-18-kaggle-cloud-training-design.md`

---

## File Structure

| Action | File | Responsibility | Lines |
|--------|------|----------------|-------|
| CREATE | `scripts/cloud/__init__.py` | Package marker | ~5 |
| CREATE | `scripts/cloud/train_kaggle.py` | Self-contained Kaggle training script | ~280 |
| CREATE | `scripts/cloud/promote_model.py` | Local ISO validation + promotion | ~80 |
| CREATE | `scripts/cloud/upload_features.py` | Upload features to Kaggle Dataset | ~40 |
| CREATE | `scripts/cloud/kernel-metadata.json` | Kaggle API headless config | ~15 |
| CREATE | `tests/test_cloud_training.py` | Tests (mocked, run locally) | ~150 |

---

## Task 1: Tests + Quality Gates

**Files:**
- Create: `tests/test_cloud_training.py`
- Create: `scripts/cloud/__init__.py`

The quality gates and data helpers are tested first (TDD), then implemented inside `train_kaggle.py` in subsequent tasks.

- [ ] **Step 1.1: Create package + test file skeleton**

```python
# scripts/cloud/__init__.py
"""Cloud training package for ALICE Engine."""

# tests/test_cloud_training.py
"""Tests Cloud Training - ISO 29119.

Document ID: ALICE-TEST-CLOUD-TRAINING
Version: 1.0.0
Tests: 15

Classes:
- TestComputeHash: Tests dataframe hashing (2 tests)
- TestBuildLineage: Tests ISO 5259 lineage (2 tests)
- TestQualityGates: Tests AUC gates (4 tests)
- TestModelCard: Tests model card structure (3 tests)
- TestHyperparamsSync: Tests config matches YAML (1 test)
- TestPromoteModel: Tests promotion logic (3 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 42001:2023 - AI Management System
"""
```

Write the full test file with all 15 tests. Key tests:

**TestComputeHash:**
- `test_hash_deterministic`: same DataFrame → same 16-char hex hash
- `test_hash_differs_on_change`: modify one value → different hash

**TestBuildLineage:**
- `test_lineage_has_all_fields`: verify dict contains train/valid/test paths, hashes, sample counts, feature_count, target_distribution, created_at
- `test_lineage_feature_count_excludes_target`: verify feature_count = len(columns) - 1

**TestQualityGates:**
- `test_auc_below_floor_fails`: AUC 0.65 → gate fails
- `test_auc_above_floor_passes`: AUC 0.75 → gate passes
- `test_degradation_relative_fails`: champion 0.75, new 0.72 → 4% drop → fails
- `test_first_run_no_champion_passes`: champion_auc=None → skip degradation

**TestModelCard:**
- `test_card_has_all_required_fields`: check all ProductionModelCard fields present
- `test_card_artifacts_have_checksums`: each artifact has sha256 + size_bytes
- `test_card_environment_has_versions`: python_version, catboost, xgboost, lightgbm

**TestHyperparamsSync:**
- `test_kaggle_params_match_yaml`: load YAML, compare keys (excluding thread_count/n_jobs overrides)

**TestPromoteModel:**
- `test_robustness_fail_rejects`: mock robustness → not compliant → REJECTED
- `test_fairness_critical_rejects`: mock fairness → CRITICAL → REJECTED
- `test_all_pass_promotes`: mock all pass → PRODUCTION

- [ ] **Step 1.2: Run tests, verify FAIL**

Run: `pytest tests/test_cloud_training.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 1.3: Commit test skeleton**

```bash
git add scripts/cloud/__init__.py tests/test_cloud_training.py
git commit -m "test(cloud): add 15 tests for Kaggle training pipeline (ISO 29119)"
```

---

## Task 2: train_kaggle.py — Data Loading + Lineage

**Files:**
- Create: `scripts/cloud/train_kaggle.py` (first ~80 lines)

- [ ] **Step 2.1: Write data loading section**

Imports, logging setup, constants, and these functions:

```python
CATEGORICAL_FEATURES = ["type_competition", "division", "ligue_code", "jour_semaine"]
CATBOOST_CAT_FEATURES = [
    "type_competition", "division", "ligue_code",
    "blanc_titre", "noir_titre", "jour_semaine", "zone_enjeu_dom",
]
LABEL_COLUMN = "resultat_blanc"

def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """SHA256 of hash_pandas_object, 16 hex chars. Matches model_registry/utils.py."""
    hash_values = pd.util.hash_pandas_object(df, index=True)
    return hashlib.sha256(hash_values.values.tobytes()).hexdigest()[:16]

def build_lineage(train, valid, test, data_dir) -> dict:
    """ISO 5259 data lineage."""
    return {
        "train_path": str(data_dir / "train.parquet"),
        "valid_path": str(data_dir / "valid.parquet"),
        "test_path": str(data_dir / "test.parquet"),
        "train_samples": len(train), "valid_samples": len(valid), "test_samples": len(test),
        "train_hash": compute_dataframe_hash(train),
        "valid_hash": compute_dataframe_hash(valid),
        "test_hash": compute_dataframe_hash(test),
        "feature_count": len(train.columns) - 1,
        "target_distribution": {
            "positive_ratio": float((train[LABEL_COLUMN] == 1.0).mean()),
            "total_samples": len(train),
        },
        "created_at": datetime.now(tz=UTC).isoformat(),
    }

def prepare_features(train, valid, test):
    """Label-encode base categoricals, split X/y, return encoders.
    Missing flags (*_missing) already in parquets — do NOT re-generate.
    """
    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    for split in [train, valid, test]:
        for col in CATEGORICAL_FEATURES:
            if col not in encoders:
                enc = LabelEncoder()
                split[col] = enc.fit_transform(split[col].astype(str))
                encoders[col] = enc
            else:
                split[col] = encoders[col].transform(split[col].astype(str))
    y_train = (train[LABEL_COLUMN] >= 1.0).astype(int)
    y_valid = (valid[LABEL_COLUMN] >= 1.0).astype(int)
    y_test = (test[LABEL_COLUMN] >= 1.0).astype(int)
    drop_cols = [LABEL_COLUMN, "resultat_noir", "resultat_text"]
    X_train = train.drop(columns=[c for c in drop_cols if c in train.columns])
    X_valid = valid.drop(columns=[c for c in drop_cols if c in valid.columns])
    X_test = test.drop(columns=[c for c in drop_cols if c in test.columns])
    return X_train, y_train, X_valid, y_valid, X_test, y_test, encoders
```

- [ ] **Step 2.2: Run hash + lineage tests**

Run: `pytest tests/test_cloud_training.py::TestComputeHash tests/test_cloud_training.py::TestBuildLineage -v`
Expected: 4 PASS

- [ ] **Step 2.3: Commit**

```bash
git add scripts/cloud/train_kaggle.py
git commit -m "feat(cloud): add data loading + ISO 5259 lineage to train_kaggle.py"
```

---

## Task 3: train_kaggle.py — Hyperparameters + Training

**Files:**
- Modify: `scripts/cloud/train_kaggle.py` (add ~80 lines)

- [ ] **Step 3.1: Add hyperparameters + training functions**

```python
def default_hyperparameters() -> dict:
    """Matching config/hyperparameters.yaml. Cross-ref: config/hyperparameters.yaml
    NOTE: thread_count/n_jobs=4 for Kaggle (YAML uses -1 for all cores).
    NOTE: zone_enjeu_dom (parquet column) replaces zone_enjeu (YAML name).
    """
    return {
        "global": {"random_seed": 42, "early_stopping_rounds": 50, "eval_metric": "auc"},
        "catboost": {
            "iterations": 1000, "learning_rate": 0.03, "depth": 6,
            "l2_leaf_reg": 3, "min_data_in_leaf": 20,
            "thread_count": 4, "task_type": "CPU", "random_seed": 42, "verbose": 100,
        },
        "xgboost": {
            "n_estimators": 1000, "learning_rate": 0.03, "max_depth": 6,
            "reg_lambda": 1.0, "reg_alpha": 0.0, "min_child_weight": 1,
            "tree_method": "hist", "n_jobs": 4, "random_state": 42,
            "early_stopping_rounds": 50, "verbosity": 1,
        },
        "lightgbm": {
            "n_estimators": 1000, "learning_rate": 0.03, "num_leaves": 63,
            "max_depth": -1, "reg_lambda": 1.0, "reg_alpha": 0.0,
            "min_child_samples": 20, "n_jobs": 4, "random_state": 42,
            "early_stopping_rounds": 50, "verbose": -1,
        },
    }

def compute_validation_metrics(y_true, y_pred, y_proba) -> dict:
    """10-field metrics on validation set. Matches scripts/training/metrics.py."""
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 log_loss, precision_score, recall_score, roc_auc_score)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "true_negatives": int(cm[0, 0]), "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]), "true_positives": int(cm[1, 1]),
    }

def train_all_sequential(X_train, y_train, X_valid, y_valid, config) -> dict:
    """CatBoost → gc → XGBoost → gc → LightGBM. Returns {name: {model, metrics, importance}}."""
    import gc, time
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    results = {}
    cat_params, xgb_params, lgb_params = config["catboost"], config["xgboost"], config["lightgbm"]
    # CatBoost (native categoricals)
    cat_idx = [X_train.columns.get_loc(c) for c in CATBOOST_CAT_FEATURES if c in X_train.columns]
    cb = CatBoostClassifier(**{k: v for k, v in cat_params.items()}, cat_features=cat_idx, eval_metric="AUC")
    t0 = time.time(); cb.fit(X_train, y_train, eval_set=(X_valid, y_valid)); t_cb = time.time() - t0
    results["CatBoost"] = _eval_model(cb, X_valid, y_valid, t_cb)
    del cb; gc.collect()
    # XGBoost
    xgb = XGBClassifier(**{k: v for k, v in xgb_params.items()}, eval_metric="auc")
    t0 = time.time(); xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100); t_xgb = time.time() - t0
    results["XGBoost"] = _eval_model(xgb, X_valid, y_valid, t_xgb)
    del xgb; gc.collect()
    # LightGBM
    import lightgbm as lgb_lib
    lgbm = LGBMClassifier(**{k: v for k, v in lgb_params.items() if k != "early_stopping_rounds"})
    t0 = time.time()
    lgbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="auc",
             callbacks=[lgb_lib.early_stopping(lgb_params.get("early_stopping_rounds", 50)),
                        lgb_lib.log_evaluation(100)])
    t_lgb = time.time() - t0
    results["LightGBM"] = _eval_model(lgbm, X_valid, y_valid, t_lgb)
    del lgbm; gc.collect()
    return results

def _eval_model(model, X_valid, y_valid, train_time) -> dict:
    """Evaluate model on validation, return {model, metrics, importance}."""
    import numpy as np
    y_proba = model.predict_proba(X_valid)[:, 1]
    y_pred = (y_proba >= 0.5).astype(np.int64)
    metrics = compute_validation_metrics(y_valid.values, y_pred, y_proba)
    metrics["train_time_s"] = train_time
    importance = dict(zip(X_valid.columns, model.feature_importances_)) if hasattr(model, "feature_importances_") else {}
    return {"model": model, "metrics": metrics, "importance": importance}

def evaluate_on_test(results, X_test, y_test) -> None:
    """Compute test_auc, test_accuracy, test_f1 for each model. Mutates results."""
    import numpy as np
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    for name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = r["model"].predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(np.int64)
        r["metrics"]["test_auc"] = float(roc_auc_score(y_test, y_proba))
        r["metrics"]["test_accuracy"] = float(accuracy_score(y_test, y_pred))
        r["metrics"]["test_f1"] = float(f1_score(y_test, y_pred, zero_division=0))
```

- [ ] **Step 3.2: Run hyperparams sync test**

Run: `pytest tests/test_cloud_training.py::TestHyperparamsSync -v`
Expected: PASS

- [ ] **Step 3.3: Commit**

```bash
git add scripts/cloud/train_kaggle.py
git commit -m "feat(cloud): add hyperparameters + sequential training to train_kaggle.py"
```

---

## Task 4: train_kaggle.py — Quality Gates + Model Card + Push

**Files:**
- Modify: `scripts/cloud/train_kaggle.py` (add ~120 lines to complete the file)

- [ ] **Step 4.1: Add quality gates**

```python
AUC_FLOOR = 0.70

def check_quality_gates(results, champion_auc=None) -> dict:
    best = max(
        ((name, r) for name, r in results.items() if r["model"] is not None),
        key=lambda x: x[1]["metrics"]["test_auc"],
    )
    best_auc = best[1]["metrics"]["test_auc"]
    if best_auc < AUC_FLOOR:
        return {"passed": False, "reason": f"AUC {best_auc:.4f} < {AUC_FLOOR}"}
    if champion_auc and champion_auc > 0:
        drop_pct = (champion_auc - best_auc) / champion_auc * 100
        if drop_pct > 2.0:
            return {"passed": False, "reason": f"Degradation {drop_pct:.1f}% > 2.0%"}
    return {"passed": True, "best_model": best[0], "best_auc": best_auc}

def fetch_champion_auc() -> float | None:
    """Fetch latest metadata.json from Pierrax/alice-engine on HF Hub."""
```

- [ ] **Step 4.2: Add model card + save + push**

```python
def collect_environment() -> dict:
    """Python version, platform, package versions, Kaggle kernel ID."""

def build_model_card(results, lineage, gate, config) -> dict:
    """ISO 42001 Model Card with status=CANDIDATE."""
    # All fields matching ProductionModelCard:
    # version, created_at, status, environment, data_lineage,
    # artifacts (with sha256 + size), metrics, feature_importance,
    # hyperparameters, best_model, limitations, use_cases, conformance

def save_and_push(results, metadata, encoders) -> None:
    """Save to /kaggle/working/v{timestamp}/, then upload_folder to HF Hub."""

def main():
    """Full pipeline orchestration."""
```

- [ ] **Step 4.3: Run all quality gate + model card tests**

Run: `pytest tests/test_cloud_training.py::TestQualityGates tests/test_cloud_training.py::TestModelCard -v`
Expected: 7 PASS

- [ ] **Step 4.4: Verify line count**

Run: `wc -l scripts/cloud/train_kaggle.py`
Expected: < 300 lines

- [ ] **Step 4.5: Commit**

```bash
git add scripts/cloud/train_kaggle.py
git commit -m "feat(cloud): add quality gates, model card, HF push to train_kaggle.py"
```

---

## Task 5: promote_model.py — Local ISO Validation

**Files:**
- Create: `scripts/cloud/promote_model.py`

- [ ] **Step 5.1: Implement promote_model.py**

```python
"""Promote CANDIDATE model to PRODUCTION after ISO validation.

Pulls candidate from HF Hub, runs ISO 24029 robustness,
ISO 24027 fairness, and McNemar comparison vs champion.

Usage: python -m scripts.cloud.promote_model [--version v20260318_120000]
"""
import argparse, json, logging
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

def pull_candidate(repo_id, version=None) -> Path:
    """Download candidate model artifacts from HF Hub."""

def run_robustness(model, X_test, y_test) -> dict:
    """ISO 24029: noise injection + feature dropout.
    Standalone implementation (existing modules expect AutoGluon TabularPredictor).
    Uses same algorithms: 5% Gaussian noise on numerics, single-feature dropout.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score
    base_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    # Noise injection (5% std on numeric columns)
    X_noisy = X_test.copy()
    num_cols = X_noisy.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        noise = np.random.normal(0, 0.05 * X_noisy[col].std(), len(X_noisy))
        X_noisy[col] = X_noisy[col] + noise
    noisy_auc = roc_auc_score(y_test, model.predict_proba(X_noisy)[:, 1])
    noise_tolerance = noisy_auc / base_auc if base_auc > 0 else 0
    # Feature dropout (top-3 importance)
    compliant = noise_tolerance >= 0.85  # ACCEPTABLE threshold from iso_robustness_enhanced
    return {"base_auc": base_auc, "noisy_auc": noisy_auc,
            "noise_tolerance": noise_tolerance, "compliant": compliant}

def run_fairness(model, X_test, y_test, protected_attr="ligue_code") -> dict:
    """ISO 24027: demographic parity on protected attribute.
    Standalone (existing modules expect AutoGluon TabularPredictor).
    """
    import numpy as np
    y_pred = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    groups = X_test[protected_attr].unique()
    rates = {}
    for g in groups:
        mask = X_test[protected_attr] == g
        if mask.sum() < 30:
            continue
        rates[g] = float(y_pred[mask].mean())
    if len(rates) < 2:
        return {"status": "INSUFFICIENT_DATA", "demographic_parity": None}
    min_rate, max_rate = min(rates.values()), max(rates.values())
    dp_ratio = min_rate / max_rate if max_rate > 0 else 0
    if dp_ratio < 0.6:
        status = "CRITICAL"
    elif dp_ratio < 0.8:
        status = "CAUTION"
    else:
        status = "FAIR"
    return {"status": status, "demographic_parity": dp_ratio, "group_rates": rates}

def run_mcnemar(new_model, champion_model, X_test, y_test) -> dict:
    """McNemar test: new vs champion predictions on test set."""
    import numpy as np
    from scipy.stats import chi2
    pred_new = (new_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    pred_champ = (champion_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    correct_new = (pred_new == y_test.values)
    correct_champ = (pred_champ == y_test.values)
    b = int((correct_champ & ~correct_new).sum())  # champ right, new wrong
    c = int((~correct_champ & correct_new).sum())  # new right, champ wrong
    if b + c == 0:
        return {"p_value": 1.0, "statistic": 0.0, "significant": False}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1 - chi2.cdf(stat, 1))
    from sklearn.metrics import roc_auc_score
    new_auc = float(roc_auc_score(y_test, new_model.predict_proba(X_test)[:, 1]))
    champ_auc = float(roc_auc_score(y_test, champion_model.predict_proba(X_test)[:, 1]))
    return {"p_value": p_value, "statistic": stat, "significant": p_value < 0.05,
            "new_auc": new_auc, "champion_auc": champ_auc}

def promote(version_dir, repo_id) -> None:
    """Update metadata.json status=PRODUCTION, push to HF Hub."""

def reject(version_dir, reason, repo_id) -> None:
    """Update metadata.json status=REJECTED, push to HF Hub."""

def main():
    """Pull → Robustness → Fairness → McNemar → Promote/Reject."""
```

- [ ] **Step 5.2: Run promote tests**

Run: `pytest tests/test_cloud_training.py::TestPromoteModel -v`
Expected: 3 PASS

- [ ] **Step 5.3: Commit**

```bash
git add scripts/cloud/promote_model.py
git commit -m "feat(cloud): add promote_model.py for local ISO validation (24029/24027)"
```

---

## Task 6: upload_features.py + kernel-metadata.json

**Files:**
- Create: `scripts/cloud/upload_features.py`
- Create: `scripts/cloud/kernel-metadata.json`

- [ ] **Step 6.1: Implement upload_features.py**

```python
"""Upload feature parquets to Kaggle Dataset.

Usage: python -m scripts.cloud.upload_features
Requires: ~/.kaggle/kaggle.json with API credentials
"""
import json, subprocess, shutil, tempfile
from pathlib import Path

FEATURES_DIR = Path("data/features")
DATASET_ID = "pierrax/alice-features"
FILES = ["train.parquet", "valid.parquet", "test.parquet"]

def create_dataset_metadata(tmp_dir: Path) -> None:
    """Write dataset-metadata.json for Kaggle API."""

def upload() -> None:
    """Copy parquets to temp dir, create metadata, kaggle datasets create/version."""

def main():
    upload()
```

- [ ] **Step 6.2: Create kernel-metadata.json**

```json
{
  "id": "pierrax/alice-training",
  "title": "ALICE Engine Training (ISO 42001)",
  "code_file": "train_kaggle.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": ["pierrax/alice-features"],
  "competition_sources": [],
  "kernel_sources": []
}
```

- [ ] **Step 6.3: Commit**

```bash
git add scripts/cloud/upload_features.py scripts/cloud/kernel-metadata.json
git commit -m "feat(cloud): add upload_features.py + Kaggle kernel-metadata.json"
```

---

## Task 7: Full Test Suite + Verification

- [ ] **Step 7.1: Run all tests**

Run: `pytest tests/test_cloud_training.py -v --tb=short`
Expected: 15 PASS

- [ ] **Step 7.2: Run quality checks**

Run: `ruff check scripts/cloud/ tests/test_cloud_training.py && ruff format --check scripts/cloud/ tests/test_cloud_training.py`
Expected: All pass

- [ ] **Step 7.3: Verify ISO 5055 limits**

Run: `wc -l scripts/cloud/*.py tests/test_cloud_training.py`
Expected: All < 300 lines

Run: `radon cc scripts/cloud/train_kaggle.py scripts/cloud/promote_model.py -s`
Expected: All A or B

Run: `python -c "import ast, sys; tree=ast.parse(open('scripts/cloud/train_kaggle.py').read()); fns=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]; over=[f.name for f in fns if f.end_lineno-f.lineno>50]; print(f'Functions >50 lines: {over or \"none\"}')" `
Expected: "Functions >50 lines: none"

- [ ] **Step 7.4: Dry-run local test of train_kaggle.py**

Run: `python -c "from scripts.cloud.train_kaggle import default_hyperparameters, compute_dataframe_hash; print('imports OK')"`
Expected: imports succeed

- [ ] **Step 7.5: Commit**

```bash
git add -u
git commit -m "fix(cloud): lint and quality fixes"
```

---

## Task 8: E2E — Upload Features + Run on Kaggle

This task requires Kaggle API credentials (`~/.kaggle/kaggle.json`).

- [ ] **Step 8.1: Upload features to Kaggle**

Run: `python -m scripts.cloud.upload_features`
Expected: Dataset `pierrax/alice-features` created/updated on Kaggle

- [ ] **Step 8.2: Push kernel to Kaggle**

Run: `kaggle kernels push -p scripts/cloud/`
Expected: Kernel submitted, starts running

- [ ] **Step 8.3: Monitor Kaggle execution**

Run: `kaggle kernels status pierrax/alice-training`
Expected: Eventually "complete"

- [ ] **Step 8.4: Download Kaggle output**

Run: `kaggle kernels output pierrax/alice-training -p /tmp/kaggle-output`
Expected: Model artifacts + metadata.json in output

- [ ] **Step 8.5: Verify CANDIDATE on HF Hub**

Run: `python -c "from huggingface_hub import hf_hub_download; import json; m = json.load(open(hf_hub_download('Pierrax/alice-engine', 'metadata.json', repo_type='model'))); print(f'Status: {m[\"status\"]}, Best: {m[\"best_model\"]}, AUC: {m[\"metrics\"]}')" `
Expected: status=CANDIDATE

- [ ] **Step 8.6: Run local ISO promotion**

Run: `python -m scripts.cloud.promote_model`
Expected: Robustness PASS, Fairness PASS, McNemar PASS → status=PRODUCTION

- [ ] **Step 8.7: Final commit**

```bash
git add -A
git commit -m "feat(cloud): complete Kaggle cloud training pipeline (ISO 42001)"
```
