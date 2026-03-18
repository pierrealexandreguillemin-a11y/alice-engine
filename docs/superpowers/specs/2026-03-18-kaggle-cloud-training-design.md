# Kaggle Cloud Training — Design Spec

**Date**: 2026-03-18
**Status**: Approved (v3 — full ISO compliance audit)
**Approach**: Kaggle training + local ISO validation (2-phase pipeline)
**ISO Compliance**: 5055, 5259, 24027, 24029, 23894, 27034, 29119, 42001

## Problem

Training requires ~7 GB RAM for feature parquets (3.5M rows, 82 columns). Local machine has 15 GB total, ~6 GB available — insufficient. Cloud training needed.

## Architecture: 2-Phase Pipeline

```
PHASE 1 — Kaggle (cloud training, FREE)
  Load features → Train CatBoost/XGBoost/LightGBM → Quality Gates
  → Push CANDIDATE model to HF Hub

PHASE 2 — Local (ISO validation, lightweight)
  Pull CANDIDATE → Robustness (ISO 24029) → Fairness (ISO 24027)
  → McNemar comparison (ISO 24029) → Promote to PRODUCTION or REJECT
```

Phase 1 runs on Kaggle (30 GB RAM, free). Phase 2 runs locally (only loads test set + model, ~1 GB RAM). No model reaches production without full ISO validation.

## Data Flow

```
LOCAL (one-time setup)
  data/features/{train,valid,test}.parquet (115 MB)
      ↓
  scripts/cloud/upload_features.py → Kaggle Dataset "pierrax/alice-features"

PHASE 1 — KAGGLE
  /kaggle/input/alice-features/{train,valid,test}.parquet
      ↓
  scripts/cloud/train_kaggle.py
      ↓  1. Load + hash data (ISO 5259 lineage)
      ↓  2. Prepare features (label encoding, missing flags already in parquets)
      ↓  3. Train CatBoost → gc → XGBoost → gc → LightGBM (sequential)
      ↓  4. Evaluate on test (14 metrics per model)
      ↓  5. Quality gates (AUC floor + relative degradation)
      ↓  6. Model Card (ISO 42001) with status=CANDIDATE
      ↓  7. Save locally → push to HF Hub
      ↓
  HF Hub: Pierrax/alice-engine
    models/v{timestamp}/
      catboost.cbm, xgboost.ubj, lightgbm.txt
      label_encoders.joblib
      metadata.json (status: CANDIDATE)

PHASE 2 — LOCAL
  scripts/cloud/promote_model.py
      ↓  1. Pull CANDIDATE from HF Hub
      ↓  2. Load test data (local, ~11 MB)
      ↓  3. ISO 24029 robustness (noise injection, feature dropout)
      ↓  4. ISO 24027 fairness (demographic parity on ligue_code)
      ↓  5. McNemar vs champion (ISO 24029 statistical)
      ↓  6. Update metadata.json status → PRODUCTION or REJECTED
      ↓  7. Push updated metadata + reports to HF Hub
```

## Kaggle Environment

| Resource | Value |
|----------|-------|
| RAM | ~30 GB |
| CPU | 4 cores |
| GPU (optional) | T4 16 GB, 30h/week free |
| Session | 12h CPU |
| Storage | 20 GB `/kaggle/working/` |
| Internet | Required (HF push) |

## Feature Parquet Schema

82 columns total, 81 features + 1 target (`resultat_blanc`).

32 categorical columns including: `type_competition`, `division`, `ligue_code`, `jour_semaine`, `blanc_titre`, `noir_titre`, `zone_enjeu_dom`, `zone_enjeu_ext`, `couleur_preferee_blanc`, `couleur_preferee_noir`, `elo_trajectory_blanc`, `elo_trajectory_noir`, `pressure_type_blanc`, `pressure_type_noir`, etc.

Missing data flags (`*_missing`) already present in parquets from feature engineering (ISO 5259). Script must NOT re-generate them.

The `prepare_features` function in the Kaggle script must label-encode the same 4 base categoricals as production (`type_competition`, `division`, `ligue_code`, `jour_semaine`). Additional categoricals (`blanc_titre`, `noir_titre`, etc.) are passed natively to CatBoost via `cat_features` in hyperparameters, and label-encoded for XGBoost/LightGBM.

## Components

### 1. `scripts/cloud/train_kaggle.py` (~280 lines, single self-contained file)

Kaggle requires a single .py file for script kernels.

#### Metrics (14 fields, matching `ModelMetrics` dataclass)

```python
METRICS_FIELDS = [
    "auc_roc", "accuracy", "precision", "recall", "f1_score", "log_loss",
    "train_time_s", "test_auc", "test_accuracy", "test_f1",
    "true_negatives", "false_positives", "false_negatives", "true_positives",
]
```

#### Hyperparameters (embedded, matching config/hyperparameters.yaml)

```python
def default_hyperparameters() -> dict:
    return {
        "global": {"random_seed": 42, "early_stopping_rounds": 50, "eval_metric": "auc"},
        "catboost": {
            "iterations": 1000, "learning_rate": 0.03, "depth": 6,
            "l2_leaf_reg": 3, "min_data_in_leaf": 20,
            "thread_count": 4,  # Kaggle: 4 cores (production: -1)
            "task_type": "CPU", "random_seed": 42, "verbose": 100,
            "cat_features": [
                "type_competition", "division", "ligue_code",
                "blanc_titre", "noir_titre", "jour_semaine", "zone_enjeu_dom",
            ],
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
            "verbose": -1,
        },
    }
# NOTE: thread_count/n_jobs=4 for Kaggle. Production YAML uses -1.
# Cross-reference: config/hyperparameters.yaml
```

#### Quality Gates (relative threshold, matching production rollback detector)

```python
AUC_FLOOR = 0.70

def check_quality_gates(results, champion_auc=None):
    best_auc = max(r["test_auc"] for r in results.values() if r["model"] is not None)
    if best_auc < AUC_FLOOR:
        return {"passed": False, "reason": f"AUC {best_auc:.4f} < floor {AUC_FLOOR}"}
    if champion_auc and champion_auc > 0:
        drop_pct = (champion_auc - best_auc) / champion_auc * 100
        if drop_pct > 2.0:  # Same as DegradationThresholds.auc_drop_pct
            return {"passed": False, "reason": f"Degradation {drop_pct:.1f}% > 2.0%"}
    return {"passed": True, "best_model": ..., "best_auc": best_auc}
```

Champion AUC resolution: fetch `metadata.json` from `Pierrax/alice-engine` on HF Hub. If not found (first run), skip degradation check.

#### Data Lineage (ISO 5259, matching production)

```python
def compute_dataframe_hash(df) -> str:
    """SHA256 of pd.util.hash_pandas_object, truncated to 16 hex chars.
    Identical to scripts/model_registry/utils.py.
    """

def build_lineage(train, valid, test) -> dict:
    """Returns lineage dict with: paths, hashes, sample counts,
    feature_count, target_distribution, timestamp.
    """
```

#### Model Card (ISO 42001, matching ProductionModelCard)

Fields: `version`, `created_at`, `status` ("CANDIDATE"), `environment`, `data_lineage`, `artifacts` (with SHA256 checksums + sizes), `metrics` (14 fields per model), `feature_importance`, `hyperparameters`, `best_model`, `limitations`, `use_cases`, `conformance`, `quality_gate_result`.

`environment` captured via: `sys.version`, `platform.*`, `catboost.__version__`, etc. Includes Kaggle kernel ID if available (`os.environ.get("KAGGLE_KERNEL_RUN_SLUG")`).

#### Error Handling

- **Partial model failure**: If 1 model OOMs, others still train. Gate evaluates successful models only.
- **HF push failure**: Artifacts saved to `/kaggle/working/` first. Push failure logged, models downloadable from Kaggle output.
- **Session timeout** (12h): Not a concern for ~30 min training.

### 2. `scripts/cloud/promote_model.py` (~80 lines, runs locally)

Phase 2: pulls CANDIDATE from HF Hub, runs ISO validations, promotes or rejects.

```python
def main():
    # 1. Pull candidate version from HF Hub
    version_dir = pull_candidate("Pierrax/alice-engine")

    # 2. Load test data (local, lightweight)
    test = pd.read_parquet("data/features/test.parquet")

    # 3. Load best model from candidate
    model, model_name = load_best_model(version_dir)

    # 4. ISO 24029 Robustness (reuse existing infrastructure)
    robustness = run_robustness_validation(model, test)
    if not robustness["compliant"]:
        reject(version_dir, "Robustness validation failed")
        return

    # 5. ISO 24027 Fairness (reuse existing infrastructure)
    fairness = run_fairness_validation(model, test, protected_attr="ligue_code")
    if fairness["status"] == "CRITICAL":
        reject(version_dir, "Fairness validation critical")
        return

    # 6. McNemar vs champion (if champion exists)
    if champion_exists():
        mcnemar = run_mcnemar(model, champion_model, test)
        # Block only if significantly WORSE
        if mcnemar["p_value"] < 0.05 and mcnemar["new_auc"] < mcnemar["champion_auc"]:
            reject(version_dir, f"Significantly worse (p={mcnemar['p_value']:.4f})")
            return

    # 7. Promote
    promote(version_dir)  # Update metadata.json status → PRODUCTION, push to HF
```

This script imports from the existing ISO validation modules:
- `scripts/autogluon/iso_robustness_enhanced.py`
- `scripts/autogluon/iso_fairness_enhanced.py`
- `scripts/comparison/mcnemar_test.py`

RAM requirement: ~1 GB (test set 11 MB + model ~10 MB + validation overhead).

### 3. `scripts/cloud/upload_features.py` (~40 lines, runs locally)

Uploads feature parquets as Kaggle Dataset `pierrax/alice-features`.

### 4. `scripts/cloud/kernel-metadata.json` (~15 lines)

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

### 5. `scripts/cloud/__init__.py` (~5 lines)

### 6. `tests/test_cloud_training.py` (~150 lines)

```
Document ID: ALICE-TEST-CLOUD-TRAINING
Version: 1.0.0
Tests: ~15

Classes:
- TestDataLoader: hash, lineage, feature_count, target_distribution (4 tests)
- TestQualityGates: AUC floor, relative degradation, partial failure, first run (4 tests)
- TestModelCard: structure vs ProductionModelCard, checksums, environment (3 tests)
- TestPromoteModel: robustness pass/fail, fairness pass/fail (4 tests)
```

Includes a test that compares `default_hyperparameters()` keys against `config/hyperparameters.yaml` to catch divergence (excluding documented overrides like `thread_count`).

## What does NOT change

- `scripts/train_models_parallel.py` — local training still works
- `scripts/training/` — untouched
- `scripts/autogluon/` — reused by promote_model.py, not modified
- `scripts/comparison/` — reused by promote_model.py, not modified
- All existing tests — untouched

## Out of scope V1

- **Stacking** : meta-learner logistic regression sur les 3 base models (`scripts/ensemble_stacking.py`). Ajouté en V2 cloud si les 3 modèles individuels justifient un ensemble.
- **AutoGluon** : multi-layer bagging + stacking auto. Sous-performait au round jan 2026 (AUC 0.7173 vs LightGBM 0.7513, McNemar p=0.0002). Reporté — pas pertinent tant que les base models n'atteignent pas un plateau.

## File inventory

| Action | File | Lines | Runs on |
|--------|------|-------|---------|
| CREATE | `scripts/cloud/__init__.py` | ~5 | - |
| CREATE | `scripts/cloud/train_kaggle.py` | ~280 | Kaggle |
| CREATE | `scripts/cloud/promote_model.py` | ~80 | Local |
| CREATE | `scripts/cloud/upload_features.py` | ~40 | Local |
| CREATE | `scripts/cloud/kernel-metadata.json` | ~15 | Kaggle API |
| CREATE | `tests/test_cloud_training.py` | ~150 | Local |

**Total**: 6 new files (~570 lines)

## Success criteria

1. `python -m scripts.cloud.upload_features` uploads features to Kaggle
2. `kaggle kernels push -p scripts/cloud/` trains on Kaggle without OOM
3. Quality gates pass (AUC >= 0.70, degradation < 2% relative)
4. CANDIDATE model + metadata pushed to `Pierrax/alice-engine`
5. `metadata.json` structure matches `ProductionModelCard` (14 metrics, checksums, env, lineage)
6. `python -m scripts.cloud.promote_model` runs ISO 24029 + 24027 + McNemar locally
7. Model promoted to PRODUCTION only after all ISO gates pass
8. All local tests pass (mocked)
9. All files < 300 lines, functions < 50 lines
10. Hyperparameters test catches any divergence with YAML
