# V9 Optuna Pipeline — Hyperparameter Optimization + Stacking

**Date:** 2026-04-07
**Status:** APPROVED
**Scope:** Optuna tuning → Training final → OOF stacking → Champion selection → ISO
**Prerequisite:** V8 features complete (197 features, FE kernel output)
**Supersedes:** V8 manual hyperparameters (postmortem: docs/postmortem/2026-04-07-skipped-optuna-tuning.md)

---

## 1. Context

V8 models (XGBoost v5, LightGBM v7, CatBoost v6) pass quality gates but
use manually tuned hyperparameters from ~18 Kaggle iterations. Optuna
Bayesian optimization (Phase 2 of METHODOLOGIE_ML_TRAINING.md) was never
executed. The gain from systematic tuning is unknown.

The stacking evaluation (2026-04-07) used incorrect methodology (train on
valid, not OOF) and is invalidated. Proper OOF stacking requires dedicated
Kaggle kernels.

## 2. Pipeline Overview

11 steps, 9 Kaggle kernels, 3 user checkpoints. Sequential — each step
gated before the next starts.

```
OPTUNA PHASE (3 kernels)
  Step 1: Optuna XGBoost  → best_params + study.db  [GATE 1]
  Step 2: Optuna CatBoost → best_params + study.db  [GATE 2]
  Step 3: Optuna LightGBM → best_params + study.db  [GATE 3]
  ── USER CHECKPOINT: review 3 best_params ──

TRAINING FINAL PHASE (3 kernels)
  Step 4: Train XGBoost (best params, full convergence) [GATE 4: 15 quality gates]
  Step 5: Train CatBoost                                [GATE 5: 15 quality gates]
  Step 6: Train LightGBM                                [GATE 6: 15 quality gates]
  ── USER CHECKPOINT: review 3 models ──

OOF STACKING PHASE (3 kernels)
  Step 7: OOF XGBoost  (5-fold CV, best params)  [GATE 7]
  Step 8: OOF CatBoost (5-fold CV, best params)  [GATE 8]
  Step 9: OOF LightGBM (5-fold CV, best params)  [GATE 9]
  ── USER CHECKPOINT: review 3 OOF sets ──

STACKING + ISO (local)
  Step 10: Meta-learner Optuna + eval  [GATE 10: quality gates on stacking]
  Step 11: ISO reports                 [GATE 11: full ISO pipeline]
```

## 3. Search Spaces (audited against fabricant docs)

### 3.1 Shared parameter

| Parameter | Range | Source |
|-----------|-------|--------|
| init_score_alpha | [0.3, 0.8] float | Custom — residual prior strength (Guo 2017) |

### 3.2 XGBoost

Source: [xgboost.readthedocs.io/parameter.html](https://xgboost.readthedocs.io/en/stable/parameter.html)

| Parameter | Range | Type | Default | Fabricant guidance |
|-----------|-------|------|---------|--------------------|
| max_depth | [3, 8] | int | 6 | "increasing makes more complex, likely to overfit" |
| eta | [1e-3, 0.1] | float, log | 0.3 | range [0,1] |
| reg_lambda | [0.1, 20] | float, log | 1 | "increasing makes more conservative" |
| reg_alpha | [1e-3, 1.0] | float, log | 0 | L1 regularization |
| subsample | [0.5, 1.0] | float | 1 | "typically >= 0.5" |
| colsample_bytree | [0.3, 1.0] | float | 1 | range (0,1] |
| min_child_weight | [20, 200] | int | 1 | "larger = more conservative" |

Fixed: n_estimators=50000, early_stopping=200, objective=multi:softprob.

### 3.3 CatBoost

Source: [catboost.ai/docs/parameter-tuning](https://catboost.ai/docs/en/concepts/parameter-tuning)

| Parameter | Range | Type | Default | Fabricant guidance |
|-----------|-------|------|---------|--------------------|
| depth | [4, 10] | int | 6 | "optimal 4-10, 6-10 recommended" |
| learning_rate | [1e-3, 0.1] | float, log | auto/0.03 | — |
| l2_leaf_reg | [0.1, 20] | float, log | 3 | "any positive, try different values" |
| rsm | [0.2, 0.8] | float | 1 | MANDATORY >50 features |
| random_strength | [0.5, 5.0] | float | 1 | "controls randomness for scoring" |
| min_data_in_leaf | [20, 200] | int | 1 | Depthwise policy |

Fixed: iterations=50000, early_stopping=200, loss=MultiClass, task_type=CPU.
CatBoost rsm incompatible GPU. CatBoostPruningCallback incompatible GPU (issue #3550).

### 3.4 LightGBM

Source: [lightgbm.readthedocs.io/Parameters-Tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)

| Parameter | Range | Type | Default | Fabricant guidance |
|-----------|-------|------|---------|--------------------|
| max_depth | [3, 8] | int | -1 | "limit to deal with over-fitting" |
| num_leaves | [7, 63] | int | 31 | CLAMP: min(val, 2^max_depth-1) |
| learning_rate | [1e-3, 0.1] | float, log | 0.1 | — |
| reg_lambda | [0.1, 20] | float, log | 0 | "use for regularization" |
| feature_fraction | [0.3, 1.0] | float | 1 | "use feature sub-sampling" |
| bagging_fraction | [0.5, 1.0] | float | 1 | Requires bagging_freq=1 |
| min_child_samples | [20, 200] | int | 20 | — |

Fixed: n_estimators=50000, early_stopping=200, objective=multiclass,
bagging_freq=1 (must be >0 to activate bagging_fraction).

## 4. Optuna Kernel Architecture (Steps 1-3)

### 4.1 Infrastructure

- **Storage:** SQLite (`/kaggle/working/optuna_{model}.db`) for resume
- **Sampler:** TPESampler(seed=42) — 10 startup trials (exploration), then Bayesian
- **Pruning:** `optuna_integration.XGBoostPruningCallback` / CatBoost / LightGBM
- **Metric:** minimize `multi_logloss` (eval metric on valid set)
- **Timeout:** 39600s (11h — 1h margin for save + logs)
- **n_trials:** 100 target, timeout decides actual count

### 4.2 Packages (Optuna >= 4.0 breaking change)

```python
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "optuna", "optuna-integration"])

from optuna_integration import XGBoostPruningCallback   # NOT optuna.integration
from optuna_integration import CatBoostPruningCallback
from optuna_integration import LightGBMPruningCallback
```

Source: [optuna-integration](https://github.com/optuna/optuna-integration),
[Optuna v4 migration](https://github.com/optuna/optuna/discussions/5573)

### 4.3 Entry point pattern

```python
import os, sys
from pathlib import Path
os.environ["ALICE_MODEL"] = "xgboost"
for p in [Path("/kaggle/input/alice-code"),
          Path("/kaggle/input/datasets/pguillemin/alice-code")]:
    if p.exists():
        sys.path.insert(0, str(p))
        break
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "optuna", "optuna-integration"])
from scripts.cloud.optuna_kaggle import main
main()
```

### 4.4 Objective function structure

```python
def objective(trial, X_train, y_train, X_valid, y_valid,
              init_scores_train, init_scores_valid, draw_lookup):
    alpha = trial.suggest_float("init_score_alpha", 0.3, 0.8)
    # ... suggest all model-specific params ...

    init_train = init_scores_train * alpha
    init_valid = init_scores_valid * alpha

    # Train with early stopping + pruning callback
    model = train_model(params, X_train, y_train, init_train,
                        X_valid, y_valid, init_valid,
                        callbacks=[pruning_callback, early_stop])

    # Return valid logloss
    return valid_logloss
```

init_scores computed ONCE before the study, alpha applied per trial.

### 4.5 Artefacts de sortie par kernel

| Fichier | Contenu | Usage |
|---------|---------|-------|
| `best_params_{model}.json` | Best hyperparams + alpha + logloss + n_trials | Gate 1-3 + Training Final |
| `optuna_{model}.db` | SQLite study (resume-safe) | Analysis + resume |
| `trial_history_{model}.csv` | All trials: params, value, duration_s | Convergence diagnostic |

### 4.6 Quality Gates (Steps 1-3)

| Gate | Condition | Rationale |
|------|-----------|-----------|
| G1 | best_logloss < 0.9766 (Elo baseline) | Model must beat baseline |
| G2 | n_trials_completed >= 10 | Minimum benchmark data |
| G3 | all params within fabricant ranges | No out-of-bounds |
| G4 | trial 1 time logged | Benchmark for budget planning |

## 5. Training Final (Steps 4-6)

Same architecture as V8 training kernels (`train_kaggle.py`), but with
Optuna best_params instead of manual hyperparameters.

### 5.1 Changes from V8

- Load `best_params_{model}.json` instead of `default_hyperparameters()`
- `init_score_alpha` from Optuna (not fixed 0.7)
- Otherwise identical: residual learning, quality gates, SHAP, calibration

### 5.2 Quality Gates (Steps 4-6) — V8 15-condition gate

Same as V8 (documented in CLAUDE.md):
T1-T2: log_loss < Elo AND < Naive
T3-T4: RPS < Elo AND < Naive
T5: E[score] MAE < Elo
T6: Brier < Naive
T7: ECE < 0.05 all classes
T8: Draw calibration bias < 0.02
T9: mean_p_draw > 1%
T10-T12: reporting conditions

ALL 15 gates must PASS. No exceptions.

### 5.3 Artefacts de sortie (per model)

Same as V8: model file, metadata.json, calibrators.joblib, encoders.joblib,
draw_rate_lookup.parquet, test/valid predictions, SHAP, feature importance.

## 6. OOF Stacking (Steps 7-9)

### 6.1 Architecture: 1 kernel per model

Each kernel runs 5-fold CV with the best Optuna params. Output: OOF
predictions for the full train+valid set + test predictions.

### 6.2 Computation budget (extrapolated, NOT measured)

| Model | Per fold estimate | 5 folds | Fits 12h? |
|-------|-------------------|---------|-----------|
| XGBoost | ~41 min | ~3.5h | YES |
| LightGBM | ~79 min (no startup) | ~6.6h | YES if from scratch each fold |
| CatBoost | ~204 min | ~17h | NO — split 2 sessions with SQLite |

**WARNING:** These are extrapolations from V8 training times, NOT measurements.
Actual times depend on Optuna-optimized params (may be faster or slower).
LightGBM startup 3h22m per fold is prohibitive if resuming from checkpoint —
MUST train from scratch each fold.

### 6.3 Fold structure

```python
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_oof = train + valid combined (1,210,446 rows)
y_oof = labels for combined set

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_oof, y_oof)):
    # Train on 80%, predict on 20%
    model = train_with_best_params(X_oof[train_idx], y_oof[train_idx],
                                    init_scores[train_idx] * alpha)
    oof_preds[val_idx] = predict_with_init(model, X_oof[val_idx],
                                            init_scores[val_idx] * alpha)
    test_preds_fold = predict_with_init(model, X_test, init_test * alpha)
    test_preds_acc += test_preds_fold / n_folds
    del model; gc.collect()  # RAM
```

### 6.4 Quality Gates (Steps 7-9)

| Gate | Condition | Rationale |
|------|-----------|-----------|
| G1 | oof_preds.shape == (n_total, 3) | Correct dimensions |
| G2 | oof_preds.sum(axis=1) ≈ 1.0 (atol=1e-4) | Valid probabilities |
| G3 | 0 NaN in oof_preds | No missing predictions |
| G4 | test_preds matches Gate G1-G3 | Same checks on test |
| G5 | OOF logloss < Elo baseline | Model still beats baseline in OOF |

### 6.5 Artefacts de sortie

| Fichier | Shape | Usage |
|---------|-------|-------|
| `oof_{model}.parquet` | (1210446, 4) [y_true + 3 probas] | Meta-learner training |
| `test_{model}.parquet` | (231532, 4) [y_true + 3 probas] | Meta-learner eval |

## 7. Stacking Meta-Learner (Step 10, local)

### 7.1 Meta-features

Assemble from OOF outputs:
- X_meta_train: (1210446, 9) — 3 models × 3 classes
- X_meta_test: (231532, 9)
- y_train, y_test: from OOF parquets (y_true column)

### 7.2 Meta-learner Optuna

Quick local Optuna (seconds per trial, 9 features):

```python
# LogisticRegression
C = trial.suggest_float("C", 0.01, 10.0, log=True)
# RidgeClassifier
alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
```

5-fold CV on X_meta_train, eval on X_meta_test.
Metric: multi_logloss for optimization, E[score] MAE for decision.

### 7.3 Decision gate (Step 10)

Compare on test set:
- Best single model (from Steps 4-6)
- Stacking LR (best C)
- Stacking Ridge (best alpha)

| Metric | Threshold | Action |
|--------|-----------|--------|
| E[score] MAE gain > 0.001 | Stacking wins | Serve ensemble |
| E[score] MAE gain <= 0.001 | No significant gain | Serve best single model |

### 7.4 Quality Gates (Step 10)

Full V8 quality gates (15 conditions) applied to stacking output.
If stacking passes all gates AND beats best single → champion = stacking.
If stacking fails any gate → champion = best single, document failure.

## 8. ISO Reports (Step 11, local)

### 8.1 Deliverables

| Norm | Artifact | Content |
|------|----------|---------|
| ISO 42001 | Model Card V9 | Optuna methodology, best params, stacking decision |
| ISO 25059 | Quality report | All gates results, baseline comparison |
| ISO 24029 | Robustness | Noise tolerance on champion model |
| ISO 24027 | Fairness | Per-group calibration on champion |
| ISO 42005 | Impact assessment | Champion serves real decisions |
| ISO 5259 | Data lineage | Same FE as V8, new training approach |

### 8.2 Quality Gate (Step 11)

All 9 ISO pipeline steps documented and validated.
Model card links to Optuna study DBs for full traceability.

## 9. Existing Code to Adapt

| File | Current | Adaptation needed |
|------|---------|-------------------|
| `scripts/training/optuna_core.py` | Binary AUC | Multiclass logloss + alpha in search |
| `scripts/training/optuna_objectives.py` | Binary AUC objectives | Multiclass objectives + init_scores |
| `scripts/ensemble/stacking.py` | Binary predict_proba[:, 1] | Multiclass (n, 3) full matrix |
| `scripts/ensemble/stacking.py` | AUC metric | Logloss + E[score] MAE |
| `config/hyperparameters.yaml` | Outdated search spaces | V9 ranges (Section 3) |
| `config/hyperparameters.yaml` | stacking.selection | Multiclass decision gate |
| `scripts/cloud/train_kaggle.py` | Hardcoded params | Load best_params.json |

## 10. References

### Fabricant documentation
- XGBoost params: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost tuning: https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
- CatBoost tuning: https://catboost.ai/docs/en/concepts/parameter-tuning
- CatBoost params: https://catboost.ai/docs/en/references/training-parameters/common
- LightGBM params: https://lightgbm.readthedocs.io/en/latest/Parameters.html
- LightGBM tuning: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html

### Optuna
- TPESampler: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
- RDB save/resume: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html
- optuna-integration (v4+): https://github.com/optuna/optuna-integration
- v4 migration: https://github.com/optuna/optuna/discussions/5573
- CatBoost pruning GPU bug: https://github.com/optuna/optuna/issues/3550
- CatBoost Optuna tutorial: https://github.com/catboost/tutorials/blob/master/hyperparameters_tuning/hyperparameters_tuning_using_optuna_and_hyperopt.ipynb

### Stacking / Ensemble
- scikit-learn StackingClassifier: https://scikit-learn.org/stable/modules/ensemble.html
- H2O Stacked Ensembles: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html
- Karaaslan & Erbay 2025 (chess ML): https://www.mdpi.com/2079-9292/15/1/1

### ISO
- ISO 42001 documentation requirements: https://www.hicomply.com/hub/documentation-required-under-iso-42001
- ISO 42001 lifecycle: https://aws.amazon.com/blogs/security/ai-lifecycle-risk-management-iso-iec-420012023-for-ai-governance/

### Project
- Postmortem Optuna skipped: docs/postmortem/2026-04-07-skipped-optuna-tuning.md
- Méthodologie ML: docs/project/METHODOLOGIE_ML_TRAINING.md §2.3
- Existing Optuna code: scripts/training/optuna_core.py, optuna_objectives.py
- Existing stacking code: scripts/ensemble/stacking.py
- Hyperparameters config: config/hyperparameters.yaml
