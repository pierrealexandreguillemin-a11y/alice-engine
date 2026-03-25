# Phase 1b: SHAP Feature Validation + Calibration Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Supersedes:** `2026-03-23-residual-learning-phase1.md` Tasks 4-6 (Tasks 1-3 completed).

**Goal:** Validate which of 177 V8 features have signal (SHAP + permutation importance on 3 saved v10 models), fix CatBoost `rsm`, fix calibration (temperature scaling), pass quality gate 9/9.

**Context:** Residual learning WORKS (Phase 1 Tasks 1-3). v10 outputs downloaded (42 MB). 166/177 features at importance=0 was a CatBoost `PredictionValuesChange` artifact — XGBoost uses 109, LightGBM 50.

**Spec:** `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md` — Phase 1 section.

**Depends on:**
- Phase 1 Tasks 1-3 (COMPLETE) — residual learning in baselines.py, kaggle_trainers.py, train_kaggle.py
- V8 FE parquets (COMPLETE, `alice-fe-v8` kernel output)
- V10 model artifacts (DOWNLOADED, `/tmp/v8_results/v20260324_171050/`)
- `alice-code` dataset (UP TO DATE, 2026-03-25)

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| NEW | `scripts/cloud/shap_analysis_kaggle.py` | SHAP + permutation on saved models |
| NEW | `scripts/cloud/kernel-metadata-shap.json` | Kernel metadata for SHAP analysis |
| MODIFY | `scripts/cloud/train_kaggle.py` | Add CatBoost `rsm`, SHAP as pipeline step |
| MODIFY | `scripts/kaggle_diagnostics.py` | Temperature scaling calibration |
| MODIFY | `scripts/kaggle_trainers.py` | CatBoost `rsm=0.3` parameter |
| MODIFY | `config/hyperparameters.yaml` | Add rsm, update calibration method |

---

## Task 1: SHAP + Permutation Importance Kernel (Kaggle)

**Files:** `scripts/cloud/shap_analysis_kaggle.py`, `scripts/cloud/kernel-metadata-shap.json`

**Why Kaggle:** FE parquets (196 cols) are kernel output of `alice-fe-v8`, not downloadable as dataset. Models + features must be together.

- [ ] **Step 1: Verify alice-training-v8 kernel_sources mount provides v10 outputs**

Cancelled kernel may not provide outputs as kernel_source. If not → upload v10 models as dataset `pguillemin/alice-models-v10`.

- [ ] **Step 2: Write kernel-metadata-shap.json**

CPU-only (SHAP doesn't need GPU). Sources: alice-code (code), alice-fe-v8 (features), alice-training-v8 or alice-models-v10 (models).

- [ ] **Step 3: Write shap_analysis_kaggle.py** (max 300 lines)

Pipeline:
1. Load FE parquets (test set for SHAP, valid for permutation baseline)
2. Load 3 models (CatBoost.cbm, XGBoost.ubj, LightGBM.txt)
3. Compute init_scores (required for correct predictions with residual models)
4. CatBoost SHAP: `model.get_feature_importance(type='ShapValues', data=pool)`
5. XGBoost/LightGBM: `shap.TreeExplainer(model).shap_values(X_test)`
6. Permutation importance (sklearn) on all 3 using `predict_with_init`
7. Save all artifacts

- [ ] **Step 4: Upload alice-code (if modules changed) + push kernel**

- [ ] **Step 5: Download SHAP results**

Expected outputs: `*_shap_importance.csv`, `permutation_importance_all.csv`, `feature_concordance.csv`

- [ ] **Step 6: Commit**

---

## Task 2: Feature Validation by Concordance

**After Task 1 results.**

- [ ] **Step 1: Classify features by cross-model agreement**

| Category | Criteria | Action |
|----------|----------|--------|
| VALIDATED | ≥2/3 models: SHAP > threshold AND permutation > 0 | Keep |
| MARGINAL | 1/3 uses, low SHAP | Keep but flag |
| NOISE | 0/3 models, permutation ≤ 0 | Remove |

- [ ] **Step 2: Domain verification** — each VALIDATED feature must have chess/FFE justification

- [ ] **Step 3: Document and commit**

---

## Task 3: CatBoost `rsm` Fix + Retrain

- [ ] **Step 1: Add `rsm: 0.3` to config/hyperparameters.yaml catboost section**

- [ ] **Step 2: Wire rsm in kaggle_trainers.py `_train_catboost()`**

- [ ] **Step 3: Upload alice-code + push training kernel v12**

Expect: CatBoost features non-zero > 30 (instead of 11).

- [ ] **Step 4: Verify + commit**

---

## Task 4: Temperature Scaling Calibration

**Design BEFORE code — this is the quality gate blocker.**

- [ ] **Step 1: Implement `find_temperature()` + `apply_temperature()` in kaggle_diagnostics.py**

Temperature scaling: `calibrated = softmax(logits / T)`, T minimizes NLL on valid.
Single scalar, preserves ratios, no renormalization. Guo et al. 2017 (ICML).

- [ ] **Step 2: Wire into train_kaggle.py** replacing isotonic per-class + renorm

- [ ] **Step 3: Test locally with v10 valid/test predictions** (already downloaded)

Verify: E[score] MAE < Elo (condition 6 — the one that failed with isotonic).

- [ ] **Step 4: If passes → push kernel v12 with temperature scaling**

- [ ] **Step 5: Commit**

---

## Task 5: Quality Gate 9/9

- [ ] **Step 1: Download v12 outputs**

- [ ] **Step 2: Verify ALL 9 conditions**

| # | Condition | Threshold |
|---|-----------|-----------|
| 1 | log_loss < naive | < 1.099 |
| 2 | log_loss < Elo | < 0.984 |
| 3 | RPS < naive | computed |
| 4 | RPS < Elo | computed |
| 5 | Brier < naive | computed |
| 6 | E[score] MAE < Elo | < Elo MAE |
| 7 | ECE < 0.05 per class | < 0.05 |
| 8 | draw_bias < 2% | < 0.02 |
| 9 | mean_p_draw > 1% | > 0.01 |

- [ ] **Step 3: If fail → diagnose, fix, re-push (max 2 attempts)**

- [ ] **Step 4: Push passing model to HF Hub `Pierrax/alice-engine`**

- [ ] **Step 5: Commit**

---

## Task 6: ISO Documentation

- [ ] **Step 1: Update TRAINING_PROGRESS.md §9 with v12 results**
- [ ] **Step 2: Update ISO_COMPLIANCE_TODOS.md scores**
- [ ] **Step 3: Regenerate fairness report (ISO 24027)**
- [ ] **Step 4: Model card with residual + temperature scaling + SHAP**
- [ ] **Step 5: Update memory**
- [ ] **Step 6: Commit**
