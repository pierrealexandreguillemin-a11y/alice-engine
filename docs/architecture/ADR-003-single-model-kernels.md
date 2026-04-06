# ADR-003: Single-Model Kaggle Kernels

## Status: ACCEPTED (2026-03-29)

## Context

v17 training kernel ran 3 models sequentially (CatBoost 3.7h + XGBoost 0.4h + LightGBM 5.2h = 9.6h).
All 3 models hit the 20K iteration cap without early stopping. The kernel timed out at 9h.
No checkpoints → all 3 trained models lost. 9.5h of GPU quota wasted (models were CPU-only).

## Decision

Split training into **1 kernel per model**, all CPU (no GPU quota):

| Kernel | Slug | Model | Budget | Role |
|--------|------|-------|--------|------|
| FE | `alice-fe-v8` | — | ~1h CPU | Feature engineering |
| Train XGB | `alice-train-xgboost` | XGBoost | ~1h CPU | Canary (fast feedback) |
| Train CB | `alice-train-catboost` | CatBoost | ~9h CPU | SHAP analysis |
| Train LGB | `alice-train-lightgbm` | LightGBM | ~10h CPU | Best model |

Model selection via `KAGGLE_KERNEL_RUN_SLUG` env var (set by Kaggle from kernel id).
Same `train_kaggle.py` code_file for all 3 — detects model from slug name.

**Deployment order**: XGBoost first (canary), then CatBoost + LightGBM in parallel.

## Key facts

- **CPU has no weekly quota** on Kaggle, only 12h per-session limit
- **GPU quota is 30h/week** shared across P100+T4 — reset Saturday
- All 3 models run CPU anyway (CatBoost rsm, XGBoost device=cpu, LightGBM pip=no GPU)
- v17 burned 9.5h GPU quota for 0% GPU usage

## Parameters

- `n_estimators`: 50,000 (was 20,000 — all 3 hit cap in v17)
- `early_stopping_rounds`: 200 (was 500)
- Checkpoints after each model (immediate save to disk)
- `enable_gpu: false` in all kernel-metadata

## Consequences

- 0 GPU quota consumed
- Each model gets up to 12h (vs 3h shared before)
- Models can converge naturally via early stopping
- Failure of one kernel doesn't affect others
- XGBoost as canary gives ~1h feedback loop
