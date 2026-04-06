# ADR-002: Feature Construction at Inference Time

**Status:** OPEN — Decision needed before wiring routes→services
**Date:** 2026-03-22
**Context:** V8 ML model trained on 196 features. CE/ALI services need predictions in real-time.

---

## Problem

The V8 ML model requires 196 features per (player, board, opponent) matchup.
Feature engineering runs in batch (1.4M rows, 74 min on Kaggle P100 CPU).
At inference time (POST /predict), we need features for ~64 matchups
(8 boards x 8 candidate players) in <500ms.

## Feature Accessibility at Inference Time

| Category | Features | #cols | Real-time? | Source |
|----------|----------|-------|------------|--------|
| Elo, diff_elo, titre | Direct from request | ~6 | Trivial | PredictRequest |
| Draw priors (avg_elo, proximity) | Formula | 3 | Trivial | Compute |
| draw_rate_prior | Lookup table | 1 | Trivial | Pre-loaded table |
| Temporal (phase, ronde_norm) | From request context | 2 | Trivial | PredictRequest |
| Enrichment (categorie, K-coeff) | Static per player | 6 | Trivial | joueurs.parquet |
| Standings, zone_enjeu | Current season scores | 16 | DB query | MongoDB / parquet |
| Recent form W/D/L | Last 5 games per player | 10 | DB query | echiquiers history |
| Color perf W/D/L | Rolling 3 seasons | 16 | **Pre-computed** | Feature store |
| Board position | Rolling last season | 4 | **Pre-computed** | Feature store |
| Club behavior | Aggregated per team | ~16 | **Pre-computed** | Feature store |
| Club reliability | Aggregated per team | 6 | **Pre-computed** | Feature store |
| Player reliability | Aggregated per player | 4 | **Pre-computed** | Feature store |
| Noyau, FFE regulatory | Current season status | 10 | DB query | Noyau tracker |
| ALI presence, patterns | Selection history | 16 | **Pre-computed** | Feature store |
| Club level / vases | Inter-team hierarchy | 16 | **Pre-computed** | Feature store |
| Player team context | Promu/relegue/gap | 6 | **Pre-computed** | Feature store |
| Composition strategy | Historical board usage | 6 | **Pre-computed** | Feature store |
| Elo trajectory | Rolling window=6 | 4 | **Pre-computed** | Feature store |
| H2H | Past confrontations (sparse) | 8 | DB query | echiquiers history |
| Pressure/clutch | Historical (sparse) | 14 | **Pre-computed** | Feature store |
| match_important, adversaire | Derived from standings | 3 | Compute from standings | Derived |
| Domicile | From request | 1 | Trivial | PredictRequest |

**Summary: ~30 features trivial, ~50 from DB, ~115 need pre-computation.**

## Options

### Option A: Feature Store (recommended)

Pre-compute player/team features weekly (or after each round).
Store in Redis/Parquet/SQLite. At inference, join pre-computed features
to the matchup row.

```
Weekly job: FE pipeline → player_features.parquet + team_features.parquet
Inference:  join(request_features, player_store, team_store) → 196 cols → predict
```

**Pros:** Fast inference (~10ms), full feature set, same code as training FE.
**Cons:** Stale features (up to 1 week), storage + refresh job needed.

### Option B: Simplified Model

Train a lightweight model on ~20-30 features computable in real-time
(Elo, draw_prior, standings, recent form). Accept lower accuracy.

**Pros:** No feature store, simple architecture, fast.
**Cons:** Lower accuracy, two models to maintain, breaks V8 calibration.

### Option C: Batch Prediction per Round

Before each round, compute features for ALL possible matchups in the group.
Store predictions as a lookup table.

**Pros:** Most accurate (full features), simple inference (lookup).
**Cons:** Combinatorial explosion (~1000 matchups/group), heavy pre-compute,
stale if team changes last-minute.

## Recommendation

**Option A (Feature Store)** for V9 scope. Reasons:
- Reuses V8 FE code (proven, tested, 196 features)
- Inference latency acceptable (<50ms with parquet joins)
- Natural integration with `make refresh-data` pipeline
- Feature store doubles as drift monitoring baseline

## Artifacts from Kernel 2 needed for CE integration

| Artifact | Saved by | Needed for |
|----------|----------|------------|
| `{Model}_model.{ext}` | `save_models()` | `model.predict_proba(X)` |
| `calibrators.joblib` | `calibrate_models()` | `_apply_calibration(proba, calibrators[name])` |
| `encoders.joblib` | `save_models()` | `encoder.transform(X[cat_cols])` |
| `metadata.json` | `save_metadata_and_push()` | `best_model` name, feature list |
| `train_feature_distributions.csv` | `save_diagnostics()` | Drift monitoring baseline |

### Model loading at inference (per library)

```python
# CatBoost — direct
from catboost import CatBoostClassifier
model = CatBoostClassifier()
model.load_model("CatBoost_model.cbm")
proba = model.predict_proba(X)  # (n, 3): [P(loss), P(draw), P(win)]

# XGBoost — direct
from xgboost import XGBClassifier
model = XGBClassifier()
model.load_model("XGBoost_model.ubj")
proba = model.predict_proba(X)  # (n, 3)

# LightGBM — Booster API (saved via booster_.save_model)
import lightgbm as lgb
booster = lgb.Booster(model_file="LightGBM_model.txt")
proba = booster.predict(X)  # (n, 3) for multiclass
# NOTE: Booster.predict(), NOT LGBMClassifier.predict_proba()
```

### Probability mapping to CE

```python
# Model output: [P(loss=0), P(draw=1), P(win=2)]
proba = model.predict_proba(X_single)  # shape (1, 3)

# Apply calibration
from scripts.kaggle_diagnostics import _apply_calibration
proba_cal = _apply_calibration(proba, calibrators["CatBoost"])

# Map to CE BoardAssignment
assignment.loss_probability = float(proba_cal[0, 0])   # class 0
assignment.draw_probability = float(proba_cal[0, 1])   # class 1
assignment.win_probability  = float(proba_cal[0, 2])   # class 2
assignment.expected_score   = proba_cal[0, 2] + 0.5 * proba_cal[0, 1]
```

### Draw rate lookup table

`build_draw_rate_lookup()` in `scripts/features/draw_priors.py` builds the
(elo_band x diff_band) → draw_rate_prior table. This table is NOT currently
saved as a standalone artifact by Kernel 2. It needs to be:
- Either saved alongside the model artifacts
- Or reconstructed from training data at inference time

**Action needed:** Add `draw_rate_lookup.parquet` to Kernel 2 output.

## Dependencies

- **Blocked by:** V8 Kernel 2 completion (model artifacts)
- **Blocks:** Wiring routes→services (#2 in next_actions), ALI (#3), V9 CE
