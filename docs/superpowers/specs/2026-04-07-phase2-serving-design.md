# Phase 2 — Serving Architecture & Ensemble Evaluation

**Date:** 2026-04-07
**Status:** DRAFT — pending stacking evaluation
**Scope:** Evaluate ensemble stacking before wiring, then design serving layer
**Prerequisite:** V8 milestone complete (3 models converged, ALL PASS)

---

## 1. Audit Findings

### 1.1 Blending Evaluation Gap

The V8_MODEL_COMPARISON (Section 11) tested only **weighted average blending**
on calibrated probabilities. Grid search found optimal weights 90/5/5
(XGB/LGB/CB), yielding -0.02% vs XGBoost alone. Conclusion was "blending
adds nothing."

**Missing from state-of-the-art evaluation:**

| Method | Principle | Done? | Expected value |
|--------|-----------|-------|----------------|
| Weighted average | Sum of P(W/D/L) × weights | Yes | No gain (tested) |
| **Stacking (meta-learner)** | LogisticRegression on 9 cols (3 models × 3 classes) | **No** | Can learn non-linear complementarities |
| **Stacking + recalibration** | Isotonic on stacking output | **No** | Preserves calibration quality |
| ssBlending (Oxford 2025) | Stratified sampling blending | No | Minor variant, low priority |
| XStacking (Garouani 2025) | SHAP meta-features in stacking | No | Overkill for 3 base learners |

**Why stacking might help here (and weighted average didn't):**

The 3 models have **documented complementary strengths**:
- CatBoost v6: best calibration (ECE 0.0110 mean, draw bias 0.0127)
- LightGBM v7: best draw recall (0.5578)
- XGBoost v5: best log loss (0.5660) and F1 macro (0.6991)

A LogisticRegression meta-learner can learn to trust CatBoost's draw
probabilities, XGBoost's win/loss discrimination, and LightGBM's draw
sensitivity. Weighted average cannot — it applies uniform weights across
all classes.

**Why it might NOT help:**

Models are highly correlated: same features (197), same algorithm family
(gradient boosting), same residual learning (Elo init_scores, alpha=0.7).
Literature on chess outcome prediction with GBDT ensembles reports
typically < 1% gain from ensembling similar base learners (Karaaslan &
Erbay, MDPI Electronics 2025).

**Decision:** Must test before concluding. Stacking is a ~2h local experiment
with existing test predictions. No excuse to skip it.

### 1.2 Architecture — FTI Pattern Alignment

Industry standard (Hopsworks 2025): Feature/Training/Inference pipelines
share a feature store to prevent training-serving skew.

**Alice alignment:**

| FTI Component | Alice Implementation | Status |
|---------------|---------------------|--------|
| Feature Pipeline | `scripts/features/pipeline.py` → parquets | Exists (Kaggle) |
| Feature Store | Pre-computed parquets (joueur, equipe, draw_rate) | Exists (training) |
| Training Pipeline | `scripts/cloud/train_*.py` | Exists |
| Inference Pipeline | `services/inference.py` (stub) | **To build** |
| Shared transforms | `compute_differentials()` — pure, stateless | Exists (reusable) |
| Model Registry | `reports/v8_*/metadata.json` + HF Hub | Partial |

Key anti-skew mechanism: `compute_differentials()` is called identically
in both FE (training) and inference. No feature is computed differently.

### 1.3 ISO 42001 Serving Requirements

| Requirement | Implementation |
|-------------|---------------|
| Prediction traceability | Audit logger (MongoDB, ISO 27001) — exists |
| Model version in predictions | `metadata.json` version field — exists |
| Inference reproducibility | init_scores + alpha stored in metadata — exists |
| Input validation | Pydantic schemas — exists |
| Monitoring & drift | Phase 5 (not Phase 2 scope) |
| Model lineage | Dataset hash + training config in metadata — exists |

### 1.4 Service Architecture Decision

**Chosen: 3 services with stable interfaces (Approach B)**

```
POST /predict
  routes.py (controller — HTTP only, orchestration)
    │
    ├─→ ALIService.predict_opponents(club_id, ronde)
    │     → list[OpponentPrediction]
    │     Phase 2: fallback Elo. Phase 3: Monte Carlo ML.
    │
    ├─→ MLPredictor.predict_matchups(joueurs × adversaires, contexte)
    │     → matrice P(W/D/L) par (joueur, échiquier)
    │     Loads model(s), assembles features, computes init_scores.
    │     Supports 1 model (XGBoost) or N models (stacking).
    │
    └─→ ComposerService.optimize(joueurs, matrice P(W/D/L), contraintes)
          → CompositionResult
          Phase 2: greedy Elo-order. Phase 4: OR-Tools.
```

**Rationale for 3 services (not monolith):**
- SRP: each service has one reason to change
- ALI evolves independently (Phase 3 adds ML lineup prediction)
- MLPredictor evolves independently (stacking, retrain, model swap)
- CE evolves independently (Phase 4 adds OR-Tools multi-team)
- Testable in isolation with clear interfaces

**MLPredictor must support both single-model and ensemble from day 1.**
The stacking evaluation may change the serving strategy. Design the
interface to be model-count-agnostic:

```python
class MLPredictor:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Returns (n, 3) P(loss, draw, win) — regardless of 1 or 3 models."""
```

---

## 2. Intermediate Step: Stacking Evaluation

Before wiring, evaluate stacking on existing artifacts (no retraining needed).

### 2.1 Available Artifacts

All 3 models saved test & valid predictions:
- `reports/v8_xgboost_v5_resume/XGBoost_test_predictions.parquet`
- `reports/v8_xgboost_v5_resume/XGBoost_valid_predictions.parquet`
- Downloaded: `lightgbm-v7-output/`, `catboost-v6-output/`

Each contains columns: `pred_loss`, `pred_draw`, `pred_win`, `true_label`.

### 2.2 Stacking Protocol

1. **Train meta-learner on valid set** (70,647 samples):
   - Input: 9 columns (3 models × 3 proba classes)
   - Target: true_label (0=loss, 1=draw, 2=win)
   - Meta-learner: `LogisticRegression(multi_class='multinomial', max_iter=1000)`
   - Also test: `MLPClassifier(hidden_layer_sizes=(16,), max_iter=500)`

2. **Evaluate on test set** (231,532 samples):
   - Metrics: log_loss, RPS, E[score] MAE, ECE per class, draw calibration bias
   - Compare vs XGBoost alone, weighted average, and stacking

3. **Recalibrate stacking output**:
   - Apply isotonic calibration (same as individual models)
   - Renormalize probabilities to sum=1
   - Re-evaluate all metrics

4. **Decision gate:**
   - If stacking improves E[score] MAE by > 0.001 → serve 3 models + meta-learner
   - If not → serve XGBoost alone, document conclusion with evidence
   - E[score] MAE is the metric that matters for the CE (not log_loss)

### 2.3 Results (2026-04-07)

**DECISION: Stack_MLP_cal WINS — serve 3 models + MLP meta-learner.**

| Metric | XGBoost v5 | Stack_MLP_cal | Delta |
|--------|-----------|---------------|-------|
| **E[score] MAE** | 0.24739 | **0.24254** | **-0.00485 (-2.0%)** |
| log_loss | **0.56604** | 0.57335 | +0.00731 (+1.3%) |
| ECE draw | 0.01555 | **0.01233** | -0.00322 (-20.7%) |
| Draw bias | +0.01460 | **+0.01113** | -0.00347 (-23.8%) |

E[score] MAE gain = +0.00485 > 0.001 threshold = **SIGNIFICANT**.
Full results: `reports/stacking_evaluation.json`, `docs/project/V8_MODEL_COMPARISON.md` Section 11.
Script: `scripts/evaluate_stacking.py`.

### 2.4 Serving Implications (DECIDED)

| Stacking result | Serving strategy | Startup cost | Inference cost |
|----------------|-----------------|-------------|----------------|
| ~~No gain~~ | ~~XGBoost only (427 MB)~~ | ~~2s~~ | ~~1 predict call~~ |
| **Gain (chosen)** | **3 models + MLP meta-learner (~536 MB)** | **~5s** | **3 predict + MLP + isotonic** |

Oracle VM has 24 GB RAM — 536 MB fits comfortably.

---

## 3. Phase 2 Wiring Architecture (after stacking decision)

### 3.1 Service Interfaces

```python
# services/ali_service.py
class ALIService:
    """Predict opponent lineup. Phase 2: Elo fallback."""
    def predict_opponents(
        self,
        opponent_club_id: str,
        round_number: int,
        opponent_players: list[dict],
        team_size: int = 8,
    ) -> list[OpponentPrediction]: ...

# services/ml_predictor.py
class MLPredictor:
    """Predict P(W/D/L) for matchups. Supports single or ensemble."""
    def __init__(self, model_dir: Path, config: dict): ...
    def load(self) -> bool: ...
    def predict(self, features: pd.DataFrame) -> np.ndarray: ...
    # Returns (n, 3): [P(loss), P(draw), P(win)]

# services/composer.py (existing, enhanced)
class ComposerService:
    """Optimize lineup using ML probabilities instead of Elo formula."""
    def optimize(
        self,
        available_players: list[dict],
        predicted_opponents: list[OpponentPrediction],
        probabilities: np.ndarray,  # NEW: ML P(W/D/L) matrix
        constraints: dict | None = None,
    ) -> CompositionResult: ...
```

### 3.2 Feature Assembly for Inference

At inference, we need 197 features for each (player, opponent, board) tuple.

**Feature sources:**
- **Pre-computed (feature store):** joueur stats, equipe stats, draw rates,
  standings — refreshed weekly via `make refresh-data`
- **Computed at request time:** diff_elo, avg_elo, est_domicile, board number,
  ronde, saison, type_competition, differentials
- **From request:** player Elo, opponent Elo, constraints

**Assembly flow:**
```
Request(player, opponent, board, context)
  → Lookup pre-computed features from parquets
  → Merge player + opponent + team features
  → compute_differentials() (same code as training FE)
  → encode categoricals (encoders.joblib)
  → 197-column feature vector
  → compute_elo_init_scores() + *= alpha
  → predict_with_init(model, X, init_scores)
  → calibrate (temperature T or isotonic)
  → P(W/D/L)
```

### 3.3 Startup Sequence (main.py lifespan)

```
1. Load model(s) from disk or HF Hub
2. Load calibrators.joblib
3. Load encoders.joblib
4. Load draw_rate_lookup.parquet
5. Load feature store parquets (joueur, equipe, standings)
6. Load metadata.json (alpha, version, quality gate)
7. Validate: all artifacts present, versions match
8. Store in app.state → ready to serve
```

---

## 4. References

### Scientific Literature
- Karaaslan & Erbay (2025). "Machine Learning Approaches for Classifying
  Chess Game Outcomes." MDPI Electronics 15(1):1.
  https://www.mdpi.com/2079-9292/15/1/1
- Guo et al. (2017). "On Calibration of Modern Neural Networks." ICML.
  (Temperature scaling — used in our calibration pipeline)
- Ash & Adams (2020). "Warm-Starting Neural Network Training." NeurIPS.
  (Prior strength tuning — basis for init_score_alpha)

### Ensemble Methods
- scikit-learn StackingClassifier documentation.
  https://scikit-learn.org/stable/modules/ensemble.html
- scikit-learn CalibratedClassifierCV (isotonic, Platt).
  https://scikit-learn.org/stable/modules/calibration.html
- ssBlending (Oxford Bioinformatics Advances 2025).
  https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf002/8030212
- Garouani et al. (2025). XStacking with SHAP meta-features.
  https://www.emergentmind.com/topics/stacked-ensemble-learning

### ML Serving Architecture
- Hopsworks FTI Pipeline Architecture (2025).
  https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines
- Hopsworks Training-Inference Skew Prevention.
  https://www.hopsworks.ai/dictionary/training-inference-skew
- ML System Design Guide (2026).
  https://www.systemdesignhandbook.com/guides/ml-system-design/

### ISO Standards
- ISO/IEC 42001:2023 — AI Management System.
  https://www.iso.org/standard/42001
- ISO/IEC 42001 overview (KPMG).
  https://kpmg.com/ch/en/insights/artificial-intelligence/iso-iec-42001.html
- Feature Store infrastructure for ML (CORE Systems 2026).
  https://core.cz/en/blog/2026/feature-store-ml-infrastructure-2026/
