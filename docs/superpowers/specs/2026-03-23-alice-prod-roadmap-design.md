# ALICE Engine — Production Roadmap Spec

**Date:** 2026-03-23
**Status:** APPROVED
**Last updated:** 2026-04-17 (Phase 1 COMPLETE, Phase 2 next)
**Scope:** From V8 training fix to production deployment (5 phases)
**Architecture:** Vercel (chess-app) → Oracle VM (FastAPI + ML) → HF Hub (model storage)

---

## Status Overview (2026-04-17)

| Phase | Status | Key result |
|-------|--------|-----------|
| **Phase 1** | **COMPLETE** | Champion MLP(32,16) stacking 0.5530 ll, ECE_draw 0.0016 |
| Phase 2 | **NEXT** | API stubs exist, wiring pending |
| Phase 3 | Blocked by Phase 2 | ALI features computed, logic pending |
| Phase 4 | Blocked by Phase 3 | OR-Tools spec ready |
| Phase 5 | Blocked by Phase 4 | Oracle VM spec ready |

---

## Context

V8 Feature Engineering COMPLETE (196 cols, 1.44M rows, Kaggle kernel).
V8 Training FAILED quality gate: 3 models ignored draw class (recall=0%),
166/177 features unused. Root cause: no residual learning on Elo baseline.

This spec defines the path from fixing training to ALICE in production.

---

## Phase 1: Training Residual (beat Elo baseline)

### Principle

The Elo formula + draw_rate_prior captures ~92% of log_loss. Instead of
learning from scratch, the ML model starts from Elo predictions and learns
CORRECTIONS — when draws are more/less likely than Elo predicts.

```
Elo baseline:  P_elo(loss), P_elo(draw), P_elo(win)
                         ↓
               log-odds = log(P / (1-P)) per class
                         ↓
               init_score for each training sample
                         ↓
     CatBoost/XGBoost/LightGBM learn CORRECTIONS only
                         ↓
     Output: P(W/D/L) = Elo + ML corrections
```

### Implementation

1. `baselines.py`: new `compute_elo_init_scores(elo_proba)` → (n, 3) log-odds
2. `kaggle_trainers.py`: pass init scores to each model
   - CatBoost: baseline predictions via Pool or custom objective
   - XGBoost: `base_margin` parameter (n × 3 matrix)
   - LightGBM: `init_score` parameter
3. Validation incrémentale:

| Step | Features | Expected outcome |
|------|----------|-----------------|
| Step 0 | Init Elo + 0 features | Reproduce Elo baseline exactly (sanity) |
| Step 1 | Init Elo + top 10 (rates, form, draw) | Beat Elo by >0.01 log_loss |
| Step 2 | Init Elo + all 177 features | Measure if sparse features help |
| Step 3 | ~~AutoGluon on Step 1 or Step 2~~ (ELIMINE -- ADR-011) | ~~Validate with AutoML~~ |

### Quality Gate (unchanged)

9 conditions: log_loss < naive, log_loss < Elo, RPS < naive, RPS < Elo,
Brier < naive, E[score] MAE < Elo, ECE < 0.05 per class, draw bias < 0.02,
draw recall > 1% (model must actually predict draws, not ignore minority class).

### ISO Deliverables

| Norm | Artifact |
|------|----------|
| ISO 42001 | Model Card V8 (residual approach documented) |
| ISO 25059 | Quality gate results + baseline comparison |
| ISO 24029 | Robustness report (noise tolerance, stability) |
| ISO 24027 | Fairness report (per-group calibration) |
| ISO 5259 | Data lineage (same FE parquets, new training approach) |
| ISO 42005 | Impact assessment (model serves real decisions) |

### Outputs — DELIVERED (2026-04-17)

All artifacts on HF Hub `Pierrax/alice-engine/v9/`:
- `LightGBM.txt` (45 MB) — champion base model, 0.5619 single
- `XGBoost.ubj` (64 MB) — 0.5622 single
- `CatBoost.cbm` (17 MB) — 0.5708 single
- `champion_dirichlet_lgb.joblib` — Dirichlet calibrator (Kull 2019)
- `encoders.joblib` — feature encoding
- `draw_rate_lookup.parquet` — init_scores inference
- `*_metadata.json` — model cards per model

**V9 Results:**
- 3 models V9 Training Final: LGB 0.5619, XGB 0.5622, CB 0.5708 (all T1-T12 PASS)
- OOF 5-fold XGB+LGB+CB: 15 kernels total
- Stacking MLP(32,16) 18f + temperature: **0.5530 log_loss, 0.0016 ECE_draw**
- AutoGluon: ELIMINATED (ADR-011, test 0.5716 > V9 LGB 0.5619)
- Kuncheva Q=0.997: 3 GBMs near-identical, stacking gain = recalibration
- 24+ experiments sweep: architecture/features/calibration (documented)

**Quality gates T1-T16: ALL PASS.**
Detail: `config/MODEL_SPECS.md`, `docs/requirements/QUALITY_GATES.md`

---

## Phase 2: Feature Store + API Wiring — STATUS: NEXT

> **Existing code:** `app/` (FastAPI stubs COMPLET), `services/` (stubs),
> `app/api/routes.py` (endpoints), `app/api/schemas.py` (Pydantic models).
> **Missing:** wiring routes → services → inference, feature store assembly,
> model loading from HF Hub at startup.
>
> **Champion model update:** The inference pipeline must support the MLP stacking
> meta-learner (not just single model predict). Flow:
> `features → 3 GBM predict → 18 meta-features → MLP → temp scaling → P(W/D/L)`
>
> **Architecture change (2026-04-15):** batch ML hebdo + CE on-demand <2s.
> Predictions are PRE-COMPUTED, not real-time. CE uses cached probas.
> See `memory/project_batch_architecture.md`.

### Feature Store

Pre-computed player/team stats, refreshed weekly (or after each round).

```
Refresh pipeline (cron or make refresh-features):
  echiquiers.parquet → FE pipeline (same code as Kaggle)
    → joueur_features.parquet  (~80k joueurs × ~40 cols)
    → equipe_features.parquet  (~23k équipes × ~30 cols)
    → draw_rate_lookup.parquet (from Kernel 2 output)
    → standings_current.parquet (current season positions)
```

At inference: assemble 196 cols by lookup + join, not recalculation.
Target: <50ms assembly time for a single matchup.

### API Wiring

```
POST /api/v1/predict
  ├── InferenceService.load_model()
  │     HF Hub → model.cbm + calibrators.joblib + encoders.joblib
  │     Loaded ONCE at startup, kept in memory
  ├── ALI: predict_lineup() → fallback Elo (Phase 2), ML (Phase 3)
  │     → predicted_opponents per board
  ├── For each (player × board × opponent):
  │     feature_store.assemble(player, opponent, context) → 196 cols
  │     model.predict_proba() → calibrate → P(W/D/L)
  └── CE: optimize()
        → E[score] = P(win) + 0.5×P(draw) → optimal lineup
```

### Model Loading + Inference (stacking pipeline)

```python
# At startup: load 3 GBMs + MLP meta-learner + calibrators from HF Hub
lgb_model = lightgbm.Booster(model_file="v9/LightGBM.txt")
xgb_model = xgboost.Booster(model_file="v9/XGBoost.ubj")
cb_model = CatBoostClassifier().load_model("v9/CatBoost.cbm")
mlp_meta = joblib.load("meta_learner/mlp_meta_learner.joblib")  # MLP(32,16)
T_scaling = joblib.load("meta_learner/temperature_T.joblib")     # T=1.02

# Batch predict (weekly, pre-compute all matchups):
# 1. Feature assembly from feature store
X = feature_store.assemble(player, opponent, context)  # 201 cols

# 2. Init scores (Elo residual)
init_scores = compute_elo_baseline(blanc_elo, noir_elo) * alpha_per_model

# 3. Three GBM predictions
p_xgb = predict_with_init(xgb_model, X, init_scores * 0.5)  # (n, 3)
p_lgb = predict_with_init(lgb_model, X, init_scores * 0.1)  # (n, 3)
p_cb  = predict_with_init(cb_model,  X, init_scores * 0.3)  # (n, 3)

# 4. Meta-features (18 = 9 probas + 9 engineered)
X_meta = build_meta_features(p_xgb, p_lgb, p_cb)  # std, max, entropy

# 5. MLP stacking + temperature scaling
p_raw = mlp_meta.predict_proba(X_meta)
p_final = apply_temperature(p_raw, T_scaling)

# 6. Output to CE cache
assignment.loss_probability  = p_final[0]  # class 0
assignment.draw_probability  = p_final[1]  # class 1
assignment.win_probability   = p_final[2]  # class 2
assignment.expected_score    = p_final[2] + 0.5 * p_final[1]
```

### ISO Deliverables

| Norm | Artifact |
|------|----------|
| ISO 42001 | Model Card updated (serving endpoint documented) |
| ISO 5259 | Feature Store lineage (hash, date, coverage) |
| ISO 27034 | Input validation (Pydantic schemas, rate limiting) |
| ISO 25059 | Serving metrics (latency P50/P99, throughput) |
| ISO 42010 | ADR-002 implemented (feature store architecture) |
| ISO 29119 | Integration tests (POST /predict E2E with real model) |

---

## Phase 3: ALI (Adversarial Lineup Inference) — STATUS: BLOCKED (Phase 2)

> **Existing code:** ALI features computed in FE pipeline (ali_presence,
> ali_patterns, ali_absence). Logic pending.
> **Research finding (2026-04-16):** copule inter-boards identifiee comme
> optimisation Phase 4 (correlation entre boards dans un match). Source:
> pair-copula construction (Springer 2023), same-game parlays (Wizard of Odds).

### Principle

ALI predicts WHO plays for the opponent (not the game outcome).
Compositions are submitted simultaneously (A02 Art. 3.6.a) — the captain
does NOT know the opponent's lineup. ALI is a probabilistic estimate.

### Monte Carlo on Historical Patterns

1. ALI features exist (ali_presence, ali_patterns, ali_absence in FE output)
2. `taux_presence_saison` = P(player selected)
3. `echiquier_prefere` + `flexibilite` = board distribution
4. `generate_scenarios(N=20)`: draw 20 probable lineups weighted by selection proba
5. For EACH scenario: ML predicts P(W/D/L) per board
6. CE optimizes on the WEIGHTED AVERAGE across scenarios

```
20 scenarios × 8 boards × ~10 candidate players = ~1600 ML predictions
Single predict_proba batch → ~10ms
```

### ISO Deliverables

| Norm | Artifact |
|------|----------|
| ISO 42001 | ALI model card (features, limitations, uncertainty) |
| ISO 42005 | Impact assessment ALI (risk of wrong opponent prediction) |
| ISO 24027 | Fairness ALI (no club size/wealth bias) |
| ISO 29119 | Tests: scenarios coherent, sum probas = 1 |
| ISO 23894 | Risk: captain overconfident in prediction |

---

## Phase 4: CE V9 Multi-Team (OR-Tools) — STATUS: BLOCKED (Phase 3)

> **Research findings (2026-04-16):**
> - Predict-then-Optimize (PtO) = architecture actuelle. Validee par
>   arxiv:2502.02861: calibration error bounds downstream regret.
>   ECE_draw 0.0016 → regret borne par 1 + 2×0.0016 ≈ 1.003.
> - Decision-Focused Learning (DFL) = Phase 5+ alternative (NeurIPS 2024).
>   Prerequis: CE fonctionnel pour differencier a travers.
> - Conformal prediction = uncertainty quantification pour mode risk-adjusted.
>   Prerequis: CE qui exploite les prediction sets.
> - Copules inter-boards: correlation 30-50% entre boards meme match.
>   Simplification actuelle: boards independants (documente).
> Detail: `config/MODEL_SPECS.md` §Paradigme PtO vs DFL.

### Problem

A club has N teams playing the SAME weekend. The captain distributes
30-50 players across all teams under FFE constraints.

```
Input:
  - 40 available players
  - 3 teams (N1, N3, Régionale)
  - 20 opponent scenarios per team (from ALI)
  - Strategy mode chosen by captain
  - FFE constraints

Output:
  - Optimal player → team → board allocation
  - E[score] per team + total
  - Alternatives with explained trade-offs
```

### OR-Tools LP/CP

```
Variables:  x[player, team, board] ∈ {0, 1}
Objective:  depends on strategy mode
Constraints:
  - 1 player = 1 team only (∑ teams x[j,e,b] ≤ 1)
  - 1 board = 1 player (∑ players x[j,e,b] = 1)
  - Elo ordering per board (A02 3.6.e, 100pt margin)
  - Locked core players (A02 3.7.f noyau)
  - Max 3 mutés per team per season (A02 3.7.g)
  - Demotion allowed, promotion conditional
```

### Strategy Modes

| Mode | OR-Tools Objective | Use Case |
|------|-------------------|----------|
| Aggressive | Max Σ E[score] priority team | Promotion/title |
| Conservative | Max min(E[score]) all teams (maximin) | Leisure club |
| Tactical round | Max P(match win) for team in danger zone | Relegation |
| Risk-adjusted | Max E[score] - λ×Var[score] | P(draw) critical here |

Risk-adjusted mode is the raison d'être of V8 MultiClass: two compositions
with same E[score] but different variance thanks to P(draw).

### ISO Deliverables

| Norm | Artifact |
|------|----------|
| ISO 42001 | CE V9 model card (solver, modes, constraints) |
| ISO 42005 | Impact assessment CE (relegation risk on bad allocation) |
| ISO 25010 | Solver performance (<2s for 50 players × 3 teams) |
| ISO 29119 | Tests: all FFE constraints respected, solutions valid |
| ISO 5259 | Noyau/mutés data validation (official FFE source) |
| ISO 42010 | CE V9 architecture in docs/architecture/ |

---

## Phase 5: Deploy + Monitoring — STATUS: BLOCKED (Phase 4)

> **Architecture change (2026-04-15):** batch ML hebdo + CE on-demand <2s.
> Model artifacts on HF Hub `Pierrax/alice-engine/v9/`. Oracle VM downloads
> at startup. Stacking pipeline (3 GBMs + MLP) fits in 24GB ARM.
> **Model update:** champion = stacking MLP, not single CatBoost.
> Inference: 3× GBM predict + MLP + temp scaling ≈ 20ms batch.

### Production Architecture

```
HF Hub (storage)              Kaggle (training)
  model .cbm ←──push────────── training kernel
       │
       │ download at startup
       ▼
Oracle VM (Always Free)        Vercel (chess-app Next.js)
  4 OCPUs ARM, 24 GB RAM        app/api/alice/predict/route.ts
  FastAPI + CatBoost in RAM  ←── fetch() HTTPS ←── User browser
  Feature store (parquet)
  Inference ~10ms
  HTTPS (Let's Encrypt)
  Always on (no cold start)
```

### Why Oracle VM (not Render, not Vercel)

| Criterion | Oracle VM | Render Free | Vercel Hobby |
|-----------|-----------|-------------|-------------|
| RAM | **24 GB** | 512 MB | 2 GB (stateless) |
| Always on | **Yes** | No (sleep 15 min) | No (serverless) |
| Model persistence | **In memory** | Reload on wake | Reload each invoc |
| Cost | **Free** | Free (limited) | Free (limited) |
| Control | **Full** (SSH, cron) | Limited | Serverless only |
| Cold start | **None** | ~30s | 1-5s |
| Python packages | **Any** | Any | 250 MB bundle limit |

### Vercel → Oracle VM Communication

```typescript
// chess-app/app/api/alice/predict/route.ts
export async function POST(request: Request) {
  const body = await request.json();
  const res = await fetch('https://alice.your-domain.com/api/v1/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return Response.json(await res.json());
}
```

- Server-to-server: no CORS issues
- Vercel timeout 60s, ALICE responds in <5s
- HTTPS via Let's Encrypt on Oracle VM

### Refresh Pipeline (automated)

```
Weekly (cron on Oracle VM):
  1. scrape.py --refresh-season 2026      ← new FFE rounds
  2. parse_dataset → echiquiers.parquet   ← update data
  3. FE pipeline → feature store refresh  ← player/team stats
  4. Drift check (PSI per-class)          ← alert if shift

Monthly (or if drift detected):
  5. Re-training (Kaggle or HF Jobs)
  6. Quality gate 8 conditions
  7. If pass → promote model on HF Hub
  8. Oracle VM pulls new model (systemd restart or hot-reload)
```

### Drift Monitoring

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| PSI P(loss) | ≥ 0.1 | ≥ 0.25 | Re-training |
| PSI P(draw) | ≥ 0.1 | ≥ 0.25 | Re-training |
| PSI P(win) | ≥ 0.1 | ≥ 0.25 | Re-training |
| Log loss drift | ≥ 5% | ≥ 10% | Urgent re-training |
| Draw rate drift | ≥ 3% abs | ≥ 5% | Verify FFE data |
| Elo shift | ≥ 50 pts | — | Normal inter-season |
| Feature store age | > 7 days | > 14 days | Force refresh |

### ISO Deliverables

| Norm | Artifact |
|------|----------|
| ISO 42001 | Deployment record, rollback procedure |
| ISO 27001 | API key auth, HTTPS, rate limiting, audit logs |
| ISO 5259 | Drift monitoring dashboard, PSI reports |
| ISO 23894 | Risk register prod (downtime, stale model, bad composition) |
| ISO 42005 | Impact assessment prod (real users) |
| ISO 25059 | ISO 25059 Report FINAL (all production metrics) |
| ISO 15289 | Operations manual updated |

---

## Dependencies (updated 2026-04-17)

```
Phase 1 (Training) ─── COMPLETE ──────────┐
  Champion: MLP stacking 0.5530           │
  Models on HF Hub v9/                    ▼
Phase 2 (Feature Store + API) ──────► Phase 3 (ALI)
  *** CURRENT ***                         │
  API stubs exist, wiring pending         ▼
                                    Phase 4 (CE V9)
                                          │
                                          ▼
                                    Phase 5 (Deploy)
```

**Phase 1 lock LIFTED.** Phase 2 can proceed immediately.
Phase 2 deliverables: feature store assembly, model loading from HF,
inference pipeline (3 GBMs → MLP stacking → temp scaling → P(W/D/L)),
integration tests, batch predict cron.
