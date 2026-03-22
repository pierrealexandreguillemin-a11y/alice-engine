# ALICE Engine — Production Roadmap Spec

**Date:** 2026-03-23
**Status:** APPROVED
**Scope:** From V8 training fix to production deployment (5 phases)
**Architecture:** Vercel (chess-app) → Oracle VM (FastAPI + ML) → HF Hub (model storage)

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
| Step 3 | AutoGluon on Step 1 or Step 2 | Validate with AutoML |

### Quality Gate (unchanged)

8 conditions: log_loss < naive, log_loss < Elo, RPS < naive, RPS < Elo,
Brier < naive, E[score] MAE < Elo, ECE < 0.05 per class, draw bias < 0.02.

### ISO Deliverables

| Norm | Artifact |
|------|----------|
| ISO 42001 | Model Card V8 (residual approach documented) |
| ISO 25059 | Quality gate results + baseline comparison |
| ISO 24029 | Robustness report (noise tolerance, stability) |
| ISO 24027 | Fairness report (per-group calibration) |
| ISO 5259 | Data lineage (same FE parquets, new training approach) |
| ISO 42005 | Impact assessment (model serves real decisions) |

### Outputs

- Model artifacts: `.cbm` / `.ubj` / `.txt` + `calibrators.joblib` + `encoders.joblib`
- `draw_rate_lookup.parquet` (45 cells, for inference)
- `metadata.json` (model card + quality gate)
- Pushed to HF Hub `Pierrax/alice-engine` if gate passes

---

## Phase 2: Feature Store + API Wiring

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

### Model Loading (per library)

```python
# CatBoost
model = CatBoostClassifier()
model.load_model("CatBoost_model.cbm")
proba = model.predict_proba(X)  # (n, 3): [P(loss), P(draw), P(win)]

# Probability mapping to CE:
assignment.loss_probability  = proba[0]  # class 0
assignment.draw_probability  = proba[1]  # class 1
assignment.win_probability   = proba[2]  # class 2
assignment.expected_score    = proba[2] + 0.5 * proba[1]
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

## Phase 3: ALI (Adversarial Lineup Inference)

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

## Phase 4: CE V9 Multi-Team (OR-Tools)

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

## Phase 5: Deploy + Monitoring

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

## Dependencies

```
Phase 1 (Training) ──────────────────────┐
  blocks everything                       │
                                          ▼
Phase 2 (Feature Store + API) ──────► Phase 3 (ALI)
  feature store + model loading           │
                                          ▼
                                    Phase 4 (CE V9)
                                          │
                                          ▼
                                    Phase 5 (Deploy)
```

Phase 1 is the lock. Everything else follows sequentially.
Only spec/ADR work can be done in parallel during Phase 1.
