# Model Card — PreferenceModel 2024 (Phase 4a)

**Document ID** : ALICE-MODEL-CARD-PREFERENCE-2024
**Version** : 1.0.0
**Date** : 2026-05-27
**ISO 42001 §6** : AI management system model documentation
**ISO 42005** : AI system impact assessment
**ISO 23894** : Risk management AI (linked to R-ALI-06)

---

## 1. Model overview

- **Purpose** : Predict `P(player → team_rank | Elo, recency, streak, brule, historical_rank)`
  for opponent-side CE-adverse simulation (Phase 4a downstream consumer
  is `services.ali.adverse_ce.AdverseCESolver` objective function).
- **Type** : Bradley-Terry-Luce MAP estimation, implemented as multinomial
  `sklearn.linear_model.LogisticRegression(solver="lbfgs")` with L2 penalty
  `C = 1 / laplace_alpha` (Laplace prior).
- **Output** : Probability vector over `team_rank ∈ {0, 1, …, n_teams_max-1}`
  (16 ranks observed in saison 2024).
- **Architecture pattern** : Top-down ancestral sampling (Pearl 1988) downstream;
  this model scores `P(player → rank)` per Bayesian decomposition.
- **Determinism** : Fully reproducible from `(input_sha256, seed, laplace_alpha)`.

---

## 2. Training data

- **Source** : `data/echiquiers.parquet` (1,747,563 rows total).
- **Subset** : `saison == 2024` after `dropna(blanc_equipe, blanc_elo)` and
  `blanc_elo > 0` filter (excludes 8,853 unrated scolaires).
- **Final train size** : 81,843 rows.
- **Input SHA-256** : `a888b29df48afbc5a6eb4607319cd44c76bec8ba962dca5dd18683bdac8ff313`
  (computed by `PreferenceModel._sha256_dataframe` over feature subset).
- **Schema validation** : ISO 5259 via `pandera.DataFrameModel` (see
  `tests/services/ali/test_preference_model.py::EchiquiersSchema`).
- **Class distribution** :
  - rank 0: 15 254 (18.6%), rank 1: 14 150, rank 2: 14 200, rank 3: 13 570
  - rank 4: 7 411, rank 5: 5 387, rank 6: 5 471, rank 7: 5 248
  - rank 8–15: 130–182 each (sparse high-board observations)

---

## 3. Features

| Feature | Type | Provenance | Notes |
|---|---|---|---|
| `elo` | int | `blanc_elo` | Elo at match time (post-filter ≥ 1) |
| `recency_decay` | float | `0.9 ** (max_ronde - ronde)` | F2 recency, exponential decay |
| `streak_count` | int | constant 1 | **MVP placeholder** (F3 grouping deferred) |
| `brule_count` | int | constant 0 | **MVP placeholder** (A02 §3.7.c grouping deferred) |
| `historical_team_rank` | int | constant 0 | **MVP placeholder** — using observed `team_rank` would cause label leakage |

Target : `team_rank = echiquier - 1`.

---

## 4. Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| `laplace_alpha` | 1.0 | Bayesian default (Bishop 2006 §3.3) |
| `C` (sklearn inverse) | 1.0 | derived `1 / laplace_alpha` |
| `solver` | `"lbfgs"` | multinomial-default since sklearn 1.5 |
| `max_iter` | 2000 | empirical (1747-row real data needs > 500) |
| `random_state` | 42 | seed |
| sklearn version | 1.8.0 | (verified `multi_class` kwarg removed in 1.7) |

---

## 5. Performance metrics

Measured on training set (no held-out split at T2 ; acceptance evaluation
deferred to Plan 3 T9 per spec).

| Metric | Value | Threshold | Status |
|---|---|---|---|
| Log-loss (16-class) | 2.0161 | < 2.7726 (log 16 baseline) | PASS — beats uniform |
| Top-1 accuracy (training) | 0.2276 | ≥ 0.0625 (= 1/16 uniform) | PASS |
| Inference batch 1000 feats | 10.9 ms | < 100 ms | PASS (~10× under budget) |
| ECE per division | deferred to Plan 3 T9 | — | — |

**Per-rank accuracy** (training set, illustrates class imbalance) :

```
rank=0  (n=15254) acc=0.614    rank=1  (n=14150) acc=0.000
rank=2  (n=14200) acc=0.095    rank=3  (n=13570) acc=0.582
rank=4  (n= 7411) acc=0.000    rank=5  (n= 5387) acc=0.000
rank=6  (n= 5471) acc=0.000    rank=7  (n= 5248) acc=0.000
rank=8-15 sparse (130-182 each) acc=0.000
```

The bimodal `{0, 3}` accuracy reflects Elo separability between top boards
of N1/N2 (rank ≤ 3) and lower divisions — known MVP limitation, mitigated
Plan 3 T9 via stratified evaluation.

---

## 6. Limitations

1. **Sparse high-rank classes (8–15)** : < 200 samples each ; Laplace prior
   mitigates but cannot recover signal absent the data.
2. **MVP feature placeholders** : `streak_count`, `brule_count`, and
   `historical_team_rank` are constant — model trained without
   ground-truth history. Phase 4a+T enrichment will populate these via
   per-player rolling aggregation (see debt
   `D-2026-05-27-preference-model-mvp-features` in
   `memory/project_debt_current.md`).
3. **`historical_team_rank` label-leakage guard** : Plan code initially
   used `historical_team_rank = team_rank` (= target). Fixed inline at T2
   to constant 0 to avoid trivial 100% recall masking the bias gate.
4. **No held-out split at T2** : training metrics only. Calibration (ECE
   per division) and out-of-sample log-loss deferred to Plan 3 T9.
5. **Convergence warning** : `lbfgs` does not fully converge in 2000 iter
   on raw Elo scale (100–2761). Standardisation deferred (would invalidate
   `input_sha256` lineage). Predictions remain monotone in Elo per
   `test_fit_synthetic_recovers_signal`.
6. **Top-down composition assumption** : downstream CE-adverse solver
   assumes opponent fills teams top-down (Bayesian ancestral order).
   Strategic-sacrifice anti-patterns are out-of-scope MVP (see ADR-016).

---

## 7. Lineage (ISO 5259)

| Stage | Hash | Producer |
|---|---|---|
| Input (echiquiers.parquet filtered) | `a888b29df48afbc5a6eb4607319cd44c76bec8ba962dca5dd18683bdac8ff313` | `PreferenceModel._sha256_dataframe` |
| Artifact (joblib bytes) | `db096d354f215829c655171b5c6e7287f5d100bb7c95337be6b9035a0ce82836` | `PreferenceModel._sha256_estimator` |
| Pandera schema validation | `EchiquiersSchema` | `tests/services/ali/test_preference_model.py::TestSchema` |
| Artifact path | `models/preference_model_2024.joblib` | `scripts/train_preference_model.py` |

Reproduce :

```bash
PYTHONPATH=. .venv/Scripts/python scripts/train_preference_model.py \
    --saison 2024 --output models/preference_model_2024.joblib \
    --alpha 1.0 --seed 42
```

---

## 8. Ethical considerations & impact assessment

- **ISO 24027 bias gate** : per-`sexe` recall gap fail-fast threshold
  `BIAS_RECALL_GAP_THRESHOLD = 0.10`. For saison 2024 training data the
  parquet does not carry a `sexe` column at the row level, so the gate
  records `bias_gate_skipped=True` in the artifact. Phase 4a+T enrichment
  will join `joueurs.parquet` to expose `sexe` and re-trigger the gate.
- **ISO 42005 impact assessment** : this model's output drives the
  opponent-side CE-adverse simulation that ultimately influences user-facing
  composition recommendations. Mispredictions skew opponent strength
  estimates → suboptimal user board allocations. Impact category : medium
  (advisory, user retains final decision). Audit logged via input + artifact
  SHA-256.
- **ISO 23894 risk register link** : R-ALI-06 (top-down conditional
  multi-team adverse CE) — top-down ancestral sampling mitigates the
  strategic-coupling assumption violation. Residual risk : acceptable for
  Phase 4a MVP, re-evaluated Plan 3 T9.
- **Fairness** : sparse class imbalance (rank 8–15) means players assigned
  to lower boards in 2024 are under-represented in fit. Mitigation : Laplace
  prior + stratified eval Plan 3 T9.
