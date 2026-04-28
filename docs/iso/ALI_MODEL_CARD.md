# ALI Model Card — Adversarial Lineup Inference (Phase 3 SOTA)

**Document ID** : ALICE-MODEL-CARD-ALI-v1.0.0
**ISO Standards** : ISO/IEC 42001:2023 (AI Management System) + ISO/IEC 25059:2023
(AI Quality Model) + ISO/IEC 24027:2021 (AI Bias) + ISO/IEC 24029:2021 (AI Robustness)
**Format** : Mitchell et al. 2019 "Model Cards for Model Reporting" (FAccT 220–229)
**Status** : DRAFT post-conformity-audit 2026-04-28 (T20 Plan 3 V2 + G1-G5 fix-on-sight).
Quantitative values populated by T22 Gates Report. Pending JALON #3 review.
**Generated** : 2026-04-28
**Authors** : Pierre-Alexandre Guillemin + Claude (LLM co-authorship, see ISO 42001
disclosure `docs/iso/AI_DEVELOPMENT_DISCLOSURE.md`).

---

## 1. Model Details

### 1.1 Identification

- **System name** : ALI (Adversarial Lineup Inference) — Phase 3 SOTA
- **System version** : `0.3.0-phase3-plan3` (commit pinned in `lineage_hash` per request)
- **Type** : Hybrid Monte Carlo + deterministic enumeration system for opponent
  team composition prediction in chess interclub matches
- **Domain** : Chess (FFE — Fédération Française des Échecs)
- **License** : Proprietary (single-author project, see Phase 7 monetization roadmap)

### 1.2 Architecture

ALI is a **hybrid 10 + 10 generator** producing 20 weighted scenarios per match :

```
opponent_club_id + round_date + competition_context
    │
    ▼
PlayerPoolLoader            → eligible roster (F7 survivor filter)
    │
    ▼
HistoryEnricher              → F2 recency decay λ=0.9 + F3 streak lag 1-3
    │
    ▼
CopulaJointSampler.fit       → Spearman rank correlation matrix on co-presence
    │
    ▼
VerifiabilityClassifier      → partition (10 PUBLIC, 4 PRIVATE FFE A02 rules)
    │
    ▼
TopKEnumerator(10)           +  MonteCarloSampler(5 LHS pairs × 2 antithetic)
    │                                       │
    └──────── merge + dedup ────────────────┘
                    │
                    ▼
ScenarioSet(20 scenarios pondérés, lineage_hash SHA-256, generated_at UTC)
```

Each scenario specifies a complete `Lineup(team_size, BoardAssignment[])` with
weights normalized to sum = 1. The downstream `StackingInferenceService`
consumes each (scenario × board) to produce P(W/D/L) per board, aggregated by
weighted expectation in `services/ali/aggregation.py`.

### 1.3 Components & Lines

| Module | Path | Lines | Role |
|--------|------|------:|------|
| Generator | `services/ali/generator.py` | ~330 | Orchestrator |
| Topk | `services/ali/topk.py` | ~150 | Branch-and-bound deterministic enumeration |
| Monte Carlo | `services/ali/monte_carlo.py` | 254 | LHS + antithetic + copula inverse-transform |
| Copula | `services/ali/joint_sampler.py` | ~200 | Gaussian copula (Sklar 1959) |
| History | `services/ali/history.py` | ~180 | F2 recency + F3 streak features |
| Pool loader | `services/ali/pool_loader.py` | ~120 | F7 survivor filter |
| Cache | `services/ali/cache.py` | 152 | In-RAM parquets + SHA-256 lineage |
| Verifiability | `services/ali/verifiability.py` | ~80 | PUBLIC / PRIVATE rule partition |
| Rule engine | `services/ffe/rule_engine.py` | ~250 | JSON-driven FFE A02 rules |
| Aggregation | `services/ali/aggregation.py` | ~270 | Per-board weighted P(W/D/L) |
| Inference | `services/inference.py` | 293 | Stacking ML pipeline |

All modules respect ISO 5055 ≤ 300 lines/file, complexity ≤ B (xenon),
ruff + mypy --strict clean.

### 1.4 ML Sub-System (StackingInferenceService)

ALI delegates per-board P(W/D/L) prediction to the ML stacking pipeline :

```
3 GBMs (LGB α=0.1, XGB α=0.5, CB α=0.3) with init_score = compute_elo_baseline
    │
    ▼
build_meta_features (18 cols : per-model probas + Elo features)
    │
    ▼
MLP(32, 16) champion (V9 stacking, sklearn 1.6.1)
    │
    ▼
Temperature scaling τ (Guo 2017 calibration)
    │
    ▼
PredictionResult(p_loss, p_draw, p_win, e_score = p_win + 0.5 × p_draw)
```

**Champion metrics V9** (training set, see `docs/project/V9_HP_SEARCH_RESULTS.md`) :
- Multiclass log loss test : **0.5530**
- ECE_draw : **0.0016** (well-calibrated, Plan 3 P3G10 threshold ≤ 0.05)
- Draw bias : within ± 1% (Tier 2 Dirichlet calibration, Kull 2019)

---

## 2. Intended Use

### 2.1 Primary Users

- **Chess club captains** (FFE registered clubs, ~989 active in 989 league
  divisions N1/N2/N3/N4/Régionale)
- Use case : decide which players to field (lineup composition) for an
  upcoming interclub round, given an opponent club identifier

### 2.2 Use Cases In Scope

1. **Adversary prediction** : `POST /api/v1/compose` with `opponent_club_id`
   → 20 weighted scenarios of opponent lineup
2. **Recomposition** : `POST /api/v1/recompose` after late player unavailability
   → re-aggregate ML predictions on updated user lineup
3. **Audit trail** : every prediction tagged with `lineage_hash` SHA-256 covering
   parquets + rules + λ + seed (ISO 5259 reproducibility)

### 2.3 Decision Latency Target

- Phase 3 SOTA in-RAM cache : ALI generate ~200 ms, ML inference 20×K boards
  ~800 ms, CE fallback ~20 ms. **Total p50 ≈ 1.0 s, p95 ≤ 1.5 s** (target).
- Phase 5 deploy on Oracle VM ARM 24 GB : capacity benchmark in
  `docs/operations/ALI_SLO.md` (TBD Phase 5).

---

## 3. Out-of-Scope (Explicit)

ALI **is not** :

- A real-time decision system (sub-second). Use case is **pre-match
  preparation** (hours/days ahead).
- A youth competition (J02) predictor — `joueurs.parquet` lacks birth dates
  (D3 deferred Phase 3.5).
- A cup competitions (Coupes fédérales) predictor — only A02 (championnats
  interclubs classiques) implemented (D4 deferred Phase 3.5).
- A multi-team optimization engine — Phase 3 CE = simple Elo-descending sort
  per team. **Phase 4 OR-Tools** (CP-SAT) will introduce true multi-team
  joint optimization with FFE constraints (brûlé 3.7.c, noyau 3.7.f, mutes
  3.7.g, foreign quota 3.7.h).
- A deterministic single-answer system. By design, returns 20 scenarios with
  weights to expose epistemic uncertainty.
- A general game-outcome predictor for chess. Trained for **interclub team
  compositions** specifically.

---

## 4. Factors (Mitchell 2019 §3)

Prediction-relevant factors that may shape ALI behavior. Documented per
Mitchell 2019 to flag dimensions on which performance should be analyzed
disaggregated.

### 3.1 Groups (relevant subpopulations)

| Group axis | Categories | Why relevant |
|------------|-----------|--------------|
| **Competition level** | N1, N2, N3, N4, Régionale | Top divisions have stable rosters and strict A02 rules ; lower divisions have higher player turnover and laxer noyau enforcement. Predictive accuracy expected to differ. |
| **Club size** | Quartiles of player pool size | Small clubs (≤ 25 players) have less historical co-presence data → copula fit on sparse samples → lower confidence. |
| **Player gender** | M, F, mixed teams | FFE A02 §3.7.i mandates ≥ 1 female on N1/N2 teams. Female-mandated lineups have less rotation latitude → distinct presence dynamics. |
| **Player category** | Senior (SE), Vétéran (VE), Junior (JU)... | Younger players rotate more. ALI does not currently model `categorie` directly (D3 deferred Phase 3.5). |
| **Player muté status** | Native (`mute=False`) vs muté | A02 §3.7.g caps mutés ≤ 3 per lineup → constraints differ. |
| **Foreign quota** | FR/UE ≥ 5 per A02 §3.7.h | Constrains pool selection differently in international clubs. |

### 3.2 Instrumentation (data acquisition)

ALI consumes two parquet datasets from the FFE public website, syncronised
via HuggingFace Hub mirror (`Pierrax/ffe-history`) :

- **Acquisition cadence** : weekly batch sync (`make refresh-data`)
- **Provenance** : FFE official rankings + interclub results (no scraping
  intermediaries, public federation export)
- **Known instrumentation biases** :
  - Historical seasons 2002-2010 have sparser coverage (early FFE
    digitization)
  - Forfeits (`type_resultat="forfait_blanc/noir"`) appear as empty
    `blanc_nom`/`noir_nom` strings — filtered by ground_truth extraction
  - Player names use FFE convention "NOM Prenom" — no normalization for
    homonyms (D-P3-06 deferred Phase 5)

### 3.3 Environments (deployment context)

Performance characterized in the following environments :

| Environment | Status | Latency target | Notes |
|-------------|--------|---------------|-------|
| **Laptop dev** (Windows + Python 3.13) | Validated | p50 ≈ 1.0 s (warm cache) | Phase 3 backtest baseline |
| **CI (GitHub Actions)** | Validated | tests in ~5 min | pytest + ruff + mypy + xenon |
| **Oracle VM ARM 24 GB** (Phase 5) | TBD | p50 ≤ 800 ms target | Capacity benchmark deferred |
| **High-load multi-tenant** (Phase 5+) | TBD | RPS ≥ 3 sustainable | Multi-club concurrent requests |

Distributional shift between environments is **expected to be minimal**
because data fetched from same FFE source ; latency may differ.

### 3.4 Factor Cross-Tabulation Plan

Phase 3.5 STRICT (D8) will produce per-group breakdown of metrics
T13-T17 stratified by competition level × club size quartile × gender —
3D table in `docs/iso/ALI_FAIRNESS_VALIDATION.md`.

---

## 5. Training Data

### 4.1 Sources

- **`data/joueurs.parquet`** : 95,851 FFE players × 989 clubs, columns include
  `nr_ffe`, `nom`, `prenom`, `elo`, `club`, `categorie`, `genre`, `mute`
- **`data/echiquiers.parquet`** : ~2 million board-level records, seasons
  2002-2026, rounds 1-19, columns include `saison`, `ronde`, `equipe_dom`,
  `equipe_ext`, `echiquier`, `blanc_nom/elo/equipe`, `noir_nom/elo/equipe`,
  `resultat_blanc`, `type_resultat`
- **Source** : FFE public data (HuggingFace `Pierrax/ffe-history`)
- **Refresh policy** : weekly batch sync (`make refresh-data`)

### 4.2 Lineage & Reproducibility

- SHA-256 hash of each parquet stored in `ALIDataCache.parquet_sig_*` at
  load time (ISO 5259)
- `ScenarioSet.lineage_hash` = SHA-256(parquets + rules + λ + n_topk + n_mc
  + seed + opponent_club_id + round_date + saison + ronde + nb_rondes_total)
- Two `/compose` calls with identical inputs produce identical `lineage_hash`
  (audit-stable, default `seed=42`, ADR-014 §Determinism)

### 4.3 Pre-processing

- `team_to_club` mapping built by majority vote (deterministic tiebreak :
  count DESC, club name ASC) — H1 fix in `cache.py::_build_team_to_club`
- F2 recency : `taux_presence_effectif = Σ_r λ^(age_r) × 1[present_r] / Σ_r λ^(age_r)`,
  λ = 0.9 (Brown 1959 exponential smoothing)
- F3 streak : `played_lag1`, `played_lag2`, `played_lag3` booleans (Box &
  Jenkins 1970)
- F7 survivor : currently identity (`joueurs.parquet` contains only active
  FFE licenses by construction). Override via `overrides[*].licence_active`
  for hypothetical scenarios.

### 4.4 Sample Sizes

- ML training (V9 champion) : **1,139,819 train + 70,647 valid + 197,843
  test** binary records (board game outcomes Win/Loss/Draw)
- ALI evaluation backtest : hold-out **season 2024**, target ≥ 100 matches
  for Bergmeir & Benítez 2012 walk-forward statistical significance (T22
  Gates Report)

### 4.5 Data Quality Dimensions (ISO/IEC 5259:2024)

ISO 5259 mandates explicit characterization of data quality across five
dimensions. Documented here for `joueurs.parquet` + `echiquiers.parquet` :

| Dimension | Rating | Evidence | Mitigation if degraded |
|-----------|--------|----------|------------------------|
| **Completeness** | High | 95,851 active FFE players covered, ~2M boards 2002-2026, no missing seasons within range | None needed |
| **Accuracy** | High | Source = FFE official federation export, no third-party intermediaries. Elo values cross-checked against monthly FFE bulletin | Pandera schema rejects malformed rows pre-commit hook (D-P3-07 schema FFE A02 deferred Phase 5) |
| **Consistency** | High | `team_to_club` majority vote with deterministic tiebreak (count DESC then alpha ASC, ISO 5259 reproducibility). FFE A02 §3.6 color alternance invariant validated by `_validate_ffe_color_invariant` ground_truth.py (D-P3-14) | `FFEDataQualityError` raised on violation, fail-fast |
| **Timeliness** | Medium | Weekly batch sync acceptable for pre-match preparation use case. Real-time updates not target. Stale > 7 days flagged by `ALIDataCache.is_stale()` | Health endpoint exposes cache age (Phase 5 §4.15.5) |
| **Validity** | High | Schema validation : Pydantic for API inputs (FFE ID format `A12345`), Pandera for parquet inputs (`scripts/backtest/schemas.py`), JSON Schema for FFE rules (`config/ffe_rules/a02.json`) | Reject invalid inputs at boundary (ISO 27034) |

**Documented data quality issues** :

- Forfeits (`type_resultat="forfait_*"`) produce empty `blanc_nom`/`noir_nom`
  → handled by `_extract_players` filter (`name.strip() == ""`), tested
  T17.5 with real Aix-Les Bains/Meximieux 2024 R1 fixture
- "Got 19" issue : when adversary pool is too small (< ~12 distinct
  lineups feasible), `_merge_and_pad` exhausts retries → `validate()` raise
  → backtest `BacktestRunner` skips with `skip_failed_matches=True`
  (D-P2-03 documented ADR-014)

---

## 6. Quantitative Analyses

### 5.1 Backtest Protocol (Walk-Forward)

- **Train / tuning** : seasons 2021-2023
- **Hold-out validation** : season 2024 (gates T13-T22 verts exigés)
- **Test final** : season 2025 (untouched in development, future)

Following Bergmeir & Benítez 2012, walk-forward evaluation is used (no
cross-validation across time, no leakage). The `BacktestRunner`
(`scripts/backtest/runner.py`) implements rolling multi-season pattern
similar to CARMELO (sports prediction).

### 5.2 Quality Gates (P3G07-P3G11)

| Gate | Metric | Threshold | Direction | Source |
|------|--------|-----------|-----------|--------|
| P3G07 | Top-K recall (union 20 scenarios) | ≥ 0.90 | ge | Spec Phase 3 §6.2 T13 |
| P3G07b | Accuracy@K (top weighted scenario) | ≥ 0.75 | ge | Pappalardo 2019 sports SOTA |
| P3G08 | Jaccard max | ≥ 0.75 | ge | Spec Phase 3 §6.2 T14 |
| P3G09a | Brier score on P(presence) | ≤ 0.20 | le | Spec Phase 3 §6.2 T15 |
| P3G09b | Brier skill score vs Elo baseline | ≥ 0.05 | ge | Pappalardo 2019 |
| P3G10 | ECE 10-bins on P(presence) | ≤ 0.05 | le | Guo 2017 (strict ALICE) |
| P3G11a | E[score] MAE team_size=8 | ≤ 1.0 | le | Spec Phase 3 §6.2 T17 |
| P3G11b | McNemar p-value (ALI vs baseline) | < 0.05 | lt | Plan 3 V2 T22 |

Statistical rigor : each metric reported with **bootstrap BCa 95% CI**
(Efron 1987, n_resamples=1000, seed=42). Gate PASS if `ci.lower ≥ threshold`
(direction `ge`) or `ci.upper ≤ threshold` (direction `le`) — point estimate
alone is insufficient.

### 5.3 Numerical Results

**Status : populated by T22 Gates Report** (`docs/iso/ALI_QUALITY_GATES_REPORT.md`).
Mitchell 2019 explicitly allows Model Card describing protocol and
characteristics independently of one specific evaluation run.

Smoke fairness (T13) and smoke robustness (T14) executed on small samples
during Plan 3 development pass all 7 P3G gates exposed by
`BacktestReport.gates_summary()`.

### 5.4 McNemar Paired Test (T22)

ALI vs baseline Elo (1-scenario top-K Elo descending) :
- Test : two-sided exact binomial if `b + c < 25`, else Yates-corrected χ²
  (Edwards 1948)
- Definition "correct" per match : `recall_ali ≥ 0.90` (T13 threshold)
- Reference : McNemar Q. 1947 Psychometrika 12(2)
- Implementation : `scripts/backtest/statistical.py::mcnemar_paired`

---

## 7. Fairness Analysis (ISO 24027)

### 6.0 Sources of Bias (ISO/IEC TR 24027:2021 §6)

ISO 24027 §6 mandates explicit enumeration of bias sources. Documented for
ALI :

#### 6.0.1 Data bias

- **Survivor bias on PRIVATE rules** : 4 of 14 A02 articles (3.7.b force
  équipes, 3.2 désignation titulaires, 3.7.f noyau, 3.7.k inscriptions)
  are unverifiable from public data. Historical compositions implicitly
  respected their own private rules → empirical samples are favorably
  biased toward valid lineups. **Impact** : low (favorable bias). **Mitigation** :
  classification PUBLIC/PRIVATE explicit (`VerifiabilityClassifier`),
  ConfidenceLevel exposed to consumer.
- **Sparsity bias on small clubs** : clubs with < 25 active players have
  fewer historical co-presence observations → copula fit on sparse data
  → wider posterior. **Impact** : medium (low confidence flagged). **Mitigation** :
  `ConfidenceLevel = "low"` for clubs with `coverage_ratio < 0.5` or
  `n_historical_rounds < 10`.
- **Temporal coverage bias** : 2002-2010 seasons have sparser FFE
  digitization than 2011+. Recent seasons over-represented in copula
  fit. **Impact** : low (recency decay λ=0.9 already prioritizes recent).
- **Forfeit / non-joué bias** : matches with high forfait rate (small
  competing clubs) have noisy ground truth. **Impact** : low (filtered
  by `_extract_players`).

#### 6.0.2 Algorithmic bias

- **Mode-dominated TopK** : 10 deterministic scenarios capture the mode
  ; if the mode is wrong (rare lineup), Top-K recall fails. **Mitigation** :
  10 MC scenarios cover the queue (LHS + antithetic).
- **Linear copula on rank correlation** : Spearman captures monotone
  dependence ; misses non-monotone interactions (rare in lineup
  selection — typically additive substitution effect). **Mitigation** :
  acceptable per Genest & Favre 2007.
- **Static `λ=0.9, n_topk=10, n_mc=5`** : tuned on 2021-2024 backtest, not
  adaptive to drift. **Impact** : medium long-term. **Mitigation** :
  Adaptive Importance Sampling Phase 5+ (D9).

#### 6.0.3 Deployment bias

- **Single-instance Phase 3** : tested only on developer laptop.
  Production deploy on Oracle VM ARM 24 GB (Phase 5) untested → potential
  performance shift. **Mitigation** : capacity benchmark Phase 5
  (`docs/operations/ALI_SLO.md`).
- **Latency-driven degraded path** : if MC rejection rate > 50 % or
  timeout > 1.5 s, `ALIModeManager` may auto-switch to Phase 2 fallback
  (Elo-only) — degraded predictions silently. **Mitigation** : warning in
  response metadata + `ali_mode_gauge` Prometheus metric (Phase 5 §4.15).

#### 6.0.4 Human bias (in design)

- **Single-author project** : design decisions reflect one developer's
  judgment. **Mitigation** : SOTA literature review documented §8 (13
  alternatives evaluated and rejected with sources).
- **LLM co-authorship** : Claude (Opus 4.7) participated in design and
  code generation. Documented `docs/iso/AI_DEVELOPMENT_DISCLOSURE.md`.
  **Mitigation** : ISO 42001 traceability + JALON peer reviews at 3
  intermediate gates.

### 6.1 Smoke Tests Executed (T13)

- **Breakdown by competition level** (N1, N2, N3, N4, Régionale) — verifies
  no systematic ALI bias by skill tier
- **Breakdown by club size** (quartiles of `joueurs_by_club` cardinality) —
  small clubs have less history, expected lower confidence
- **Stratified sampling** (`scripts/backtest/stratified_sampler.py`) ensures
  balanced backtest representation per stratum

### 6.2 ConfidenceLevel Metadata (Phase 3 §4.13)

Each `ScenarioSet` carries (Phase 5 deploy) :
- `n_historical_rounds` : volume of past data per club
- `coverage_ratio` : fraction of pool with ≥ 3 historical rounds
- `confidence_tier` : `"high"` (cov ≥ 0.8 ∧ n ≥ 20), `"medium"` (cov ≥ 0.5
  ∧ n ≥ 10), `"low"` otherwise
- `sample_warnings` : human-readable notes ("club has < 5 rounds this season")

API consumers should display the tier to inform captain about prediction
reliability — fairness via **transparency**, not via hard correction.

### 6.3 Out-of-Scope Phase 3

- **Gender breakdown** : full audit deferred Phase 3.5 STRICT (D8). Smoke tests
  do not yet partition by `joueurs.genre` because most interclub teams are
  male-dominated by design (FFE A02 §3.7.i N1/N2 minimum female only).
- **Age (J02 youth)** : not supported Phase 3 (D3 deferred Phase 3.5)

---

## 8. Robustness Analysis (ISO 24029)

### 7.1 Smoke Tests Executed (T14)

- **Elo perturbation** : add ±50 noise to opponent Elos → expected Top-K
  recall variation < 10 % (smoke threshold). Phase 3.5 STRICT will tighten
  to < 5 % (D8).
- **Stress suite** (`scripts/backtest/robustness.py`) : feature noise,
  missing data, distributional shift on `taux_presence_effectif`

### 7.2 Determinism (T18)

- Two backtest runs with `seed=42` produce **bit-identical** :
  - `ScenarioSet.lineage_hash` (covered by `test_scenario_set_bit_identical_same_seed`)
  - `MatchStats` per-match metrics rounded 12 decimals
  - Bootstrap CI bounds (lower, point, upper)
- Sanity counterpart : `seed=42` vs `seed=123` produce diverging hashes
  (interdit "seed silencieusement ignoré")
- Reference : Henderson et al. 2018 "Deep RL that Matters" (AAAI)

### 7.3 Edge Cases (T17)

5 edge cases covered by `tests/backtest/test_edge_cases.py` :
1. Round 1 (no history) — ground truth extractable, no crash
2. Pool < team_size — runner skips silently (defensive guard)
3. Last round of season — extractable as normal (no off-by-one)
4. Inexistent round — `KeyError` fail-fast (no silent fallback)
5. Match with forfaits — `_extract_players` filters `name.strip() == ""`
   (real fixture Aix-Les Bains/Meximieux 2024 R1)

### 7.4 Property-Based Hardening (T19 + T19.5)

22 Hypothesis property tests covering :
- Bounds invariants (recall, accuracy, jaccard, brier, ECE, BSS, p_value
  ∈ valid intervals)
- Identity properties (recall(obs, obs)=1, jaccard(A, A)=1, brier perfect=0)
- Containment (bootstrap CI lower ≤ point ≤ upper)
- Sanity counterparts (different seeds → diverging outputs)
- **Degenerate inputs** (T19.5) : empty scenario_set, var=0 bootstrap
  (revealed and fixed `bootstrap_ci NaN` bug, commit 2629cfd, fix-on-sight),
  baseline ≤ 0 BSS, McNemar boundary cases

Reference : Hughes & Claessen 2000 QuickCheck (ICFP), MacIver 2019 Hypothesis.

---

## 9. SOTA Comparative Audit (ISO 42001 §8.2 — D-P3-10 résorbée)

ISO 42001 requires "state-of-the-art documented" + "alternatives considered".
This section explicitly lists alternatives evaluated and rejected, with
justification.

### 8.1 SOTA Components Adopted

| Component | Choice | Source literature |
|-----------|--------|-------------------|
| Joint sampling | Gaussian copula | Sklar 1959, Genest & Favre 2007, Nelsen 2006 |
| MC diversity | Latin Hypercube + antithetic variates | McKay 1979, Hammersley & Morton 1956, Owen 2013 |
| Recency weighting | Exponential decay λ=0.9 | Brown 1959, Silver 2012 (FiveThirtyEight) |
| Streak modeling | Autoregressive lag 1-3 | Box & Jenkins 1970, Pappalardo 2019 |
| Survivor filter | Active license at round date | Brown, Goetzmann, Ross, Ibbotson 1992 |
| ML calibration | Temperature scaling + Dirichlet | Guo et al. 2017, Kull et al. 2019 |
| Optim paradigm | Predict-then-Optimize (PtO) | Elmachtoub & Grigas 2022 |
| Backtest | Walk-forward strict | Bergmeir & Benítez 2012, 2018 |
| Rules | JSON declarative | LegalRuleML (W3C) |
| Significance | McNemar paired + bootstrap BCa | McNemar 1947, Edwards 1948, Efron 1987 |
| Property testing | Hypothesis (Python) | Hughes & Claessen 2000 QuickCheck |

### 8.2 Alternatives Rejected (with reasons)

| Alternative | Rejection rationale |
|-------------|---------------------|
| **Decision-Focused Learning (DFL)** | Wilder, Dilkina, Tambe 2019. ADR : PtO retained for explainability — each scenario interpretable. DFL couples ML and CE too tightly, harder to audit ISO 42001. |
| **Gibbs sampler** | O(N²) × iterations, ill-conditioned for pools > 30 players. Copula : O(N) sample via Cholesky. Comparison empirical Plan 3 backtest : copula expected to win on convergence + variance. |
| **IID Monte Carlo** | 20 IID scenarios may be quasi-identical (variance wasted). LHS + antithetic guarantees stratified coverage + neg-correlated pairs (Owen 2013 ch. 10). |
| **Conformal prediction** (Vovk, Gammerman, Shafer 2005) | Provides distribution-free CI on E[score]. Deferred Phase 4+ (D15) : requires CE multi-objective consumption (e.g. minimax worst-case). Plan 2 delivers point E[score] sufficient for Phase 4 OR-Tools baseline. |
| **Deep RNN / Transformer for lineup prediction** | Overkill for N ~ 40 players/club. Deep models require 10^4+ training samples per club — infeasible. Tabular features + copula better fit for sample-efficient inference. |
| **Holt-Winters / Kalman filter** for F2 recency | Adds complexity without proven gain over exponential smoothing in this regime. λ=0.9 tuned empirically on backtest. Holt-Winters justified only with strong seasonality (rounds within season ≠ chess clubs). |
| **ARIMA** for F3 streak | Lag 1-3 booleans capture short-term dynamics sufficient for ~11-round seasons. Full ARIMA needs ≥ 30 observations per series ; chess clubs typically have 10-20 per season. |
| **Inverse-Probability Weighting** for F7 survivor | Justified when license_active is observed but treatment unobserved. Here `joueurs.parquet` already filters to active FFE only — IPW redundant. |
| **JSON Logic / Open Policy Agent (Rego)** for rules | Custom JSON schema sufficient for FFE A02 (14 rules). External engines add deployment dependency without justifying gain at this scale. Reviewed in ADR-013. |
| **Polars / DuckDB lazy cache** | Pandas in-RAM cache already meets latency target (`<200 ms` ALI generate). Migration cost not justified Phase 3. Possible Phase 5+ if cache memory pressure observed. |
| **Adaptive Importance Sampling (AIS)** | Veach & Guibas 1995, Cornuet et al. 2012, Bugallo et al. 2017. Deferred Phase 5+ (D9) : requires production feedback loop volume to detect drift. Without observed drift, gain marginal vs fixed MIS. |
| **`zone_enjeu` modulation** (Koning 2000, Carling 2015) | Context-dependent presence (accession/maintenance/middle of standings). Deferred Phase 4+ (D13) : couples ALI to live standings data, structural dependency on CE OR-Tools. |
| **AutoGluon** | Tested commit 4baa57f, ADR-011 ELIMINATED. No residual learning support, calibration incompatible with CE downstream. Test logloss 0.5716 worse than V9 LGB 0.5619. Postmortem : `docs/postmortem/2026-04-16-autogluon-v9-time-allocation-failure.md`. |

---

## 10. Limitations

### 9.1 Architectural Limitations

1. **`_EXPECTED_SCENARIOS = 20` invariant strict** (`services/ali/scenario.py`).
   By design : 10 TopK + 10 MC. If pool < ~12 distinct lineups generable,
   `validate()` raises ; runner skips match (D-P2-03 résorbée Plan 3 T23,
   ADR-014 §Invariants).
2. **Default `seed=42` audit-stable** : two `/compose` calls with identical
   inputs return identical 20 scenarios. Caller can override via
   `ComposeRequest.seed` for variance exploration (D-P2-04 résorbée Plan 3
   T23, ADR-014 §Determinism).
3. **Composition Engine (CE) in Phase 3 = Elo-descending sort + E[score]
   weighting**. True multi-team joint optimization deferred Phase 4 (CP-SAT
   OR-Tools). Without Phase 4, ALICE optimizes lineups one team at a time
   (no cross-team constraint resolution like brûlé / noyau).

### 9.2 Statistical Limitations

4. **Bootstrap CI degenerate on var=0 input** (`scripts/backtest/bootstrap.py`
   L91, fix-on-sight commit 2629cfd). When all per-match metric values are
   identical (e.g. all `recall = 1.0` on saturated sample), BCa undefined ;
   guard returns degenerate CI `lower = upper = point` rather than NaN.
   Detected by Hypothesis T19 falsifying example `[0.0, 0.0]`. **Action for
   audit consumers** : check `n_resamples=0` flag in `BootstrapCI` to detect
   degenerate cases.
5. **PRIVATE rules (4 of 14 A02 articles) supposed respected by adversary**.
   Articles 3.7.b (force équipes), 3.2 (désignation titulaires), 3.7.f
   (noyau), 3.7.k (inscriptions) cannot be verified from public data.
   `VerifiabilityClassifier` partitions PUBLIC vs PRIVATE ; MC sampler
   validates only against PUBLIC. **Survivor bias favorable** : historical
   compositions already respected their own noyau, so empirical samples
   are implicitly compliant. Bias quantified Phase 3.5 STRICT (D8).

### 9.3 Domain & Coverage Limitations

6. **Youth competitions (J02)** : age filtering not implemented (D3
   deferred Phase 3.5). `joueurs.parquet` has `categorie` field but not
   exact birth dates.
7. **Cup competitions** : only A02 championnats interclubs implemented
   (D4 deferred Phase 3.5).
8. **Seasonal scope** : training 2002-2024, hold-out 2024, test 2025.
   Performance may degrade if FFE rules change significantly post-2025
   (e.g. team_size adjustments per division) — no automatic retraining
   pipeline yet (D6/D7 deferred Phase 5).

### 9.4 Operational Limitations

9. **Latency** : `~ 1.0 s` p50 measured locally on laptop with warm cache.
   Production deploy on Oracle VM ARM 24 GB (Phase 5) may differ ; capacity
   benchmark and SLO documented in `docs/operations/ALI_SLO.md` (Phase 5).
10. **Single-instance deployment** : no horizontal scaling design Phase 3.
    Multi-tenant SaaS scope Phase 5+ (rate limiting per `user_club_id`,
    audit log MongoDB partitioned by `tenant_id` — crochets en place,
    plein wiring Phase 5).
11. **Drift monitoring absent in production** : Adaptive Importance Sampling
    + drift dashboard deferred Phase 5+ (D9). Currently static λ=0.9 and
    static `n_topk=10, n_mc_pairs=5` — calibrated on backtest 2021-2024,
    not adaptive to season 2025+ shifts.

---

## 11. Ethical Considerations

### 10.1 Personal Data (RGPD)

- **No personal data beyond FFE public**. Names, Elo ratings, club
  membership, game results — all are publicly available on FFE website
  by long-standing federation policy. No private contact info, no
  health/genetic data, no children-specific data.
- **Scope of impact** : adversary prediction = strategic decision support,
  not individual profiling for commercial / political / health purposes
  (ISO 42005 impact assessment — `docs/iso/AI_RISK_ASSESSMENT.md`).

### 10.2 Audit & Repudiation

- Every `/compose` call logged in MongoDB (ISO 27001 A.8.15) with
  `lineage_hash`, `rule_uuids_applied`, `model_versions`, timestamp,
  `tenant_id` (Phase 5).
- Captains receiving predictions cannot deny having received them
  (non-repudiation property).

### 10.3 Failure Modes

- **Wrong adversary club** : if user provides invalid `opponent_club_id`,
  Pydantic validation rejects (FFE ID format `A12345`). Unknown FFE ID
  → 404 with explicit error.
- **Pool too small** : if opponent has fewer eligible players than required,
  `ValueError` from `ScenarioGenerator.generate` ; route returns
  500 + audit logs the failure. No silent degradation.
- **ML inference failure** : `strict=False` (production default) →
  fallback to Elo baseline per board (degraded observability, ISO 25010).
  `strict=True` (backtest) → fail-fast, ISO 42001 explicability.

### 10.4 Misuse Potential

- **Low**. Predictions could theoretically inform betting markets, but
  chess interclub matches are not bet on at scale. The data is FFE
  public ; ALI adds modeling but does not introduce information asymmetry
  beyond publicly observable frequencies.

---

## 12. Recommendations for Users (Mitchell 2019 §9)

Practical guidance for chess club captains and downstream API consumers
to make best use of ALI given its limitations.

### 11.1 When ALI is reliable to follow

- **Stable rosters** : opponent club has ≥ 20 historical rounds and ≥ 80 %
  of active pool with ≥ 3 prior rounds (`ConfidenceLevel = "high"`)
- **Top divisions (N1-N3)** : strict A02 rule enforcement → predictions
  more constrained → higher recall expected
- **Mid-season (rounds 4-9)** : enough history to fit copula, no
  end-of-season collapse effects

### 11.2 When to apply human override

- **Small clubs** (`ConfidenceLevel = "low"`) : trust your local knowledge
  over MC scenarios — adversary captain may field unexpected youth or
  guests
- **Cup matches** (D4 not implemented) : ALI predictions invalid, do
  **not** use `/compose` for federation cups
- **Last round of season with relegation/promotion at stake** :
  `zone_enjeu` not yet modeled (D13 Phase 4+) → predictions miss
  motivational shifts
- **Pool change > 30 % vs prior rounds** : drift not detected by Phase 3
  static system → seasonal injuries, transfers, retirements

### 11.3 How to interpret 20 scenarios

- **Top scenario (max weight)** = highest-likelihood lineup. Use as
  starting hypothesis.
- **Top-K recall ≥ 0.90** (gate P3G07) means the union of 20 scenarios
  contains 90 %+ of who actually plays — pool of 8 boards × 20 scenarios
  = 160 lineup slots, expect ~10-12 distinct opponents covered out of
  team_size=8
- **Brier skill score > 0.05** means ALI strictly improves over Elo-only
  baseline. If BSS < 0, **stop using ALI for that club** and report to
  developer.

### 11.4 Audit & dispute

- Every prediction has `lineage_hash`. If a captain disputes a prediction
  retrospectively, federate audit log against `lineage_hash` to recover
  exact input parquets + rules + seed used.
- Disputes should be filed as GitHub issues citing `lineage_hash` prefix
  + match identifiers (saison, ronde, opponent_club_id).

### 11.5 Operational guardrails for API consumers

- **Catch `400 ValueError` from `pool too small`** : opponent club has
  < team_size active players → display "ALI unavailable for this club"
  rather than retrying
- **Check `ali_mode` in response metadata** : if `"fallback"`, ALI
  predictions are not available, predictions came from Phase 2 Elo-only
- **Honor rate limit** : current Phase 3 limit per IP, Phase 5 per
  `user_club_id` (slowapi configuration)
- **Display `ConfidenceLevel`** to end-users (capitaine UI) — fairness
  via transparency, not via hard correction

### 11.6 Recommendations against misuse

- **Do not chain `seed` variations to find a desired lineup** : the
  determinism is for audit, not for cherry-picking. If 20 scenarios with
  default seed don't include your favored lineup, that's a signal it's
  unlikely — not a prompt to retry until it appears.
- **Do not aggregate predictions across multiple clubs you are not
  managing** — that may infringe federation expectations of strategic
  privacy (even if data is public).

---

## 13. Determinism, Reproducibility, Versioning

- **Lineage hash** : SHA-256 covering `(parquets sigs + rules sigs +
  λ + n_topk + n_mc + seed + opponent_club_id + round_date + saison +
  ronde + nb_rondes_total)`. Two identical inputs → identical hash.
- **Model versioning Phase 5** : DVC tracking of `reports/backtest/*.parquet`
  + commit-pinning of model weights (D6/D7 deferred Phase 5).
- **Reproducibility check** : `python -m pytest tests/backtest/test_determinism.py`
  (4 tests, runs in ~ 3 min on laptop).

---

## 14. Maintenance & Update Plan

- **Weekly** : `make refresh-data` re-syncs FFE parquets (HuggingFace
  `Pierrax/ffe-history`)
- **Per release** : full backtest hold-out 2024 + JALON #3 peer review
  (`superpowers:requesting-code-review` skill)
- **Per ADR change** : review `docs/architecture/adr/ADR-014-*` and
  retrigger T22 Gates Report
- **Drift monitor (Phase 5+)** : weekly KL divergence + PSI on input
  features ; alert if KL > 0.1 (warning) or > 0.3 (critical)

---

## 15. Compliance Cross-References

| ISO Standard | Section in this Card | External Doc |
|--------------|---------------------|--------------|
| 42001 §8.2 (model docs) | All sections | `docs/iso/ALI_MODEL_CARD.md` (this) |
| 42005 (impact) | §11 Ethical | `docs/iso/AI_RISK_ASSESSMENT.md` |
| 23894 (risk register) | §10 Limitations | `docs/iso/AI_RISK_REGISTER.md` (T21) |
| 25059 (quality gates) | §6 Quantitative | `docs/iso/ALI_QUALITY_GATES_REPORT.md` (T22) |
| 24027 (fairness, sources of bias) | §7 Fairness | T13 smoke + Phase 3.5 STRICT (D8) |
| 24029 (robustness) | §8 Robustness | T14 smoke + Phase 3.5 STRICT (D8) |
| 5259 (data lineage + 5 quality dimensions) | §5.2 + §5.5 | `lineage_hash` SHA-256 + Pandera |
| 27001 (audit) | §11.2 Audit | MongoDB audit log |
| 27034 (input validation) | §11.3 Failure modes | Pydantic schemas + RuleEngine |
| 25059 §Security | §11.4 Misuse + Threat Model | `docs/security/ALI_THREAT_MODEL.md` (STRIDE + OWASP API Top 10) |
| 5055 (code quality) | §1.3 Components | All modules ≤ 300 lines, xenon B |
| 29119 (testing) | §8.4 Property tests | ~245 unit tests + Hypothesis |
| 42010 (ADRs) | All architectural decisions | `docs/architecture/adr/ADR-013` + `ADR-014` |
| 23053 (AI lifecycle) | §14 Maintenance | `docs/iso/AI_LIFECYCLE.md` (Phase 5) |
| 22989 (terminology) | Glossary cross-refs | `docs/iso/AI_GLOSSARY.md` (Phase 5) |
| 38507 (governance) | §11 Ethical + §14 | `docs/iso/AI_GOVERNANCE.md` (Phase 5) |
| 25012 (data quality) | §5 Training data | Pandera schemas pre-commit hook |
| 25024 (data measurement) | §5.4 Sample sizes | `compute_data_lineage()` |
| 25010 (system quality) | §10.4 Operational | Latency targets + degraded paths |

---

## 16. References

### 16.1 Statistical & ML Methods

- Sklar A. 1959. "Fonctions de répartition à n dimensions et leurs marges."
  *Publications de l'Institut Statistique de l'Université de Paris* 8 : 229-231.
- McKay M. D., Beckman R. J., Conover W. J. 1979. "A comparison of three
  methods for selecting values of input variables in the analysis of output
  from a computer code." *Technometrics* 21(2) : 239-245.
- Hammersley J. M., Morton K. W. 1956. "A new Monte Carlo technique :
  antithetic variates." *Mathematical Proceedings of the Cambridge
  Philosophical Society* 52(3) : 449-475.
- Brown R. G. 1959. "Statistical forecasting for inventory control."
  McGraw-Hill (origin of exponential smoothing).
- Box G. E. P., Jenkins G. M. 1970. "Time Series Analysis : Forecasting
  and Control." Holden-Day.
- Brown S. J., Goetzmann W., Ross S. A., Ibbotson R. G. 1992. "Survivorship
  Bias in Performance Studies." *Review of Financial Studies* 5(4) : 553-580.
- Genest C., Favre A.-C. 2007. "Everything you always wanted to know about
  copula modeling but were afraid to ask." *Journal of Hydrologic Engineering*
  12(4) : 347-368.
- Nelsen R. B. 2006. "An Introduction to Copulas." Springer.
- Owen A. B. 2013. "Monte Carlo theory, methods and examples." Online book.
- Guo C., Pleiss G., Sun Y., Weinberger K. Q. 2017. "On Calibration of Modern
  Neural Networks." *ICML* : 1321-1330.
- Kull M., Perelló-Nieto M., Kängsepp M., Silva Filho T., Song H., Flach P.
  2019. "Beyond temperature scaling : obtaining well-calibrated multiclass
  probabilities with Dirichlet calibration." *NeurIPS* 32.
- Efron B. 1987. "Better Bootstrap Confidence Intervals." *Journal of the
  American Statistical Association* 82(397) : 171-185.
- McNemar Q. 1947. "Note on the sampling error of the difference between
  correlated proportions or percentages." *Psychometrika* 12(2) : 153-157.
- Edwards A. L. 1948. "Note on the correction for continuity in testing the
  significance of the difference between correlated proportions."
  *Psychometrika* 13(3) : 185-187.
- Bergmeir C., Benítez J. M. 2012. "On the use of cross-validation for time
  series predictor evaluation." *Information Sciences* 191 : 192-213.
- Bergmeir C., Hyndman R. J., Koo B. 2018. "A note on the validity of
  cross-validation for evaluating autoregressive time series prediction."
  *Computational Statistics & Data Analysis* 120 : 70-83.

### 16.2 Domain (Sports Prediction)

- Pappalardo L., Cintia P., Rossi A., Massucco E., Ferragina P., Pedreschi D.,
  Giannotti F. 2019. "PlayeRank : Data-driven Performance Evaluation and
  Player Ranking in Soccer via a Machine Learning Approach." *ACM TIST* 10(5).
  arXiv : 1902.01957.
- Silver N. 2012. "The Signal and the Noise." Penguin (FiveThirtyEight
  methodology).

### 16.3 ML Engineering

- Mitchell M., Wu S., Zaldivar A., Barnes P., Vasserman L., Hutchinson B.,
  Spitzer E., Raji I. D., Gebru T. 2019. "Model Cards for Model Reporting."
  *FAccT (FAT*)* : 220-229.
- Henderson P., Islam R., Bachman P., Pineau J., Precup D., Meger D. 2018.
  "Deep Reinforcement Learning that Matters." *AAAI*.
- Hughes J., Claessen K. 2000. "QuickCheck : a lightweight tool for random
  testing of Haskell programs." *ICFP* : 268-279.
- MacIver D. R. 2019. "Hypothesis : A new approach to property-based testing
  in Python." PyCon talks + Hypothesis library docs.

### 16.4 Decision-Theoretic

- Elmachtoub A. N., Grigas P. 2022. "Smart Predict, then Optimize."
  *Management Science* 68(1) : 9-26.
- Wilder B., Dilkina B., Tambe M. 2019. "Melding the Data-Decisions Pipeline :
  Decision-Focused Learning for Combinatorial Optimization." *AAAI* :
  1658-1665.

### 16.5 Survivor Bias / Statistical Pitfalls

- Veach E., Guibas L. 1995. "Optimally combining sampling techniques for
  Monte Carlo rendering." *SIGGRAPH* : 419-428.
- Cornuet J.-M., Marin J.-M., Mira A., Robert C. P. 2012. "Adaptive
  multiple importance sampling." *Scandinavian Journal of Statistics*
  39(4) : 798-812.
- Bugallo M. F., Elvira V., Martino L., Luengo D., Míguez J., Djurić P. M.
  2017. "Adaptive Importance Sampling : The Past, the Present, and the
  Future." *IEEE Signal Processing Magazine* 34(4) : 60-79.

---

## 17. Document Control

| Field | Value |
|-------|-------|
| Document ID | ALICE-MODEL-CARD-ALI-v1.0.0 |
| Created | 2026-04-28 |
| Last updated | 2026-04-28 |
| Status | DRAFT (T20 Plan 3 V2) |
| Author | Pierre-Alexandre Guillemin |
| LLM Co-author | Claude Opus 4.7 (1M context) — see `docs/iso/AI_DEVELOPMENT_DISCLOSURE.md` |
| Reviewers | TBD JALON #3 (`superpowers:code-reviewer` agent) |
| Approval | Pending JALON #3 |
| Next review | Per release cycle (post-T22 Gates Report) |
| Supersedes | N/A (first ALI Model Card) |

**Related documents** :
- ADR-014 : `docs/architecture/adr/ADR-014-ali-mc-hybride-sota.md`
- Spec Phase 3 : `docs/superpowers/specs/2026-04-19-phase3-ali-monte-carlo-design.md`
- Plan V2 : `docs/superpowers/plans/2026-04-20-phase3-plan3-validation.md`
- Architecture : `docs/architecture/ALI_ARCHITECTURE.md`
- Quality Gates Report : `docs/iso/ALI_QUALITY_GATES_REPORT.md` (T22 pending)
- Risk Register : `docs/iso/AI_RISK_REGISTER.md` (T21 pending)
