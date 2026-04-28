# AI Risk Register - ALICE Engine

**Document ID:** ALICE-RISK-REG-001
**Version:** 1.0.0
**Date:** 2026-01-14
**Classification:** Internal
**Conformité:** ISO/IEC 23894:2023 - AI Risk Management

---

## 1. Risk Matrix (Impact × Probability)

```
                         IMPACT
              Low (1)    Medium (2)    High (3)    Critical (4)
         ┌─────────────┬─────────────┬────────────┬─────────────┐
High (4) │    4        │     8       │    12 ⚠️   │    16 🔴    │
         ├─────────────┼─────────────┼────────────┼─────────────┤
Med (3)  │    3        │     6       │     9 ⚠️   │    12 ⚠️    │
P        ├─────────────┼─────────────┼────────────┼─────────────┤
R        │    2        │     4       │     6      │     8       │
O Low (2)├─────────────┼─────────────┼────────────┼─────────────┤
B        │    1        │     2       │     3      │     4       │
Very     └─────────────┴─────────────┴────────────┴─────────────┘
Low (1)

Legend:
🔴 Critical (12-16): Immediate action required
⚠️ High (8-11): Action required within sprint
🟡 Medium (4-7): Monitor and plan mitigation
🟢 Low (1-3): Accept and monitor
```

---

## 2. Active Risk Register

### 2.1 Model Performance Risks

| ID | Risk | P | I | Score | Status | Owner | Due |
|----|------|---|---|-------|--------|-------|-----|
| **MPR-01** | Data drift degrades predictions | 3 | 3 | **9** ⚠️ | ✅ Mitigated | MLOps | Ongoing |
| **MPR-02** | Concept drift (rule changes FFE) | 2 | 3 | 6 | 🟡 Monitor | ML Eng | Seasonal |
| **MPR-03** | Overfitting on training data | 2 | 2 | 4 | ✅ Mitigated | ML Eng | Ongoing |
| **MPR-04** | Adversarial input manipulation | 1 | 2 | 2 | ✅ Mitigated | ML Eng | Ongoing |

**Controls:**
- MPR-01: `scripts/model_registry/drift_monitor.py` (PSI, KS-test)
- MPR-02: Annual review of FFE rule changes
- MPR-03: 5-fold CV + early stopping (50 rounds)
- MPR-04: `scripts/robustness/` adversarial tests

---

### 2.2 Data Quality Risks

| ID | Risk | P | I | Score | Status | Owner | Due |
|----|------|---|---|-------|--------|-------|-----|
| **DQR-01** | Missing Elo ratings | 2 | 2 | 4 | ✅ Mitigated | Data | Ongoing |
| **DQR-02** | Incorrect match results | 1 | 3 | 3 | ✅ Mitigated | Data | Ongoing |
| **DQR-03** | Out-of-distribution inputs | 3 | 2 | 6 | ✅ Mitigated | ML Eng | Ongoing |
| **DQR-04** | Data poisoning attack | 1 | 4 | 4 | 🟡 Monitor | SecOps | Quarterly |

**Controls:**
- DQR-01: Schema validation (`scripts/model_registry/validation.py`)
- DQR-02: FFE official source verification
- DQR-03: `scripts/model_registry/input_validator.py` (OOD detection)
- DQR-04: Data source authentication + checksums

---

### 2.3 Fairness & Bias Risks

| ID | Risk | P | I | Score | Status | Owner | Due |
|----|------|---|---|-------|--------|-------|-----|
| **FBR-01** | Division bias (N1 vs DEP) | 2 | 3 | 6 | ✅ Mitigated | ML Eng | Ongoing |
| **FBR-02** | Elo range bias | 2 | 2 | 4 | ✅ Mitigated | ML Eng | Ongoing |
| **FBR-03** | Temporal bias (season start) | 2 | 2 | 4 | 🟡 Monitor | ML Eng | Seasonal |
| **FBR-04** | Club size bias | 1 | 2 | 2 | 🟢 Accepted | ML Eng | - |

**Controls:**
- FBR-01: `scripts/monitoring/bias_tracker.py` (EEOC 80% rule)
- FBR-02: Bias by Elo range monitoring
- FBR-03: Early season data augmentation
- FBR-04: Accepted (minor impact on predictions)

**Thresholds (ISO 24027):**
- Demographic Parity Ratio ≥ 0.80
- Equalized Odds Diff ≤ 0.10

---

### 2.4 Security Risks

| ID | Risk | P | I | Score | Status | Owner | Due |
|----|------|---|---|-------|--------|-------|-----|
| **SEC-01** | API key exposure | 1 | 4 | 4 | ✅ Mitigated | SecOps | Ongoing |
| **SEC-02** | Model extraction | 1 | 2 | 2 | 🟢 Accepted | SecOps | - |
| **SEC-03** | Injection attacks | 1 | 4 | 4 | ✅ Mitigated | SecOps | Ongoing |
| **SEC-04** | Dependency vulnerabilities | 2 | 3 | 6 | ✅ Mitigated | DevOps | Weekly |

**Controls:**
- SEC-01: Gitleaks pre-commit hook + env vars
- SEC-02: Low priority (chess domain, low value)
- SEC-03: Pydantic validation + input sanitization
- SEC-04: pip-audit + dependabot

---

### 2.5 Operational Risks

| ID | Risk | P | I | Score | Status | Owner | Due |
|----|------|---|---|-------|--------|-------|-----|
| **OPR-01** | Model serving downtime | 1 | 2 | 2 | 🟢 Accepted | DevOps | - |
| **OPR-02** | Version conflicts | 2 | 2 | 4 | ✅ Mitigated | MLOps | Ongoing |
| **OPR-03** | Rollback failure | 1 | 3 | 3 | ✅ Mitigated | MLOps | Ongoing |
| **OPR-04** | Monitoring gaps | 2 | 2 | 4 | ✅ Mitigated | MLOps | Ongoing |

**Controls:**
- OPR-01: Render auto-restart + health endpoint
- OPR-02: `scripts/model_registry/versioning.py`
- OPR-03: `rollback_to_version()` + retention policy
- OPR-04: Structured logging + health checks

---

### 2.6 Phase 2 Serving Risks

| ID | Risk | P | I | Score | Status | Owner | Due |
|----|------|---|---|-------|--------|-------|-----|
| **R-PH2-01** | Feature store stale >14 days | Medium | Low | /health age monitoring + cron alerting | 🟡 Monitor | MLOps | Ongoing |
| **R-PH2-02** | Silent fallback LGB+Dirichlet | Low | Low | Fallback flag in /health + response metadata | ✅ Mitigated | ML Eng | Ongoing |
| **R-PH2-03** | Model corruption HF download | Low | Very Low | SHA-256 checksums + local cache | ✅ Mitigated | MLOps | Ongoing |
| **R-PH2-04** | MongoDB unavailable | Medium | Medium | Parquet fallback + default Elo 1500 | 🟡 Monitor | DevOps | Ongoing |

**Controls:**
- R-PH2-01: `/health` endpoint reports `feature_store_age`; cron alerting if >14 days
- R-PH2-02: `fallback_mode` flag exposed in `/health` and response metadata
- R-PH2-03: SHA-256 checksum verification in `scripts/serving/model_loader.py` + local cache
- R-PH2-04: Feature store parquets as offline fallback; default Elo 1500 when MongoDB unavailable

---

### 2.7 Phase 3 ALI Risks (R-ALI-01..05)

Risques spécifiques au système ALI (Adversarial Lineup Inference) Plan 3
SOTA, identifiés via audit Model Card 2026-04-28 (Mitchell 2019 §9
Limitations). Cross-référencés `docs/iso/ALI_MODEL_CARD.md` §10 et §7.0
"Sources of Bias".

| ID | Risk | P | I | Score | Status | Owner | Due |
|----|------|---|---|-------|--------|-------|-----|
| **R-ALI-01** | PRIVATE rules unverifiable (4/14 A02 articles supposed respected by adversary) | 3 | 2 | **6** 🟡 | 🟡 Monitor | ML Eng | Phase 3.5 STRICT (D8) |
| **R-ALI-02** | Pool too small → ScenarioSet validate() raise → backtest match skip | 2 | 2 | **4** 🟢 | ✅ Mitigated | ML Eng | Ongoing |
| **R-ALI-03** | Bootstrap BCa CI degenerate on var=0 input → NaN propagation in gates | 1 | 2 | **2** 🟢 | ✅ Mitigated | ML Eng | Ongoing |
| **R-ALI-04** | Season-over-season drift undetected (no AIS / drift dashboard) | 4 | 3 | **12** ⚠️ | 🟡 Monitor | MLOps | Phase 5+ (D9) |
| **R-ALI-05** | Single-instance deployment, no horizontal scaling Phase 3 | 3 | 2 | **6** 🟡 | 🟡 Monitor | DevOps | Phase 5 |

**R-ALI-01 — PRIVATE rules unverifiable**
- **Description** : 4 of 14 FFE A02 articles (3.7.b force équipes, 3.2
  désignation titulaires, 3.7.f noyau, 3.7.k inscriptions) cannot be
  verified from public data. ALI MC sampler validates only against PUBLIC
  rules → may generate scenarios that violate adversary's private noyau.
- **Impact rationale** : Medium (2). Survivor bias is **favorable** —
  historical compositions implicitly respected their own private rules,
  so empirical samples are biased toward valid lineups. Risk = ALI
  predictions over-represent rare/unrealistic lineups missed by survivor
  filter.
- **Mitigation Phase 3** : `VerifiabilityClassifier` partitions PUBLIC vs
  PRIVATE explicitly (`services/ali/verifiability.py`). `ConfidenceLevel`
  metadata exposed to consumer warns if pool history < 10 rounds.
- **Mitigation Phase 3.5 STRICT (D8)** : quantitative bias breakdown by
  level + club size + gender to verify <10% Top-K recall variance per
  stratum. Document `docs/iso/ALI_FAIRNESS_VALIDATION.md`.
- **Status** : Monitor. Bias suspected favorable but not yet quantified.
  D8 BLOQUANT Phase 4.

**R-ALI-02 — Pool too small**
- **Description** : Adversary club with < 12 distinct lineups feasible
  (small pool + tight rules) causes `_merge_and_pad` to exhaust 5 retry
  rounds → `ScenarioSet.validate()` raises → request fails or backtest
  match skipped.
- **Impact rationale** : Low (2). Per-match degradation, not systemic.
  Rare clubs affected (small clubs in N4/Régionale).
- **Mitigation Phase 3** : `BacktestRunner.skip_failed_matches=True`
  default. Production `/compose` returns `400 ValueError` with explicit
  error message documented ADR-014 §Invariants. Caller-side fallback to
  Elo-only baseline expected.
- **Status** : Mitigated. Error message clarifies cause to consumer
  (commit cdf6a7c).

**R-ALI-03 — Bootstrap CI degenerate**
- **Description** : When all per-match metric values are identical
  (saturated sample, e.g. `recall=1.0` everywhere), BCa method
  mathematically undefined → SciPy returns NaN → P3G gates broken
  silently.
- **Impact rationale** : Medium (2) if undetected. Pollutes
  `BacktestReport.gates_summary()` and downstream T22 report.
- **Mitigation Phase 3** : Guard `np.var(arr) == 0.0` returns degenerate
  CI `lower=upper=point` with `n_resamples=0` flag (commit 2629cfd).
  Detected by Hypothesis property test T19 (`bootstrap_ci([0, 0])`
  falsifying example). Property tests T19.5 cover degenerate inputs
  for all 5 ALI metrics.
- **Status** : Mitigated. Audit consumers should check `n_resamples=0`
  flag in `BootstrapCI`.

**R-ALI-04 — Drift undetected (Phase 5+ blocker for prod scale)**
- **Description** : Static λ=0.9 + static `n_topk=10, n_mc_pairs=5` are
  calibrated on backtest seasons 2021-2024. In production, FFE rules
  evolve, player roster turns over (retirements, mutations), team
  strategies shift. ALI predictions degrade silently without alert.
- **Impact rationale** : High (3) at scale. Bad predictions undermine
  product trust ; without drift alert, captains may follow stale ALI
  recommendations for weeks before noticing.
- **Probability rationale** : High (4). Drift is **expected** in any
  production ML system over months. FFE-specific drivers : annual rule
  reviews, end-of-season retirement waves, club restructuring.
- **Mitigation Phase 3** : NONE. Static calibration accepted as Phase 3
  deliverable.
- **Mitigation Phase 5+ (D9 — required before commercialisation)** :
  Adaptive Importance Sampling (Veach & Guibas 1995, Cornuet et al. 2012,
  Bugallo et al. 2017) + drift dashboard (PSI on `taux_presence`, KL
  divergence on copula correlations, weekly Brier baseline). Alert if
  KL > 0.1 (warning), KL > 0.3 (critical).
- **Status** : Monitor. Risk score 12 reflects expected severity at
  prod scale ; tolerable Phase 3 (controlled hold-out 2024 only).
- **HIGHEST RISK ALI** in current register. Must be addressed Phase 5
  before any user-facing deployment.

**R-ALI-05 — Single-instance deployment**
- **Description** : Phase 3 architecture = single uvicorn process on
  developer laptop. No horizontal scaling design, no container
  orchestration, no failover. Phase 5 multi-tenant SaaS will require
  scaling that's not yet engineered.
- **Impact rationale** : Medium (2). Latency degradation under load,
  not data corruption. Failure isolated (no shared state across
  hypothetical instances).
- **Mitigation Phase 3** : N/A. Single-instance acceptable for Phase 3
  validation.
- **Mitigation Phase 5** : Oracle VM ARM 24 GB benchmark
  (`docs/operations/ALI_SLO.md` TBD). Capacity test : p95 ≤ 1500ms,
  RPS ≥ 3 sustainable, memory ≤ 1.5 GB. Multi-tenant rate limit per
  `user_club_id` (slowapi already wired Phase 3 hooks).
- **Status** : Monitor. Tracé en dette via `CLAUDE.md` Deploy SaaS
  manquant (Phase 5 scope étendu 2026-04-19).

**Controls (R-ALI-01..05):**
- R-ALI-01 : `services/ali/verifiability.py` PUBLIC/PRIVATE partition +
  `services/ali/confidence.py` ConfidenceLevel metadata (Phase 3 §4.13)
- R-ALI-02 : `BacktestRunner.skip_failed_matches=True` default +
  `ADR-014 §Invariants ScenarioSet`
- R-ALI-03 : `scripts/backtest/bootstrap.py` L91 guard + `tests/backtest/
  test_properties_hypothesis.py` (15) + `test_properties_degenerate.py` (7)
- R-ALI-04 : NONE Phase 3. Phase 5+ Required : `services/observability/
  drift_tracker.py` (designed Phase 3 §4.15.3 spec, not implemented)
- R-ALI-05 : Phase 5 deploy plan : Oracle VM + capacity benchmark +
  load test (`scripts/benchmark/ali_benchmark.py` designed Phase 3 §6ter)

**Cross-references** :
- Sources de biais détaillées : `docs/iso/ALI_MODEL_CARD.md` §7.0
- Limitations narratives : `docs/iso/ALI_MODEL_CARD.md` §10
- Threat model security : `docs/security/ALI_THREAT_MODEL.md`
- Resorption phases : `memory/project_debt_current.md` (D8, D9)

---

## 3. Risk Treatment Actions

### 3.1 Immediate Actions (Score ≥ 9)

| ID | Action | Deadline | Status |
|----|--------|----------|--------|
| MPR-01 | Implement KS-test drift detection | 2026-01-14 | ✅ Done |
| DQR-03 | Add OOD input rejection | 2026-01-14 | ✅ Done |
| FBR-01 | Deploy continuous bias monitoring | 2026-01-14 | ✅ Done |
| R-ALI-04 | AIS + drift dashboard (D9) — REQUIRED before any prod deploy | Phase 5 | 📋 Planned |

### 3.2 Planned Actions (Score 6-8)

| ID | Action | Deadline | Status |
|----|--------|----------|--------|
| SEC-04 | Weekly pip-audit automation | 2026-01-20 | 🔄 In Progress |
| MPR-02 | FFE 2025-2026 rule review | 2026-02-01 | 📋 Planned |
| FBR-03 | Season start data handling | 2026-09-01 | 📋 Planned |
| R-ALI-01 | Quantitative bias breakdown (D8 STRICT) | Phase 3.5 | 📋 Planned |
| R-ALI-05 | Oracle VM capacity benchmark + multi-tenant rate limit | Phase 5 | 📋 Planned |

### 3.3 Monitoring Actions (Score 1-5)

| ID | Frequency | Metric | Threshold |
|----|-----------|--------|-----------|
| All | Daily | PSI score | < 0.10 |
| All | Weekly | Bias metrics | DP ≥ 0.80 |
| All | Per inference | OOD ratio | < 30% |
| All | Weekly | Vulnerability scan | 0 critical |

---

## 4. Key Risk Indicators (KRIs)

### 4.1 Real-time KRIs

| KRI | Current | Target | Status |
|-----|---------|--------|--------|
| PSI (Elo distribution) | 0.05 | < 0.10 | 🟢 |
| OOD rejection rate | 2% | < 5% | 🟢 |
| API latency p99 | 180ms | < 500ms | 🟢 |
| Error rate | 0.1% | < 1% | 🟢 |

### 4.2 Weekly KRIs

| KRI | Current | Target | Status |
|-----|---------|--------|--------|
| Model AUC-ROC | 0.72 | ≥ 0.70 | 🟢 |
| Demographic parity | 0.87 | ≥ 0.80 | 🟢 |
| Test coverage | 91% | ≥ 80% | 🟢 |
| Critical vulns | 0 | 0 | 🟢 |

---

## 5. Escalation Matrix

| Score | Level | Response Time | Escalation To |
|-------|-------|---------------|---------------|
| 12-16 | 🔴 Critical | < 4 hours | Risk Committee + AI Owner |
| 9-11 | ⚠️ High | < 24 hours | ML Lead + Risk Manager |
| 6-8 | 🟡 Medium | < 1 week | Team Lead |
| 1-5 | 🟢 Low | Next sprint | Team |

---

## 6. Risk Review Schedule

| Review Type | Frequency | Participants | Output |
|-------------|-----------|--------------|--------|
| Daily monitoring | Daily | MLOps | Dashboard update |
| Risk sync | Weekly | ML + DevOps | Status update |
| Risk committee | Monthly | All leads | Risk report |
| Full assessment | Quarterly | All + Stakeholders | Updated register |

---

## 7. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-14 | ALICE Team | Initial register with P1 mitigations |
| 2.0.0 | 2026-04-28 | Pierre + Claude (LLM co-auth) | T21 Plan 3 V2 : add §2.7 Phase 3 ALI Risks (R-ALI-01..05) — PRIVATE rules survivor, pool too small, bootstrap degenerate, drift undetected (highest score 12), single-instance deploy. Cross-refs Model Card §7.0 + §10. |

---

## 8. Related Documents

- [AI_RISK_ASSESSMENT.md](AI_RISK_ASSESSMENT.md) - Full risk assessment
- [AI_POLICY.md](AI_POLICY.md) - AI governance policy
- [STATEMENT_OF_APPLICABILITY.md](STATEMENT_OF_APPLICABILITY.md) - ISO controls

---

**Next Review:** 2026-05-28 (post-T22 Gates Report) + Phase 3.5 STRICT (D8 bloquant Phase 4)
**Risk Status:** 🟡 AMBER — Phase 3 shipped with R-ALI-04 (drift detection) score 12 ⚠️ at Monitor.
Tolerable Phase 3 (controlled hold-out 2024 backtest only). MUST be addressed Phase 5
before any user-facing deployment. Other R-ALI risks Mitigated or planned Phase 3.5/5.
