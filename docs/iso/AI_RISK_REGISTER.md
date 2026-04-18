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

## 3. Risk Treatment Actions

### 3.1 Immediate Actions (Score ≥ 9)

| ID | Action | Deadline | Status |
|----|--------|----------|--------|
| MPR-01 | Implement KS-test drift detection | 2026-01-14 | ✅ Done |
| DQR-03 | Add OOD input rejection | 2026-01-14 | ✅ Done |
| FBR-01 | Deploy continuous bias monitoring | 2026-01-14 | ✅ Done |

### 3.2 Planned Actions (Score 6-8)

| ID | Action | Deadline | Status |
|----|--------|----------|--------|
| SEC-04 | Weekly pip-audit automation | 2026-01-20 | 🔄 In Progress |
| MPR-02 | FFE 2025-2026 rule review | 2026-02-01 | 📋 Planned |
| FBR-03 | Season start data handling | 2026-09-01 | 📋 Planned |

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

---

## 8. Related Documents

- [AI_RISK_ASSESSMENT.md](AI_RISK_ASSESSMENT.md) - Full risk assessment
- [AI_POLICY.md](AI_POLICY.md) - AI governance policy
- [STATEMENT_OF_APPLICABILITY.md](STATEMENT_OF_APPLICABILITY.md) - ISO controls

---

**Next Review:** 2026-02-14
**Risk Status:** 🟢 GREEN (All high risks mitigated)
