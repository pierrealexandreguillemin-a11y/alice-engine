# AI Risk Assessment - ALICE Engine

**Document ID:** ALICE-RISK-001
**Version:** 1.0.0
**Date:** 2026-01-10
**Classification:** Internal
**Conformité:** ISO/IEC 23894:2023

---

## 1. Executive Summary

This document presents the comprehensive AI risk assessment for the ALICE (Analytical Learning for Intelligent Chess Evaluation) machine learning system. The assessment follows ISO/IEC 23894:2023 guidelines for AI risk management.

**Risk Assessment Summary:**

| Risk Level | Count | Status |
|------------|-------|--------|
| Critical | 0 | All mitigated |
| High | 2 | Mitigated |
| Medium | 4 | Mitigated |
| Low | 3 | Accepted |

---

## 2. Scope and Context

### 2.1 System Description

ALICE is a machine learning system for predicting chess match outcomes in French Federation (FFE) team championships. The system uses gradient boosting models (CatBoost, XGBoost, LightGBM) trained on historical match data.

### 2.2 Stakeholders

| Stakeholder | Interest | Impact |
|-------------|----------|--------|
| Chess clubs | Composition decisions | Direct |
| Players | Match predictions | Direct |
| FFE | Tournament integrity | Indirect |
| Developers | System reliability | Direct |

### 2.3 Risk Appetite

ALICE operates in a low-stakes domain (chess predictions). Risk appetite is:
- **Low** for bias and fairness risks
- **Medium** for accuracy degradation
- **High** for minor prediction errors

---

## 3. Risk Identification

### 3.1 Risk Categories (ISO 23894)

```
R1: Data Quality Risks
R2: Model Performance Risks
R3: Fairness and Bias Risks
R4: Security Risks
R5: Operational Risks
R6: Compliance Risks
```

---

## 4. Risk Register

### R1: Data Quality Risks

#### R1.1 Incomplete Training Data

| Attribute | Value |
|-----------|-------|
| **ID** | R1.1 |
| **Description** | Missing values in Elo ratings, player titles, or match results |
| **Likelihood** | Medium |
| **Impact** | Medium |
| **Risk Level** | Medium |
| **Mitigation** | Data validation pipeline, completeness checks |
| **Controls** | `scripts/parse_dataset/`, data quality tests |
| **Residual Risk** | Low |
| **Owner** | Data Steward |

#### R1.2 Data Drift

| Attribute | Value |
|-----------|-------|
| **ID** | R1.2 |
| **Description** | Distribution shift in input features over time |
| **Likelihood** | Medium |
| **Impact** | High |
| **Risk Level** | High |
| **Mitigation** | PSI monitoring with thresholds |
| **Controls** | `scripts/model_registry/drift.py` |
| **Residual Risk** | Low |
| **Owner** | MLOps Engineer |

**Thresholds:**
- PSI Warning: >= 0.10
- PSI Critical: >= 0.25

#### R1.3 Label Noise

| Attribute | Value |
|-----------|-------|
| **ID** | R1.3 |
| **Description** | Incorrect match outcomes in training data |
| **Likelihood** | Low |
| **Impact** | Medium |
| **Risk Level** | Low |
| **Mitigation** | Data validation against FFE official results |
| **Controls** | Data ingestion validation |
| **Residual Risk** | Low |
| **Owner** | Data Steward |

---

### R2: Model Performance Risks

#### R2.1 Accuracy Degradation

| Attribute | Value |
|-----------|-------|
| **ID** | R2.1 |
| **Description** | Model performance drops below acceptable thresholds |
| **Likelihood** | Medium |
| **Impact** | High |
| **Risk Level** | High |
| **Mitigation** | Continuous monitoring, automatic alerts |
| **Controls** | MLflow tracking, metrics thresholds |
| **Residual Risk** | Low |
| **Owner** | ML Engineer |

**Thresholds (config/hyperparameters.yaml):**
- AUC-ROC minimum: 0.70
- Accuracy minimum: 0.60
- F1 Score minimum: 0.55

#### R2.2 Overfitting

| Attribute | Value |
|-----------|-------|
| **ID** | R2.2 |
| **Description** | Model memorizes training data, fails on new data |
| **Likelihood** | Medium |
| **Impact** | Medium |
| **Risk Level** | Medium |
| **Mitigation** | Cross-validation, early stopping, regularization |
| **Controls** | 5-fold CV, 50 rounds early stopping |
| **Residual Risk** | Low |
| **Owner** | ML Engineer |

#### R2.3 Adversarial Inputs

| Attribute | Value |
|-----------|-------|
| **ID** | R2.3 |
| **Description** | Manipulated inputs cause incorrect predictions |
| **Likelihood** | Low |
| **Impact** | Medium |
| **Risk Level** | Low |
| **Mitigation** | Robustness testing (ISO 24029) |
| **Controls** | `scripts/robustness/adversarial_tests.py` |
| **Residual Risk** | Low |
| **Owner** | ML Engineer |

---

### R3: Fairness and Bias Risks

#### R3.1 Division Bias

| Attribute | Value |
|-----------|-------|
| **ID** | R3.1 |
| **Description** | Systematic prediction errors for certain divisions (N1, N2, REG, DEP) |
| **Likelihood** | Medium |
| **Impact** | Medium |
| **Risk Level** | Medium |
| **Mitigation** | Bias detection and monitoring by division |
| **Controls** | `scripts/fairness/bias_detection.py` |
| **Residual Risk** | Low |
| **Owner** | ML Engineer |

**Thresholds (ISO 24027 / EEOC):**
- SPD Warning: |SPD| >= 0.1
- SPD Critical: |SPD| >= 0.2
- DIR Acceptable: 0.8 <= DIR <= 1.25

#### R3.2 Elo Rating Bias

| Attribute | Value |
|-----------|-------|
| **ID** | R3.2 |
| **Description** | Unfair predictions for certain Elo ranges |
| **Likelihood** | Low |
| **Impact** | Medium |
| **Risk Level** | Low |
| **Mitigation** | Bias analysis by Elo range |
| **Controls** | `compute_bias_by_elo_range()` |
| **Residual Risk** | Low |
| **Owner** | ML Engineer |

#### R3.3 Title Bias

| Attribute | Value |
|-----------|-------|
| **ID** | R3.3 |
| **Description** | Systematic errors for titled vs non-titled players |
| **Likelihood** | Low |
| **Impact** | Low |
| **Risk Level** | Low |
| **Mitigation** | Fairness monitoring by title |
| **Controls** | Bias detection framework |
| **Residual Risk** | Accepted |
| **Owner** | ML Engineer |

---

### R4: Security Risks

#### R4.1 Model Extraction

| Attribute | Value |
|-----------|-------|
| **ID** | R4.1 |
| **Description** | Unauthorized copying of trained models |
| **Likelihood** | Low |
| **Impact** | Low |
| **Risk Level** | Low |
| **Mitigation** | Access controls, model encryption |
| **Controls** | File permissions, Git access |
| **Residual Risk** | Accepted |
| **Owner** | DevOps |

#### R4.2 Data Poisoning

| Attribute | Value |
|-----------|-------|
| **ID** | R4.2 |
| **Description** | Malicious data injected into training set |
| **Likelihood** | Low |
| **Impact** | High |
| **Risk Level** | Medium |
| **Mitigation** | Data validation, source verification |
| **Controls** | FFE data verification |
| **Residual Risk** | Low |
| **Owner** | Data Steward |

---

### R5: Operational Risks

#### R5.1 Infrastructure Failure

| Attribute | Value |
|-----------|-------|
| **ID** | R5.1 |
| **Description** | Model serving infrastructure becomes unavailable |
| **Likelihood** | Low |
| **Impact** | Medium |
| **Risk Level** | Low |
| **Mitigation** | Backup systems, failover procedures |
| **Controls** | Infrastructure redundancy |
| **Residual Risk** | Accepted |
| **Owner** | DevOps |

#### R5.2 Version Conflicts

| Attribute | Value |
|-----------|-------|
| **ID** | R5.2 |
| **Description** | Incompatible model versions deployed |
| **Likelihood** | Medium |
| **Impact** | Medium |
| **Risk Level** | Medium |
| **Mitigation** | Version control, model registry |
| **Controls** | MLflow, semantic versioning |
| **Residual Risk** | Low |
| **Owner** | MLOps Engineer |

---

### R6: Compliance Risks

#### R6.1 ISO Non-Compliance

| Attribute | Value |
|-----------|-------|
| **ID** | R6.1 |
| **Description** | Failure to meet ISO 42001/5259/24027/24029 requirements |
| **Likelihood** | Low |
| **Impact** | Medium |
| **Risk Level** | Low |
| **Mitigation** | Regular audits, compliance monitoring |
| **Controls** | `scripts/audit_iso_conformity.py`, `docs/iso/IMPLEMENTATION_STATUS.md` |
| **Residual Risk** | Low |
| **Owner** | Quality Auditor |

---

## 5. Risk Treatment Plan

### 5.1 Implemented Controls

| Control ID | Control Description | Risks Addressed |
|------------|---------------------|-----------------|
| C1 | Data validation pipeline | R1.1, R1.3 |
| C2 | PSI drift monitoring | R1.2, R2.1 |
| C3 | Cross-validation | R2.2 |
| C4 | Early stopping | R2.2 |
| C5 | Bias detection | R3.1, R3.2, R3.3 |
| C6 | Robustness testing | R2.3 |
| C7 | MLflow tracking | R5.2, R6.1 |
| C8 | ISO audit scripts | R6.1 |

### 5.2 Control Effectiveness

| Control | Implementation | Effectiveness | Last Tested |
|---------|----------------|---------------|-------------|
| C1 | `scripts/parse_dataset/` | High | 2026-01-10 |
| C2 | `scripts/model_registry/drift.py` | High | 2026-01-10 |
| C3 | `config/hyperparameters.yaml` | High | 2026-01-10 |
| C4 | `config/hyperparameters.yaml` | High | 2026-01-10 |
| C5 | `scripts/fairness/bias_detection.py` | High | 2026-01-10 |
| C6 | `scripts/robustness/adversarial_tests.py` | High | 2026-01-10 |
| C7 | MLflow integration | High | 2026-01-10 |
| C8 | `scripts/audit_iso_conformity.py` | High | 2026-01-10 |

---

## 6. Risk Monitoring

### 6.1 Key Risk Indicators (KRIs)

| KRI | Threshold | Frequency | Owner |
|-----|-----------|-----------|-------|
| PSI Score | < 0.10 | Daily | MLOps |
| AUC-ROC | >= 0.70 | Per training | ML Engineer |
| SPD by Division | < 0.10 | Weekly | ML Engineer |
| Test Coverage | >= 80% | Per commit | DevOps |
| Vulnerability Count | 0 critical | Weekly | DevOps |

### 6.2 Reporting

- **Daily:** Automated monitoring dashboards
- **Weekly:** Risk summary report
- **Monthly:** Risk committee review
- **Quarterly:** Full risk reassessment

---

## 7. Conclusion

The ALICE system has a comprehensive risk management framework in place. All identified critical and high risks have been mitigated to acceptable levels through implemented controls. Continuous monitoring ensures early detection of emerging risks.

**Overall Risk Status:** GREEN (Acceptable)

---

## 8. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-10 | ALICE Team | Initial release |

---

**Approval:**

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Risk Manager | ____________ | ____________ | ____________ |
| AI Owner | ____________ | ____________ | ____________ |
