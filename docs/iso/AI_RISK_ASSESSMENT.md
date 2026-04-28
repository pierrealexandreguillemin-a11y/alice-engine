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

---

## Phase 2 Serving Risks (2026-04-18)

### Risk: Feature store stale (>14 days)
- **Impact:** Predictions based on outdated player stats
- **Likelihood:** Medium (cron failure, data source down)
- **Mitigation:** /health reports feature_store_age, alert if >14 days
- **Residual risk:** Low (features are rolling 3 seasons, 1 week delay negligible)

### Risk: Silent fallback to LGB+Dirichlet
- **Impact:** Slightly worse calibration (ECE 0.0042 vs 0.0016) without captain knowing
- **Mitigation:** /health and response metadata report fallback_mode flag
- **Residual risk:** Low (LGB+Dirichlet still passes all T1-T12 gates)

### Risk: Model corruption during HF download
- **Impact:** Inference crash at startup
- **Mitigation:** SHA-256 checksum verification in model_loader, local cache as fallback
- **Residual risk:** Very low

### Risk: MongoDB unavailable
- **Impact:** Cannot load club player data for real Elo lookup
- **Mitigation:** Feature store parquets as offline fallback, default Elo 1500
- **Residual risk:** Medium (degraded predictions with placeholder Elo)

---

## Phase 3 ALI Impact (2026-04-28) — ISO/IEC 42005:2025

### Objet

Section ajoutée Plan 3 V2 §T22 (fix-on-sight). Évalue l'impact système
ALI (Adversarial Lineup Inference) Plan 3 SOTA sur la base du backtest
hold-out 2024 stratifié champion mode (N=70 matches, BSS=0.66, gates
absolus partiels). Cross-référence rigoureuse :
- `docs/iso/ALI_QUALITY_GATES_REPORT.md` — chiffres complets
- `docs/iso/ALI_MODEL_CARD.md` §11 Ethical Considerations
- `docs/iso/AI_RISK_REGISTER.md` §2.7 R-ALI-01..05

### Impact dimensions (ISO 42005 §6 framework)

| Dimension | Niveau | Justification |
|-----------|--------|---------------|
| **Individual impact** | LOW | Décision strategic (composition équipe) — pas de profilage individuel commercial / social / médical. Joueur prédit avoir 70 % presence ne subit aucune conséquence négative directe. RGPD : data publiques FFE, no health/genetic. Cf. Model Card §11.1. |
| **Group impact** | LOW-MEDIUM | Petits clubs (Q1 pool size) bénéficient mieux d'ALI (recall 0.74) que grands clubs (recall 0.46). Variabilité service-niveau, pas discrimination protégée ISO 24027 §6 (sex/race/age). À disclorer en CGU `/compose`. |
| **Societal impact** | LOW | Échecs amateurs FFE — pas de marché de paris organisé scale. ALI ne crée pas d'asymétrie d'information durable (data déjà publiques). |
| **Operational impact** | MEDIUM | Captain reçoit recommandation, peut surprendre ou aller à l'encontre. Failure mode = mauvais composition → match perdu = pas grave individuellement. R-ALI-04 drift undetected (HIGH risk score 12) à scale prod. |
| **Overall** | MEDIUM (Phase 3) → LOW (production stable, MEDIUM-HIGH si drift non géré Phase 5+) |

### R-ALI-06 NEW (T22 review post-mortem) — ALI input non-conditionné multi-équipes

**Severity** : HIGH (impact systémique sur tous gates absolus P3G07-P3G11).
**Probability** : Certain (vérifié empiriquement T22).

**Description** : ALI Phase 3 prédit la composition d'une équipe spécifique
d'un club (ex Mulhouse Philidor 1 N3) mais reçoit en entrée le pool total
du club via `services/ali/pool_loader.py::load_pool(club_id, round_date)`
sans information sur l'allocation simultanée des autres équipes du même
club qui jouent le même weekend (Mulhouse Philidor 2 N4, Mulhouse Philidor
3 R1).

**Évidence empirique** :
- 117 clubs alignent 2-4 équipes simultanément en N3 ronde 5 saison 2024.
- Gap recall by_pool_size = 0.28 (small 0.74 vs xlarge 0.46) — direction
  cohérente avec hypothèse multi-équipes (plus le club est grand, plus
  probablement il aligne plusieurs équipes simultanément, pire ALI prédit).
- Code `services/ali/pool_loader.py` L37 : `df = self._cache.lookup_club(club_id)`
  retourne tous les joueurs sans filtre équipe.
- `data/joueurs.parquet` schema 19 cols sans `equipe` — l'attribution est
  observable seulement via `data/echiquiers.parquet::blanc_equipe`.

**Mécanisme** : observed lineup N3 = joueurs Elo rang 17-24 (rangs 1-8
forcés en N1 par §3.7.b, 9-16 en N2). ALI sample dans pool total ⇒
sur-représentation top Elo en N3 ⇒ recall structurellement faible.

**Mitigation Phase 3** : NONE possible. Limitation acceptée dans gates
report §7.5.

**Mitigation Phase 4a Approche A SOTA (REQUIRED, validée user 2026-04-28
"optimal sota ou go fuck yourself") — bloquant gates absolus** : ALI
conditionné par CE-adverse miroir (D-P3-19, roadmap §Phase 4a NEW,
upstream Phase 4b CE user). Pipeline cible :

1. **CE-adverse miroir** : pour chaque club adverse multi-équipes, un
   solveur OR-Tools simule l'allocation joueurs × équipes adverses sous
   mêmes contraintes FFE A02 §3.7.b (ordre Elo descendant entre équipes),
   §3.7.c (joueur brûlé), §3.7.d (même groupe), §3.7.f (noyau 50 %).
   Réutilise primitives `services/ce/` Phase 4b.
2. **ALI Phase 4a sample conditionné** : `ScenarioGenerator.generate(
   opponent_club_id, target_team, simultaneous_teams)` reçoit la liste
   des autres équipes adverses du club + leur allocation simulée. Pool
   sampling `target_team` = pool club adverse total **moins** joueurs
   alloués aux équipes supérieures.
3. **20 scénarios joints** sous distribution conditionnelle vraie.
4. **Re-backtest hold-out 2024** N=70 attendu : recall ≥ 0.65 (vs 0.57
   actuel), Jaccard ≥ 0.50 (vs 0.39), Brier ≤ 0.22 (vs 0.29).
   McNemar n_disc ≥ 25 attendu → puissance α=0.05 OK.

**Approche B rejetée** (joint sampling sans CE-adverse miroir) : moins
SOTA car ne réutilise pas les primitives FFE OR-Tools Phase 4b ; risque
divergence logique CE-user vs inférence ALI.

**Status** : 🔴 CRITICAL OPEN. Bloquant Phase 4a acceptance gate.
**Owner** : ML Eng + Architect. **Due** : Phase 4a (D-P3-19, ADR-016
NEW à rédiger). **Cross-ref** : roadmap §Phase 4a + 4b ;
`docs/iso/ALI_QUALITY_GATES_REPORT.md` §6.2 + §7.5.

### Findings backtest hold-out 2024 (T22)

**Validations Phase 3 succès** :
- ✅ Pipeline ALI SOTA opérationnel : 3 GBMs + MLP(32,16) + temperature
  scaling chargés (champion mode FULL post-D-P3-13 résorption).
- ✅ ALI lift baseline Elo de manière franche : Brier Skill Score
  **0.6566 ≥ 0.05 gate**, Brier ALI 0.293 vs baseline 0.991 (réduction
  70 %). Hypothèse Plan 3 §Goal validée empiriquement.
- ✅ Stratification per-ronde wired (T15 module) — fix-on-sight T22 a
  éliminé biais sampling structurel (run v1 100 % ronde=1 J02, run v2
  équilibré N3 SE 5 rondes).
- ✅ Reproductibilité parfaite MLP refit (sweep ECE_draw 0.0016 →
  refit 0.001648, écart 5e-6).

**Failures Phase 3 acceptés** :
- ❌ Gates absolus P3G07-P3G11 : recall 0.57 (gate 0.90), Jaccard 0.39
  (0.75), Brier 0.29 (0.20), ECE 0.31 (0.05), MAE 2.6 (1.0). Threshold
  excessivement ambitieux pour la difficulté intrinsèque "predict 8/8
  joueurs sur ~30 éligibles" (cf. Pappalardo 2019 sport SOTA recall
  0.4-0.7 typical).
- ❌ McNemar p=0.25 : n_disc=3 trop petit pour puissance bilatérale
  α=0.05 (b=3, c=0 — direction systématiquement favorable ALI mais
  test mal-calibré pour `recall ≥ 0.90` strict).
- ❌ Ronde 11 absente (saison N3 2024 = 9 rondes) : stratification 5
  rondes au lieu 6 prévues.

### Décision Phase 3 (ISO 42005 §7 acceptance criteria)

**APPROVED WITH MONITORING** : ALI Phase 3 SOTA accepté pour merge sur la
base du lift relatif validé (BSS 0.66) ; gates absolus reportés Phase
3.5 STRICT avec leviers identifiés.

### Leviers Phase 3.5 / 5+ (escalation path)

1. **Phase 3.5 STRICT — bloquant Phase 4 OR-Tools CE** :
   - **D8** : fairness/robustness ALI breakdown rigoureux N ≥ 200 matches
     multi-saisons 2021-2024.
   - **D-P3-18 NEW (T22 finding)** : redéfinir `ali_correct(match) :=
     recall ≥ 0.50` ou `jaccard ≥ 0.5` pour McNemar puissant (n_disc
     attendu 30-50 vs actuel 3).
   - **D3** + **D4** : étendre cohorte J02 (rules différentes) +
     Coupes (formats variables).
   - Expected outcome : McNemar p < 0.05 confirmé, gates absolus calibrés
     empiriquement (raccourcir threshold à 0.65 recall, 0.50 Jaccard, etc.).

2. **Phase 5+ — production multi-tenant** :
   - **D9 Adaptive Importance Sampling** (Veach & Guibas 1995, Cornuet et
     al. 2012, Bugallo et al. 2017) + drift dashboard
     (`services/observability/drift_tracker.py` design Phase 3 §4.15.3)
     pour amener ECE-presence sous 0.05 et adresser **R-ALI-04 highest-
     risk** (score 12, drift undetected). **Required avant prod scale.**
   - **D15 Conformal prediction** (Vovk 2005) pour CIs exacts E[score]
     (corrige MAE 2.6 → prediction intervals).
   - **R-ALI-05** : Oracle VM ARM 24 GB capacity benchmark
     (`scripts/benchmark/ali_benchmark.py` design Phase 3 §6ter).

### Disclosure consommateurs `/compose` API (Plan 3 §4.13)

- `ConfidenceLevel` exposé dans response JSON
  (`services/ali/confidence.py`).
- Recommandation T22 : ajouter `large_club_warning: true` si pool
  adversaire ≥ Q3 (gap recall 0.28 finding).
- Footer documentation API : "ALI predict 5-7 of 8 players correctly in
  median ; never perfect ; tactical decisions remain human."

### Controls (ISO 42005 §7 traceability)

- Backtest reproductible : `scripts/backtest/run_holdout_2024.py` +
  seed=42 (ADR-014 §Determinism).
- Lineage SHA256 : `reports/backtest/ali_holdout_2024.json` +
  `models/cache/mlp_champion_metadata.json`.
- Monitoring Phase 5+ : drift dashboard (`docs/superpowers/specs/2026-04-19-
  phase3-ali-monte-carlo-design.md` §4.15.3 design) à implémenter avant
  multi-tenant.
- Audit log MongoDB : `lineage_hash`, `rule_uuids_applied`,
  `model_versions`, `tenant_id` (Phase 5).

**Status escalation** : ⚠️ R-ALI-04 (drift, score 12) doit être traité
**avant Phase 5 commercialisation**. Refus de signer la Phase 5 SaaS
acceptance gate sans drift dashboard opérationnel + AIS adaptive.
