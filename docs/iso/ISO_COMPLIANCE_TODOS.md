# ISO Compliance TODOs - Plan de reprise

> Session: 2026-01-11
> Status: En attente de reprise

## Scores actuels

| Norme | Score | Status |
|-------|-------|--------|
| ISO 5055 (Code Quality) | 100% | ✅ Complete |
| ISO 27001 (Security) | ~90% | 🔶 P2/P3 restants |
| ISO 42001 (AI Management) | 100% | ✅ Complete |
| ISO 5259 (Data Quality ML) | 100% | ✅ Complete |
| ISO 23894 (AI Risk) | 82% | 🔶 P1/P2 restants |
| ISO 24029 (Robustness) | 85% | 🔶 P1/P2 restants |
| ISO 24027 (Bias) | 92% | 🔶 P1/P2 restants |

---

## P1 - Critiques (6 items)

### ISO 23894 (AI Risk Management)

- [ ] **Drift monitoring** - Détection distribution shift (PSI, KS-test)
  - Fichier: `scripts/model_registry/drift_monitor.py`

- [ ] **Risk register** - Matrice risques AI (impact × probabilité)
  - Fichier: `docs/iso/AI_RISK_REGISTER.md`

### ISO 24029 (Robustness)

- [ ] **Adversarial testing** - Tests perturbations adverses (FGSM-like)
  - Fichier: `tests/test_adversarial.py`

- [ ] **Input validation bounds** - Rejection si features hors distribution
  - Fichier: `app/schemas/prediction.py`

### ISO 24027 (Bias)

- [ ] **Bias monitoring continu** - Métriques fairness en production
  - Fichier: `scripts/monitoring/bias_tracker.py`

---

## P2 - Importants (9 items)

### ISO 27001 (Security)

- [ ] Rotation des clés API - `rotate_api_key()` avec expiration
  - Fichier: `app/api/auth.py`

- [ ] Chiffrement données au repos - AES-256 pour cache modèles
  - Fichier: `scripts/model_registry/security_encryption.py`

- [ ] Audit log MongoDB - Logger les accès DB (read/write)
  - Fichier: `app/services/mongodb.py`

### ISO 23894 (AI Risk)

- [ ] Alerting automatique - Webhook Slack/Email si drift > seuil
  - Fichier: `scripts/alerts/drift_alerter.py`

- [ ] Model rollback - Mécanisme retour version N-1 si dégradation
  - Fichier: `scripts/model_registry/versioning.py`

### ISO 24029 (Robustness)

- [ ] Confidence calibration - Platt scaling / isotonic regression
  - Fichier: `scripts/model_registry/calibration.py`

- [ ] Uncertainty quantification - Intervalle de confiance prédictions
  - Fichier: `scripts/prediction/uncertainty.py`

### ISO 24027 (Bias)

- [ ] Protected attributes check - Validation absence features sensibles
  - Fichier: `scripts/data/bias_validator.py`

- [ ] Fairness report automatique - Génération rapport post-training
  - Fichier: `scripts/model_registry/fairness_report.py`

---

## P3 - Nice-to-have (5 items)

### ISO 27001 (Security)

- [ ] HTTPS enforcement - Middleware redirect HTTP→HTTPS
  - Fichier: `app/main.py`

- [ ] Security headers - CSP, X-Frame-Options, HSTS
  - Fichier: `app/middleware/security.py`

### ISO 23894 (AI Risk)

- [ ] Risk dashboard - Visualisation risques temps réel
  - Fichier: `app/api/routes.py` endpoint `/risks`

### ISO 24029 (Robustness)

- [ ] Stress test pipeline - Tests charge + données extrêmes
  - Fichier: `tests/test_stress.py`

### ISO 24027 (Bias)

- [ ] Bias mitigation - Reweighting / adversarial debiasing
  - Fichier: `scripts/training/debiasing.py`

---

## Ordre de traitement recommandé

1. **P1 ISO 23894**: Drift monitoring (fondation pour alerting)
2. **P1 ISO 24029**: Input validation bounds (sécurité prédictions)
3. **P1 ISO 24029**: Adversarial testing (robustesse)
4. **P1 ISO 24027**: Bias monitoring (équité production)
5. **P1 ISO 23894**: Risk register (documentation)
6. Puis P2 par ordre de dépendance
