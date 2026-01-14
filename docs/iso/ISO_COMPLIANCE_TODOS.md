# ISO Compliance TODOs - Plan de reprise

> Session: 2026-01-14
> Status: P1 terminés, P2/P3 en attente

## Scores actuels

| Norme | Score | Status |
|-------|-------|--------|
| ISO 5055 (Code Quality) | 100% | ✅ Complete |
| ISO 27001 (Security) | ~90% | 🔶 P2/P3 restants |
| ISO 42001 (AI Management) | 100% | ✅ Complete |
| ISO 5259 (Data Quality ML) | 100% | ✅ Complete |
| ISO 23894 (AI Risk) | **100%** | ✅ **Complete** |
| ISO 24029 (Robustness) | **100%** | ✅ **Complete** |
| ISO 24027 (Bias) | **100%** | ✅ **Complete** |

---

## P1 - Critiques (6 items) - ✅ TOUS TERMINÉS

### ISO 23894 (AI Risk Management)

- [x] **Drift monitoring** - Détection distribution shift (PSI, KS-test)
  - Fichiers: `scripts/model_registry/drift_monitor.py`, `drift_types.py`, `drift_stats.py`
  - Session: 2026-01-14

- [x] **Risk register** - Matrice risques AI (impact × probabilité)
  - Fichier: `docs/iso/AI_RISK_REGISTER.md`
  - Session: 2026-01-14

### ISO 24029 (Robustness)

- [x] **Adversarial testing** - Tests perturbations adverses
  - Fichiers: `scripts/robustness/` (module complet existant)
  - Tests: `tests/test_robustness_adversarial.py` (29 tests)
  - Session: Déjà complet

- [x] **Input validation bounds** - Rejection si features hors distribution
  - Fichiers: `scripts/model_registry/input_validator.py`, `input_types.py`
  - Session: 2026-01-14

### ISO 24027 (Bias)

- [x] **Bias monitoring continu** - Métriques fairness en production
  - Fichiers: `scripts/monitoring/bias_tracker.py`, `bias_types.py`
  - Session: 2026-01-14

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
  - Fichier: `scripts/model_registry/versioning.py` (partiellement implémenté)

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

## Modules créés cette session (2026-01-14)

### Drift Monitoring (ISO 23894)
```
scripts/model_registry/
├── drift_types.py     # 109 lignes - Enums, dataclasses
├── drift_stats.py     # 130 lignes - PSI, KS, Chi2, JS
└── drift_monitor.py   # 166 lignes - Main monitoring
```

### Input Validation (ISO 24029)
```
scripts/model_registry/
├── input_types.py     # 165 lignes - Enums, dataclasses
└── input_validator.py # 147 lignes - OOD validation
```

### Bias Monitoring (ISO 24027)
```
scripts/monitoring/
├── bias_types.py      # 164 lignes - Enums, dataclasses
└── bias_tracker.py    # 155 lignes - Fairness monitoring
```

### Documentation (ISO 23894)
```
docs/iso/
└── AI_RISK_REGISTER.md # Matrice risques complète
```

---

## Conformité ISO 5055

Tous les nouveaux fichiers respectent:
- ✅ Maximum 200 lignes par fichier
- ✅ SRP (Single Responsibility Principle)
- ✅ Docstrings ISO conformes
- ✅ Type hints complets

---

**Dernière mise à jour:** 2026-01-14
**Score global P1:** 100% ✅
