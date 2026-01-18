# ISO Standards Mapping - ALICE Engine

> **Document Type**: Quality Records (QR) - ISO 15289
> **Version**: 2.2.0
> **Date**: 18 Janvier 2026
> **Objectif**: Traçabilité explicite des normes ISO par fichier/dossier

---

## Normes ML/AI Applicables

| Norme | Focus | Priorité |
|-------|-------|----------|
| **ISO/IEC 42001:2023** | AI Management System (certifiable) | Critique |
| **ISO/IEC 23894:2023** | AI Risk Management | Critique |
| **ISO/IEC 5259:2024** | Data Quality for ML | Critique |
| **ISO/IEC 25059:2023** | AI Quality Model | Important |
| **ISO/IEC 24029** | Neural Network Robustness | Important |
| **ISO/IEC TR 24027** | Bias in AI | Important |

---

## Matrice Exigences

| Norme | Exigence | Fichier(s) Implémentation |
|-------|----------|--------------------------|
| ISO 42001 | Model Card | `scripts/model_registry.py` |
| ISO 42001 | Traçabilité | `scripts/model_registry.py` |
| ISO 42001 | Explicabilité | `scripts/model_registry.py` |
| ISO 5259 | Qualité données | `scripts/feature_engineering.py` |
| ISO 5259 | Lineage | `scripts/model_registry.py` |
| ISO 27001 | Intégrité | `scripts/model_registry.py` |
| ISO 27001 | Auditabilité | `scripts/model_registry.py` |

---

## Mapping Complet par Fichier

### Scripts ML/AI Core

| Fichier | Normes | Exigences Couvertes |
|---------|--------|---------------------|
| `scripts/model_registry.py` | ISO 42001, 5259, 27001 | Model Card, Lineage, Intégrité, Chiffrement, Signatures |
| `scripts/feature_engineering.py` | ISO 5259, 42001 | Qualité features, Traçabilité transformations |
| `scripts/ffe_rules_features.py` | ISO 5259, 25012 | Validation règles métier, Qualité données |
| `scripts/train_models_parallel.py` | ISO 42001, 23894 | Gouvernance training, Gestion risques |
| `scripts/ensemble_stacking.py` | ISO 42001, 25059 | Métriques qualité, Explicabilité |
| `scripts/evaluate_models.py` | ISO 25059, 29119 | Benchmarks, Tests modèles |
| `scripts/parse_dataset.py` | ISO 5259, 25012 | Parsing qualité, Validation schéma |
| `scripts/ml_types.py` | ISO 5055, 5259 | Types stricts, Qualité code |

### Scripts AutoGluon (ISO Validation)

| Fichier | Normes | Exigences Couvertes |
|---------|--------|---------------------|
| `scripts/autogluon/trainer.py` | ISO 42001, 5055 | Pipeline AutoGluon, MLflow tracking |
| `scripts/autogluon/iso_fairness_enhanced.py` | ISO 24027 Clause 7-8 | Root cause, equalized odds, mitigations |
| `scripts/autogluon/iso_robustness_enhanced.py` | ISO 24029-1/2 | Bruit, dropout, consistance, monotonicité |
| `scripts/autogluon/iso_impact_assessment_enhanced.py` | ISO 42005:2025 | 10-step process, monitoring triggers |
| `scripts/autogluon/iso_model_card.py` | ISO 42001 | Model Card JSON |
| `scripts/autogluon/run_iso_validation_enhanced.py` | ISO 24027/24029/42005 | Runner validation complète |

### Scripts Baseline (Comparaison Indépendante)

| Fichier | Normes | Exigences Couvertes |
|---------|--------|---------------------|
| `scripts/baseline/catboost_baseline.py` | ISO 24029, 5055 | CatBoost isolé, réutilise training |
| `scripts/baseline/xgboost_baseline.py` | ISO 24029, 5055 | XGBoost isolé |
| `scripts/baseline/lightgbm_baseline.py` | ISO 24029, 5055 | LightGBM isolé |
| `scripts/baseline/run_baselines.py` | ISO 24029, 5055 | Runner séquentiel |

### Scripts Serving (Déploiement)

| Fichier | Normes | Exigences Couvertes |
|---------|--------|---------------------|
| `scripts/serving/pyfunc_wrapper.py` | ISO 42001, 5055 | MLflow PyFunc pour Render |
| `scripts/serving/deploy_to_mlflow.py` | ISO 42001 | Déploiement automatisé |

### Scripts Agents (Architecture AG-A)

| Fichier | Normes | Exigences Couvertes |
|---------|--------|---------------------|
| `scripts/agents/semantic_memory.py` | ISO 42001, 24027, 24029 | Base connaissance ISO |
| `scripts/agents/iterative_refinement.py` | ISO 42001, 24027 | Corrections automatiques |

### Scripts Alerts (ISO 23894)

| Fichier | Normes | Exigences Couvertes |
|---------|--------|---------------------|
| `scripts/alerts/alert_types.py` | ISO 23894, 5055 | Types alertes, severity levels |
| `scripts/alerts/drift_alerter.py` | ISO 23894 | Slack webhook, cooldown, auto-alerting |

### Scripts Calibration (ISO 24029)

| Fichier | Normes | Exigences Couvertes |
|---------|--------|---------------------|
| `scripts/calibration/calibrator_types.py` | ISO 24029, 5055 | Types calibration, métriques |
| `scripts/calibration/calibrator.py` | ISO 24029 | Platt scaling, isotonic regression, ECE |

### Scripts Uncertainty (ISO 24029)

| Fichier | Normes | Exigences Couvertes |
|---------|--------|---------------------|
| `scripts/uncertainty/uncertainty_types.py` | ISO 24029, 5055 | Types intervalles, métriques |
| `scripts/uncertainty/conformal.py` | ISO 24029 | Conformal prediction, couverture garantie |

### Scripts AIMMS (ISO 42001 - AI Management System)

| Fichier | Normes | Exigences Couvertes |
|---------|--------|---------------------|
| `scripts/aimms/aimms_types.py` | ISO 42001, 5055 | Types lifecycle, configs, résultats |
| `scripts/aimms/postprocessor.py` | ISO 42001 Clause 8.2/9.1 | Orchestration calibration→uncertainty→alerting |
| `scripts/aimms/run_iso42001_postprocessing.py` | ISO 42001, 5055 | Runner post-training (<50 lignes) |

### Scripts Comparison

| Fichier | Normes | Exigences Couvertes |
|---------|--------|---------------------|
| `scripts/comparison/mcnemar_test.py` | ISO 24029, 29119 | Test McNemar 5x2cv |
| `scripts/comparison/run_mcnemar.py` | ISO 24029, 5055 | Runner comparaison |

### Services

| Fichier | Normes | Exigences Couvertes |
|---------|--------|---------------------|
| `services/inference.py` | ISO 42001, 27001 | Prédictions traçables, Sécurité |
| `services/data_loader.py` | ISO 5259, 25012 | Chargement données validées |
| `services/composer.py` | ISO 25010, 5055 | Qualité code, Architecture |

### API

| Fichier | Normes | Exigences Couvertes |
|---------|--------|---------------------|
| `app/api/routes.py` | ISO 27001, 27034 | Sécurité endpoints, Input validation |
| `app/api/schemas.py` | ISO 5259, 25012 | Validation schémas, Types stricts |
| `app/config.py` | ISO 27001 | Gestion secrets, Configuration sécurisée |

### Tests

| Fichier | Normes | Exigences Couvertes |
|---------|--------|---------------------|
| `tests/test_model_registry.py` | ISO 29119, 42001 | 74 tests - intégrité, signatures, drift |
| `tests/test_feature_engineering.py` | ISO 29119, 5259 | Tests features, validation schéma |
| `tests/test_ffe_rules_features.py` | ISO 29119, 25012 | 66 tests - règles métier FFE |
| `tests/test_ensemble_stacking.py` | ISO 29119, 42001 | Tests ensemble learning |
| `tests/aimms/test_aimms.py` | ISO 29119, 42001 | 13 tests - AIMMS postprocessor |

### Documentation

| Fichier/Dossier | Normes | Type ISO 15289 |
|-----------------|--------|----------------|
| `docs/requirements/CDC_ALICE.md` | ISO 15289, 42001 | ConOps |
| `docs/requirements/FEATURE_SPECIFICATION.md` | ISO 5259, 15289 | SyRS |
| `docs/requirements/REGLES_FFE_ALICE.md` | ISO 25012, 15289 | SyRS |
| `docs/architecture/` | ISO 42010, 15289 | AD |
| `docs/api/` | ISO 26514, 15289 | IDD |
| `docs/iso/IMPLEMENTATION_STATUS.md` | ISO 15289 | QR |
| `docs/iso/ISO_MAPPING.md` | ISO 15289 | QR |

---

## Référence Rapide par Norme

### ISO/IEC 42001:2023 - AI Management System

Fichiers concernés:
- `scripts/model_registry.py` - Model Card, versioning
- `scripts/train_models_parallel.py` - Gouvernance training
- `scripts/ensemble_stacking.py` - Explicabilité
- `scripts/aimms/` - **AIMMS Lifecycle (Clause 8.2, 9.1, 10.2)**
  - `postprocessor.py` - Orchestration calibration→uncertainty→alerting
  - `run_iso42001_postprocessing.py` - Runner post-training
- `services/inference.py` - Traçabilité prédictions
- `docs/requirements/CDC_ALICE.md` - Spécifications AI

### ISO/IEC 5259:2024 - Data Quality for ML

Fichiers concernés:
- `scripts/model_registry.py` - Data lineage
- `scripts/feature_engineering.py` - Qualité features
- `scripts/ffe_rules_features.py` - Validation règles
- `scripts/parse_dataset.py` - Parsing qualité
- `app/api/schemas.py` - Validation schémas
- `docs/requirements/FEATURE_SPECIFICATION.md` - Spécifications features

### ISO/IEC 23894:2023 - AI Risk Management

Fichiers concernés:
- `scripts/model_registry/drift.py` - Drift monitoring (PSI, KS)
- `scripts/alerts/drift_alerter.py` - Alerting automatique (Slack webhook)
- `scripts/train_models_parallel.py` - Gestion risques training

### ISO 27001 - Information Security

Fichiers concernés:
- `scripts/model_registry.py` - Intégrité (SHA-256), Signatures (HMAC), Chiffrement (AES-256)
- `app/api/routes.py` - Sécurité API
- `app/config.py` - Gestion secrets

### ISO/IEC 24029 - Neural Network Robustness

Fichiers concernés:
- `scripts/calibration/calibrator.py` - Confidence calibration (Platt, isotonic)
- `scripts/uncertainty/conformal.py` - Uncertainty quantification (conformal prediction)
- `scripts/autogluon/iso_robustness_enhanced.py` - Tests robustesse (bruit, dropout)
- `scripts/robustness/` - Module robustesse adversariale

### ISO 29119 - Software Testing

Fichiers concernés:
- `tests/` - Tous les fichiers de tests
- Coverage minimum: 80%

---

## Template En-tête Fichier Python

```python
"""
Module: nom_module.py
Description: Description du module

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System
- ISO/IEC 5259:2024 - Data Quality for ML
- ISO/IEC 27001 - Information Security

Author: ALICE Engine Team
Last Updated: YYYY-MM-DD
"""
```

---

*Dernière MAJ: 2026-01-18 | ALICE Engine v0.6.0 - AIMMS Integration*
