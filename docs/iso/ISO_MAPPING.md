# ISO Standards Mapping - ALICE Engine

> **Document Type**: Quality Records (QR) - ISO 15289
> **Version**: 1.0.0
> **Date**: 9 Janvier 2026
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

### Scripts ML/AI

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
- `scripts/model_registry.py` - Drift monitoring
- `scripts/train_models_parallel.py` - Gestion risques training

### ISO 27001 - Information Security

Fichiers concernés:
- `scripts/model_registry.py` - Intégrité (SHA-256), Signatures (HMAC), Chiffrement (AES-256)
- `app/api/routes.py` - Sécurité API
- `app/config.py` - Gestion secrets

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

*Dernière MAJ: 2026-01-09 | ALICE Engine v0.4.0*
