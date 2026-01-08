# ISO Standards Reference — ALICE Engine

## Normes Actives

| Norme | Focus | Priorité |
|-------|-------|----------|
| **ISO 27001** | ISMS, gestion risques sécurité | 🔴 Critique |
| **ISO 27034** | Secure coding, OWASP | 🔴 Critique |
| **ISO 5055** | Qualité code, CWE auto | 🔴 Critique |
| **ISO 25010** | Qualité système (FURPS+) | 🟠 Important |
| **ISO 25012** | Qualité données | 🟠 Important |
| **ISO 25019** | SaaS/Cloud | 🟠 Important |
| **ISO 29119** | Tests logiciels | 🟠 Important |
| **ISO 42010** | Architecture | 🟠 Important |
| **ISO 12207** | Cycle de vie | 🟡 Utile |
| **ISO 90003** | Qualité processus | 🟡 Utile |
| **ISO 15289** | Contenu documentation cycle de vie | 🟠 Important |
| **ISO 26514** | Information utilisateur logiciel | 🟠 Important |
| **ISO 26515** | Documentation en environnement agile | 🟡 Utile |
| **ISO 25065** | UX/Accessibilité | 🟡 Utile |

---

## Documentation (ISO 15289 + ISO 26514)

### Structure docs/ conforme ISO 15289

```
docs/
├── architecture/       # ISO 42010 - Architecture Description
│   ├── ARCHITECTURE.md         # Vue d'ensemble architecture
│   ├── DATA_MODEL.md           # Modèle de données
│   └── DECISIONS.md            # ADR (Architecture Decision Records)
│
├── api/                # ISO 26514 - Information for Users (API)
│   └── API_CONTRACT.md         # Contrat API OpenAPI
│
├── requirements/       # ISO 15289 - Requirements Specification
│   ├── CDC_ALICE.md            # Cahier des charges
│   └── CONTEXTE_*.md           # Contextes métier
│
├── operations/         # ISO 15289 - Operations Documentation
│   ├── DEPLOIEMENT_RENDER.md   # Guide déploiement
│   └── MAINTENANCE.md          # Procédures maintenance
│
├── development/        # ISO 15289 - Development Documentation
│   ├── CONTRIBUTING.md         # Guide contribution
│   └── PYTHON-HOOKS-SETUP.md   # Setup développeur
│
├── iso/                # ISO 15289 - Quality Records
│   └── IMPLEMENTATION_STATUS.md # Auto-généré
│
└── project/            # ISO 15289 - Project Documentation
    ├── ANALYSE_INITIALE_ALICE.md # Analyse initiale
    ├── BILAN_PARSING.md          # Resultats parsing dataset
    └── CHANGELOG.md              # Journal des modifications
```

### Types de documents ISO 15289

| Type | Code | Exemples ALICE |
|------|------|----------------|
| **Concept of Operations** | ConOps | CDC_ALICE.md |
| **System Requirements** | SyRS | CONTEXTE_*.md |
| **Architecture Description** | AD | ARCHITECTURE.md |
| **Interface Design** | IDD | API_CONTRACT.md |
| **Software User Documentation** | SUD | README.md |
| **Operations Manual** | OpsMan | DEPLOIEMENT_RENDER.md |
| **Quality Records** | QR | IMPLEMENTATION_STATUS.md |
| **Data Quality Report** | DQR | BILAN_PARSING.md |

### Contenu minimal par document (ISO 26514)

Chaque document technique doit contenir :

1. **En-tête**
   - Titre
   - Version
   - Date dernière mise à jour
   - Auteur/Responsable

2. **Introduction**
   - Objectif du document
   - Audience cible
   - Prérequis

3. **Corps**
   - Contenu structuré avec titres hiérarchiques
   - Exemples de code si applicable
   - Schémas/diagrammes si nécessaire

4. **Références**
   - Documents liés
   - Normes applicables

---

## Architecture SRP

```
feature/
├── feature.controller.ts   # Routes, HTTP uniquement
├── feature.service.ts      # Logique métier pure
├── feature.repository.ts   # Accès données uniquement
├── feature.validator.ts    # Validation Zod
├── feature.types.ts        # Interfaces/Types
└── feature.test.ts         # Tests unitaires
```

**Règles :**
- 1 fichier = 1 responsabilité
- Controller → Service → Repository (jamais l'inverse)
- Service = pur, testable, sans I/O direct
- Repository = seul à toucher MongoDB

---

## Pyramide de Tests

```
        ╱╲
       ╱E2E╲         5%  — Playwright (flux critiques)
      ╱──────╲
     ╱ Intég. ╲     15%  — API routes, DB memory
    ╱──────────╲
   ╱  Unitaires ╲   80%  — Vitest, logique pure
  ╱──────────────╲
```

| Type | Cible | Outil |
|------|-------|-------|
| Unit | Services, Flat-Six rules | Vitest |
| Intégration | Routes API, DB | Supertest + mongodb-memory-server |
| E2E | Workflows complets | Playwright |
| Accessibilité | WCAG 2.1 AA | axe-core |
| Sécurité | OWASP Top 10 | ZAP DAST |

---

## Checklist Sécurité (27034)

- [ ] Input validation (Zod)
- [ ] Output encoding
- [ ] Auth/Authz chaque route
- [ ] SQL/NoSQL injection (mongo-sanitize)
- [ ] XSS (Helmet CSP)
- [ ] CSRF tokens
- [ ] Rate limiting
- [ ] Secrets en env vars
- [ ] Logs sans données sensibles
- [ ] Dépendances à jour

---

## Qualité Code (5055)

```bash
# Avant chaque commit
npm run lint          # 0 erreurs
npm run typecheck     # 0 erreurs
npm run test          # 100% pass
npm audit             # 0 critical/high
```

**CWE prioritaires :**
- CWE-89: Injection
- CWE-79: XSS
- CWE-287: Auth bypass
- CWE-522: Credentials faibles
- CWE-798: Hardcoded secrets

---

## Qualité Données (25012)

| Critère | Implémentation |
|---------|----------------|
| Exactitude | Validation Zod stricte |
| Complétude | Required fields + defaults |
| Cohérence | Transactions MongoDB |
| Unicité | Indexes unique (ffeId, email) |
| Traçabilité | createdAt, updatedAt, audit logs |

---

## Multi-tenant (25019)

```typescript
// TOUJOURS filtrer par clubId
const players = await Player.find({ clubId: req.user.clubId });

// JAMAIS
const players = await Player.find({ _id: id }); // ❌ Fuite données
```

---

## JSDoc Standard

```typescript
/**
 * @description Valide composition via Flat-Six
 * @param {string} compositionId - ID MongoDB
 * @param {string} clubId - Isolation tenant
 * @returns {Promise<ValidationResult>}
 * @throws {NotFoundError} Composition inexistante
 * @throws {ValidationError} Règles FFE violées
 * @see ISO 25010 - Fiabilité
 */
```

---

## Commandes Rapides

```bash
# Dev
npm run dev

# Tests
npm run test              # Unitaires
npm run test:integration  # Intégration
npm run test:e2e          # E2E
npm run test:coverage     # Couverture

# Qualité
npm run lint
npm run typecheck
npm audit
npm run security:check    # Snyk

# Build
npm run build
```

---

## Contexte Hostile — Rappels

- **Mineurs** → RGPD renforcé, consentement parental
- **Paiements** → PCI-DSS awareness
- **Concurrence** → Obfuscation, rate limiting agressif
- **Attaques** → WAF, monitoring, alertes
- **Données FFE** → Scraping légal, cache, respect robots.txt

---

---

## Python Implementation (ALICE Engine)

### FFE Rules Module (`scripts/ffe_rules_features.py`)

Implementation complète des règles FFE en Python avec typage strict (ISO/IEC 5055):

```python
# Types disponibles
from scripts.ffe_rules_features import (
    TypeCompetition,      # Enum: A02, F01, C01, C03, C04, J02, J03, REG, DEP
    NiveauCompetition,    # Enum: TOP16, N1, N2, N3, N4, REGIONAL, DEPARTEMENTAL
    Sexe,                 # Enum: MASCULIN, FEMININ
    Joueur,               # dataclass: id_fide, nom, elo, sexe, nationalite, mute
    Equipe,               # dataclass: nom, club, division, ronde, groupe
    ReglesCompetition,    # TypedDict: taille_equipe, seuil_brulage, noyau, etc.
)

# Fonctions de détection
detecter_type_competition(nom: str) -> TypeCompetition
get_niveau_equipe(equipe: str) -> int  # 1=Top16, 10=plus faible
get_regles_competition(type_comp: TypeCompetition) -> ReglesCompetition

# Règle joueur brûlé (A02 Art. 3.7.c)
est_brule(joueur_id, equipe_cible, historique, seuil=3) -> bool
matchs_avant_brulage(joueur_id, equipe_sup, historique, seuil=3) -> int

# Règle noyau (A02 Art. 3.7.f)
get_noyau(equipe_nom, historique_noyau) -> set[int]
calculer_pct_noyau(composition_ids, equipe_nom, historique) -> float
valide_noyau(composition_ids, equipe, historique, regles) -> bool

# Zones d'enjeu (classement)
calculer_zone_enjeu(position, nb_equipes, division) -> str

# Validation composition
valider_composition(composition, equipe, hist_brulage, hist_noyau, regles) -> list[str]
```

### Feature Engineering (`scripts/feature_engineering.py`)

Pipeline ML intégrant les features FFE:

| Feature | Type | Source |
|---------|------|--------|
| `nb_equipes` | int | Multi-équipes joueur |
| `niveau_max` | int | Niveau hiérarchique max joué |
| `niveau_min` | int | Niveau hiérarchique min joué |
| `type_competition` | cat | A02, F01, C01, etc. |
| `multi_equipe` | bool | Joueur dans plusieurs équipes |
| `zone_enjeu` | cat | montee/danger/mi_tableau |
| `niveau_hierarchique` | int | Niveau équipe (1-10) |

### Tests (`tests/test_ffe_rules_features.py`)

66 tests couvrant:
- Détection type compétition (12 tests)
- Niveau équipe (8 tests)
- Joueur brûlé (6 tests)
- Noyau (9 tests)
- Zones d'enjeu (7 tests)
- Validation composition (8 tests)
- Règles par compétition (7 tests)
- Mouvement joueurs (3 tests)

---

## Model Registry — Production Models (ISO 42001 / 5259 / 27001)

### Vue d'ensemble

Module `scripts/model_registry.py` centralisant la normalisation des modèles ML production:

| Fonctionnalité | Norme ISO | Status |
|----------------|-----------|--------|
| Checksums SHA-256 | 27001 (Integrity) | ✅ |
| Git commit tracking | 42001 (Reproducibility) | ✅ |
| Data lineage | 5259 (Data Quality) | ✅ |
| Model Card | 42001 (AI Governance) | ✅ |
| ONNX export | 42001 (Portability) | ✅ |
| Feature importance | 42001 (Explainability) | ✅ |
| Validation intégrité | 27001 (Security) | ✅ |
| Rollback mechanism | 27001 (Recovery) | ✅ |
| Signature HMAC-SHA256 | 27001 (Authenticity) | ✅ |
| Schema validation | 5259 (Data Quality) | ✅ |
| Retention policy | 27001 (Lifecycle) | ✅ |
| Chiffrement AES-256 | 27001 (Confidentiality) | ✅ |
| Drift monitoring | 5259/42001 (Monitoring) | ✅ |

### Dataclasses Production

```python
from scripts.model_registry import (
    # Core
    DataLineage,           # Traçabilité données train/valid/test
    EnvironmentInfo,       # Environnement d'entraînement
    ModelArtifact,         # Artefact modèle avec checksum
    ProductionModelCard,   # Model Card ISO 42001

    # Validation
    SchemaValidationResult,  # Résultat validation schema

    # Drift Monitoring
    DriftMetrics,          # Métriques drift par ronde
    DriftReport,           # Rapport drift saison
)
```

### Fonctions Clés

```python
# === INTÉGRITÉ (ISO 27001) ===
compute_file_checksum(path)           # SHA-256 hex (64 chars)
validate_model_integrity(artifact)    # Vérifie checksum
load_model_with_validation(artifact)  # Charge avec vérification

# === SIGNATURE (ISO 27001) ===
generate_signing_key()                # Clé HMAC 32 bytes
compute_model_signature(path, key)    # HMAC-SHA256
verify_model_signature(path, sig, key)  # Vérification

# === CHIFFREMENT (ISO 27001) ===
generate_encryption_key()             # Clé AES-256 (32 bytes)
encrypt_model_file(path, key)         # AES-256-GCM + nonce
decrypt_model_file(path, key)         # Déchiffrement authentifié
encrypt_model_directory(version_dir)  # Batch chiffrement
decrypt_model_directory(version_dir)  # Batch déchiffrement

# === DATA LINEAGE (ISO 5259) ===
compute_data_lineage(train_path, ...) # Traçabilité complète
compute_dataframe_hash(df)            # Hash pandas déterministe

# === SCHEMA VALIDATION (ISO 5259) ===
validate_dataframe_schema(df)         # Valide colonnes/types
validate_train_valid_test_schema(...)  # Cohérence splits

# === DRIFT MONITORING (ISO 5259/42001) ===
compute_psi(baseline, current)        # Population Stability Index
compute_drift_metrics(round, preds, actuals, ...)  # Métriques ronde
create_drift_report(season, version, elo)  # Nouveau rapport
add_round_to_drift_report(report, ...)    # Ajouter ronde
check_drift_status(report)            # Recommandation

# === VERSIONING (ISO 42001) ===
save_production_models(models, ...)   # Sauvegarde normalisée
list_model_versions(models_dir)       # Liste versions
rollback_to_version(models_dir, ver)  # Rollback
apply_retention_policy(dir, max=10)   # Nettoyage anciennes versions
```

### Seuils Drift Monitoring

| Métrique | Warning | Critical |
|----------|---------|----------|
| PSI | ≥ 0.1 | ≥ 0.25 |
| Accuracy drop | ≥ 5% | - |
| ELO shift | ≥ 50 pts | - |

### Recommandations Drift

| Status | Signification | Action |
|--------|---------------|--------|
| `OK` | Modèle stable | Aucune |
| `MONITOR_CLOSELY` | Légère dégradation | Surveiller |
| `RETRAIN_RECOMMENDED` | Drift significatif | Planifier retraining |
| `RETRAIN_URGENT` | Drift critique | Retraining immédiat |

### Tests (`tests/test_model_registry.py`)

74 tests couvrant:
- Checksums et hash (4 tests)
- Git info (2 tests)
- Package versions (2 tests)
- Environment info (2 tests)
- Data lineage (2 tests)
- Model artifacts (2 tests)
- Model card (1 test)
- Version listing (2 tests)
- Rollback (2 tests)
- Validate integrity (3 tests)
- Feature importance (5 tests)
- HMAC signature (8 tests)
- Schema validation (7 tests)
- Retention policy (6 tests)
- AES-256 encryption (12 tests)
- Drift monitoring (15 tests)

---

*Dernière MAJ: 2026-01-08 | ALICE Engine v0.3.0*
