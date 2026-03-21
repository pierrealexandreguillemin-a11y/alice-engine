# ISO Standards Reference — ALICE Engine

## Normes Actives

### Normes Générales Logiciel

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

### Normes ML/AI (ALICE Engine)

| Norme | Focus | Priorité | Exigences Clés |
|-------|-------|----------|----------------|
| **ISO/IEC 42001:2023** | AI Management System (certifiable) | 🔴 Critique | Model Card, Traçabilité, Gouvernance AI |
| **ISO/IEC 42005:2025** | AI System Impact Assessment | 🔴 Critique | Évaluation impacts individus/groupes/société |
| **ISO/IEC 23894:2023** | AI Risk Management | 🔴 Critique | Évaluation risques AI, Mitigation biais |
| **ISO/IEC 5259:2024** | Data Quality for ML | 🔴 Critique | Qualité données entraînement, Lineage |
| **ISO/IEC 25059:2023** | AI Quality Model | 🟠 Important | Métriques qualité modèles, Benchmarks |
| **ISO/IEC 24029** | Neural Network Robustness | 🟠 Important | Tests adversariaux, Robustesse |
| **ISO/IEC TR 24027** | Bias in AI | 🟠 Important | Détection/mitigation biais, Fairness |

### Matrice Exigences ML/AI

| Norme | Exigence | Implémentation ALICE |
|-------|----------|---------------------|
| ISO 42001 | Model Card | `ProductionModelCard` dans `model_registry.py`. V8: MultiClass 3-way, ~156 features |
| ISO 42001 | Traçabilité | Git commit tracking, versioning modèles. V8: `alice-training-v8` slug |
| ISO 42001 | Explicabilité | Feature importance per class (MultiClass). SHAP pending. |
| ISO 5259 | Qualité données | `DataLineage`, `validate_dataframe_schema()`. V8: forfaits excluded |
| ISO 5259 | Lineage | `compute_data_lineage()` train/valid/test. V8: ~156 features, temporal split ≤2022/2023/≥2024 |
| ISO 27001 | Intégrité | SHA-256 checksums, HMAC signatures |
| ISO 27001 | Confidentialité | Chiffrement AES-256-GCM |
| ISO 27001 | Auditabilité | Logs, retention policy, drift reports |
| ISO 23894 | Risques AI | Drift monitoring PSI, alertes seuils |
| ISO 24027 | Biais | `scripts/fairness/bias_detection.py` - SPD, EOD, DIR |
| ISO 24029 | Robustesse | `scripts/robustness/adversarial_tests.py` - Tests adversariaux |
| ISO 42005 | Impact Assessment | `scripts/autogluon/iso_impact_assessment.py` - V7/AutoGluon era. V8: redo after MultiClass training |

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
app/
├── api/
│   ├── routes.py           # Routes FastAPI, HTTP uniquement
│   └── schemas.py          # Validation Pydantic (request/response)
├── config.py               # Configuration centralisée
└── main.py                 # Point d'entrée FastAPI

services/
├── inference.py            # Logique métier ALI (pure, testable)
├── composer.py             # Logique métier CE (pure, testable)
└── data_loader.py          # Repository - accès données uniquement

scripts/
├── model_registry.py       # Gestion modèles ML
├── feature_engineering.py  # Pipeline features
└── ffe_rules_features.py   # Règles métier FFE
```

**Règles :**
- 1 fichier = 1 responsabilité
- Routes → Services → DataLoader (jamais l'inverse)
- Services = purs, testables, sans I/O direct
- DataLoader = seul à toucher MongoDB/fichiers

---

## Pyramide de Tests

```
        ╱╲
       ╱E2E╲         5%  — Playwright (flux critiques)
      ╱──────╲
     ╱ Intég. ╲     15%  — API routes, DB memory
    ╱──────────╲
   ╱  Unitaires ╲   80%  — Pytest, logique pure
  ╱──────────────╲
```

| Type | Cible | Outil |
|------|-------|-------|
| Unit | Services, FFE rules, ML pipelines | Pytest |
| Intégration | Routes FastAPI, DB | Pytest + httpx + mongomock |
| E2E | Workflows complets | Playwright |
| Sécurité | OWASP Top 10 | Bandit + pip-audit |
| Complexité | Cyclomatic | Xenon (max B) |

---

## Checklist Sécurité (27034)

- [x] Input validation (Pydantic)
- [x] Output encoding (FastAPI auto)
- [ ] Auth/Authz chaque route (JWT)
- [x] NoSQL injection (Pydantic + motor)
- [x] Rate limiting (slowapi)
- [x] Secrets en env vars (.env + pydantic-settings)
- [x] Logs sans données sensibles (structlog)
- [x] Dépendances à jour (pip-audit pre-push)
- [x] Secrets détection (gitleaks pre-commit)
- [x] Static analysis (Bandit, Ruff S rules)

---

## Qualité Code (5055)

```bash
# Pre-commit (automatique)
ruff check .          # 0 erreurs lint
ruff format .         # Format uniforme
mypy app/ services/   # 0 erreurs types
bandit -r app/        # 0 vulnérabilités

# Pre-push (automatique)
pytest --cov-fail-under=70   # Coverage minimum
xenon --max-absolute=B       # Complexité max
pip-audit --strict           # 0 vulns dépendances
```

**CWE prioritaires :**
- CWE-89: Injection (SQL/NoSQL)
- CWE-287: Auth bypass
- CWE-522: Credentials faibles
- CWE-798: Hardcoded secrets (gitleaks)
- CWE-502: Deserialization (pickle)

---

## Qualité Données (25012 + 5259)

| Critère | Implémentation |
|---------|----------------|
| Exactitude | Validation Pydantic v2 stricte |
| Complétude | Required fields + `Field(default=...)` |
| Cohérence | motor async + mongomock tests |
| Unicité | Indexes unique (ffeId, clubId) |
| Traçabilité | `DataLineage` dans `model_registry.py` |
| Lineage | `compute_data_lineage()` train/valid/test |
| Schema | `validate_dataframe_schema()` |

---

## Multi-tenant (25019)

```python
# TOUJOURS filtrer par clubId (isolation tenant)
async def get_players(club_id: str, db: AsyncIOMotorDatabase) -> list[Player]:
    return await db.players.find({"clubId": club_id}).to_list(None)

# JAMAIS - fuite de données inter-clubs
async def get_player_bad(player_id: str, db: AsyncIOMotorDatabase):
    return await db.players.find_one({"_id": player_id})  # ❌ Pas de clubId
```

**Règles CDC_ALICE.md §2.2 :**
- Isolation stricte par `clubId`
- Modèle global par défaut
- Modèle spécifique par club si >50-100 matchs historiques

---

## Docstring Standard (Google Style)

```python
def predict_lineup(
    club_id: str,
    opponent_club_id: str,
    round_number: int,
) -> PredictionResult:
    """Prédit la composition adverse probable (ALI).

    Analyse l'historique des compositions adverses pour estimer
    la probabilité de présence de chaque joueur.

    Args:
        club_id: Identifiant FFE du club utilisateur (isolation tenant).
        opponent_club_id: Identifiant FFE du club adverse.
        round_number: Numéro de la ronde (1-11 typiquement).

    Returns:
        PredictionResult contenant les joueurs probables avec probabilités.

    Raises:
        ClubNotFoundError: Club adverse inexistant.
        InsufficientDataError: Historique insuffisant (<10 matchs).

    ISO Compliance:
        - ISO/IEC 42001:2023 - AI Management (traçabilité prédictions)
        - ISO/IEC 5259:2024 - Data Quality (validation entrées)

    See Also:
        CDC_ALICE.md §2.1 F.1 - Prédiction de Composition Adverse
    """
```

---

## Commandes Rapides

```bash
# Installation
make install              # pip install -r requirements.txt + dev
make hooks                # pre-commit install (tous hooks)

# Dev
uvicorn app.main:app --reload

# Tests
make test                 # pytest tests/
make test-cov             # pytest --cov --cov-fail-under=70

# Qualité (pre-commit)
make quality              # ruff + mypy + bandit
make lint                 # ruff check + format
make typecheck            # mypy

# Sécurité (pre-push)
pip-audit --strict        # Vulnérabilités dépendances
xenon --max-absolute=B    # Complexité cyclomatique

# Artefacts
make graphs               # Génère graphs/dependencies.svg
make iso-docs             # MAJ docs/iso/IMPLEMENTATION_STATUS.md

# Validation complète
make all                  # install + quality + test-cov + graphs
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

Pipeline ML V8 — MultiClass 3-way (win/draw/loss). ~156 features across 13 categories.

**Spec complète :** `docs/superpowers/specs/2026-03-21-multiclass-v8-design.md`

| Category | Features | Columns | Status V8 |
|----------|----------|---------|-----------|
| 1. Match context | echiquier, est_domicile, type_competition, division, etc. | 12 | Unchanged |
| 2. Player strength | Elo, titres, categorie_age, K_coefficient | 10 | +4 new (age, K) |
| 3. Player form W/D/L | win_rate/draw_rate/expected_score (recent, color, trend) | 20 | Refactored from 12 binary |
| 4. Draw priors | avg_elo, elo_proximity, draw_rate_prior, player, equipe | 8 | All new (Pawnalyze) |
| 5. Presence/availability | taux_presence, regularite, role_type, echiquier_prefere | 16 | Wire existing code |
| 6. Pressure | clutch_win/draw (zone_enjeu, not score_dom) | 6 | Fixed (leakage) |
| 7. H2H | h2h_win_rate, h2h_draw_rate, h2h_exists | 4 | Refactored W/D/L |
| 8. Standings | position, ecart, points_cumules, zone_enjeu | 16 | Unchanged (draw-aware) |
| 9. Club behavior | rotation, noyau, win/draw_rate_home | 16 | Wire + refactor W/D/L |
| 10. Vases communiquants | joueur_promu/relegue, team_rank, reinforcement_rate | 16 | All new (REGLES_FFE §4) |
| 11. FFE regulatory | est_dans_noyau, ffe_nb_equipes, joueur_fantome | 20 | Unchanged |
| 12. Elo trajectory | momentum, elo_trajectory | 4 | Unchanged |
| 13. Composition strategy | decalage_position, echiquier_moyen | 8 | Minor fix (rolling) |

**Key V8 changes vs V7 :**
- Target: 3-class (loss=0, draw=1, win=2). Forfaits (2.0) excluded from everything.
- All result-derived features decomposed into (win_rate, draw_rate) instead of `.mean()`
- All player features stratified by `type_competition` (national ≠ regional)
- All player features rolling (3 seasons) instead of global career
- `score_dom`/`score_ext` removed (match score leakage)
- `clutch_factor` fixed: uses `zone_enjeu` instead of `score_dom`

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

**V8 note:** MultiClass 3-way requires monitoring PSI per class (3 distributions).
Current PSI code works per-column — call 3x (P(loss), P(draw), P(win)). Phase C scope.

| Métrique | Warning | Critical | V8 Adaptation |
|----------|---------|----------|---------------|
| PSI | ≥ 0.1 | ≥ 0.25 | Per-class: PSI(P_loss), PSI(P_draw), PSI(P_win) |
| Log loss drift | ≥ 5% increase | ≥ 10% increase | NEW — replaces accuracy drop |
| ELO shift | ≥ 50 pts | - | Unchanged |
| Draw rate drift | ≥ 3% absolute | ≥ 5% | NEW — mean(P(draw)) vs observed |

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

---

## Mapping Fichiers → Normes ISO

### Scripts ML/AI

| Fichier | Normes Applicables | Exigences |
|---------|-------------------|-----------|
| `scripts/model_registry.py` | ISO 42001, 5259, 27001 | Model Card, Lineage, Intégrité, Chiffrement |
| `scripts/feature_engineering.py` | ISO 5259, 42001 | Qualité features, Traçabilité transformations |
| `scripts/ffe_rules_features.py` | ISO 5259, 25012 | Validation règles métier, Qualité données |
| `scripts/train_models_parallel.py` | ISO 42001, 23894 | Gouvernance training, Gestion risques |
| `scripts/ensemble_stacking.py` | ISO 42001, 25059 | Métriques qualité, Explicabilité |
| `scripts/evaluate_models.py` | ISO 25059, 29119 | Benchmarks, Tests modèles |
| `scripts/parse_dataset.py` | ISO 5259, 25012 | Parsing qualité, Validation schéma |
| `scripts/fairness/bias_detection.py` | ISO 24027, 42001 | Détection biais, Fairness metrics |
| `scripts/robustness/adversarial_tests.py` | ISO 24029, 42001 | Tests robustesse, Perturbations |

### Scripts AutoGluon (ML Training Pipeline)

| Fichier | Normes Applicables | Exigences |
|---------|-------------------|-----------|
| `scripts/autogluon/trainer.py` | ISO 42001, 5055 | Pipeline AutoGluon, Traçabilité MLflow |
| `scripts/autogluon/run_training.py` | ISO 42001, 5055 | Runner Phase 3, <50 lignes |
| `scripts/autogluon/config.py` | ISO 42001 | Configuration YAML, Presets |
| `scripts/autogluon/iso_robustness.py` | ISO 24029, 5055 | Validation robustesse basique |
| `scripts/autogluon/iso_robustness_enhanced.py` | ISO 24029-1/2, 5055 | **Tests formels: bruit, dropout, consistance, monotonicité** |
| `scripts/autogluon/iso_fairness.py` | ISO 24027, 5055 | Validation fairness basique |
| `scripts/autogluon/iso_fairness_enhanced.py` | ISO 24027 Clause 7-8, 5055 | **Root cause analysis, equalized odds, mitigations** |
| `scripts/autogluon/iso_model_card.py` | ISO 42001, 5055 | Génération Model Card JSON |
| `scripts/autogluon/iso_impact_assessment.py` | ISO 42005, 5055 | Impact Assessment basique |
| `scripts/autogluon/iso_impact_assessment_enhanced.py` | ISO 42005:2025, 5055 | **10-step process, monitoring triggers, lifecycle** |
| `scripts/autogluon/iso_validator.py` | ISO 42001, 24029, 24027, 42005 | Orchestration validations ISO |
| `scripts/autogluon/run_iso_validation_enhanced.py` | ISO 24027/24029/42005, 5055 | **Runner validation complète** |

### Scripts Comparison (Statistical Tests)

| Fichier | Normes Applicables | Exigences |
|---------|-------------------|-----------|
| `scripts/comparison/mcnemar_test.py` | ISO 24029, 5055 | Test McNemar 5x2cv (Dietterich 1998) |
| `scripts/comparison/run_mcnemar.py` | ISO 24029, 5055 | Runner Phase 4.4, <50 lignes |
| `scripts/comparison/pipeline.py` | ISO 24029, 42001 | Pipeline comparaison complète |

### Scripts Reports

| Fichier | Normes Applicables | Exigences |
|---------|-------------------|-----------|
| `scripts/reports/generate_iso25059.py` | ISO 25059, 5055 | Rapport final qualité AI, <50 lignes |

### Scripts Baseline (Comparaison Indépendante)

| Fichier | Normes Applicables | Exigences |
|---------|-------------------|-----------|
| `scripts/baseline/catboost_baseline.py` | ISO 24029, 5055 | CatBoost isolé, réutilise `scripts/training` |
| `scripts/baseline/xgboost_baseline.py` | ISO 24029, 5055 | XGBoost isolé, réutilise `scripts/training` |
| `scripts/baseline/lightgbm_baseline.py` | ISO 24029, 5055 | LightGBM isolé, réutilise `scripts/training` |
| `scripts/baseline/run_baselines.py` | ISO 24029, 5055 | Runner séquentiel, comparaison AutoGluon |
| `scripts/baseline/types.py` | ISO 5055 | Types partagés (BaselineMetrics) |

### Scripts Serving (Déploiement MLflow/Render)

| Fichier | Normes Applicables | Exigences |
|---------|-------------------|-----------|
| `scripts/serving/pyfunc_wrapper.py` | ISO 42001, 5055 | MLflow PyFunc wrapper universel |
| `scripts/serving/deploy_to_mlflow.py` | ISO 42001, 5055 | Script déploiement Render |

### Scripts Agents (Architecture AG-A/MLZero)

| Fichier | Normes Applicables | Exigences |
|---------|-------------------|-----------|
| `scripts/agents/semantic_memory.py` | ISO 42001, 24027, 24029 | Base connaissance ISO, seuils, mitigations |
| `scripts/agents/iterative_refinement.py` | ISO 42001, 24027 | Corrections automatiques fairness/robustness |

### Scripts AIMMS (AI Management System Lifecycle)

| Fichier | Normes Applicables | Exigences |
|---------|-------------------|-----------|
| `scripts/aimms/aimms_types.py` | ISO 42001, 5055 | Types lifecycle (LifecyclePhase), configs, résultats |
| `scripts/aimms/postprocessor.py` | ISO 42001 Clause 8.2/9.1/10.2 | Orchestration: calibration → uncertainty → alerting |
| `scripts/aimms/run_iso42001_postprocessing.py` | ISO 42001, 5055 | Runner post-training (<50 lignes) |

### Hooks Claude Code

| Fichier | Normes Applicables | Exigences |
|---------|-------------------|-----------|
| `.claude/hooks/pre_check.py` | ISO 5055, 27034 | PreToolUse guards, Validation pré-exécution |
| `.claude/settings.json` | ISO 27034 | Configuration hooks sécurisée |

### Services

| Fichier | Normes Applicables | Exigences |
|---------|-------------------|-----------|
| `services/inference.py` | ISO 42001, 27001 | Prédictions traçables, Sécurité API |
| `services/data_loader.py` | ISO 5259, 25012 | Chargement données validées |
| `services/composer.py` | ISO 25010, 5055 | Qualité code, Architecture |

### API

| Fichier | Normes Applicables | Exigences |
|---------|-------------------|-----------|
| `app/api/routes.py` | ISO 27001, 27034 | Sécurité endpoints, Input validation |
| `app/api/schemas.py` | ISO 5259, 25012 | Validation schémas, Types stricts |
| `app/config.py` | ISO 27001 | Gestion secrets, Configuration sécurisée |

### Tests

| Fichier | Normes Applicables | Exigences |
|---------|-------------------|-----------|
| `tests/test_model_registry.py` | ISO 29119, 42001 | Tests intégrité, coverage |
| `tests/test_feature_engineering.py` | ISO 29119, 5259 | Tests features, validation |
| `tests/test_ffe_rules_features.py` | ISO 29119, 25012 | Tests règles métier |
| `tests/test_fairness_bias_detection.py` | ISO 29119, 24027 | Tests détection biais (29 tests) |
| `tests/test_robustness_adversarial.py` | ISO 29119, 24029 | Tests robustesse (29 tests) |

### Documentation

| Dossier/Fichier | Normes Applicables | Type ISO 15289 |
|-----------------|-------------------|----------------|
| `docs/requirements/CDC_ALICE.md` | ISO 15289 | ConOps |
| `docs/requirements/FEATURE_SPECIFICATION.md` | ISO 5259, 15289 | SyRS |
| `docs/architecture/` | ISO 42010, 15289 | AD |
| `docs/api/` | ISO 26514, 15289 | IDD |
| `docs/iso/IMPLEMENTATION_STATUS.md` | ISO 15289 | QR |

---

## En-têtes Fichiers Python (Template)

Chaque fichier Python doit inclure un docstring avec les normes applicables:

```python
"""
Module: nom_module.py
Description: Description du module

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (Model Card, Traçabilité)
- ISO/IEC 5259:2024 - Data Quality for ML (Lineage, Validation)
- ISO/IEC 27001 - Information Security (Intégrité, Chiffrement)

Author: ALICE Engine Team
Last Updated: YYYY-MM-DD
"""
```

---

---

## Ressources Externes (Templates & Best Practices)

### AutoGluon Pipeline

| Ressource | URL | Usage |
|-----------|-----|-------|
| AutoGluon GitHub | https://github.com/autogluon/autogluon | Repo officiel, exemples |
| AWS SageMaker Pipeline | https://github.com/aws-samples/automl-pipeline-with-autogluon-sagemaker-lambda | Template pipeline cloud |
| TabArena Benchmark | https://github.com/autogluon/tabarena | Benchmarks tabulaires 2025 |
| Custom Feature Generator | `examples/tabular/example_custom_feature_generator.py` | Feature engineering avancé |

### Ensemble Stacking (CatBoost/XGBoost/LightGBM)

| Ressource | URL | Usage |
|-----------|-----|-------|
| Medium Tutorial | https://medium.com/@stevechesa/stacking-ensembles | Guide stacking complet |
| Kaggle Notebook | https://www.kaggle.com/code/ayanabil11/lightgbm-xgboost-and-catboost-stacking | Template Kaggle |
| Research Paper | https://www.researchgate.net/publication/397047638 | Framework théorique |
| Unifiedbooster | https://thierrymoudiki.github.io/blog/2024/08/05/python/r/unibooster | Interface unifiée |

### MLflow + AutoGluon Integration

| Ressource | URL | Usage |
|-----------|-----|-------|
| PyFunc Wrapper | https://github.com/psmishra7/autogluon_mlflow | Notebook intégration |
| MLOps Pipeline | https://github.com/AnderGarro/autogluon_airflow_mlflow | Airflow + AutoGluon + MLflow |
| Databricks Guide | https://community.databricks.com/t5/machine-learning/autogluon-mlflow-integration/td-p/111423 | Best practices Databricks |

### Notes AutoGluon 2024-2025

- **Kaggle 2024**: Top 3 dans 15/18 compétitions tabulaires, 7 premières places
- **AutoGluon-Assistant (AG-A)**: Pilotage par LLM (Bedrock/OpenAI)
- **TabArena 2025**: Remplace TabRepo, benchmark vivant
- **MLflow Flavor**: Pas encore officiel, utiliser PyFunc wrapper

---

## Résultats Pipeline ISO (2026-01-17)

### Dernière Validation

| Norme | Métrique | Valeur | Seuil | Status |
|-------|----------|--------|-------|--------|
| ISO 24027 | Demographic Parity | **70.20%** | ≥80% | ⚠️ CAUTION |
| ISO 24027 | Equalized Odds | 52.50% | ≥80% | ⚠️ |
| ISO 24029 | Noise Tolerance | **99.28%** | ≥95% | ✅ ROBUST |
| ISO 24029 | Stability Score | 95.58% | ≥90% | ✅ |
| ISO 24029 | Consistency | 95.60% | ≥95% | ✅ |
| ISO 42005 | Impact Level | MEDIUM | - | ✅ |
| ISO 42005 | Recommendation | APPROVED_WITH_MONITORING | - | ✅ |

### Notes Importantes

1. **ligue_code vide (118k samples)**: Ce sont les compétitions NATIONALES (N1-N4, Top 16, Coupes, UNSS) où le code ligue n'est pas applicable. **Ce n'est PAS une erreur de données.**

2. **Feature critique**: `diff_elo` (différence ELO blanc-noir) - impact de 7.5% sur accuracy si dropout.

3. **Prochaine review**: 2026-03-03 (monitoring trimestriel)

---

*Dernière MAJ: 2026-01-18 | ALICE Engine v0.6.1 - AIMMS Integration*
