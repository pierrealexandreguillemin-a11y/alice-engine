# Methodologie Entrainement ML - Alice-Engine

> **Document Type**: Technical Specification - ISO 15289
> **Version**: 1.0.0
> **Date creation**: 8 Janvier 2026
> **Derniere MAJ**: 8 Janvier 2026
> **Conformite**: ISO/IEC 42001, 23894, 25059, 5259, 29119

---

## Table des matieres

1. [Executive Summary](#1-executive-summary)
2. [Gap Analysis - Etat Actuel](#2-gap-analysis---etat-actuel)
3. [Standards Industrie ML](#3-standards-industrie-ml)
4. [Conformite ISO pour ML/IA](#4-conformite-iso-pour-mlia)
5. [Methodologie CatBoost (Fabricant)](#5-methodologie-catboost-fabricant)
6. [Pipeline Entrainement Conforme](#6-pipeline-entrainement-conforme)
7. [Checklists de Conformite](#7-checklists-de-conformite)
8. [Plan d'Implementation](#8-plan-dimplementation)

---

## 1. Executive Summary

### 1.1 Verdict Global

| Composant | Statut | Conformite |
|-----------|--------|------------|
| Feature Engineering | ✅ Bon | 70% |
| Data Splits | ⚠️ Risque leakage | 50% |
| Model Training | ❌ **Inexistant** | 0% |
| Experiment Tracking | ❌ **Inexistant** | 0% |
| Model Persistence | ❌ **Inexistant** | 0% |
| Production Inference | ⚠️ Stub only | 10% |
| ISO Conformite | ⚠️ Partiel | 40% |

### 1.2 Gaps Critiques

```
BLOQUANTS PRODUCTION:
1. Aucun script d'entrainement (models jamais sauvegardes)
2. Data leakage: features calculees AVANT split temporel
3. Aucun experiment tracking (MLflow, W&B)
4. Inference service incomplet (TODO dans le code)
```

### 1.3 Effort Estime

| Phase | Taches | Effort |
|-------|--------|--------|
| Phase 1 (Critique) | Training, persistence, fix leakage | 1-2 semaines |
| Phase 2 (Important) | MLOps, CV, metrics | 1 semaine |
| Phase 3 (Production) | Monitoring, CI/CD | 1 semaine |

---

## 2. Gap Analysis - Etat Actuel

### 2.1 Fichiers Analyses

| Fichier | Lignes | Role | Statut |
|---------|--------|------|--------|
| `scripts/evaluate_models.py` | 400+ | Evaluation 3 modeles | ⚠️ Incomplet |
| `scripts/feature_engineering.py` | 689 | Pipeline features | ⚠️ Leakage |
| `services/inference.py` | 150+ | Service prediction | ❌ Stub |
| `app/api/routes.py` | 130+ | Endpoints API | ⚠️ /train stub |

### 2.2 Gaps Detailles

#### GAP-001: Pas de Script d'Entrainement
**Severite**: CRITIQUE
**Fichier**: `scripts/evaluate_models.py`

```python
# Actuel (lignes 131-229): Modeles entraines mais JAMAIS sauvegardes
model_cb, time_cb = train_catboost(X_train, y_train, X_valid, y_valid, cat_features)
metrics_cb_valid = evaluate_model(model_cb, X_valid, y_valid, "CatBoost")
# MANQUANT: model_cb.save_model("models/catboost_v1.cbm")
```

**Impact**: Impossible de deployer en production

---

#### GAP-002: Data Leakage dans Feature Engineering
**Severite**: CRITIQUE
**Fichier**: `scripts/feature_engineering.py` (lignes 588-598)

```python
# PROBLEME: Features calculees sur TOUT le dataset AVANT split
club_reliability = extract_club_reliability(df)  # Utilise donnees futures!
player_reliability = extract_player_reliability(df)

# PUIS split temporel
train, valid, test = temporal_split(df)  # Trop tard!
```

**Impact**: Metriques d'evaluation invalides (trop optimistes)

---

#### GAP-003: Aucun Experiment Tracking
**Severite**: CRITIQUE

```
MANQUANT:
- MLflow / Weights & Biases / Neptune
- Versioning des hyperparametres
- Comparaison entre runs
- Artefacts versionnes
```

**Actuel**: Seul un CSV de resultats (`ml_evaluation_results.csv`)

---

#### GAP-004: Hyperparametres Hardcodes
**Severite**: HAUTE
**Fichier**: `scripts/evaluate_models.py` (lignes 143-151)

```python
model = CatBoostClassifier(
    iterations=500,          # Pourquoi 500?
    learning_rate=0.05,      # Pourquoi 0.05?
    depth=6,                 # Pourquoi 6?
    random_seed=42,          # OK
)
```

**Impact**: Pas de tuning systematique, performances sous-optimales

---

#### GAP-005: Inference Service Incomplet
**Severite**: CRITIQUE
**Fichier**: `services/inference.py` (lignes 54-74)

```python
def load_model(self) -> bool:
    # TODO: Charger le modele CatBoost ou XGBoost
    # self.model = CatBoostClassifier()
    # self.model.load_model(self.model_path)
    self.is_loaded = False  # Toujours False!
    return False
```

**Impact**: API retourne toujours fallback, jamais de vraie prediction

---

#### GAP-006: Metriques Insuffisantes
**Severite**: MOYENNE
**Fichier**: `scripts/evaluate_models.py` (lignes 232-254)

```python
# Actuel: seulement AUC et Accuracy
return {
    "auc_roc": auc,
    "accuracy": acc,
}

# MANQUANT:
# - Precision, Recall, F1
# - Matrice de confusion
# - Courbe ROC
# - Calibration
# - Feature importance / SHAP
```

---

### 2.3 Tableau Recapitulatif des Gaps

| ID | Gap | Fichier | Ligne | Severite | Effort |
|----|-----|---------|-------|----------|--------|
| GAP-001 | Pas de training script | evaluate_models.py | 131-229 | CRITIQUE | 2j |
| GAP-002 | Data leakage features | feature_engineering.py | 588-598 | CRITIQUE | 1j |
| GAP-003 | Pas d'experiment tracking | N/A | N/A | CRITIQUE | 2j |
| GAP-004 | Hyperparams hardcodes | evaluate_models.py | 143-151 | HAUTE | 1j |
| GAP-005 | Inference stub | inference.py | 54-74 | CRITIQUE | 1j |
| GAP-006 | Metriques insuffisantes | evaluate_models.py | 232-254 | MOYENNE | 0.5j |
| GAP-007 | Pas de cross-validation | N/A | N/A | HAUTE | 1j |
| GAP-008 | Pas de model registry | N/A | N/A | HAUTE | 1j |
| GAP-009 | Pas de tests ML | tests/ | N/A | HAUTE | 2j |
| GAP-010 | Dependencies non lockees | requirements.txt | ALL | MOYENNE | 0.5j |

---

## 3. Standards Industrie ML

### 3.1 MLOps Maturity Model (Google/Netflix/Meta)

| Niveau | Description | Alice-Engine |
|--------|-------------|--------------|
| **Level 0** | Manuel, silos | ← Actuel |
| **Level 1** | Automation partielle, CI/CD ML | Cible court terme |
| **Level 2** | Full automation, continuous training | Cible moyen terme |

### 3.2 Best Practices Industrie

#### 3.2.1 Lifecycle Management

```
1. Experimental Phase → 2. Production Phase → 3. Monitoring Phase
       ↑                                              ↓
       └──────────── Retraining Loop ←───────────────┘
```

**Recommandations**:
- Model Registry central (MLflow, DVC)
- Experiment tracking obligatoire
- Versionning code + data + models

#### 3.2.2 Cross-Validation Temporelle

```
INTERDIT pour donnees temporelles:
- K-Fold standard (leakage futur → passe)

RECOMMANDE:
- Expanding Window (anchored):
  Train 2020 → Test 2021
  Train 2020-2021 → Test 2022
  Train 2020-2022 → Test 2023

- Walk-Forward (sliding):
  Window fixe qui avance dans le temps
```

**Implementation scikit-learn**:
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

#### 3.2.3 Hyperparameter Tuning

**Recommande**: Optuna (Bayesian optimization + pruning)

```python
import optuna

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
    }
    model = CatBoostClassifier(**params, verbose=0)
    model.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=50)
    return model.get_best_score()['validation']['AUC']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

#### 3.2.4 Data Quality Validation

**Outil recommande**: Great Expectations

```python
import great_expectations as gx

# Definir expectations
expectation_suite = gx.ExpectationSuite("training_data")
expectation_suite.add_expectation(
    gx.ExpectColumnToExist(column="elo_blanc")
)
expectation_suite.add_expectation(
    gx.ExpectColumnValuesToBeBetween(
        column="elo_blanc",
        min_value=0,
        max_value=3000
    )
)
```

#### 3.2.5 Reproducibility Standards

**Tier Bronze** (minimum):
- Code, data, models publics

**Tier Silver** (recommande):
- + Installation single-command
- + Ordre d'execution documente
- + Seeds deterministes
- + OS/system requirements

```python
# Configuration globale des seeds
import random
import numpy as np

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    # Pour CatBoost: random_seed=seed dans params
```

---

## 4. Conformite ISO pour ML/IA

### 4.1 Standards ISO Applicables

| Standard | Titre | Application Alice |
|----------|-------|-------------------|
| **ISO/IEC 42001:2023** | AI Management System | Gouvernance IA |
| **ISO/IEC 23894:2023** | AI Risk Management | Gestion risques ML |
| **ISO/IEC 25059:2023** | AI System Quality | Qualite systeme |
| **ISO/IEC 5259:2024** | Data Quality for ML | Qualite donnees |
| **ISO/IEC 24029** | Neural Network Robustness | Tests robustesse |
| **ISO/IEC TR 24027** | Bias in AI Systems | Detection biais |
| **ISO/IEC 22989:2022** | AI Terminology | Vocabulaire |

### 4.2 ISO/IEC 42001 - AI Management System (AIMS)

**Structure (Clauses 4-10)**:

| Clause | Focus | Implementation Alice |
|--------|-------|---------------------|
| 4 | Contexte | Definir scope AIMS |
| 5 | Leadership | Politique IA, responsabilites |
| 6 | Planning | Objectifs, risk assessment |
| 7 | Support | Ressources, competences, docs |
| 8 | Operation | Dev systeme IA, controles |
| 9 | Performance | Monitoring, audit interne |
| 10 | Amelioration | Actions correctives |

**Controles Annex A**:
- [ ] Risk management controls
- [ ] Model and data governance
- [ ] Transparency and explainability
- [ ] Human oversight mechanisms
- [ ] Bias mitigation
- [ ] Security controls

### 4.3 ISO/IEC 23894 - AI Risk Management

**Processus (base ISO 31000)**:

```
1. Etablissement contexte
2. Identification risques IA
   - Drift (data, concept)
   - Biais
   - Decisions black-box
   - Inputs adverses
3. Analyse risques
4. Evaluation risques
5. Traitement risques
6. Monitoring continu
```

**Risques identifies pour Alice**:

| Risque | Probabilite | Impact | Mitigation |
|--------|-------------|--------|------------|
| Data drift | Haute | Moyen | Monitoring features |
| Predictions biaisees | Moyenne | Haut | Tests fairness |
| Model staleness | Haute | Moyen | Retraining schedule |
| Adversarial inputs | Faible | Haut | Input validation |

### 4.4 ISO/IEC 25059 - AI Quality Model

**Caracteristiques qualite**:

| Caracteristique | Definition | Metrique Alice |
|-----------------|------------|----------------|
| **Accuracy** | Exactitude outputs | AUC-ROC, F1 |
| **Interpretability** | Explicabilite | SHAP values |
| **Robustness** | Performance stable | Tests perturbation |
| **Fairness** | Equite groupes | Demographic parity |
| **Privacy** | Protection donnees | Anonymisation Elo |
| **Security** | Resistance attaques | Input validation |

### 4.5 ISO/IEC 5259 - Data Quality for ML

**5 Parties du standard**:

| Partie | Titre | Application |
|--------|-------|-------------|
| Part 1 | Overview, Terminology | Vocabulaire commun |
| Part 2 | Data Quality Measures | Metriques qualite |
| Part 3 | Data Quality Management (certifiable) | DQMS |
| Part 4 | Process Framework | Processus par type ML |
| Part 5 | Governance Framework | Gouvernance donnees |

**Checklist Data Quality**:
- [ ] Schema validation (colonnes, types)
- [ ] Statistical checks (distributions)
- [ ] Missing values policy
- [ ] Outlier detection
- [ ] Temporal consistency
- [ ] Cross-feature validation

---

## 5. Methodologie CatBoost (Fabricant)

### 5.1 Configuration Recommandee (Doc Officielle)

```python
from catboost import CatBoostClassifier, Pool

# Configuration production-ready
model = CatBoostClassifier(
    # Core params
    iterations=1000,
    learning_rate=0.03,          # Plus bas = plus stable
    depth=6,                      # 4-10 selon complexite

    # Regularization
    l2_leaf_reg=3,               # 1-10
    random_strength=1,           # 0-10

    # Categorical handling (natif)
    cat_features=categorical_cols,
    one_hot_max_size=10,         # One-hot si <10 categories

    # Early stopping
    early_stopping_rounds=50,
    eval_metric='AUC',
    use_best_model=True,

    # Reproducibility
    random_seed=42,

    # Performance
    task_type='CPU',             # ou 'GPU' si disponible
    thread_count=-1,

    # Class imbalance
    auto_class_weights='Balanced',

    verbose=100,
)
```

### 5.2 Cross-Validation Native CatBoost

```python
from catboost import Pool, cv

cv_dataset = Pool(X, y, cat_features=categorical_cols)

params = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.03,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
}

cv_results = cv(
    cv_dataset,
    params,
    fold_count=5,
    stratified=True,
    shuffle=True,
    partition_random_seed=42,
    early_stopping_rounds=50,
    plot=False,
)

print(f"CV AUC: {cv_results['test-AUC-mean'].max():.4f}")
```

### 5.3 Feature Importance (SHAP natif)

```python
# PredictionValuesChange (default)
importance = model.get_feature_importance()

# SHAP Values (natif CatBoost - rapide)
shap_values = model.get_feature_importance(
    data=train_pool,
    type='ShapValues'
)

# LossFunctionChange
importance_lfc = model.get_feature_importance(
    data=train_pool,
    type='LossFunctionChange'
)
```

### 5.4 Serialization Production

```python
# Format recommande: CBM (natif, plus rapide)
model.save_model("models/catboost_production.cbm")

# Format portable: JSON
model.save_model("models/catboost_production.json", format="json")

# Chargement
loaded_model = CatBoostClassifier()
loaded_model.load_model("models/catboost_production.cbm")

# Verification
assert (model.predict(X_test) == loaded_model.predict(X_test)).all()
```

### 5.5 Gestion Classes Desequilibrees

```python
# Option 1: Auto-balancing (recommande)
model = CatBoostClassifier(
    auto_class_weights='Balanced'
)

# Option 2: Poids manuels
# Si 900 negatifs, 100 positifs:
model = CatBoostClassifier(
    class_weights=[1, 9]  # [class_0, class_1]
)

# Option 3: scale_pos_weight (binaire)
model = CatBoostClassifier(
    scale_pos_weight=9
)
```

---

## 6. Pipeline Entrainement Conforme

### 6.1 Architecture Cible

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE ML CONFORME                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  DATA    │───→│ FEATURE  │───→│ TRAINING │───→│ SERVING  │  │
│  │ QUALITY  │    │   ENG    │    │          │    │          │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │               │               │               │         │
│       ▼               ▼               ▼               ▼         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │   GX     │    │ DVC/Git  │    │  MLflow  │    │ Monitor  │  │
│  │ Validate │    │ Version  │    │  Track   │    │  Drift   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Script d'Entrainement Conforme

```python
# scripts/train_model.py - A CREER

import mlflow
import optuna
from catboost import CatBoostClassifier, Pool, cv
from pathlib import Path
import json
from datetime import datetime

# Configuration
CONFIG = {
    "model_dir": Path("models"),
    "experiment_name": "alice-engine-training",
    "random_seed": 42,
}

def set_seeds(seed: int):
    """Reproducibilite globale."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

def load_data_with_validation():
    """Charge donnees avec validation qualite."""
    # TODO: Ajouter Great Expectations
    pass

def train_with_cv(X, y, cat_features, params):
    """Entrainement avec cross-validation temporelle."""
    pool = Pool(X, y, cat_features=cat_features)

    cv_results = cv(
        pool,
        params,
        fold_count=5,
        stratified=True,
        early_stopping_rounds=50,
    )

    return cv_results

def train_final_model(X_train, y_train, X_valid, y_valid,
                      cat_features, params):
    """Entraine modele final avec early stopping."""
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)

    model = CatBoostClassifier(**params)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=50,
        verbose=100,
    )

    return model

def save_model_with_metadata(model, metrics, params, version):
    """Sauvegarde modele + metadata (model card)."""
    model_path = CONFIG["model_dir"] / f"catboost_v{version}.cbm"
    metadata_path = CONFIG["model_dir"] / f"catboost_v{version}_metadata.json"

    # Sauvegarder modele
    model.save_model(str(model_path))

    # Sauvegarder metadata (model card)
    metadata = {
        "version": version,
        "training_date": datetime.now().isoformat(),
        "framework": "CatBoost",
        "framework_version": "1.2.7",
        "metrics": metrics,
        "params": params,
        "feature_importance": dict(zip(
            model.feature_names_,
            model.get_feature_importance().tolist()
        )),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return model_path, metadata_path

def main():
    set_seeds(CONFIG["random_seed"])

    # MLflow tracking
    mlflow.set_experiment(CONFIG["experiment_name"])

    with mlflow.start_run():
        # 1. Load data
        X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()

        # 2. Hyperparameter tuning (Optuna)
        best_params = tune_hyperparameters(X_train, y_train)
        mlflow.log_params(best_params)

        # 3. Cross-validation
        cv_results = train_with_cv(X_train, y_train, cat_features, best_params)
        mlflow.log_metric("cv_auc_mean", cv_results["test-AUC-mean"].max())

        # 4. Train final model
        model = train_final_model(
            X_train, y_train, X_valid, y_valid,
            cat_features, best_params
        )

        # 5. Evaluate on test set
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # 6. Save model with metadata
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path, _ = save_model_with_metadata(
            model, metrics, best_params, version
        )

        # 7. Log artifacts
        mlflow.log_artifact(str(model_path))
        mlflow.catboost.log_model(model, "model")

if __name__ == "__main__":
    main()
```

### 6.3 Fix Data Leakage

```python
# scripts/feature_engineering.py - A CORRIGER

def extract_features_no_leakage(df: pd.DataFrame):
    """
    Features calculees APRES split temporel pour eviter leakage.
    """
    # 1. SPLIT TEMPOREL D'ABORD
    train, valid, test = temporal_split(df)

    # 2. Features calculees PER-SPLIT
    train_features = compute_reliability_features(train)
    valid_features = compute_reliability_features(
        pd.concat([train, valid])  # Valid peut voir train
    )
    test_features = compute_reliability_features(
        pd.concat([train, valid, test])  # Sequentiel
    )

    # 3. Joindre features aux splits
    train = train.merge(train_features, on="joueur_id")
    valid = valid.merge(valid_features, on="joueur_id")
    test = test.merge(test_features, on="joueur_id")

    return train, valid, test
```

---

## 7. Checklists de Conformite

### 7.1 Checklist Pre-Entrainement

| # | Item | Statut | Responsable |
|---|------|--------|-------------|
| 1 | [ ] Data quality validated (schema, stats) | | |
| 2 | [ ] No data leakage in features | | |
| 3 | [ ] Temporal split correct | | |
| 4 | [ ] Class balance analyzed | | |
| 5 | [ ] Features documented | | |
| 6 | [ ] Seeds configured globally | | |
| 7 | [ ] MLflow experiment created | | |

### 7.2 Checklist Entrainement

| # | Item | Statut | Responsable |
|---|------|--------|-------------|
| 1 | [ ] Cross-validation performed | | |
| 2 | [ ] Hyperparameters tuned (Optuna) | | |
| 3 | [ ] Early stopping configured | | |
| 4 | [ ] All metrics logged (AUC, F1, Precision, Recall) | | |
| 5 | [ ] Feature importance calculated | | |
| 6 | [ ] Model saved (.cbm format) | | |
| 7 | [ ] Metadata/model card generated | | |

### 7.3 Checklist Post-Entrainement

| # | Item | Statut | Responsable |
|---|------|--------|-------------|
| 1 | [ ] Model loaded and verified | | |
| 2 | [ ] Predictions match saved model | | |
| 3 | [ ] Inference latency acceptable | | |
| 4 | [ ] Model registered in MLflow | | |
| 5 | [ ] Documentation updated | | |
| 6 | [ ] Tests passed | | |

### 7.4 Checklist ISO Conformite

| Standard | Requirement | Status |
|----------|-------------|--------|
| **ISO 42001** | AI policy defined | [ ] |
| **ISO 42001** | Risk assessment done | [ ] |
| **ISO 42001** | Human oversight documented | [ ] |
| **ISO 23894** | AI risks identified | [ ] |
| **ISO 23894** | Risk treatment plan | [ ] |
| **ISO 25059** | Quality metrics defined | [ ] |
| **ISO 5259** | Data quality validated | [ ] |
| **ISO 24027** | Bias assessment done | [ ] |
| **ISO 29119** | Tests documented | [ ] |

---

## 8. Plan d'Implementation

### 8.1 Phase 1 - Critique (Semaine 1-2)

| Tache | Priorite | Effort | Livrable |
|-------|----------|--------|----------|
| Creer `train_model.py` | P0 | 2j | Script fonctionnel |
| Fix data leakage | P0 | 1j | Feature eng corrige |
| Installer MLflow | P0 | 0.5j | Tracking operationnel |
| Implementer model persistence | P0 | 1j | Models/ avec .cbm |
| Completer inference.py | P0 | 1j | Service fonctionnel |

### 8.2 Phase 2 - Important (Semaine 3)

| Tache | Priorite | Effort | Livrable |
|-------|----------|--------|----------|
| Ajouter Optuna tuning | P1 | 1j | Best params |
| Implementer CV temporelle | P1 | 1j | TimeSeriesSplit |
| Ajouter metriques completes | P1 | 0.5j | F1, confusion, SHAP |
| Config hyperparams YAML | P1 | 0.5j | config/hyperparams.yaml |
| Tests ML pipeline | P1 | 2j | tests/test_ml_pipeline.py |

### 8.3 Phase 3 - Production (Semaine 4)

| Tache | Priorite | Effort | Livrable |
|-------|----------|--------|----------|
| Data validation (GX) | P2 | 1j | Expectations suite |
| Model monitoring | P2 | 1j | Drift detection |
| CI/CD pipeline | P2 | 1j | GitHub Actions |
| Documentation complete | P2 | 1j | Docs a jour |
| Lock dependencies | P2 | 0.5j | requirements.lock |

### 8.4 Metriques de Succes

| Metrique | Actuel | Cible Phase 1 | Cible Final |
|----------|--------|---------------|-------------|
| AUC-ROC | 0.7527 | 0.75+ | 0.80+ |
| Model persistence | Non | Oui | Oui + versioning |
| Experiment tracking | Non | MLflow | MLflow + registry |
| Test coverage ML | 0% | 50% | 80% |
| Data leakage | Oui | Non | Non + validated |
| ISO conformite | 40% | 60% | 80% |

---

## Annexes

### A. Comparaison CatBoost vs XGBoost vs LightGBM

| Critere | CatBoost | XGBoost | LightGBM |
|---------|----------|---------|----------|
| **Vitesse train** | Lent (5min) | Moyen (10s) | Rapide (8s) |
| **Categorical natif** | **Meilleur** | Manuel | Bon |
| **Out-of-box** | **Meilleur** | Bon | Bon |
| **GPU** | Bon | **Meilleur** | Bon |
| **Memoire** | Moyen | Eleve | **Meilleur** |
| **Documentation** | Bonne | **Meilleure** | Bonne |

**Choix Alice-Engine**: CatBoost (categorical features, minimal tuning)

### B. Ressources

**Documentation officielle**:
- [CatBoost Docs](https://catboost.ai/docs/)
- [MLflow Docs](https://mlflow.org/docs/latest/)
- [Optuna Docs](https://optuna.org/)
- [Great Expectations](https://greatexpectations.io/)

**Standards ISO**:
- [ISO/IEC 42001:2023](https://www.iso.org/standard/42001)
- [ISO/IEC 23894:2023](https://www.iso.org/standard/77304.html)
- [ISO/IEC 5259 Series](https://www.iso.org/standard/81088.html)

---

## Historique des modifications

| Version | Date | Auteur | Modifications |
|---------|------|--------|---------------|
| 1.0.0 | 2026-01-08 | Claude Code | Creation initiale |

---

*Document genere selon ISO 15289 - Technical Specification*
*Conformite: ISO/IEC 42001, 23894, 25059, 5259, 29119*
