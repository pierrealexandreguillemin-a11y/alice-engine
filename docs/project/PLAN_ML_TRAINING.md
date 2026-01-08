# Plan RAZ - Entrainement ML Conforme

> **Version**: 1.0.0
> **Date**: 8 Janvier 2026
> **Statut**: APPROUVE - Pret pour implementation
> **Conformite**: ISO/IEC 42001, 5259, 29119 | MLOps Level 1

---

## 1. Decision RAZ

### 1.1 Justification

| Probleme | Impact | Decision |
|----------|--------|----------|
| Data leakage features | Metriques invalides | RAZ features |
| Modeles non sauvegardes | Pas de production | RAZ training |
| Pas de stacking | Sous-optimal | Ajouter ensemble |
| Pas d'experiment tracking | Pas de reproductibilite | Ajouter MLflow |

### 1.2 Ce qui est conserve

- [x] Dataset parse (echiquiers.parquet, joueurs.parquet)
- [x] Regles FFE (ffe_rules_features.py)
- [x] Architecture API (FastAPI)
- [x] Tests existants

### 1.3 Ce qui est refait

- [ ] Feature engineering (correction leakage)
- [ ] Scripts d'entrainement (parallelise + stacking)
- [ ] Model persistence + metadata
- [ ] Experiment tracking

---

## 2. Architecture Cible

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE ML CONFORME - ALICE ENGINE                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐                                                        │
│  │   RAW DATA   │                                                        │
│  │  .parquet    │                                                        │
│  └──────┬───────┘                                                        │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────┐     TEMPORAL SPLIT FIRST (no leakage)                 │
│  │   SPLIT      │────────────────────────────────────────┐              │
│  │  TEMPOREL    │                                        │              │
│  └──────┬───────┘                                        │              │
│         │                                                │              │
│    ┌────┴────┬────────────┐                              │              │
│    ▼         ▼            ▼                              │              │
│ ┌──────┐ ┌──────┐    ┌──────┐                           │              │
│ │TRAIN │ │VALID │    │ TEST │                           │              │
│ │≤2022 │ │ 2023 │    │>2023 │                           │              │
│ └──┬───┘ └──┬───┘    └──┬───┘                           │              │
│    │        │           │                                │              │
│    ▼        ▼           ▼                                │              │
│  ┌──────────────────────────────────┐                   │              │
│  │  FEATURE ENGINEERING PER SPLIT   │ ← Pas de leakage  │              │
│  │  (fiabilite, forme, FFE rules)   │                   │              │
│  └──────────────┬───────────────────┘                   │              │
│                 │                                        │              │
│                 ▼                                        │              │
│  ┌──────────────────────────────────────────────────────┴────────────┐ │
│  │                    PARALLEL TRAINING                               │ │
│  │  ┌───────────┐    ┌───────────┐    ┌───────────┐                  │ │
│  │  │ CatBoost  │    │  XGBoost  │    │ LightGBM  │                  │ │
│  │  │  Thread 1 │    │  Thread 2 │    │  Thread 3 │                  │ │
│  │  └─────┬─────┘    └─────┬─────┘    └─────┬─────┘                  │ │
│  │        │                │                │                         │ │
│  │        ▼                ▼                ▼                         │ │
│  │  ┌───────────────────────────────────────────────────────────┐    │ │
│  │  │              K-FOLD OUT-OF-FOLD PREDICTIONS               │    │ │
│  │  │                    (5-fold CV)                            │    │ │
│  │  └───────────────────────────┬───────────────────────────────┘    │ │
│  │                              │                                     │ │
│  │                              ▼                                     │ │
│  │  ┌───────────────────────────────────────────────────────────┐    │ │
│  │  │                 STACKING META-LEARNER                     │    │ │
│  │  │              (LogisticRegression / Ridge)                 │    │ │
│  │  └───────────────────────────┬───────────────────────────────┘    │ │
│  └──────────────────────────────┼────────────────────────────────────┘ │
│                                 │                                       │
│                                 ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      MODEL SELECTION                              │  │
│  │  if stacking_auc > best_single_auc + 0.01:                       │  │
│  │      use_stacking = True                                          │  │
│  │  else:                                                            │  │
│  │      use_best_single = True                                       │  │
│  └──────────────────────────────┬───────────────────────────────────┘  │
│                                 │                                       │
│                                 ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      MODEL REGISTRY                               │  │
│  │  models/                                                          │  │
│  │  ├── catboost_v{version}.cbm                                     │  │
│  │  ├── xgboost_v{version}.ubj                                      │  │
│  │  ├── lightgbm_v{version}.txt                                     │  │
│  │  ├── stacking_meta_v{version}.joblib                             │  │
│  │  └── model_metadata.json                                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Correction Data Leakage

### 3.1 Probleme Actuel

```python
# feature_engineering.py:588-619 - PROBLEME
club_reliability = extract_club_reliability(df)  # TOUT le dataset
player_reliability = extract_player_reliability(df)
# ...
train, valid, test = temporal_split(df_clean)  # Split APRES = LEAKAGE
```

### 3.2 Solution

```python
# NOUVEAU: Split AVANT, features PER SPLIT
def extract_features_no_leakage(df: pd.DataFrame) -> tuple:
    """
    1. Split temporel D'ABORD
    2. Features calculees PER SPLIT (train ne voit que train)
    """
    # 1. Split temporel
    train_raw, valid_raw, test_raw = temporal_split(df)

    # 2. Features UNIQUEMENT sur donnees visibles
    # Train: features sur train only
    train_features = compute_all_features(train_raw)
    train = train_raw.merge(train_features, on="joueur_id")

    # Valid: features sur train + valid (valid peut voir train)
    valid_history = pd.concat([train_raw, valid_raw])
    valid_features = compute_all_features(valid_history)
    valid = valid_raw.merge(valid_features, on="joueur_id")

    # Test: features sur train + valid + test
    test_history = pd.concat([train_raw, valid_raw, test_raw])
    test_features = compute_all_features(test_history)
    test = test_raw.merge(test_features, on="joueur_id")

    return train, valid, test
```

---

## 4. Entrainement Parallele

### 4.1 Implementation ThreadPoolExecutor

```python
# scripts/train_models_parallel.py
from concurrent.futures import ThreadPoolExecutor, as_completed
import mlflow

def train_all_models_parallel(
    X_train, y_train, X_valid, y_valid, cat_features
) -> dict[str, tuple]:
    """
    Entraine CatBoost, XGBoost, LightGBM en parallele.
    Retourne {model_name: (model, train_time, metrics)}
    """
    results = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(train_catboost, X_train, y_train,
                          X_valid, y_valid, cat_features): "CatBoost",
            executor.submit(train_xgboost, X_train, y_train,
                          X_valid, y_valid): "XGBoost",
            executor.submit(train_lightgbm, X_train, y_train,
                          X_valid, y_valid, cat_features): "LightGBM",
        }

        for future in as_completed(futures):
            name = futures[future]
            model, train_time = future.result()
            metrics = evaluate_model(model, X_valid, y_valid)
            results[name] = (model, train_time, metrics)

            # Log to MLflow
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)

    return results
```

### 4.2 Configuration Hyperparametres

```yaml
# config/hyperparameters.yaml
catboost:
  iterations: 1000
  learning_rate: 0.03
  depth: 6
  l2_leaf_reg: 3
  early_stopping_rounds: 50
  random_seed: 42
  task_type: CPU  # ou GPU

xgboost:
  n_estimators: 1000
  learning_rate: 0.03
  max_depth: 6
  reg_lambda: 1
  early_stopping_rounds: 50
  random_state: 42
  tree_method: hist

lightgbm:
  n_estimators: 1000
  learning_rate: 0.03
  num_leaves: 63  # 2^6 - 1
  reg_lambda: 1
  early_stopping_rounds: 50
  random_state: 42
  verbose: -1
```

---

## 5. Stacking Ensemble

### 5.1 Architecture Stacking

```
LEVEL 0 (Base Models):
┌─────────────────────────────────────────────────────────┐
│  5-Fold Cross-Validation                                │
│                                                         │
│  Fold 1: Train[2,3,4,5] → Predict[1] ──┐               │
│  Fold 2: Train[1,3,4,5] → Predict[2] ──┤               │
│  Fold 3: Train[1,2,4,5] → Predict[3] ──┼─→ OOF Preds   │
│  Fold 4: Train[1,2,3,5] → Predict[4] ──┤               │
│  Fold 5: Train[1,2,3,4] → Predict[5] ──┘               │
│                                                         │
│  CatBoost OOF: [p1, p2, ..., pN]                       │
│  XGBoost OOF:  [p1, p2, ..., pN]                       │
│  LightGBM OOF: [p1, p2, ..., pN]                       │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
LEVEL 1 (Meta-Learner):
┌─────────────────────────────────────────────────────────┐
│  X_meta = [catboost_oof, xgboost_oof, lightgbm_oof]    │
│  y_meta = y_train                                       │
│                                                         │
│  meta_model = LogisticRegression(C=1.0)                │
│  meta_model.fit(X_meta, y_meta)                        │
│                                                         │
│  Coefficients → Poids optimal de chaque modele         │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Implementation Stacking

```python
# scripts/ensemble_stacking.py
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import numpy as np

def create_stacking_ensemble(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_folds: int = 5,
) -> tuple:
    """
    Cree un ensemble stacking avec out-of-fold predictions.

    Returns:
        meta_model: Meta-learner entraine
        oof_predictions: Predictions OOF pour train
        test_predictions: Predictions moyennees pour test
    """
    n_train = len(X_train)
    n_test = len(X_test)
    n_models = len(models)

    # Matrices OOF
    oof_preds = np.zeros((n_train, n_models))
    test_preds = np.zeros((n_test, n_models))

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for model_idx, (name, model_class) in enumerate(models.items()):
        print(f"\n[Stacking] Processing {name}...")

        fold_test_preds = np.zeros((n_test, n_folds))

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Clone et entraine le modele
            model = clone_model(model_class, name)
            model.fit(X_tr, y_tr)

            # OOF predictions
            oof_preds[val_idx, model_idx] = model.predict_proba(X_val)[:, 1]

            # Test predictions (moyennees sur les folds)
            fold_test_preds[:, fold_idx] = model.predict_proba(X_test)[:, 1]

        # Moyenne des predictions test sur les folds
        test_preds[:, model_idx] = fold_test_preds.mean(axis=1)

    # Meta-learner
    print("\n[Stacking] Training meta-learner...")
    meta_model = LogisticRegression(C=1.0, random_state=42)
    meta_model.fit(oof_preds, y_train)

    # Poids appris
    weights = dict(zip(models.keys(), meta_model.coef_[0]))
    print(f"[Stacking] Learned weights: {weights}")

    return meta_model, oof_preds, test_preds
```

---

## 6. Metriques Completes

### 6.1 Metriques Requises (ISO 25010)

| Metrique | Formule | Seuil Alice |
|----------|---------|-------------|
| AUC-ROC | Area under ROC | ≥ 0.75 (cible 0.80) |
| Accuracy | (TP+TN)/Total | ≥ 0.65 |
| Precision | TP/(TP+FP) | ≥ 0.60 |
| Recall | TP/(TP+FN) | ≥ 0.60 |
| F1-Score | 2*P*R/(P+R) | ≥ 0.60 |
| Log Loss | -log(p) | ≤ 0.60 |

### 6.2 Implementation

```python
def compute_all_metrics(y_true, y_pred, y_proba) -> dict:
    """Calcule toutes les metriques conformes ISO 25010."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, log_loss, confusion_matrix
    )

    cm = confusion_matrix(y_true, y_pred)

    return {
        "auc_roc": roc_auc_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba),
        "confusion_matrix": cm.tolist(),
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1]),
    }
```

---

## 7. Model Registry

### 7.1 Structure Fichiers

```
models/
├── v20260108_120000/
│   ├── catboost.cbm              # Format natif CatBoost
│   ├── xgboost.ubj               # Format UBJSON XGBoost
│   ├── lightgbm.txt              # Format texte LightGBM
│   ├── stacking_meta.joblib      # Meta-learner
│   ├── metadata.json             # Model card
│   └── feature_importance.json   # SHAP values
├── current -> v20260108_120000/  # Symlink vers production
└── registry.json                  # Index des versions
```

### 7.2 Model Card (metadata.json)

```json
{
  "version": "v20260108_120000",
  "created_at": "2026-01-08T12:00:00Z",
  "framework": {
    "catboost": "1.2.7",
    "xgboost": "2.1.0",
    "lightgbm": "4.3.0",
    "sklearn": "1.4.0"
  },
  "dataset": {
    "train_samples": 1139819,
    "valid_samples": 70647,
    "test_samples": 197843,
    "features": 12,
    "target": "resultat_blanc"
  },
  "metrics": {
    "catboost": {"auc": 0.7527, "f1": 0.68},
    "xgboost": {"auc": 0.7384, "f1": 0.67},
    "lightgbm": {"auc": 0.7506, "f1": 0.68},
    "stacking": {"auc": 0.7650, "f1": 0.70}
  },
  "selected_model": "stacking",
  "selection_reason": "stacking_auc > best_single + 0.01",
  "hyperparameters": {
    "catboost": {"iterations": 1000, "depth": 6},
    "xgboost": {"n_estimators": 1000, "max_depth": 6},
    "lightgbm": {"n_estimators": 1000, "num_leaves": 63}
  },
  "stacking_weights": {
    "catboost": 0.45,
    "xgboost": 0.20,
    "lightgbm": 0.35
  }
}
```

---

## 8. Plan d'Execution

### 8.1 Phase 1 - Fix Leakage (Jour 1)

| Tache | Fichier | Statut |
|-------|---------|--------|
| Modifier temporal_split pour AVANT features | feature_engineering.py | [x] |
| Creer compute_features_per_split() | feature_engineering.py | [x] |
| Creer run_feature_engineering_v2() | feature_engineering.py | [x] |
| Regenerer train/valid/test.parquet | data/features/ | [ ] A executer |
| Valider pas de leakage | test_data_leakage.py | [ ] |

### 8.2 Phase 2 - Training Parallele (Jour 2-3)

| Tache | Fichier | Statut |
|-------|---------|--------|
| Creer config/hyperparameters.yaml | config/ | [x] |
| Implementer train_models_parallel.py | scripts/ | [x] |
| Ajouter MLflow tracking | scripts/ | [x] |
| Tester entrainement parallele | - | [ ] A executer |

### 8.3 Phase 3 - Stacking (Jour 4)

| Tache | Fichier | Statut |
|-------|---------|--------|
| Implementer ensemble_stacking.py | scripts/ | [x] |
| Comparer single vs stacking | - | [ ] A executer |
| Sauvegarder meilleur modele | models/ | [ ] A executer |

### 8.4 Phase 4 - Production (Jour 5)

| Tache | Fichier | Statut |
|-------|---------|--------|
| Mettre a jour inference.py | services/ | [ ] |
| Creer model registry | models/ | [ ] |
| Tests integration | tests/ | [ ] |
| Documentation | docs/ | [ ] |

---

## 9. Criteres de Succes

| Metrique | Baseline (single) | Cible (stacking) | Seuil minimal |
|----------|-------------------|------------------|---------------|
| AUC-ROC | 0.7527 | **0.78+** | 0.75 |
| F1-Score | 0.68 | **0.72+** | 0.65 |
| Gain stacking | - | **+2-5%** | +1% |
| Temps training | 5 min | **<3 min** (parallel) | <10 min |

---

## 10. Rollback Plan

Si les resultats sont inferieurs au baseline:

1. **Rollback features**: Revenir aux features originales
2. **Rollback modele**: Utiliser single model CatBoost
3. **Investigation**: Analyser logs MLflow pour comprendre regression

---

## Historique

| Version | Date | Auteur | Modifications |
|---------|------|--------|---------------|
| 1.0.0 | 2026-01-08 | Claude Code | Creation plan RAZ |

---

*Plan conforme ISO/IEC 42001, 5259, 29119*
