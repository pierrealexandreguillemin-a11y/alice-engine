# Evaluation ML - CatBoost vs XGBoost vs LightGBM

> **Date**: 4 Janvier 2026
> **Dataset**: 1,139,819 echiquiers (train), 70,647 (valid), 197,843 (test)
> **Split**: Temporel (2002-2022 / 2023 / 2024-2026)

---

## 1. Resultats comparatifs

### Test Set (2024-2026)

| Modele | AUC-ROC | Accuracy | Train (s) | Inference (ms) |
|--------|---------|----------|-----------|----------------|
| **CatBoost** | **0.7527** | **68.30%** | 292.9 | 414 |
| LightGBM | 0.7506 | 68.22% | 8.5 | 534 |
| XGBoost | 0.7384 | 67.44% | 10.0 | 177 |

### Validation Set (2023)

| Modele | AUC-ROC | Accuracy |
|--------|---------|----------|
| **CatBoost** | **0.7528** | **68.12%** |
| LightGBM | 0.7500 | 67.90% |
| XGBoost | 0.7440 | 67.52% |

---

## 2. Analyse

### Precision (AUC-ROC)

```
CatBoost: 0.7527 (+1.4% vs XGBoost)
LightGBM: 0.7506 (+1.2% vs XGBoost)
XGBoost:  0.7384 (baseline)
```

CatBoost confirme les benchmarks academiques: meilleure gestion des
categories (division, ligue_code, titre FIDE) sans encodage manuel.

### Vitesse entrainement

```
LightGBM: 8.5s   (reference)
XGBoost:  10.0s  (1.2x plus lent)
CatBoost: 292.9s (34x plus lent)
```

CatBoost est lent car il utilise des splits ordered boosting
(anti-overfitting) sur 500 iterations completes.

### Vitesse inference

```
XGBoost:  177ms (reference)
CatBoost: 414ms (2.3x plus lent)
LightGBM: 534ms (3.0x plus lent)
```

Pour 197k predictions, tous les modeles sont en dessous de 1s.
En production (1 prediction), la difference est negligeable (~2us).

---

## 3. Recommandation

### Modele retenu: CatBoost

| Critere | Poids | CatBoost | Justification |
|---------|-------|----------|---------------|
| Precision | Critique | **Best** | +1.4% AUC = moins d'erreurs |
| Categories | Important | **Native** | Pas d'encodage (division, ligue) |
| Train time | Faible | Acceptable | 5 min offline, 1x/saison |
| Inference | Faible | OK | <1ms/prediction suffisant |

### Configuration recommandee

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=['type_competition', 'division', 'ligue_code',
                  'blanc_titre', 'noir_titre', 'jour_semaine'],
    early_stopping_rounds=50,
    eval_metric='AUC',
    random_seed=42,
)
```

---

## 4. Features utilisees

### Numeriques (6)

- `blanc_elo`: Elo joueur blancs
- `noir_elo`: Elo joueur noirs
- `diff_elo`: blanc_elo - noir_elo
- `echiquier`: Position dans l'equipe (1-8)
- `niveau`: Niveau de la division (1=N1, 8=D4)
- `ronde`: Numero de ronde

### Categorielles (6)

- `type_competition`: national, regional, coupe, etc.
- `division`: Nationale 1, Nationale 2, etc.
- `ligue_code`: IDF, HDF, PACA, etc.
- `blanc_titre`: GM, IM, FM, (vide)
- `noir_titre`: GM, IM, FM, (vide)
- `jour_semaine`: Lundi, Samedi, Dimanche

---

## 5. Prochaines ameliorations

1. **Ajouter features fiabilite** (deja extraites):
   - `taux_forfait_club`
   - `taux_presence_joueur`
   - `forme_recente`
   - `echiquier_moyen`

2. **Optimiser hyperparametres** (Optuna):
   - `depth`: [4, 6, 8, 10]
   - `learning_rate`: [0.01, 0.05, 0.1]
   - `l2_leaf_reg`: [1, 3, 5, 10]

3. **Split temporel ajuste**:
   - Train: 2018-2023 (donnees recentes)
   - Valid: 2024
   - Test: 2025-2026

---

## 6. Fichiers generes

- `data/ml_evaluation_results.csv`: Resultats bruts
- `scripts/evaluate_models.py`: Script d'evaluation

---

*Genere le 4 Janvier 2026*
*Script: evaluate_models.py*
