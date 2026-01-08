# Evaluation ML - CatBoost vs XGBoost vs LightGBM

> **Date**: 4 Janvier 2026
> **Derniere MAJ**: 8 Janvier 2026
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

## 4. Interpretation et limites

### 4.1 Echelle AUC-ROC

| Plage AUC | Interpretation | Alice-Engine |
|-----------|----------------|--------------|
| 0.50 - 0.60 | Mediocre (a peine mieux que hasard) | |
| 0.60 - 0.70 | Acceptable | |
| **0.70 - 0.80** | **Bon** | **0.7527 ← actuel** |
| 0.80 - 0.90 | Tres bon | ← cible |
| 0.90 - 1.00 | Excellent | |

**Verdict**: Performance "bonne" mais pas "excellente". Marge d'amelioration.

### 4.2 Accuracy et taux d'erreur

```
Accuracy:    68.30%
Taux erreur: 31.70% (environ 1 prediction sur 3 incorrecte)
```

**Contexte**: Pour la prediction de compositions d'echecs, ce taux est acceptable
car le modele genere des probabilites (scenarios multiples), pas une prediction unique.

### 4.3 Ecarts entre modeles

| Comparaison | Ecart AUC | Ecart Accuracy | Significativite |
|-------------|-----------|----------------|-----------------|
| CatBoost vs XGBoost | +1.43% | +0.86% | **Significatif** |
| CatBoost vs LightGBM | +0.21% | +0.08% | **Faible** |

**Analyse**:
- CatBoost domine clairement XGBoost (+1.4% AUC)
- CatBoost et LightGBM sont quasi ex-aequo (+0.21% AUC)
- Le choix CatBoost reste justifie par sa gestion native des categories

### 4.4 Limites identifiees

| Limite | Impact | Mitigation |
|--------|--------|------------|
| AUC < 0.80 | Predictions moyennement fiables | Hyperparameter tuning (Phase 5) |
| 32% erreurs | 1 prediction sur 3 fausse | Scenarios multiples, probabilites |
| Ecart faible vs LightGBM | Choix CatBoost discutable | Justifie par categories natives |
| Dataset ancien (2002-2022) | Patterns obsoletes possibles | Reentrainer sur 2018-2024 |

### 4.5 Actions correctives prevues

1. **Phase 5 - Hyperparameter Tuning** (Optuna)
   - Cible: AUC 0.80+
   - Parametres: depth, learning_rate, l2_leaf_reg, iterations

2. **Features supplementaires**
   - `taux_forfait_club` - fiabilite club
   - `forme_recente` - 5 derniers matchs
   - `objectif_equipe` - montee/maintien/descente

3. **Split temporel ajuste**
   - Reentrainer sur donnees recentes (2018-2024)
   - Eviter patterns obsoletes des annees 2000

---

## 5. Features utilisees

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

## 6. Prochaines ameliorations

### 6.1 Features reglementaires FFE (Priorite: HAUTE)

> **Dataset disponible**: `docs/requirements/REGLES_FFE_ALICE.md` (1,153 lignes)
> **Implementation**: `scripts/ffe_rules_features.py` (845 lignes, 66 tests)
> **Statut**: Implemente mais NON integre dans ML

| Feature | Type | Impact attendu | Source |
|---------|------|----------------|--------|
| `joueur_brule` | bool | **Eleve** | `est_brule()` |
| `matchs_avant_brulage` | int 0-3 | **Moyen** | `matchs_avant_brulage()` |
| `est_dans_noyau` | bool | **Moyen** | `get_noyau()` |
| `pct_noyau_equipe` | float | **Moyen** | `calculer_pct_noyau()` |
| `zone_enjeu` | cat | **Eleve** | `calculer_zone_enjeu()` |
| `joueur_mute` | bool | **Faible** | Champ `mute` dataset |

**Hypothese**: AUC +5-10% attendu car:
- Joueur brule = prediction certaine (ne peut pas jouer)
- Zone enjeu influence directement les choix de composition

### 6.2 Features fiabilite (deja extraites)

- `taux_forfait_club`
- `taux_presence_joueur`
- `forme_recente`
- `echiquier_moyen`

### 6.3 Optimiser hyperparametres (Optuna)

- `depth`: [4, 6, 8, 10]
- `learning_rate`: [0.01, 0.05, 0.1]
- `l2_leaf_reg`: [1, 3, 5, 10]

### 6.4 Split temporel ajuste

- Train: 2018-2023 (donnees recentes)
- Valid: 2024
- Test: 2025-2026

---

## 7. Fichiers generes

- `data/ml_evaluation_results.csv`: Resultats bruts
- `scripts/evaluate_models.py`: Script d'evaluation

---

*Genere le 4 Janvier 2026*
*Mis a jour le 8 Janvier 2026*
*Script: evaluate_models.py*
