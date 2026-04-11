# Model Specifications — Per-Model Architecture & Hyperparameter Guide

**Ce fichier est la SEULE source de vérité pour les décisions hyperparamètres.**
**Il est dans config/ = lu AVANT chaque action sur un modèle.**
**Chaque param a une justification SPÉCIFIQUE à l'architecture du modèle.**

---

## ALICE Engine — Ce que les modèles SERVENT

### Objectif métier
Un club FFE a N équipes qui jouent le MÊME week-end. Le capitaine distribue
les joueurs disponibles du club sur N équipes × K échiquiers (4 à 16 selon la division) sous contraintes FFE.
ALICE optimise cette allocation pour TOUT le club, pas une seule équipe.

### Pipeline complet
```
ALI (Adversarial Lineup Inference)
  → Prédit QUI joue chez l'adversaire (20 scénarios Monte Carlo par match)

ML (ce que les modèles produisent)
  → Pour chaque (joueur × échiquier × adversaire_prédit × équipe):
    P(win), P(draw), P(loss) calibrées

CE (Composition Engine, OR-Tools)
  → Optimise l'allocation joueurs × équipes × échiquiers
  → Contraintes FFE : noyau, mutés (max 3/saison), ordre Elo (100pts), 1 joueur = 1 équipe
  → Modes : agressif (max E[score] prioritaire), conservateur (maximin), 
    tactique (max P(match_win) zone danger), risk-adjusted (E - λ×Var)
```

### Ce que le downstream (CE) consomme
- **P(win), P(draw), P(loss)** — 3 probas CALIBRÉES par (joueur, échiquier, adversaire, équipe)
- **E[score] = P(win) + 0.5×P(draw)** — objectif principal
- **Var[score] = P(win)×(1-E)² + P(draw)×(0.5-E)² + P(loss)×(0-E)²** — mode risk-adjusted
- **P(match_win)** = convolution de K boards (4-16) — mode tactique
- Volume : 20 scénarios × N matchs × candidats × K boards par match = milliers de prédictions ML par ronde
- Latence cible : batch predict <50ms pour le lot complet

### Conséquences pour le ML
1. **Calibration > accuracy** — probas fausses = compositions fausses, même si accuracy OK
2. **P(draw) est le signal critique** — draws=13.7% mais 45% de la variance E[score]. C'est ce qui distingue une compo "sûre" d'une compo "risquée"
3. **3-class multiclass obligatoire** — binaire empêche le CE de distinguer draw/loss
4. **log_loss = métrique d'optimisation** — RPS + E[score] MAE = métriques de décision
5. **Résidual learning** — Elo baseline capture 92%, ML cherche les corrections (form, contexte, historique club)
6. **Per-board, pas per-match** — K prédictions indépendantes (4-16 selon division), agrégées par le CE
7. **Multi-équipe = contrainte CE, pas ML** — le ML prédit board par board, le CE alloue

### Quality gates (15 conditions, TOUTES obligatoires)
T1-T2: log_loss < Elo ET < Naïf
T3-T4: RPS < Elo ET < Naïf
T5: E[score] MAE < Elo
T6: Brier < Naïf
T7: ECE < 0.05 per class
T8: Draw calibration bias < 0.02
T9: mean_p_draw > 1%
T10-T12: reporting

### Inference flow (production)
```
Pour chaque match du club (N équipes × 20 scénarios ALI):
  Pour chaque (joueur_candidat × échiquier):
    features = feature_store.assemble(joueur, adversaire_prédit, contexte_match)  # 196 cols
    init_scores = compute_elo_baseline(joueur.elo, adversaire.elo) * alpha_per_model
    raw_logits = model.predict(features)
    final_logits = raw_logits + init_scores
    probas = temperature_scale(softmax(final_logits))
    → CE reçoit [P(loss), P(draw), P(win)] pour cette cellule

CE.optimize(all_probas, contraintes_ffe, mode_strategie)
  → Composition optimale pour TOUTES les équipes du club
```

---

## XGBoost — Depth-wise (level-by-level)

### Architecture
- Croissance : **depth-wise** (expand ALL nodes at same level)
- Régularisation principale : lambda (L2), alpha (L1), min_child_weight
- Gradient response : ROBUSTE aux petits gradients (construit l'arbre entier quelle que soit la magnitude)
- Compensation : plus d'itérations pour petits gradients → early stopping plus tard, même résultat final

### Conséquences pour le tuning
- **init_score_alpha : QUASI-INDIFFÉRENT** (gap 0.001 logloss entre 0.5 et 0.8)
- subsample et min_child_weight sont les leviers principaux
- colsample_bytree=1.0 optimal quand alpha bas (plus de signal à capturer)
- Paysage HP plat → tous les params contribuent ~également

### Hyperparamètres V9 (saison=2022, Grid 82 combos)

| Param | Optimal V9 | V8 | Gap | Source |
|-------|-----------|-----|-----|--------|
| init_score_alpha | 0.5 | 0.7 | 0.004 | Grid v4 : monotonique mais plat |
| subsample | 0.8 | 0.7 | 0.004 | Grid v4 : 0.8>0.7>0.6 |
| colsample_bytree | 1.0 | 0.5 | 0.003 | Grid v4 : 1.0>0.75>0.5 |
| min_child_weight | 50 | 50 | = | Grid v4 : 50>125>200 |
| max_depth | 8 (fixed) | 4 | — | fANOVA 0%, Optuna v6 all use 8 |
| eta | 0.05 (fixed) | 0.005 | — | Coupled with early_stopping=200 |
| reg_lambda | 4.0 (fixed) | 10.0 | — | Probst 2019 (valid XGB) : low tunability |
| reg_alpha | 0.01 (fixed) | 0.5 | — | Probst 2019 : quasi-nul |

### Interactions entre params (XGBoost)
- **eta × n_estimators** : couplés. eta=0.05 + early_stopping=200 → converge en ~800-1600 iters
- **subsample × colsample** : multiplicatifs. sub=0.8 × col=1.0 = chaque arbre voit 80% rows, 100% features
- **min_child_weight × depth** : MCW contrôle la granularité, depth la profondeur. Sur 62K train, MCW=50 OK
- **reg_lambda × depth** : lambda régularise les leaf values. Avec depth=8, lambda=4.0 empêche l'overshoot
- **base_margin shape** : (n_samples × n_classes) FLATTENED pour DMatrix. Contient des raw margins (log-odds), PAS des probas

### Categorical handling
- Pas de support natif. LabelEncoder obligatoire avant training.
- `enable_categorical=True` existe (expérimental) mais NON TESTÉ sur ALICE

### Histogram binning
- `tree_method="hist"` : bins 256 par défaut. Rapide, mémoire O(n_features × 256)
- Pas de paramètre `border_count` (CatBoost-specific)

### Sources fabricant
- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
- https://xgboost.readthedocs.io/en/stable/prediction.html
- Probst et al. 2019 (JMLR) — VALIDE pour XGBoost uniquement

---

## LightGBM — Leaf-wise (best-first)

### Architecture
- Croissance : **leaf-wise** (split la feuille avec le PLUS GRAND gradient)
- Régularisation principale : num_leaves (THE complexity param), min_child_samples
- Gradient response : **HYPER-SENSIBLE** aux magnitudes de gradient
  - Petits gradients (high alpha) → pas de feuille discriminante → early stop en sous-apprentissage
  - Grands gradients (low alpha) → leaf-wise trouve les patterns efficacement
- Risque overfitting : plus élevé que depth-wise sur petits datasets

### Conséquences pour le tuning
- **init_score_alpha : PARAM #1** (gap 0.05 logloss, 92.6% fANOVA)
- Bas alpha = large residuals = leaf-wise exploite les features
- feature_fraction=1.0 optimal quand alpha bas (besoin de toutes les features)
- num_leaves bas (15-30) optimal sur 62K (cap naturel à ~135 feuilles)
- min_child_samples : "very important for leaf-wise" (doc officielle)
- bagging_fraction : quasi-nul (1.4% fANOVA), fixé à 0.8

### Hyperparamètres V9 (saison=2022, Grid 82 combos + Optuna 100 trials)

| Param | Optimal V9 | V8 | Gap | Source |
|-------|-----------|-----|-----|--------|
| init_score_alpha | **0.4** | 0.7 | **0.051** | Grid v2 : monotonique strict. TPE ne descend pas sous 0.5 |
| num_leaves | 15 | 15 | = | Grid v2 : 15>135≡255 (cap naturel) |
| feature_fraction | 1.0 | 0.5 | 0.004 | Grid v2 : 1.0>0.65>0.3 |
| min_child_samples | 275 | 200 | 0.001 | Grid v2 : léger avantage, robuste |
| max_depth | 8 (fixed) | 4 | — | Safety cap leaf-wise |
| learning_rate | 0.05 (fixed) | 0.03 | — | Coupled with early_stopping=200 |
| reg_lambda | 4.0 (fixed) | 10.0 | — | van Rijn 2018 : moderate importance |
| bagging_fraction | 0.8 (fixed) | 0.7 | — | fANOVA 1.4%, near-optimal |

### Interactions entre params (LightGBM)
- **num_leaves × min_child_samples** : contrainte dure. Max effective leaves = N_train / min_child_samples. Sur 62K avec mcs=275 → max ~225 feuilles. leaves=255 > 225 → capped naturellement
- **num_leaves × max_depth** : CLAMP obligatoire `min(num_leaves, 2^max_depth - 1)`. Avec max_depth=8 → clamp à 255
- **feature_fraction × alpha** : alpha bas = grands résidus = plus de signal. ff=1.0 nécessaire pour capturer tout le signal. Alpha haut + ff=1.0 = overfitting
- **bagging_fraction × bagging_freq** : bagging_freq DOIT être > 0 pour activer bagging_fraction. Sinon ignoré silencieusement
- **learning_rate × early_stopping** : lr=0.05 + ES=200 = converge en ~1500-2200 iters sur 62K

### GOSS (Gradient-based One-Side Sampling)
- Activé automatiquement pour grands datasets (boosting_type="gbdt" par défaut)
- Conserve les instances à GRAND gradient + échantillon des petits gradients
- Implication pour residual learning : avec alpha bas, plus d'instances ont des grands gradients → GOSS conserve plus d'info → meilleur apprentissage

### Categorical handling
- `categorical_feature="auto"` détecte les colonnes pandas Categorical
- Support natif des catégories (pas besoin de LabelEncoder pour LightGBM seul)
- MAIS pipeline ALICE encode quand même (compatibilité cross-model avec XGBoost)

### Sources fabricant
- https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
- https://lightgbm.readthedocs.io/en/latest/Parameters.html
- https://lightgbm.readthedocs.io/en/latest/Features.html (GOSS, EFB)
- "leaf-wise can converge much faster, but may over-fit if not used with appropriate parameters"
- "min_child_samples: very important to prevent over-fitting in leaf-wise tree"
- van Rijn & Hutter 2018 (KDD) — LightGBM feature_fraction #5

---

## CatBoost — Oblivious trees (symmetric)

### Architecture
- Croissance : **oblivious** (même split à tous les nœuds d'un niveau)
- Régularisation principale : depth + l2_leaf_reg
- Particularités :
  - depth = EXACTEMENT le nombre de splits (pas de croissance asymétrique)
  - min_data_in_leaf : **ZÉRO EFFET** (structure symétrique, confirmé empiriquement)
  - rsm (feature subsampling) : **OBLIGATOIRE >50 features** (sinon mêmes 11 features dominantes)
  - rsm INCOMPATIBLE GPU (pairwise only) → task_type=CPU obligatoire
  - init_model + Pool(baseline=) = **CRASH** ("baseline for continuation not supported")
  - snapshot_file exige MÊMES params (iterations, lr, tout)
- Gradient response : **TBD** (entre XGBoost et LightGBM, structure fixe par niveau)

### Conséquences pour le tuning
- **depth : param #1** pour CatBoost (contrôle directement la complexité oblivious)
- **l2_leaf_reg : param #2** (doc officielle : "first params to tune" avec depth)
- rsm [0.2, 0.7] : obligatoire pour exploration des features
- init_score_alpha : sensibilité **TBD** (2 trials seulement sur full data, Grid v2 en cours)
- min_data_in_leaf : NE PAS TUNER (zéro effet, confirmé grid 3 valeurs identiques)

### Hyperparamètres V9 (saison=2022, Grid en cours)

| Param | Optimal V9 | V8 | Gap | Source |
|-------|-----------|-----|-----|--------|
| init_score_alpha | TBD (~0.55) | 0.7 | TBD | Optuna v3 (2 trials only) : 0.547 |
| depth | TBD (~5) | 4 | TBD | Optuna : 5>10. Grid en cours |
| l2_leaf_reg | TBD (~1.8) | 10 | TBD | Optuna : 1.8>11.2. Grid en cours |
| rsm | TBD (~0.63) | 0.3 | TBD | Optuna : 0.63>0.50 |
| learning_rate | 0.05 (fixed) | 0.03 | — | Coupled with early_stopping=200 |
| random_strength | 2.0 (fixed) | 3.0 | — | Low priority |
| min_data_in_leaf | 200 (fixed) | 200 | = | ZERO effect oblivious trees |

### Interactions entre params (CatBoost)
- **depth × l2_leaf_reg** : depth haute (8-10) = plus de feuilles = plus d'overfitting → l2_leaf_reg doit augmenter. depth=4 + l2=3 OK, depth=8 → l2=[5,15] nécessaire
- **depth × mémoire** : mémoire = O(2^depth × n_trees). Chaque +1 depth = ×2 mémoire
- **rsm × depth** : oblivious trees = 4 splits/tree (depth=4). Sans rsm, 4 mêmes features choisies. rsm=0.5 force l'exploration
- **learning_rate × iterations** : lr=0.05 + ES=200 → converge en ~1000-3000 iters. lr=0.03 (V8) → ~37K iters

### Feature Combinations (spécificité CatBoost)
- CatBoost combine automatiquement les features catégorielles ("greedy feature combinations")
- Les cat_features natives (type_competition, division, etc.) bénéficient de ce traitement
- XGBoost/LightGBM n'ont PAS cette capacité → nos features différentielles compensent

### Categorical handling
- **Support NATIF** des catégories via `cat_features` param ou Pool
- N'encode PAS les catégories avant CatBoost (perte d'info)
- Pipeline ALICE : `cat_features` listées dans config, passées à Pool directement
- Les features non-cat sont numériques (label-encoded pour compatibilité XGB/LGB)

### Ordered Boosting (anti-leakage natif)
- CatBoost utilise des résidus "ordonnés" (random permutation) pour éviter le target leakage
- C'est automatique, pas de param à régler
- Avantage théorique sur XGB/LGB pour les petits datasets ou les features catégorielles à haute cardinalité

### Sources fabricant
- https://catboost.ai/docs/en/concepts/parameter-tuning
- https://catboost.ai/docs/en/references/training-parameters/common
- https://catboost.ai/docs/en/concepts/algorithm-main-stages (ordered boosting)
- CatBoost Optuna tutorial : github.com/catboost/tutorials (l2_leaf_reg tuné, min_data_in_leaf absent)
- Springer 2020 "CatBoost for big data: an interdisciplinary review"
- rsm GPU crash : CatBoostError "rsm on GPU is supported for pairwise modes only"

---

## Règle d'or : alpha × architecture

| Architecture | Alpha sensitivity | Mécanisme | Conséquence |
|-------------|------------------|-----------|-------------|
| Depth-wise (XGB) | ~0.001 | Construit l'arbre entier, compense via n_iter | Alpha quasi-indifférent |
| Leaf-wise (LGB) | ~0.05 | Split le plus grand gradient → petits gradients = sous-apprentissage | Alpha = param #1 |
| Oblivious (CB) | ~TBD | Structure fixe par niveau | Probablement modéré |

**NE JAMAIS appliquer le même alpha aux 3 modèles.**

---

---

## Prediction API (CRITIQUE pour inference)

**Chaque modèle a une API de prédiction DIFFÉRENTE pour residual learning.**
`predict_proba()` sans init_scores = RÉSULTATS FAUX.

| Library | Training | Prediction correcte | Bug connu |
|---------|----------|-------------------|-----------|
| **XGBoost** | `DMatrix.set_base_margin()` + `xgb.train()` | `DMatrix.set_base_margin(init) + bst.predict()` | #5288: XGBClassifier.fit(base_margin=) broken multiclass |
| **LightGBM** | `fit(init_score=, eval_init_score=[])` | `predict(raw_score=True) + init + softmax` | #1978: predict() ne supporte pas init_score |
| **CatBoost** | `Pool(baseline=)` | `predict(type='RawFormulaVal') + init + softmax` | #1554: predict_proba(Pool(baseline=)) ne normalise pas |

```python
# Pattern commun d'inference (predict_with_init)
raw_logits = model.predict(X, raw_score=True)  # (n, 3)
final_logits = raw_logits + init_scores * alpha
probas = softmax(final_logits, axis=1)
```

---

## Resume / Checkpoint

| Library | Mécanisme | Format | Startup | Limitation |
|---------|-----------|--------|---------|-----------|
| **XGBoost** | `TrainingCheckPoint(interval=5000)` | Binaire .ubj | Secondes | `EarlyStopping(save_best=True)` OBLIGATOIRE (sinon retourne LAST pas best) |
| **LightGBM** | Custom callback + `save_model()` | **Texte uniquement** | 65K=3h22m, 90K=4h32m | PAS de format binaire (issue #372). Si startup > 30% session → from scratch |
| **CatBoost** | `snapshot_file` + `snapshot_interval` | Binaire .cbm | Secondes | MÊMES params requis. `init_model + Pool(baseline=)` = **CRASH** |

**Règle resume :** Si startup > 30% du temps session → from scratch avec lr plus haut.

---

## GPU Compatibilité

| Library | GPU | Limitation |
|---------|-----|-----------|
| **XGBoost** | `tree_method="hist", device="cuda"` | OK sur T4/P100 |
| **LightGBM** | **CPU only** (pip). GPU requiert compilation OpenCL | Ne pas tenter sur Kaggle |
| **CatBoost** | `task_type="GPU"` natif CUDA | **rsm INCOMPATIBLE GPU** ("pairwise only"). Force CPU quand rsm set |

---

## Model File Formats & Tailles (V8)

| Library | Format | Taille V8 | Chargement |
|---------|--------|-----------|-----------|
| **XGBoost** | `.ubj` (binaire) | 427 MB | Rapide |
| **LightGBM** | `.txt` (texte) | 86 MB | **LENT** (parsing string, scale linéaire) |
| **CatBoost** | `.cbm` (binaire) | 23 MB | Rapide (18× plus petit que XGB) |

---

## SHAP / Explainabilité

| Library | Méthode native | Fiabilité | Alternative |
|---------|---------------|-----------|-------------|
| **XGBoost** | `shap.TreeExplainer(booster)` | ✓ Fiable (109/177 features non-zero) | gain importance OK |
| **LightGBM** | `shap.TreeExplainer(booster)` | ✓ Fiable (50/177 features non-zero) | gain importance OK |
| **CatBoost** | `get_feature_importance(type='ShapValues')` | ✓ SEULE méthode fiable | **PredictionValuesChange = BIAISÉ** (166/177=0 artefact residual) |

**CatBoost PredictionValuesChange est INTERDIT.** Toujours utiliser ShapValues.
**TreeSHAP budget :** TOUJOURS subsample 20K (231K × 85K trees = 14 HEURES).

---

## Calibration

| Library | Post-hoc recommandé | Interaction |
|---------|-------------------|-------------|
| **XGBoost** | Temperature scaling (préserve E[score]) | raw probas structurellement miscalibrées |
| **LightGBM** | Temperature scaling | idem |
| **CatBoost** | Temperature scaling | `probability_calibration=True` existe mais isotonic per-class + renorm DÉGRADE E[score] |

**Isotonic per-class + renorm = DANGER** (ICML 2025 arXiv:2512.09054). Utiliser temperature scaling.

---

## Mise à jour

Ce fichier est mis à jour à chaque nouveau résultat empirique.
Dernière mise à jour : 2026-04-11 (Grid XGB v4 + Grid LGB v2 + Optuna LGB v7).
CatBoost : à compléter quand Grid CB v2 termine.
