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

### Hyperparamètres V9 (saison=2022, 590 configs testées)

| Param | Optimal V9 | V8 | Gap | Source |
|-------|-----------|-----|-----|--------|
| init_score_alpha | **0.5** | 0.7 | 0.002 | Grid v4 + Gaps : plat 0.3-0.7 (delta 0.001) |
| max_depth | **6** | 4 | 0.002 | **Gaps : 6>4>8 à tous les alphas.** V8=4, Grid=8, les deux sous-optimaux |
| subsample | **0.8** | 0.7 | 0.001 | Grid v4 + Gaps R2 : confirmé à depth=6 |
| colsample_bytree | **1.0** | 0.5 | 0.001 | Grid v4 + Gaps R2 : confirmé à depth=6 |
| colsample_bynode | **0.7** | 1.0 (default) | 0 (loss) | **Tier 2 : meme logloss, draw_bias -22% (0.0029 vs 0.0037)** |
| gamma | **1.0** | 0 (default) | 0 (loss) | **Tier 2 : empeche splits noise. Ameliore draw calibration** |
| max_delta_step | **1** | 0 (default) | 0 (loss) | **Tier 2 : limite leaf outputs. Aide draw imbalance (13.7%)** |
| min_child_weight | **50** | 50 | = | Grid v4 + Gaps R2 : confirmé à depth=6 |
| eta | 0.05 (fixed) | 0.005 | — | Coupled with early_stopping=200 |
| reg_lambda | 4.0 (fixed) | 10.0 | — | Probst 2019 (valid XGB) : low tunability |
| reg_alpha | 0.01 (fixed) | 0.5 | — | Probst 2019 : quasi-nul |

### Draw calibration (Tier 2 finding, critique pour CE production)
- **bynode=0.7 + gamma=1.0 + max_delta_step=1** = meme logloss que defaults mais draw_bias -22%, ECE draw -17%
- Le CE utilise P(draw) dans E[score] = P(win) + 0.5×P(draw). Draw_bias impacte directement les compositions
- Ces 3 params sont des "gains gratuits" : aucune perte de logloss, amelioration calibration

### Interactions entre params (XGBoost)
- **eta × n_estimators** : couplés. eta=0.05 + early_stopping=200 → converge en ~800-1600 iters
- **subsample × colsample_bytree × colsample_bynode** : multiplicatifs. 0.8 × 1.0 × 0.7 = chaque split voit 80% rows, 70% features
- **gamma × depth** : gamma=1.0 empeche les splits avec gain<1.0. Avec depth=6 (64 feuilles max), regularise sans sous-exploiter
- **max_delta_step × class imbalance** : draw=13.7%. max_delta_step=1 empeche les leaf outputs extremes sur la classe minoritaire
- **min_child_weight × depth** : MCW contrôle la granularité, depth la profondeur. Sur 62K train, MCW=50 OK
- **base_margin shape** : (n_samples × n_classes) FLATTENED pour DMatrix. Contient des raw margins (log-odds), PAS des probas

### Categorical handling
- Pas de support natif. LabelEncoder obligatoire avant training.
- `enable_categorical=True` existe (expérimental) mais NON TESTÉ sur ALICE

### Histogram binning
- `tree_method="hist"` : bins 256 par défaut. Rapide, mémoire O(n_features × 256)
- Pas de paramètre `border_count` (CatBoost-specific)

### V8 Convergence Data (empirique)

| Version | LR | Rounds | Best val | Test log_loss | Training time | Notes |
|---------|-----|--------|----------|---------------|---------------|-------|
| v2 | 0.1 (default) | ~2/3000 | ~0.956 | — | — | **EXPLOSION** : log_loss 2.367 à iter 100 |
| v3 | 0.03 | 33/3000 | ~0.98 | — | — | Pas de residual, au-dessus Elo baseline |
| v5 | 0.03 | ~1/3000 | 0.984 | — | — | base_margin bug (#5288), résultat invalide |
| v18 | 0.005 | 50K (ES) | 0.5742 | 0.5742 | 4,182s (~70 min) | **Premier ALL-PASS**, T=0.928 |
| resume v5 | 0.005 | 86,490 | 0.5126 | **0.5660** | 4h42m total | **Champion V8**, delta 50K→86.5K = -1.4% test |

- Delta 85K→86.5K : val 0.51269→0.51255 (Δ=0.00014) — modèle à l'optimum
- 197/201 features avec gain > 0

### Feature Usage
- **109/177 features non-zero** (real signal, contrairement à l'artefact CatBoost)
- Permutation importance : 197 features × 5 repeats × 17s/repeat = **4h39m** — budgeter AVANT de lancer

### Temperature Scaling
- **T = 0.928** (mild) — résidual learning avec alpha produit déjà un modèle quasi-calibré
- Formula : `softmax(logits / T)`, préserve les ratios E[score]

### Bugs connus (XGBoost)
| Bug | Impact | Fix | Réf |
|-----|--------|-----|-----|
| `XGBClassifier.fit(base_margin=)` broken multiclass | Prédictions fausses | `xgb.train()` + `DMatrix.set_base_margin()` | #5288, #3505, #11872 |
| `xgb.train()` retourne LAST pas best | Modèle sous-optimal sauvé | `EarlyStopping(save_best=True)` **OBLIGATOIRE** | — |
| SHAP `reshape(N,-1,3)` scramble multiclass | Axes inversés | Auto-detect layout `(N,C,F+1)` vs `(N,F+1,C)` | — |
| LR=0.1 + residual → explosion | log_loss 2.367 à iter 100 | LR ∈ [0.001, 0.01] pour residual | V8 v2 postmortem |

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
- **init_score_alpha : PARAM #1** (gap 0.054 logloss 0.1→0.7, 92.6% fANOVA)
- Tendance monotone confirmée de 0.7 à 0.1 : chaque -0.1 alpha = -0.003 à -0.025 logloss
- Plancher à alpha ~0.1 (gradient s'aplatit : -0.008 → -0.005 → -0.003 → -0.0015)
- feature_fraction=1.0 optimal quand alpha bas (besoin de toutes les features)
- num_leaves bas (15) optimal sur 62K (cap naturel à ~135 feuilles)
- min_child_samples : "very important for leaf-wise" (doc officielle)
- bagging_fraction : quasi-nul (1.4% fANOVA), fixé à 0.8
- reg_lambda : **PLAT** (range 0.0006 à alpha=0.3). Lambda=4.0 fixe = correct.

### Hyperparamètres V9 (saison=2022, 590 configs testées)

| Param | Optimal V9 | V8 | Gap | Source |
|-------|-----------|-----|-----|--------|
| init_score_alpha | **0.1** | 0.7 | **0.054** | Gaps R1+R2 : monotone 0.7→0.1, plancher à 0.1 (gradient -0.0015) |
| num_leaves | **15** | 15 | = | Grid v2 + **Tier 2 : 15>31>63>127.** Plus de feuilles = pire draw_bias |
| feature_fraction | **1.0** | 0.5 | 0.004 | Grid v2 : 1.0>0.65>0.3 |
| min_child_samples | **275** | 200 | 0.001 | Grid v2 : léger avantage, robuste |
| min_gain_to_split | **0** (default) | 0.01 (yaml) | 0 | **Tier 2 : 0 meilleur que 0.01 en logloss ET draw calibration** |
| lambda_l1 | **0** (default) | 0 | = | **Tier 2 : L1 reg n'ameliore rien (201 features OK sans pruning)** |
| max_depth | 8 (fixed) | 4 | — | Safety cap leaf-wise |
| learning_rate | 0.05 (fixed) | 0.03 | — | Coupled with early_stopping=200 |
| reg_lambda | **4.0** (fixed) | 10.0 | — | Gaps R1 : plat (range 0.0006). van Rijn 2018 : moderate |
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

### V8 Convergence Data (empirique)

| Version | LR | Rounds | Best val | Test log_loss | Training time | Notes |
|---------|-----|--------|----------|---------------|---------------|-------|
| v2 | 0.1 (default) | 3/3000 | 0.967 | — | — | **EXPLOSION** : log_loss 3.598 à iter 100 |
| v5 | 0.03 | 9/3000 | **0.8849** | — | — | Beats Elo. Overshoot : LR=0.03 sur residuals ~0.008 = 3.75× correction |
| v3 (resume) | 0.003 | 50K | 0.5364 | — | — | TIMEOUT (3 bugs post-training) |
| v4 (resume) | 0.005 | 65K | 0.5204 | — | — | Training done, post-training TIMEOUT |
| **v7** | 0.03 | 50K (ES) | 0.5134 | **0.5721** | 20,973s (~5.8h) | **ALL-PASS** |

- Resume chain v3→v4→v5 : chaque reprise = 3h22m startup (65K model .txt)
- LR=0.03 fonctionne en final mais overshoot en début (9 iters)

### Feature Usage
- **50/177 features non-zero** (le plus sélectif des 3 modèles)
- Leaf-wise = seulement les features à fort gradient split → beaucoup de features ignorées

### Temperature Scaling
- Valeur T : à mesurer sur V9 (V8 non documenté pour LGB spécifiquement)
- Même approche que XGBoost : `softmax(logits / T)`

### Bugs connus (LightGBM)
| Bug | Impact | Fix | Réf |
|-----|--------|-----|-----|
| `predict()` ne supporte PAS init_score | Inference fausse sans workaround | `predict(raw_score=True) + init + softmax` | #1978, #1778 |
| `LGBMClassifier.save_model()` n'existe pas | AttributeError | `model.booster_.save_model()` | #4841 |
| Model .txt = 65K trees → **3h22m startup** | Resume impossible si >30% session | Restart from scratch si startup > 3h | #372 |
| `init_model` rend n_estimators **ADDITIF** | Training runaway (n_total = n_init + n_new) | Calculer n_remaining = n_target - n_init | — |
| LR=0.03 + residual → overshoot en 9 iters | Sous-optimal si early stop trop tôt | LR ∈ [0.001, 0.01] pour residual | V8 v5 |

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
- Gradient response : **MODEREMENT SENSIBLE** — structure fixe par niveau absorbe partiellement les petits gradients (comme depth-wise), mais depth bas (5) + oblivious = moins de feuilles pour compenser → alpha=0.3 > 0.5 (delta 0.0011-0.0024 selon depth)

### Conséquences pour le tuning
- **depth : param #1** pour CatBoost (contrôle directement la complexité oblivious)
- **l2_leaf_reg : param #2** (doc officielle : "first params to tune" avec depth)
- rsm [0.2, 0.7] : obligatoire pour exploration des features
- init_score_alpha : **MODEREMENT SENSIBLE** (delta 0.0024 entre 0.3 et 0.5 à depth=5). Plus sensible que XGB (0.001) mais beaucoup moins que LGB (0.038)
- min_data_in_leaf : NE PAS TUNER (zéro effet, confirmé grid 3 valeurs identiques)
- depth=8 **NETTEMENT PIRE** (+0.004-0.005 vs depth=5). Ne pas utiliser.

### Hyperparamètres V9 (saison=2022, 545 configs testées)

| Param | Optimal V9 | V8 | Gap | Source |
|-------|-----------|-----|-----|--------|
| init_score_alpha | **0.3** | 0.7 | 0.0024 | Gaps CB : 0.3>0.4>0.5>0.7 à depth=5 |
| depth | **5** | 4 | 0.0012 | Gaps CB : 5>4>6>>8. Grid v2 : 4>7 |
| l2_leaf_reg | **8.0** | 10 | 0.001 | Grid v2 : 8~15>1. Optuna mean 7.7 |
| rsm | **0.7** | 0.3 | 0.004 | Grid v2 : 0.7>0.45. Optuna mean 0.574 |
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

### V8 Convergence Data (empirique)

| Version | LR | Rounds | Best val | Test log_loss | Training time | Notes |
|---------|-----|--------|----------|---------------|---------------|-------|
| v2 | 0.1 (default) | 39/3000 | 0.9399 | — | — | Divergence (log_loss 1.037 à iter 100) |
| v3 | 0.005 | 50K | 0.5468 | 0.5895 | 8h11m | **PAS CONVERGÉ** (val still decreasing at 50K) |
| v4 | 0.01 | 150K cap | — | — | — | Fresh, was RUNNING |
| **v6** | 0.03 | 50K (ES) | 0.5265 | **0.5753** | 20,607s (~5.7h) | **ALL-PASS**, T=0.935 |

- v3 à LR=0.005 : 8h11m et pas convergé → LR=0.01 minimum pour CatBoost avec oblivious trees
- v6 : `probability_calibration=True` ON par défaut, mais temperature scaling (T=0.935) utilisé en post-hoc

### Feature Usage
- **Sans rsm** : 11/177 features non-zero (oblivious trees = 4 splits/tree, mêmes features choisies)
- **Avec rsm=0.3** : >30 features non-zero (exploration forcée)
- `PredictionValuesChange` : 166/177 = 0.0 = **ARTEFACT** (pas un vrai signal). Ne JAMAIS utiliser pour feature selection
- Seule méthode fiable : `ShapValues` (SHAP natif CatBoost)

### Temperature Scaling
- **T = 0.935** — mild, confirme que residual learning produit un modèle quasi-calibré
- `probability_calibration=True` (natif CatBoost) existe mais isotonic per-class + renorm **DÉGRADE** E[score]

### Bugs connus (CatBoost)
| Bug | Impact | Fix | Réf |
|-----|--------|-----|-----|
| `predict_proba(Pool(baseline=))` non normalisé | Probas ne somment pas à 1 | `predict(type='RawFormulaVal') + init + softmax` | #1554, #1550 |
| `init_model + Pool(baseline=)` combinés | **CRASH fatal** | Utiliser `Pool(baseline=)` SEUL, jamais init_model | — |
| `snapshot_file` exige mêmes params | Resume cassé si 1 param change | Vérifier params identiques ou restart from scratch | — |
| `PredictionValuesChange` importance | 166/177=0 artefact avec residual+oblivious | Utiliser `ShapValues` uniquement | V8 v10 |
| rsm incompatible GPU | CatBoostError "pairwise only" | `task_type=CPU` obligatoire quand rsm set | — |
| LR=0.005 + 50K rounds pas convergé | 8h11m sans convergence | LR ≥ 0.01 pour oblivious trees, ou 150K cap | V8 v3 |

### Sources fabricant
- https://catboost.ai/docs/en/concepts/parameter-tuning
- https://catboost.ai/docs/en/references/training-parameters/common
- https://catboost.ai/docs/en/concepts/algorithm-main-stages (ordered boosting)
- CatBoost Optuna tutorial : github.com/catboost/tutorials (l2_leaf_reg tuné, min_data_in_leaf absent)
- Springer 2020 "CatBoost for big data: an interdisciplinary review"
- rsm GPU crash : CatBoostError "rsm on GPU is supported for pairwise modes only"

---

## Règle d'or : alpha × architecture

| Architecture | Alpha optimal | Alpha sensitivity | Mécanisme |
|-------------|--------------|------------------|-----------|
| Leaf-wise (LGB) | **0.1** | **0.054** (0.1→0.7) | GOSS drop petits gradients → grands résidus = plus de signal leaf-wise |
| Oblivious (CB) | **0.3** | **0.0024** (0.3→0.5 à depth=5) | Structure fixe par niveau, modérément sensible |
| Depth-wise (XGB) | **0.5** | **0.001** (0.3→0.7) | Construit l'arbre entier, compense via n_iter |

**NE JAMAIS appliquer le même alpha aux 3 modèles.** (ADR-008, confirmé par 590 configs)

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

## Init Scores — Formule exacte

```python
# 1. Elo → probas (modèle logistique FIDE, avantage blanc dynamique)
p_win  = 1 / (1 + 10**((elo_noir - elo_blanc_adj) / 400))
p_loss = 1 / (1 + 10**((elo_blanc_adj - elo_noir) / 400))
p_draw = 1 - p_win - p_loss  # + draw_rate_lookup.parquet (Elo band × diff band)

# 2. Probas → log-odds centrés (init_scores)
log_p = log(clip(proba, 1e-4, 1-1e-4))          # clip évite log(0)
init_scores = log_p - mean(log_p, axis=1)        # centrage per-row, |scores| < 20

# 3. Alpha per-model (ADR-008)
init_scores_scaled = init_scores * alpha_per_model

# 4. Shapes par librairie
# XGBoost : (n_samples * n_classes,) FLATTENED → DMatrix.set_base_margin()
# LightGBM : (n_samples, n_classes) → fit(init_score=)
# CatBoost : (n_samples, n_classes) → Pool(baseline=)
```

**Règle** : init_scores calculés AVANT tout filtrage de features (blanc_elo/noir_elo requis).

---

## Learning Rate & Corrections Résiduelles

Les corrections résiduelles (ce que le ML ajoute à Elo) sont **petites** (~0.008 en magnitude).
Un LR trop haut fait que chaque arbre dépasse la correction cible.

| LR | Correction effective par arbre | Résultat empirique |
|----|-------------------------------|-------------------|
| 0.1 (default) | 12.5× overshoot | **Explosion** : log_loss > 2.0 en 100 iters (V8 v2) |
| 0.03 | 3.75× overshoot | Overshoot puis correction (converge si ES=200, mais 9 iters au best) |
| 0.01 | 1.25× overshoot | CatBoost converge (v4) |
| 0.005 | 0.625× target | XGBoost converge (v18, 86.5K rounds). CatBoost PAS convergé à 50K |
| 0.001 | trop lent | Non testé (budget Kaggle 12h insuffisant) |

**Guideline** : LR ∈ [0.005, 0.05] avec early_stopping=200. Couplé : LR plus bas = plus d'itérations.
V9 fixe LR=0.05 + ES=200 pour tous (converge en 800-3000 iters sur 62K train).

---

## V8 Benchmark Final (test set, 231,532 samples)

| Métrique | XGBoost v5 | LightGBM v7 | CatBoost v6 | Elo baseline |
|----------|-----------|-------------|-------------|-------------|
| **Test log_loss** | **0.5660** | 0.5721 | 0.5753 | 0.9766 |
| vs Elo (%) | **-42.0%** | -41.4% | -41.1% | — |
| **Test RPS** | **0.0891** | 0.0899 | 0.0899 | — |
| **E[score] MAE** | **0.2474** | 0.2487 | 0.2497 | — |
| **Brier** | **0.3414** | 0.3454 | 0.3442 | — |
| Accuracy | **0.7463** | 0.7435 | 0.7460 | — |
| F1 Macro | **0.6991** | 0.6966 | 0.6976 | — |
| Recall Draw | 0.5522 | **0.5578** | 0.5347 | — |
| ECE Loss | 0.0107 | 0.0117 | **0.0097** | — |
| ECE Draw | 0.0156 | 0.0183 | **0.0143** | — |
| ECE Win | 0.0094 | 0.0124 | **0.0091** | — |
| Draw bias | 0.0146 | 0.0177 | **0.0127** | — |
| Model size | 427 MB | 86 MB | **23 MB** | — |
| Training time | 4,182s (~70 min) | 20,973s (~5.8h) | 20,607s (~5.7h) | — |

**XGBoost = champion V8** (meilleur log_loss, RPS, E[score] MAE).
**CatBoost = meilleure calibration** (ECE et draw bias les plus bas).
**LightGBM = meilleur recall draw** (0.5578).
Tous les 3 over-predict draws (mean P(draw) > observed 12.04%).

### Erreur par bande Elo (XGBoost)
| Avg Elo | Error rate | Draw rate | Interprétation |
|---------|-----------|-----------|----------------|
| <1200 | 20.5% | 4.9% | Facile (peu de draws) |
| 1400–1600 | 23.5% | 10.6% | — |
| 1600–1800 | 29.0% | 17.2% | — |
| 1800–2000 | 34.1% | 23.0% | — |
| >2400 | 36.7% | 46.3% | Near-random (draws dominent) |
| \|diff_elo\| > 400 | 12% | — | Très prévisible |
| \|diff_elo\| < 50 | 26-33% | — | Near-random floor |

---

## Data Distribution (contexte pour le ML)

### Dataset sizes (après exclusion forfeits)
| Split | Rows | Saisons | Usage |
|-------|------|---------|-------|
| Train | 1,090,150 | 2002–2022 | Entraînement |
| Valid | 70,647 | 2023 | Early stopping + calibration |
| Test | 231,532 | 2024–2026 | Évaluation finale |
| V9 HP search | ~62K train + 71K valid | 2022 only | Optuna/Grid (ADR-009) |

### Class balance
- Loss : 43.1% — Draw : **14.3%** — Win : 42.6%
- Draw = classe minoritaire mais **45% de la variance E[score]** → signal critique pour le CE

### Draw rate par facteur
| Facteur | Min | Max | Ratio |
|---------|-----|-----|-------|
| Elo moyen | 4.9% (<1200) | 45.8% (>2400) | ×9.4 |
| Compétition | 9.6% (régional) | 20.9% (national) | ×2.2 |
| Diff Elo | 4.3% (>400) | 18.4% (<50) | ×4.3 |
| Échiquier | 9.5% (board 8) | 16.3% (board 1) | ×1.7 |

### Features
- **196 colonnes** (après V8 FE) : 128 initiales + 8 draw priors + 16 vases communiquants + 24 différentiels + 4 player strength - 8 remplacées - 2 leaky
- **Features sparse** : color perf coverage 7.4% (NaN 93%), H2H pairs ≥3 confrontations 0.8% (NaN >99%)
- **61 features mortes** (fixé) : temporal split bug excluait saison courante de valid/test history
- Avantage blanc dynamique : **+8.5 à +32.4** Elo (pas +35 fixe), lookup par niveau

---

## Stacking / Ensemble de Production

### Résultats V8 (test set)
| Méthode | Test log_loss | E[score] MAE | ECE Draw | Draw bias |
|---------|-------------|-------------|----------|-----------|
| XGBoost seul | **0.56604** | 0.24739 | 0.01555 | 0.01460 |
| Blend 90/5/5 | 0.56590 | 0.24754 | 0.01578 | — |
| Stack_LR | 0.60414 | 0.24633 | 0.03011 | — |
| Stack_MLP | 0.58250 | 0.24460 | 0.01591 | — |
| **Stack_MLP_cal** | 0.57335 | **0.24254** | **0.01233** | **0.01113** |

### Recommandation production
**Stack_MLP_cal** = MLP meta-learner (hidden=16, max_iter=500) sur 3 modèles calibrés + isotonic post-hoc.
- E[score] MAE : **-2.0%** vs XGBoost seul (0.24254 vs 0.24739)
- ECE draw : **-20.7%** (0.01233 vs 0.01555)
- Draw bias : **-23.8%** (0.01113 vs 0.01460)
- log_loss : +1.3% (trade-off accepté car CE utilise E[score], pas log_loss)

**Seuil de décision** : gain E[score] MAE > 0.001 → stacking. Gain = 0.00485 → **SIGNIFICATIF**.

### Architecture stacking
- Input meta-learner : (n, 9) = 3 modèles × 3 probas calibrées
- Meta-learner : `MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, early_stopping=True)`
- Post-hoc : `CalibratedClassifierCV(method='isotonic', cv='prefit')`
- Mémoire totale : ~536 MB (3 modèles + MLP) — Oracle VM 24 GB OK

---

## Serving Pipeline (Phase 2)

### Feature assembly
- Pre-computed (feature store, refresh hebdo) : `joueur_features.parquet`, `equipe_features.parquet`, `draw_rate_lookup.parquet`, `standings_current.parquet`
- Computed at request : `diff_elo`, `avg_elo`, `est_domicile`, `echiquier`, `ronde`, `saison`, `type_competition` + `compute_differentials()`
- Assembly target : **<50 ms** pour un matchup

### Batch prediction (ALI → ML)
- 20 scénarios ALI × N matchs × K boards × ~10 candidats = **~1600 prédictions ML** par ronde
- Single `predict_proba` batch → **~10 ms**
- CE OR-Tools solver : **<2s** pour 50 joueurs × 3 équipes

### Startup (8 étapes)
1. Load models (XGB .ubj 427MB + LGB .txt 86MB + CB .cbm 23MB)
2. Load `calibrators.joblib` (temperature T per model)
3. Load `encoders.joblib` (LabelEncoder per feature)
4. Load `draw_rate_lookup.parquet` (45 cells)
5. Load feature store parquets
6. Load `metadata.json` (alpha per model, version, quality gates)
7. Validate : artefacts present, versions match
8. Store in `app.state` → ready

### Drift monitoring (production)
| Metric | Warning | Critical |
|--------|---------|----------|
| PSI per class | ≥ 0.10 | ≥ 0.25 |
| Log loss drift | ≥ 5% | ≥ 10% |
| Draw rate drift | ≥ 3% abs | ≥ 5% abs |
| Feature store age | > 7 days | > 14 days |

---

## fANOVA — Fiabilité par source

| Source | Modèles couverts | Valide pour ALICE ? |
|--------|-----------------|-------------------|
| Probst et al. 2019 (JMLR) | glmnet, rpart, kknn, svm, ranger, **XGBoost** | **XGBoost SEUL**. Citations pour CatBoost/LightGBM = INVALIDES |
| van Rijn & Hutter 2018 (KDD) | **LightGBM** (feature_fraction #5) | Oui pour LightGBM |
| CatBoost official docs | **CatBoost** (depth + l2_leaf_reg = "first to tune") | Oui pour CatBoost |
| V9 Grid/Optuna fANOVA | 3 modèles (données empiriques ALICE) | **Source la plus fiable** — même dataset, même setup |

---

## Mise à jour

Ce fichier est mis à jour à chaque nouveau résultat empirique.
Dernière mise à jour : 2026-04-11 (Grid XGB v4 + Grid LGB v2 + Optuna LGB v7 + findings consolidés).
CatBoost : à compléter quand Grid CB v2 termine.
