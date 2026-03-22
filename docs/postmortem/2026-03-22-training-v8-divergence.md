# Post-mortem: V8 Training Divergence (Kernel 2, v2)

**Date:** 2026-03-22
**Kernel:** `pguillemin/alice-training-v8` v2
**Duration:** 4 min 16s (FE load to completion)
**Outcome:** Quality gate FAILED — `log_loss >= elo baseline`

---

## Symptom

Les 3 modèles divergent après très peu d'itérations :

| Modèle | Best iter | Best log_loss | Iter 100 | Pattern |
|--------|-----------|---------------|----------|---------|
| CatBoost | 39 / 3000 | 0.9399 | 1.037 | Overfit doux |
| XGBoost | ~2 / 3000 | ~0.956 | 2.367 | Divergence catastrophique |
| LightGBM | 3 / 3000 | 0.967 | 3.598 | Divergence catastrophique |

## Root Cause

**Hyperparamètres trop agressifs pour le dataset :**

1. `max_depth=8` + 177 features (dont ~50% sparse) = arbres trop profonds, overfitting immédiat
2. Pas de `learning_rate` explicite → valeurs par défaut 0.1 (XGB) / 0.1 (LGB) / auto (CB) trop élevées
3. `min_data_in_leaf=20` trop faible pour 1.1M lignes → splits sur le bruit
4. `num_leaves=255` (LGB) trop élevé → complexité excessive
5. Pas de subsampling → modèle voit toutes les données, mémorise le bruit

Le pattern (best très tôt puis explosion) est typique d'un learning rate trop élevé
combiné avec des features bruitées. Le modèle apprend les corrélations Elo en <10 itérations,
puis commence à memoriser les artefacts des features sparse.

## Context: pourquoi la baseline Elo est forte

- Elo prédit ~60% des parties correctement en binaire
- La draw_rate_prior (lookup table elo_band × diff_band) capture les nulles par bande Elo
- Ensemble, Elo + draw_rate_prior fait un log_loss ~0.92-0.93
- Pour battre ça, le ML doit apporter de l'information au-delà de (Elo, diff_elo, draw_prior)
- Les features comportementales (forme, H2H, pression) sont très sparse et bruitées

## Fix Applied (v3)

| Paramètre | v2 | v3 | Rationale |
|-----------|----|----|-----------|
| learning_rate | default (0.1-0.3) | **0.03** | Convergence lente, moins d'overfitting |
| max_depth | 8 | **6** | Arbres moins profonds |
| num_leaves (LGB) | 255 | **63** | Consistant avec depth=6 |
| min_data_in_leaf | 20 | **50** | Régularisation splits |
| min_child_weight (XGB) | 1 | **10** | Idem XGBoost |
| reg_lambda | 1.0 | **3.0** | L2 plus forte |
| reg_alpha | 0.0 | **0.1** | L1 légère (feature selection implicite) |
| subsample | 1.0 | **0.8** | Bagging (XGB/LGB) |
| colsample_bytree | 1.0 | **0.8** | Feature bagging (XGB/LGB) |
| n_estimators | 3000 | **5000** | Plus d'iters à LR basse |
| early_stopping | 100 | **200** | Patience accrue |

## Expected Outcome (v3)

- Best iteration devrait être > 200 (pas 3-39)
- Log_loss devrait descendre progressivement, pas exploser
- Objectif : log_loss < 0.93 (battre Elo baseline)
- Si échec : envisager feature selection (dropper features >90% NaN)

## Lessons Learned

1. **Ne pas copier les hyperparams V7 binaire** — MultiClass 3-way est plus difficile à optimiser
2. **Avec 177 features dont beaucoup sparse, régulariser agressivement**
3. **Le learning rate par défaut (0.1) est presque TOUJOURS trop élevé** pour un premier run
4. **L'Elo baseline est un adversaire redoutable en échecs** — c'est normal que les premières tentatives échouent

## Tracking

| Version | Commit | Changement | Résultat |
|---------|--------|-----------|----------|
| v1 | affcb73 | Premier push | ERROR (path /notebooks/ manquant) |
| v2 | 179a027 | Fix path notebooks/ | COMPLETE — gate FAILED (divergence) |
| v3 | 1d0289d | Hyperparams conservatifs (LR=0.03, depth=6) | EN COURS |
