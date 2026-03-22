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
| v3 | 1d0289d | Hyperparams conservatifs (LR=0.03, depth=6) | COMPLETE — gate FAILED (CatBoost 0.926 vs Elo ~0.92) |

## V3 Results (2026-03-22 21:28)

| Modèle | Best iter | Best log_loss | v2 → v3 |
|--------|-----------|---------------|---------|
| CatBoost | 42 | 0.9261 | 0.940 → 0.926 (-0.014) |
| XGBoost | 33 | ~0.98 | Pire (GPU device mismatch?) |
| LightGBM | 20 | 0.9349 | 0.967 → 0.935 (-0.032) |

**Elo baseline ~0.92.** CatBoost à 0.006 du seuil. Le tuning manuel atteint ses limites.

## Root Cause révisé (post-analyse artefacts v3)

Le problème n'est PAS les hyperparamètres. C'est plus fondamental.

### Constat brutal

- **166/177 features à importance 0.0** (CatBoost). Le modèle = `diff_elo` + poignée de rates
- **Les 3 modèles ne prédisent JAMAIS draw** (recall=0% sur les 3). Classe 14% ignorée
- **Accuracy ~55%** = equivalent à "toujours prédire le favori Elo"
- Le modèle est un classifieur binaire loss/win déguisé en 3 classes

### Pourquoi

On a construit 177 features d'un coup sans validation incrémentale. Le modèle
se noie dans le bruit des features sparse (93% NaN color_perf, 99% NaN H2H)
et n'a pas le temps d'apprendre les patterns draw avant de diverger.

L'erreur fondamentale : les features draw V8 (draw_rate_blanc, draw_rate_equipe,
H2H_draw_rate) sont CORRECTES et NECESSAIRES, mais noyées dans un océan de bruit.
Le modèle ne les trouve pas.

### Leçon

"Les modèles apprendront" n'est pas une stratégie. Avec 177 features dont 94% bruit,
même CatBoost ne peut pas trouver le signal draw. L'approche YAGNI aurait dû être
appliquée aux FEATURES (start small, add if gain), pas à la technique de renforcement Elo.

## Next Step : Residual Learning + AutoGluon

### Approche B : Residual Learning (prioritaire)

Au lieu de prédire P(W/D/L) from scratch, partir de la prediction Elo baseline
et apprendre les CORRECTIONS :

```
init_score = log(P_elo(loss)), log(P_elo(draw)), log(P_elo(win))
modèle apprend : "quand est-ce que draw > ou < ce que l'Elo prédit ?"
```

CatBoost: `init_model` ou calcul des log-odds initiaux
XGBoost: `base_margin` parameter
LightGBM: `init_score` parameter

**Pourquoi c'est le bon move :**
- La baseline Elo capture déjà ~92% de log_loss
- Le modèle n'a qu'à trouver les 8% restants
- Les features draw V8 (draw_rate, H2H, pression) deviennent UTILES
  car le modèle cherche spécifiquement les corrections draw
- L'objectif ALICE = P(draw) calibrées pour CE. Residual = focalisé dessus

### Approche C : AutoGluon (validation)

AutoGluon sur les mêmes parquets pour comparer. S'il bat la baseline Elo
sans residual learning → les features ont du signal, c'était notre approche.
S'il échoue aussi → les features elles-mêmes ont un problème.
