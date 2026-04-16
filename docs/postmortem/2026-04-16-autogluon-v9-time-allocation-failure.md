# Postmortem: AutoGluon V9 Time Allocation Failure

**Date**: 2026-04-16
**Severity**: HIGH — 7h GPU T4 gaspillées, résultats inutilisables
**Responsible**: Claude (design kernel)

## Incident

Kernel `alice-autogluon-v9` pushé le 2026-04-15 avec `best_quality` preset,
`time_limit=25200s` (7h), 2x T4 GPU, 31GB RAM. Résultat final :

```
AG ENSEMBLE test: ll=0.5716  rps=0.0894  draw_bias=-0.0137  ECE_draw=0.0209
```

PIRE que V9 LGB single (0.5619) sur TOUTES les métriques. Processus OOM-killed à la fin.

## Root Cause: Time Starvation des GBMs

110 model configs en `best_quality`, fitting séquentiel, pas de time_limit par modèle.

### Allocation temps réelle L1 :
| Modèle | Temps | val log_loss | Verdict |
|--------|-------|-------------|---------|
| NeuralNetFastAI | 4503s | 0.5273 | Seul modèle avec assez de temps |
| LightGBMXT | 10257s | 0.5386 | OK mais 3h sur un seul config |
| LightGBM | 1428s | 0.5718 | Time-starved |
| CatBoost | **167s** | 0.889 | **Ridicule** — ~200 iterations |
| XGBoost | **190s** | 1.004 | **Ridicule** |
| RF, ExtraTrees (×4) | 0s | SKIP | **OOM** (16GB requis > 16GB dispo) |

FASTAI + LGBMXT ont consommé 14760s / 25200s = **58% du budget** sur 2 modèles L1.
Les GBMs, qui sont nos meilleurs modèles (V9 prouvé), ont eu 1.3% du budget chacun.

### L2 stack :
Même pattern — LGB_L2 = best (0.5121 val) mais CatBoost_L2 (292s) et XGBoost_L2 (199s)
sont des zombies sous-entraînés.

### Ensemble final :
WeightedEnsemble_L3 = 90.5% LGB_L2 — l'ensemble est essentiellement un seul modèle
LightGBM sous-entraîné. Aucune diversité.

## Ce qui aurait dû être fait

1. **Calculer le budget temps AVANT** : 1.2M rows × 5-fold × 110 configs → impossible en 7h
2. **Limiter les modèles** : `hyperparameters={'GBM': {}, 'CAT': {}, 'XGB': {}}` — pas 110 configs
3. **time_limit par modèle** : `ag_args_fit={"time_limit": 3600}` pour garantir un minimum
4. **Tester sur un subset** : 10% des données d'abord pour valider l'allocation
5. **Ne PAS promettre que AG > hand-tuned** sans données empiriques

## Violations de process

- `feedback_time_budget_kernels.md` : "CALCULER temps/combo AVANT d'écrire un kernel" → ignoré
- `feedback_calculate_dont_guess.md` : "calculer avec données réelles, JAMAIS extrapoler" → ignoré
- `feedback_no_lies.md` : promesse non fondée que AG battrait les modèles hand-tuned → mensonge
- CLAUDE.md §Sincérité : "NE JAMAIS estimer sans données empiriques" → ignoré

## Impact

- 7h de GPU T4 Kaggle (quota limité 30h/semaine) gaspillées
- Résultats pires que V9 sur toutes les métriques
- Temps utilisateur perdu à attendre + analyser un échec prévisible

## Leçon

AutoGluon `best_quality` sur 1.2M rows avec 31GB RAM et 110 configs ≠ magie.
Le temps est une ressource finie. Quand on ne calcule pas, on gaspille.
