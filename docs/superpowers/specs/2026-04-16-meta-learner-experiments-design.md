# Meta-Learner Experiments — Design Spec

**Date**: 2026-04-16
**Objectif**: trouver le meilleur meta-learner pour combiner XGB+LGB+CB V9
**Critere champion**: ECE_draw + |draw_bias| first, log_loss second (CE downstream)
**Baseline**: LGB + Dirichlet (0.5541, ECE_draw 0.0042, bias -0.0017)

## Contexte

MLP(32,16) sur 18 features (9 probas + 9 engineered) bat LGB+Dirichlet :
- log_loss 0.5531 vs 0.5541
- ECE_draw 0.0025 vs 0.0042
- draw_bias -0.0006 vs -0.0017

Question : ajouter des features brutes (top SHAP) ameliore-t-il encore ?

## Recherche SOTA (2024-2025)

### Restacking (features brutes + predictions base)
- scikit-learn `StackingClassifier(passthrough=True)` — standard
- Risque : meta-learner bypass les base models → devient un concurrent
- Mitigation : OOF obligatoire + regularisation forte + features limitees

### Regularisation meta-learner NN
- NeurIPS 2024 ("Better by Default"): MLPs pre-tuned + dropout standard
- TabM (ICLR 2025): weight sharing comme regularisation efficace
- Best practice : early_stopping + validation_fraction = regularisation implicite

### Calibration post-stacking
- Si meta-learner degrade calibration → Dirichlet post-hoc (Kull 2019)
- MLP(32,16) 18feat ECE_draw=0.0025 = deja excellent → post-hoc optionnel

## Plan d'experiences

### Axe 1 : nombre de features brutes (top SHAP)

| Exp | Features | Total | Hypothese |
|-----|---------|-------|-----------|
| E1 | 9 probas only | 9 | Baseline MLP |
| E2 | 9 probas + 9 engineered | 18 | **Current best** |
| E3 | 18 + top 10 SHAP consensus | 28 | Features les plus informatives |
| E4 | 18 + top 30 SHAP consensus | 48 | Plus de contexte |
| E5 | 18 + ALL 201 features | 219 | Risque overfitting eleve |

### Axe 2 : architecture MLP (sur best feature set)

| Exp | Architecture | Params approx |
|-----|-------------|--------------|
| A1 | MLP(16) | ~320 |
| A2 | MLP(32,16) | ~1200 |
| A3 | MLP(64,32) | ~4000 |
| A4 | MLP(128,64,32) | ~14000 |

### Axe 3 : regularisation (sur best combo)

| Exp | Config |
|-----|--------|
| R1 | alpha=1e-4 (default sklearn) |
| R2 | alpha=1e-3 |
| R3 | alpha=1e-2 |
| R4 | alpha=1e-1 (forte regularisation) |

### Axe 4 : calibration post-MLP

| Exp | Methode |
|-----|---------|
| C1 | MLP raw (pas de post-calibration) |
| C2 | MLP + temperature scaling (Guo 2017) |
| C3 | MLP + Dirichlet (Kull 2019) |

## Protocole

1. **OOF split** : meta-learner entraine sur OOF (1.21M rows), evalue sur test (231K)
2. **Metriques** : log_loss, ECE_draw, draw_bias, RPS (toutes les 4, pas juste logloss)
3. **Seed** : random_state=42, early_stopping=True, validation_fraction=0.1
4. **Guard** : si AUCUN MLP bat LGB+Dirichlet → LGB+Dirichlet reste champion
5. **Documentation** : resultats dans MODEL_SPECS.md, code dans scripts/train_meta_learner.py

## Top features consensus (SHAP cross-model)

Presentes dans le top 20 des 3 modeles (CB SHAP + XGB gain + LGB gain) :
1. draw_rate_home_dom
2. draw_rate_noir
3. draw_rate_blanc
4. win_rate_home_dom
5. est_domicile_blanc
6. diff_form (XGB), saison (CB/LGB)
7. draw_rate_recent_noir
8. draw_rate_recent_blanc
9. draw_trend_blanc
10. win_rate_normal_noir / win_rate_normal_blanc
11. ronde
12. diff_elo
13. draw_rate_equipe_ext
14. win_rate_black_noir / win_rate_white_blanc
15. draw_rate_black_noir / draw_rate_white_blanc

## Sources

- scikit-learn StackingClassifier passthrough: sklearn docs
- NeurIPS 2024: "Better by Default: Strong Pre-Tuned MLPs" 
- TabM ICLR 2025: parameter-efficient ensembling
- Kull 2019 (NeurIPS): Dirichlet calibration
- Guo 2017 (ICML): temperature scaling
- MLMastery: stacking best practices (OOF, simple meta-learner)
- GeeksforGeeks: restacking with raw features
