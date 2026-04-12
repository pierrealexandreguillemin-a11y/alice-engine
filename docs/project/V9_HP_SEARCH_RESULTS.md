# V9 HP Search — Resultats complets pour Training Final

**Date** : 2026-04-11
**Dataset HP search** : saison=2022 (~62K train, ~71K valid) — ADR-009
**Dataset Training Final** : 2002-2022 (1,139,799 train, 70,647 valid, 231,532 test)
**Elo baseline logloss** : 0.9766 (HP search) / 0.874 (V8 full, via metadata.json)

---

## 1. XGBoost

### 1.1 V8 converge (full dataset 1.1M, test=231K)

Source : `reports/v8_xgboost_v5_resume/metadata.json`

| Param | Valeur V8 |
|-------|-----------|
| eta | 0.005 |
| max_depth | 4 |
| lambda | 10.0 |
| alpha (L1) | 0.5 |
| min_child_weight | 50 |
| subsample | 0.7 |
| colsample_bytree | 0.5 |
| init_score_alpha | 0.7 |
| tree_method | hist |
| n_estimators | 50000 (ES=200) |
| **rounds converges** | **86,490** |
| **val logloss** | **0.5125** |
| **test logloss** | **0.5660** |
| test RPS | 0.0891 |
| test E[score] MAE | 0.2474 |
| test accuracy | 0.7463 |
| test F1 macro | 0.6991 |
| recall draw | 0.5522 |
| ECE draw | 0.0156 |
| draw bias | 0.0146 |
| training time | 4,182s (~70 min) |
| model size | 427 MB |
| temperature | 0.928 (v18), 0.971 (resume v5) |
| features used (gain>0) | 197/201 |

### 1.2 Grid v4 (62K, 82/82 combos completes)

Source : `grid_results/xgboost_v4/grid_best_xgboost.json`
Fixe : eta=0.05, depth=8, lambda=4.0, alpha_reg=0.01

| # | logloss | init_alpha | subsample | colsample | mcw | iters |
|---|---------|------------|-----------|-----------|-----|-------|
| 1 | **0.5198** | 0.5 | 0.8 | 1.0 | 50 | 804 |
| 2 | 0.5201 | 0.65 | 0.8 | 1.0 | 50 | 873 |
| 3 | 0.5204 | 0.5 | 0.8 | 0.75 | 50 | 993 |
| 4 | 0.5208 | 0.5 | 0.7 | 1.0 | 50 | 788 |
| 5 | 0.5210 | 0.8 | 0.8 | 1.0 | 50 | 808 |
| V8 baseline | 0.5238 | 0.7 | 0.7 | 0.5 | 50 | 926 |

**Tendances Grid** : alpha plat (0.5→0.8 = 0.0012 delta), sub 0.8>0.7>0.6, col 1.0>0.75>0.5, mcw 50>125>200.
**v8_logloss** : 0.5238 (config V8 evaluee sur 62K)

### 1.3 Optuna v8 (62K, 122 trials = 6 old + 100 new + 16 pruned)

Source : `optuna_studies/xgboost_v8/best_params_xgboost.json`

**Anciens trials (8 params tunes, resumed depuis etude precedente) :**

| # | logloss | alpha | eta | lambda | sub | col | mcw | dur |
|---|---------|-------|-----|--------|-----|-----|-----|-----|
| 1 | **0.5116** | 0.672 | 0.046 | 3.24 | 0.698 | 0.631 | 132 | 8,852s |
| 2 | 0.5133 | 0.695 | 0.067 | 4.85 | 0.623 | 0.999 | 154 | 4,446s |
| 3 | 0.5140 | 0.695 | 0.067 | 4.78 | 0.627 | 0.985 | 149 | 4,586s |
| 4 | 0.5174 | 0.487 | 0.029 | 2.39 | 0.578 | 0.341 | 176 | 10,823s |
| 5 | 0.5179 | 0.333 | 0.092 | 7.25 | 0.549 | 0.779 | 99 | 2,814s |
| 6 | 0.5205 | 0.487 | 0.054 | 2.39 | 0.578 | 0.341 | 176 | 7,793s |

**Nouveaux trials (4 params tunes, eta=0.05/depth=8/lambda=4.0/alpha_reg=0.01 fixes) :**

| # | logloss | alpha | sub | col | mcw | dur |
|---|---------|-------|-----|-----|-----|-----|
| 1 | **0.5201** | 0.710 | 0.800 | 0.769 | 50 | 146s |
| 2 | 0.5202 | 0.715 | 0.795 | 0.881 | 52 | 146s |
| 3 | 0.5202 | 0.717 | 0.794 | 0.916 | 65 | 152s |
| 4 | 0.5203 | 0.719 | 0.800 | 0.935 | 60 | 168s |
| 5 | 0.5205 | 0.760 | 0.799 | 0.898 | 58 | 160s |

**Note** : les anciens trials venaient d'une etude avec search space elargi (eta, depth, lambda, alpha_reg tunes). Le meilleur resultat global (0.5116) utilise eta=0.046, pas 0.05.

### 1.4 Synthese XGBoost — convergence cross-methodes

| Param | V8 (1.1M) | Grid (62K) | Optuna old (62K) | Optuna new (62K) | Convergence |
|-------|-----------|------------|------------------|------------------|-------------|
| eta | 0.005 | 0.05 (fixe) | **0.046** | 0.05 (fixe) | ~0.05, tunable |
| depth | 4 | 8 (fixe) | 8 | 8 (fixe) | **8** |
| lambda | 10.0 | 4.0 (fixe) | 3.24 | 4.0 (fixe) | **3-4** |
| alpha_reg | 0.5 | 0.01 (fixe) | 0.011 | 0.01 (fixe) | **~0.01** |
| init_alpha | 0.7 | **0.5** | 0.67 | 0.71 | **0.5-0.7** (plat) |
| subsample | 0.7 | **0.8** | 0.70 | **0.80** | **0.8** |
| colsample | 0.5 | **1.0** | 0.63 | 0.85-0.94 | **0.9-1.0** |
| mcw | 50 | **50** | 132 | **50-65** | **50** |

---

## 2. LightGBM

### 2.1 V8 converge (full dataset 1.1M, test=231K)

Source : `config/hyperparameters.yaml` section lightgbm + MODEL_SPECS.md

| Param | Valeur V8 |
|-------|-----------|
| learning_rate | 0.03 (v7), 0.003 (yaml) |
| num_leaves | 15 |
| max_depth | 4 |
| reg_lambda | 10.0 |
| reg_alpha | 0.5 |
| min_child_samples | 200 |
| subsample | 0.7 |
| colsample_bytree | 0.5 |
| init_score_alpha | 0.7 |
| n_estimators | 50000 (ES=200) |
| **rounds converges** | **~16,100** (v7, lr=0.03) |
| **test logloss** | **0.5721** |
| test RPS | 0.0899 |
| test E[score] MAE | 0.2487 |
| recall draw | 0.5578 (meilleur des 3) |
| ECE draw | 0.0183 |
| draw bias | 0.0177 |
| training time | 20,973s (~5.8h) |
| model size | 86 MB (.txt) |
| features used (gain>0) | 50/177 |

### 2.2 Grid v2 (62K, 82/82 combos completes)

Source : `grid_results/lightgbm_v2/grid_best_lightgbm.json`
Fixe : lr=0.05, depth=8, lambda=4.0, bagging=0.8

| # | logloss | init_alpha | leaves | ff | mcs | iters |
|---|---------|------------|--------|-----|-----|-------|
| 1 | **0.5357** | **0.4** | 15 | 1.0 | 275 | 1,582 |
| 2 | 0.5360 | **0.4** | 15 | 1.0 | 50 | 1,876 |
| 3 | 0.5367 | **0.4** | 15 | 0.65 | 50 | 1,851 |
| 4 | 0.5370 | **0.4** | 15 | 1.0 | 500 | 1,535 |
| 5 | 0.5372 | **0.4** | 15 | 0.65 | 275 | 1,804 |
| V8 baseline | 0.5735 | 0.7 | 15 | 0.5 | 200 | 1,757 |

**Tendances Grid** : alpha **DOMINANT** (0.4 monopole top-20, gap 0.5735-0.5357=0.0378 vs V8). Leaves 15 optimal. ff 1.0>0.65>0.3. mcs quasi-plat.
**Alpha = param #1** : 100% du top-20 a alpha=0.4 (confirme ADR-008).

### 2.3 Optuna v7 (62K, 100/100 trials completes)

Source : `optuna_studies/lightgbm_v7/best_params_lightgbm.json`
Fixe : lr=0.05, depth=8, lambda=4.0, bagging=0.8
Alpha range : [0.4, 0.8] (etendu depuis v6 qui etait [0.5, 0.8])

| # | logloss | alpha | leaves | ff | mcs | dur |
|---|---------|-------|--------|-----|-----|-----|
| 1 | **0.5460** | 0.500 | 26 | 0.880 | 166 | 86s |
| 2 | 0.5470 | 0.509 | 29 | 0.897 | 68 | 79s |
| 3 | 0.5470 | 0.500 | 38 | 0.618 | 166 | 59s |
| 4 | 0.5471 | 0.508 | 39 | 0.930 | 210 | 69s |
| 5 | 0.5472 | 0.507 | 22 | 0.908 | 188 | 85s |

**Alpha top-20** : mean=0.504, range=[0.500, 0.521]. TPE converge sur la borne basse 0.50 mais ne descend **JAMAIS** sous 0.5.
**Grid > Optuna** : 0.5357 vs 0.5460 = delta 0.0103, entierement du a alpha (Grid couvre 0.4, TPE non).

### 2.4 Synthese LightGBM — convergence cross-methodes

| Param | V8 (1.1M) | Grid (62K) | Optuna v7 (62K) | Convergence |
|-------|-----------|------------|-----------------|-------------|
| lr | 0.03 | 0.05 (fixe) | 0.05 (fixe) | **0.05** (avec ES=200) |
| num_leaves | 15 | **15** | 26 | **15** (Grid + V8 concordent) |
| max_depth | 4 | 8 (fixe) | 8 (fixe) | **8** (safety cap) |
| reg_lambda | 10.0 | 4.0 (fixe) | 4.0 (fixe) | **4.0** |
| init_alpha | 0.7 | **0.4** | 0.50 (borne) | **0.4** (Grid dominant) |
| ff | 0.5 | **1.0** | 0.88 | **1.0** |
| mcs | 200 | **275** | 166 | **275** |
| bagging | 0.7 | 0.8 (fixe) | 0.8 (fixe) | **0.8** |

---

## 3. CatBoost

### 3.1 V8 converge (full dataset 1.1M, test=231K)

Source : `config/hyperparameters.yaml` section catboost + MODEL_SPECS.md

| Param | Valeur V8 |
|-------|-----------|
| learning_rate | 0.03 (v6 actual), 0.005 (yaml) |
| depth | 4 |
| l2_leaf_reg | 10 |
| min_data_in_leaf | 200 |
| rsm | 0.3 |
| random_strength | 3 |
| border_count | 128 |
| init_score_alpha | 0.7 |
| iterations | 50000 (ES=200) |
| **rounds converges** | **~37,000** (v6, lr=0.03) |
| **test logloss** | **0.5753** |
| test RPS | 0.0899 |
| test E[score] MAE | 0.2497 |
| recall draw | 0.5347 |
| ECE draw | 0.0143 (meilleur des 3) |
| draw bias | 0.0127 (meilleur des 3) |
| training time | 20,607s (~5.7h) |
| model size | 23 MB |
| temperature | 0.935 |

### 3.2 Grid v1 (62K, 32/32 combos — SANS l2_leaf_reg)

Source : `grid_results/catboost/grid_search_catboost.csv`
Fixe : lr=0.05. **SANS l2_leaf_reg** (erreur corrigee dans Grid v2).

| # | logloss | init_alpha | depth | rsm | mdil | iters |
|---|---------|------------|-------|-----|------|-------|
| 1 | **0.5639** | 0.5 | 6 | 0.5 | 50 | 6,002 |
| 2 | 0.5639 | 0.5 | 6 | 0.5 | 100 | 6,002 |
| 3 | 0.5639 | 0.5 | 6 | 0.5 | 200 | 6,002 |
| 4 | 0.5642 | 0.5 | 4 | 0.5 | 50 | 10,170 |
| 5 | 0.5642 | 0.5 | 4 | 0.5 | 100 | 10,170 |
| V8 baseline | 0.5662 | 0.7 | 4 | 0.3 | 200 | 11,979 |

**Tendances** : depth 6>4>8>10, rsm 0.5~0.7>0.3, **mdil = ZERO effet** (3 valeurs identiques par combo). Alpha plat (0.5 vs 0.7 = 0.0023).

### 3.3 Grid v2 (62K, 70/70 combos — AVEC l2_leaf_reg)

Source : `grid_results/catboost_v2/grid_best_catboost.json`
Fixe : lr=0.05, mdil=200, random_strength=2.0

| # | logloss | init_alpha | depth | l2_leaf_reg | rsm | iters |
|---|---------|------------|-------|-------------|-----|-------|
| 1 | **0.5256** | 0.5 | 4 | 8.0 | 0.7 | 6,968 |
| 2 | 0.5257 | 0.5 | 4 | 15.0 | 0.7 | 6,993 |
| 3 | 0.5263 | 0.5 | 4 | 15.0 | 0.45 | 7,600 |
| 4 | 0.5264 | 0.5 | 7 | 15.0 | 0.7 | 3,748 |
| 5 | 0.5264 | 0.65 | 4 | 15.0 | 0.7 | 6,729 |
| 6 | 0.5265 | 0.65 | 4 | 8.0 | 0.7 | 7,021 |
| 7 | 0.5267 | 0.8 | 4 | 8.0 | 0.7 | 7,079 |
| 8 | 0.5269 | 0.5 | 4 | 1.0 | 0.45 | 6,763 |
| 9 | 0.5270 | 0.65 | 4 | 15.0 | 0.45 | 7,174 |
| 10 | 0.5270 | 0.5 | 4 | 8.0 | 0.45 | 6,336 |
| V8 baseline | 0.5284 | 0.7 | 4 | 10 (yaml) | 0.3 | — |

**Tendances Grid v2** : depth **4** domine top-10 (9/10). rsm **0.7>0.45**. l2 **8-15** (l2=1.0 nettement pire). Alpha **plat** (0.5/0.65/0.8 = delta 0.0011).
**v8_logloss** : 0.5284 (config V8 evaluee sur 62K)
**Grid v2 > Grid v1** : 0.5256 vs 0.5639 = delta 0.0383 — l2_leaf_reg dans le search space = gain massif.

### 3.4 Optuna v3 (62K, 2 trials — sur ancien dataset full, NON COMPARABLE)

Source : `optuna_studies/catboost_v3/best_params_catboost.json`
**ATTENTION** : ces 2 trials tournaient sur le full 1.1M (avant ADR-009). Logloss non comparable avec les resultats sur 62K.

| # | logloss | alpha | depth | l2_leaf_reg | rsm | dur |
|---|---------|-------|-------|-------------|-----|-----|
| 1 | **0.5211** | 0.547 | 5 | 1.81 | 0.633 | 17,798s |
| 2 | 0.5267 | 0.612 | 10 | 11.25 | 0.499 | 17,598s |

### 3.5 Optuna v4 (62K, 56/56 trials completes)

Source : `optuna_studies/catboost_v4/best_params_catboost.json`
Fixe : lr=0.05, random_strength=2.0, mdil=200
Resume : inclut les 2 trials de v3 (renumerotes 0-1) + 54 nouveaux trials

| # | logloss | alpha | depth | l2_leaf_reg | rsm | dur |
|---|---------|-------|-------|-------------|-----|-----|
| 1 | **0.5253** | 0.640 | 5 | 15.0 | 0.458 | 653s |
| 2 | 0.5256 | 0.537 | 5 | 3.1 | 0.656 | 591s |
| 3 | 0.5257 | 0.550 | 5 | 11.5 | 0.479 | 712s |
| 4 | 0.5258 | 0.792 | 5 | 13.3 | 0.450 | 766s |
| 5 | 0.5258 | 0.523 | 4 | 8.4 | 0.531 | 762s |
| 6 | 0.5258 | 0.575 | 4 | 5.9 | 0.698 | 797s |
| 7 | 0.5259 | 0.549 | 4 | 8.5 | 0.532 | 779s |
| 8 | 0.5260 | 0.502 | 4 | 6.3 | 0.653 | 722s |
| 9 | 0.5260 | 0.500 | 4 | 8.2 | 0.587 | 688s |
| 10 | 0.5261 | 0.504 | 4 | 8.7 | 0.501 | 704s |

**Top-20 distributions** : alpha mean=0.569 [0.50, 0.79], depth 4 ou 5 (9 chaque), l2 mean=7.7 [3.1, 15.0], rsm mean=0.574 [0.45, 0.70].
**Grid v2 ~ Optuna v4** : 0.5256 vs 0.5253 = delta 0.0003 — concordance.

### 3.6 Synthese CatBoost — convergence cross-methodes (COMPLET)

| Param | V8 (1.1M) | Grid v1 (62K) | Grid v2 (62K) | Optuna v4 (62K) | Convergence |
|-------|-----------|---------------|---------------|-----------------|-------------|
| lr | 0.03 | 0.05 (fixe) | 0.05 (fixe) | 0.05 (fixe) | **0.05** |
| depth | 4 | **6** | **4** | 4-5 | **4** (Grid v2 + V8 concordent) |
| l2_leaf_reg | 10 | non tune | **8-15** | mean 7.7 | **8** |
| rsm | 0.3 | 0.5 | **0.7** | mean 0.574 | **0.7** |
| mdil | 200 | zero effet | 200 (fixe) | 200 (fixe) | **200** (zero effet) |
| init_alpha | 0.7 | 0.5 | **0.5** | mean 0.569 | **0.5** (plat, delta 0.0011) |
| random_strength | 3 | non tune | 2.0 (fixe) | 2.0 (fixe) | **2.0** |

**Finding ADR-008 (CORRIGE par gaps)** : alpha CatBoost = **moderement sensible** (delta 0.0024 a depth=5), PAS quasi-indifferent. La synthese Grid v2 [0.5,0.8] montrait 0.0011 mais le gaps grid etendu a [0.3,0.8] revele un gradient plus net.

---

## 3.7 Gap-Filling Round 1 (3 kernels, 44 combos)

### Gaps XGB : alpha × depth (10 combos)

| alpha | depth=4 | depth=6 | depth=8 |
|-------|---------|---------|---------|
| 0.3 | 0.5189 | 0.5183 | 0.5197 |
| 0.5 | 0.5197 | **0.5178** | 0.5198 |
| 0.7 | 0.5203 | 0.5190 | 0.5209 |

**Finding** : depth=6 bat depth=4 ET depth=8 a tous les alphas. V8=4, V9 Grid=8 fixe, les deux sous-optimaux.

### Gaps LGB : alpha × lambda (17 combos)

| alpha | lam=1 | lam=4 | lam=10 | lam=15 |
|-------|-------|-------|--------|--------|
| 0.3 | 0.5280 | **0.5278** | 0.5279 | 0.5284 |
| 0.4 | 0.5360 | 0.5357 | 0.5364 | 0.5359 |
| 0.5 | 0.5473 | 0.5463 | 0.5462 | 0.5459 |
| 0.7 | 0.5725 | 0.5714 | 0.5720 | 0.5719 |

**Findings** : alpha=0.3 >> 0.4 (delta 0.0079). Lambda plat (range 0.0006).

### Gaps CB : alpha × depth (17 combos)

| alpha | d=4 | d=5 | d=6 | d=8 |
|-------|-----|-----|-----|-----|
| 0.3 | 0.5267 | **0.5245** | 0.5260 | 0.5294 |
| 0.4 | 0.5258 | 0.5254 | 0.5253 | 0.5301 |
| 0.5 | 0.5256 | 0.5257 | 0.5256 | 0.5296 |
| 0.7 | 0.5269 | 0.5267 | 0.5267 | 0.5313 |

**Findings** : alpha=0.3 + depth=5 = best. depth=8 nettement pire. Alpha moderement sensible (pas plat).

## 3.8 Gap-Filling Round 2 (1 kernel, 11 combos)

### LGB alpha extension (3 combos) — plancher trouve

| Alpha | Logloss | Delta vs precedent |
|-------|---------|-------------------|
| 0.7 | 0.5714 | — |
| 0.5 | 0.5459 | -0.0255 |
| 0.4 | 0.5357 | -0.0102 |
| 0.3 | 0.5278 | -0.0079 |
| 0.2 | 0.5224 | -0.0054 |
| 0.15 | 0.5195 | -0.0029 |
| **0.1** | **0.5180** | -0.0015 |

Gradient s'aplatit. Alpha=0.1 = plancher.

### XGB depth=6 refinement (8 combos) — params confirmes

| sub | col | mcw | logloss |
|-----|-----|-----|---------|
| 0.8 | 1.0 | 50 | **0.5178** |
| 0.7 | 1.0 | 50 | 0.5187 |
| 0.8 | 0.75 | 50 | 0.5194 |
| 0.7 | 0.75 | 50 | 0.5203 |
| 0.8 | 1.0 | 100 | 0.5204 |
| 0.8 | 0.75 | 100 | 0.5213 |
| 0.7 | 1.0 | 100 | 0.5221 |
| 0.7 | 0.75 | 100 | 0.5227 |

sub=0.8, col=1.0, mcw=50 confirme optimal a depth=6.

---

## 3.9 Synthese finale — Params Training Final (545 configs)

| Param | XGBoost | LightGBM | CatBoost |
|-------|---------|----------|----------|
| **init_alpha** | **0.5** | **0.1** | **0.3** |
| **depth** | **6** | 8 (cap) | **5** |
| eta/lr | 0.05 | 0.05 | 0.05 |
| lambda/l2 | 4.0 | 4.0 | 8.0 |
| sub/bagging | 0.8 | 0.8 | — |
| col/ff/rsm | 1.0 | 1.0 | 0.7 |
| mcw/mcs/mdil | 50 | 275 | 200 |
| leaves | — | 15 | — |
| **Best logloss (62K)** | **0.5178** | **0.5180** | **0.5245** |
| **V8 baseline (62K)** | 0.5238 | 0.5735 | 0.5284 |
| **Gain vs V8** | -0.0060 | -0.0555 | -0.0039 |

### Alpha × architecture (finding original ALICE)

| Architecture | Alpha optimal | Sensitivity | Mecanisme |
|-------------|--------------|-------------|-----------|
| Leaf-wise (LGB) | **0.1** | 0.054 | GOSS drop petits gradients |
| Oblivious (CB) | **0.3** | 0.0024 | Structure fixe, moderement sensible |
| Depth-wise (XGB) | **0.5** | 0.001 | Arbre complet, compense via iterations |

---

## 3.10 Audit pre-Training Final — transfert 62K → 1.1M (2026-04-12)

### Params verifies (evidence empirique ALICE, 545 configs sur 62K)

Tous les params du §3.9 ont ete testes sur saison=2022 (~62K train, ~71K valid).
Le transfert vers le full dataset (1.1M) est justifie par AUTOMATA (NeurIPS 2022) :
les HP DIRECTIONS sont stables entre subset et full data (speedup 3-30×).

### Params potentiellement sensibles au scaling (surveiller)

**min_child_weight / min_child_samples :**

| Modele | Valeur | % de 62K | % de 1.1M | Doc fabricant |
|--------|--------|----------|-----------|--------------|
| XGB mcw | 50 | 0.08% | 0.005% | "larger = more conservative" |
| LGB mcs | 275 | 0.44% | 0.024% | "hundreds or thousands for large dataset" |
| CB mdil | 200 | — | — | zero effect oblivious trees |

Decision : GARDER les valeurs 62K. AUTOMATA preserve le ranking. ES=200 compense.
Si Training Final sous-performe → mcs/mcw est le premier param a re-examiner.

Source : [XGBoost tuning](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html),
[LightGBM tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)

**num_leaves LGB = 15 :**

Sur 62K : 15 feuilles = 4100 samples/feuille. Sur 1.1M : 15 feuilles = 76000 samples/feuille.
Grid v2 : 15 >> 135 >> 255 sur 62K. Decision : GARDER 15. Si sous-performance → tester 31.

Source : [LightGBM Parameters-Tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)

**early_stopping_rounds = 200 :**

Regle industrie : ~10% du n_estimators. Avec eta=0.05, convergence attendue 1000-3000 iters.
ES=200 = 7-20% → dans la plage. Decision : GARDER.

Source : [Google ML — Overfitting GBDT](https://developers.google.com/machine-learning/decision-forests/overfitting-gbdt)

### Params explores mais gardes aux defauts

**max_bin / border_count :**

| Modele | Default | Decision | Raison |
|--------|---------|----------|--------|
| XGB | 256 (hist) | GARDER | Standard pour 1.1M |
| LGB | 255 | GARDER | 63 = speed, 255 = accuracy. Accuracy prioritaire |
| CB | 254 | GARDER | 1024 pour "golden features" (doc CB), mais pas de donnees |

Source : [LightGBM max_bin issue #6016](https://github.com/lightgbm-org/LightGBM/issues/6016),
[CatBoost parameter-tuning](https://catboost.ai/docs/en/concepts/parameter-tuning)

**XGB reg_alpha (L1) = 0.01 :**

V8 = 0.5, V9 = 0.01. Jamais explore. Probst 2019 : quasi-nul pour XGB.
Sur 201 features dont certaines sparse (H2H 0.8%, color perf 7.4%), L1 pourrait aider.
Decision : GARDER 0.01. Second-order effect.

### Params NON explores signales par la litterature

**CatBoost grow_policy : SymmetricTree vs Depthwise vs Lossguide**

On utilise SymmetricTree (oblivious, default). Depthwise et Lossguide existent mais
n'ont JAMAIS ete testes. Lossguide = equivalent leaf-wise de LGB.

Doc CatBoost : "symmetric trees give better quality in many cases" mais pas toujours.
[CatBoost grow_policy issue #1348](https://github.com/catboost/catboost/issues/1348)

Decision : NE PAS CHANGER. Changer grow_policy invalide TOUTE la recherche CB
(depth, l2, rsm optimises pour oblivious). C'est un pivot architectural, pas un tuning.

**Calibration : Dirichlet calibration (Kull et al. 2019, NeurIPS)**

Temperature scaling = 1 parametre T global. Dirichlet calibration = matrice K×K + softmax,
capture les interactions inter-classes. Pour notre 3-class W/D/L ou le draw est critique
(45% de la variance E[score]), les interactions inter-classes importent.

Source : [Beyond temperature scaling (arXiv:1910.12656)](https://ar5iv.labs.arxiv.org/html/1910.12656)

Decision : AJOUTER Dirichlet calibration en parallele de temperature scaling dans le
Training Final. Comparer sur valid set. Si Dirichlet ameliore ECE draw → l'utiliser.
Implementation : `netcal.scaling.DirichletCalibration` ou log-transform + LogisticRegression.

### Conclusion

### Audit contexte production (2026-04-12)

**Ce que le CE (`composer.py`) consomme reellement :**
- `win_probability`, `draw_probability`, `loss_probability` per (joueur × echiquier × scenario)
- `expected_score = win_prob + draw_prob * 0.5` (ligne 116)
- `confidence` (champ API CDC §3.3, actuellement en dur a 0.5)
- 20 scenarios ALI × K boards × N candidats = ~1600 predictions par ronde

**Pourquoi le draw est le signal CRITIQUE (pas le logloss global) :**
- Draw = 13.7% des parties MAIS 45% de la variance E[score]
- Mode risk-adjusted : Var[score] = f(P(draw)) — distingue compo "sure" vs "risquee"
- Mode tactique : P(match_win) = convolution K boards, sensible a P(draw)
- draw_bias +2% × 8 boards = +0.08 points/match systematique → fausse les rankings

**CE actuel = formules Elo, PAS le modele ML (composer.py ligne 64) :**
```python
DRAW_PROBABILITY_FACTOR = 0.35  # draw FIXE pour tout le monde
```
Notre ML predit draw de 5% a 46% selon contexte. Le cablage CE→ML est le deliverable Phase 2.

**Pawnalyze (5M parties OTB) :** LightGBM, seulement white_elo + black_elo (2 features).
Notre approche (201 features + residual learning) est PLUS riche.
Source : https://blog.pawnalyze.com/tournament/2022/02/27/Elo-Rating-Accuracy-Is-Machine-Learning-Better.html

**ChessEngineLab :** draw rate = f(avg_elo) × relative_draw_rate(diff_elo). Lineaire.
Notre draw_rate_prior (lookup elo_band × diff_band) est l'equivalent.
Source : https://chessenginelab.substack.com/p/game-outcomes

**Consequences pour les params Training Final :**
1. **Dirichlet calibration = OBLIGATOIRE** (pas optionnel). Temperature scaling = 1 param global,
   Dirichlet = matrice K×K, capture interactions W/D/L. Draw_bias impacte directement E[score].
   Source : Kull et al. 2019 NeurIPS (arXiv:1910.12656)
2. **ECE draw + draw_bias = metriques de selection** (pas juste logloss). Le best logloss n'est
   pas forcement le best pour le CE si la calibration draw se degrade.
3. **CatBoost posterior_sampling** = utile pour le champ `confidence` de l'API CDC.
   Pas de la "recherche" — c'est un besoin produit documente.
4. **Chaque grid Tier 2 doit mesurer ECE draw et draw_bias en plus du logloss.**

### Plan d'action Training Final

**Tier 0 — OBLIGATOIRE dans le kernel Training Final :**
- Dirichlet calibration (implementation + comparaison avec temperature scaling)
- Logger ECE draw + draw_bias par modele (pas juste logloss)
- CatBoost posterior_sampling en parallele du modele standard

**Tier 1 — Flags simples, zero cout :**
- CB score_function=NewtonL2 (second derivees, potentiel amelioration calibration)
- CB border_count=1024 (meilleure resolution golden features)
- CB leaf_estimation_iterations=3 (meilleures leaf values)
- LGB min_gain_to_split=0.01 (previent splits noise avec alpha=0.1)
- LGB lambda_l1=0.1 (L1 sur 201 features dont certaines sparse)

**Tier 2 — Grids rapides (avant Training Final, sur 62K) :**
- LGB num_leaves {15, 31, 63} — 76K samples/leaf sur 1.1M = potentiellement sous-utilise
- XGB colsample_bynode {0.7, 0.8, 1.0} — regulariseur per-split jamais explore
- XGB gamma {0, 0.1, 1.0} — min loss reduction, previent noise splits
- XGB max_delta_step {0, 1, 5} — pour class imbalance (draw 13.7%)
- **CHAQUE combo evalue sur logloss + ECE draw + draw_bias**

**Tier 3 — V10 (changements architecturaux) :**
- XGB grow_policy=lossguide, LGB boosting=dart, CB boosting=Ordered, CB langevin

---

## 4. Features V8 (201 colonnes apres encodage, 219 avant)

Source : `reports/v8_xgboost_v5_resume/metadata.json` (feature_count=219)
Source : `reports/v8_xgboost_v5_resume/train_feature_distributions.csv` (201 lignes)

### 4.1 Contexte match (8 cols)
`saison`, `division`, `ligue_code`, `niveau`, `type_competition`, `ronde`, `jour_semaine`, `echiquier`

### 4.2 Joueur identite (6 cols)
`blanc_titre`, `blanc_elo`, `noir_titre`, `noir_elo`, `diff_elo`
+ `est_domicile_blanc`

### 4.3 Club fiabilite (6 cols)
`taux_forfait_dom`, `taux_non_joue_dom`, `fiabilite_score_dom`
`taux_forfait_ext`, `taux_non_joue_ext`, `fiabilite_score_ext`

### 4.4 Joueur presence (8 cols)
`taux_presence_blanc`, `joueur_fantome_blanc`, `taux_presence_noir`, `joueur_fantome_noir`
`taux_presence_saison_blanc`, `derniere_presence_blanc`, `regularite_blanc`
`taux_presence_saison_noir`, `derniere_presence_noir`, `regularite_noir`
+ `rondes_manquees_consecutives_blanc`, `taux_presence_global_blanc`
+ `rondes_manquees_consecutives_noir`, `taux_presence_global_noir`

### 4.5 Forme recente (10 cols, rolling 5 matchs meme niveau)
`win_rate_recent_blanc`, `draw_rate_recent_blanc`, `expected_score_recent_blanc`, `win_trend_blanc`, `draw_trend_blanc`
`win_rate_recent_noir`, `draw_rate_recent_noir`, `expected_score_recent_noir`, `win_trend_noir`, `draw_trend_noir`

### 4.6 Position echiquier (6+6 cols)
`echiquier_moyen_blanc`, `echiquier_std_blanc`, `echiquier_moyen_noir`, `echiquier_std_noir`
`role_type_blanc`, `echiquier_prefere_blanc`, `flexibilite_echiquier_blanc`
`role_type_noir`, `echiquier_prefere_noir`, `flexibilite_echiquier_noir`

### 4.7 Performance couleur (8+8 cols, rolling 3 saisons)
`win_rate_white_blanc`, `draw_rate_white_blanc`, `win_rate_black_blanc`, `draw_rate_black_blanc`
`win_adv_white_blanc`, `draw_adv_white_blanc`, `couleur_preferee_blanc`, `data_quality_blanc`
(idem _noir)

### 4.8 Multi-equipe FFE (4+4 cols)
`ffe_nb_equipes_blanc`, `ffe_niveau_max_blanc`, `ffe_niveau_min_blanc`, `ffe_multi_equipe_blanc`
(idem _noir)

### 4.9 Classement equipe (7+7 cols)
`zone_enjeu_dom`, `niveau_hierarchique_dom`, `position_dom`, `ecart_premier_dom`, `ecart_dernier_dom`, `points_cumules_dom`, `nb_equipes_dom`
(idem _ext)

### 4.10 Composition equipe (7+7 cols)
`nb_joueurs_utilises_dom`, `rotation_effectif_dom`, `noyau_stable_dom`, `profondeur_effectif_dom`, `renforce_fin_saison_dom`, `win_rate_home_dom`, `draw_rate_home_dom`, `club_utilise_marge_100_dom`
(idem _ext, sauf draw/win_rate_home_ext)

### 4.11 Noyau (2+2 cols)
`est_dans_noyau_blanc`, `pct_noyau_equipe_dom`, `est_dans_noyau_noir`, `pct_noyau_equipe_ext`

### 4.12 Decalage position (3+3 cols)
`decalage_position_blanc`, `joueur_decale_haut_blanc`, `joueur_decale_bas_blanc`
(idem _noir)

### 4.13 Draw priors (7 cols)
`draw_rate_blanc`, `draw_rate_noir`, `draw_rate_equipe_dom`, `draw_rate_equipe_ext`
`avg_elo`, `elo_proximity`, `draw_rate_prior`

### 4.14 Vases communiquants (5+5 cols)
`team_rank_in_club_dom`, `club_nb_teams_dom`, `reinforcement_rate_dom`, `stabilite_effectif_dom`, `elo_moyen_evolution_dom`
(idem _ext)

### 4.15 Promotion/relegation joueur (3+3 cols)
`joueur_promu_blanc`, `joueur_relegue_blanc`, `player_team_elo_gap_blanc`
(idem _noir)

### 4.16 Trajectoire Elo (2+2 cols)
`elo_trajectory_blanc`, `momentum_blanc`, `elo_trajectory_noir`, `momentum_noir`

### 4.17 Pression/clutch (7+7 cols)
`win_rate_normal_blanc`, `draw_rate_normal_blanc`, `win_rate_pression_blanc`, `draw_rate_pression_blanc`, `clutch_win_blanc`, `clutch_draw_blanc`, `pressure_type_blanc`
(idem _noir)

### 4.18 H2H (4 cols)
`h2h_win_rate`, `h2h_draw_rate`, `h2h_nb_confrontations`, `h2h_exists`

### 4.19 Titre/categorie (7 cols)
`blanc_titre_num`, `noir_titre_num`, `diff_titre`
`elo_type_blanc`, `categorie_blanc`, `k_coefficient_blanc`
`elo_type_noir`, `categorie_noir`, `k_coefficient_noir`

### 4.20 Temporel (4 cols)
`phase_saison`, `ronde_normalisee`, `adversaire_niveau_dom`, `adversaire_niveau_ext`, `match_important`

### 4.21 Differentiels joueur (8 cols, P0)
`diff_form`, `diff_win_rate_recent`, `diff_draw_rate`, `diff_draw_rate_recent`
`diff_win_rate_normal`, `diff_clutch`, `diff_momentum`, `diff_derniere_presence`

### 4.22 Differentiels equipe (6 cols, P0)
`diff_position`, `diff_points_cumules`, `diff_profondeur`, `diff_stabilite`
`diff_win_rate_home`, `diff_draw_rate_home`

### 4.23 Interactions board/match (8 cols, P1)
`zone_danger_dom`, `zone_montee_dom`, `form_in_danger`, `color_match`
`decalage_important`, `marge100_decale`, `flex_decale`, `promu_vs_strong`

### 4.24 Incertitude (2 cols, P1)
`elo_uncertainty`, `k_asymmetry`

### 4.25 Top-10 features par importance (XGBoost V8 gain)

| # | Feature | Gain |
|---|---------|------|
| 1 | est_domicile_blanc | 138.3 |
| 2 | diff_form | 82.3 |
| 3 | draw_rate_home_dom | 77.5 |
| 4 | h2h_draw_rate | 47.8 |
| 5 | draw_rate_noir | 43.0 |
| 6 | draw_rate_blanc | 39.4 |
| 7 | win_rate_home_dom | 39.0 |
| 8 | diff_points_cumules | 39.2 |
| 9 | draw_rate_recent_noir | 32.0 |
| 10 | h2h_win_rate | 34.1 |

---

## 5. Data distribution

Source : `reports/v8_xgboost_v5_resume/metadata.json`

| | Train | Valid | Test |
|---|-------|-------|------|
| Rows | 1,139,799 | 70,647 | 231,532 |
| Saisons | 2002-2022 | 2023 | 2024-2026 |
| Loss (%) | 41.2 | — | — |
| Draw (%) | 13.7 | — | — |
| Win (%) | 40.8 | — | — |
| SHA-256 | 87d306ee | 3f6cf7c2 | 9ee5bd72 |

---

## 6. Fichiers source des donnees

| Donnee | Fichier local |
|--------|--------------|
| Grid XGB v4 | `grid_results/xgboost_v4/grid_search_xgboost.csv` (82 lignes) |
| Grid LGB v2 | `grid_results/lightgbm_v2/grid_search_lightgbm.csv` (82 lignes) |
| Grid CB v1 | `grid_results/catboost/grid_search_catboost.csv` (32 lignes) |
| Optuna XGB v8 | `optuna_studies/xgboost_v8/trial_history_xgboost.csv` (122 trials) |
| Optuna LGB v7 | `optuna_studies/lightgbm_v7/trial_history_lightgbm.csv` (100 trials) |
| Optuna CB v3 | `optuna_studies/catboost_v3/trial_history_catboost.csv` (2 trials) |
| V8 metadata | `reports/v8_xgboost_v5_resume/metadata.json` |
| V8 features | `reports/v8_xgboost_v5_resume/train_feature_distributions.csv` (201 features) |
| V8 importance | `reports/v8_xgboost_v5_resume/xgboost_treeshap_importance.csv` |
| Config HP | `config/hyperparameters.yaml` |
| Config modeles | `config/MODEL_SPECS.md` |
| Grid CB v1 | `grid_results/catboost/grid_search_catboost.csv` (32 lignes, sans l2_leaf_reg) |
| Grid CB v2 | `grid_results/catboost_v2/grid_search_catboost.csv` (70 lignes, avec l2_leaf_reg) |
| Optuna CB v3 | `optuna_studies/catboost_v3/trial_history_catboost.csv` (2 trials, full data) |
| Optuna CB v4 | `optuna_studies/catboost_v4/trial_history_catboost.csv` (56 trials, 62K) |
| **Gaps XGB** | `grid_results/gaps_xgboost/grid_gaps_xgboost.csv` (10 combos, alpha×depth) |
| **Gaps LGB** | `grid_results/gaps_lightgbm/grid_gaps_lightgbm.csv` (17 combos, alpha×lambda) |
| **Gaps CB** | `grid_results/gaps_catboost/grid_gaps_catboost.csv` (17 combos, alpha×depth) |
| **Gaps R2 LGB** | `grid_results/gaps_round2/grid_gaps2_lightgbm.csv` (3 combos, alpha floor) |
| **Gaps R2 XGB** | `grid_results/gaps_round2/grid_gaps2_xgboost.csv` (8 combos, depth=6 refine) |
