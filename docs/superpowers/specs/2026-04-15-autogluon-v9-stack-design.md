# AutoGluon V9 + Stack Leger — Design Spec

**Date:** 2026-04-15
**Status:** APPROVED
**Scope:** AG benchmark kernel + OOF stacking + batch architecture prod
**Prerequisite:** V9 Training Final v4 COMPLETE (3 modeles T1-T12 ALL PASS)

---

## 1. Context

V9 Training Final v4 produit 3 modeles converges (LGB 0.5619, XGB 0.5622, CB 0.5708).
Tous passent T1-T12. La question est : peut-on faire mieux avec AutoGluon ou un stack
des 3 modeles avant de passer a Phase 2 (API + CE) ?

### Findings cles

1. **AG extreme preset inutile a 1.1M rows** — retombe sur best_quality (AG 1.4 release notes)
2. **AG ne supporte PAS init_scores** — pas de residual learning (confirme doc officielle AG 1.5)
3. **AG stacked ensemble = ~100s pour 16K rows** — 2000x trop lent pour real-time
4. **Architecture batch resout la latence** — ML batch hebdo + CE on-demand <2s
5. **Calibration > accuracy** pour le CE (Wheatcroft 2021, arxiv:2303.06021)
6. **Stack_MLP_cal V8** : E[score] MAE -2%, ECE draw -20.7%, draw_bias -23.8%

### Sources

- AG 1.5 docs : https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html
- AG 1.4 extreme : https://auto.gluon.ai/1.4.0/whats_new/v1.4.0.html
- AG predict_proba_oof : https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.predict_proba_oof.html
- AG deployment : https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-deployment.html
- Wheatcroft 2021 : https://journals.sagepub.com/doi/full/10.3233/JSA-200462
- Calibration vs accuracy : https://arxiv.org/abs/2303.06021
- Constantinou 2012 (RPS) : http://constantinou.info/downloads/papers/solvingtheproblem.pdf

---

## 2. Architecture Production (decision architecturale)

### Batch ML + CE on-demand

```
BATCH HEBDOMADAIRE (apres resultats ronde precedente, ~nuit):
  1. Update feature store (stats joueurs, equipes, classements)     ~5 min
  2. ALI: 20 scenarios adversaire par match                         ~secondes
  3. ML: predict ALL (joueur x board x adversaire_predit)           ~10s-100s
  4. Cache probas dans MongoDB (par ronde, par club)                ~secondes

A LA DEMANDE (capitaine ouvre l'app, <2s):
  CE solver (OR-Tools):
    Input : probas cachees + contraintes FFE + joueurs dispos + mode strategie
    Output : composition optimale N equipes
    Temps : <2 secondes
```

### Pourquoi batch

Les predictions ML sont INDEPENDANTES de l'allocation joueurs. Le CE optimise
l'allocation avec des probas PRE-CALCULEES. Quand le capitaine change une
contrainte (joueur indisponible, mode tactique), seul le CE re-tourne — pas le ML.

### Impact sur le choix de champion

En batch, la latence ML n'est plus un critere. Meme AG stacked ensemble (~100s)
est acceptable pour un batch nocturne. Le critere de selection = CALIBRATION
(ECE draw + draw_bias), car P(draw) = 45% de la variance E[score] pour le CE.

---

## 3. Deux kernels a lancer

### 3.1 Kernel A — AutoGluon benchmark

**Objectif** : AG best_quality bat-il V9 0.5619 ? AG best_single vs AG stacked ?

**Configuration (CORRIGEE v4, verifiee WebFetch AG 1.5 docs) :**
| Param | Valeur | Justification |
|-------|--------|---------------|
| preset | `best_quality` | `extreme` inutile >30K rows |
| eval_metric | `log_loss` | Multiclass 3-class (W/D/L). AG override default accuracy. |
| problem_type | `multiclass` | 3 classes : loss=0, draw=1, win=2 |
| calibrate | `True` | Temperature scaling integre (AG 1.2+) |
| num_bag_folds | 5 | OOF pour stacking interne |
| num_stack_levels | 1 | V8 postmortem : L2/L3 overfit |
| dynamic_stacking | `False` | Force stack levels (best_quality override sinon) |
| use_bag_holdout | `True` | REQUIS avec tuning_data + num_bag_folds (crash v1 sans) |
| time_limit | **28800** | **8h (1h marge sur 9h GPU session)** |
| num_gpus | **1** | **T4 GPU pour NN_TORCH/FASTAI (CPU = OOM skip)** |
| enable_gpu | **true** | **Metadata kernel** |
| ag_args_fit | `{"max_memory_usage_ratio": 1.5}` | SANS prefixe "ag." (WebFetch doc AG 1.5) |

**Features** : 201 features V9 (FE kernel alice-fe-v8) + 3 features Elo proba
(P_elo_win, P_elo_draw, P_elo_loss) = 204 features.

Rationale Elo probas : sans init_scores, les features Elo brutes (diff_elo, avg_elo)
ne donnent pas la transformation Elo->proba. Ajouter les probas Elo donne a AG
l'information equivalente a notre baseline Elo, sous forme de features.

**Input** : memes parquets que V9 (train/valid/test depuis alice-fe-v8)
**Output** :
- predictions_test.parquet (231532, 3) calibrees
- predictions_valid.parquet (70647, 3) calibrees
- OOF predictions via predict_proba_oof()
- leaderboard.csv (tous les modeles AG avec metriques)
- metadata.json (model card, quality gates T1-T12)
- feature_importance.csv (AG built-in)

**Quality gates** : T1-T12 identiques a V9 (via kaggle_quality_gates.py)

**Entry point** : `scripts/cloud/train_autogluon_v9.py` (nouveau)
**Metadata** : `kernel-metadata-autogluon-v9.json`
**Slug** : `pguillemin/alice-autogluon-v9`
**GPU T4** : `enable_gpu: true`, `enable_internet: true` (pip install autogluon)
**Accelerator** : NvidiaTeslaT4 (sm_75, compatible cu128)

### 3.2 Kernel B — Stack leger (OOF 3 modeles V9)

**Objectif** : 5-fold OOF sur les 3 V9 modeles converges → meta-learner local.

**ATTENTION** : ce kernel est DIFFERENT de Steps 7-9 du plan V9 original.
Le plan V9 prevoyait 3 kernels separes (1 par modele). Ici on fait 1 kernel
qui entraine les 3 modeles en 5-fold sequentiel.

**Configuration par modele** : identique V9 Training Final (hyperparameters.yaml)
- XGB : alpha=0.5, depth=6, eta=0.05, Tier 2 draw (bynode+gamma+mds)
- LGB : alpha=0.1, leaves=15, lr=0.05, mcs=275
- CB : alpha=0.3, depth=5, rsm=0.7, l2=8

**Process** :
```
Pour chaque fold k (1-5):
  train_k = train sans fold k
  valid_k = fold k
  Pour chaque modele (XGB, LGB, CB):
    1. Compute init_scores (Elo baseline × alpha_per_model)
    2. Train avec early_stopping=200
    3. Calibrate (temperature scaling sur valid_k)
    4. Predict calibre sur valid_k → OOF predictions fold k
    5. Predict calibre sur test → accumulate test predictions
  Checkpoint apres chaque modele
Test predictions = moyenne des 5 folds
```

**Output** :
- oof_predictions.parquet : (1210446, 9) — 3 modeles x 3 classes, + y_true
- test_predictions.parquet : (231532, 9) — moyennees sur 5 folds
- metadata.json (model card, quality gates)

**Time budget** :
| Composant | Estimation | Total |
|-----------|-----------|-------|
| XGB : 5 folds x ~1h | ~5h | 5h |
| LGB : 5 folds x ~50min | ~4h | 9h |
| CB : 5 folds x ~7h | **~35h** | **DEPASSE 12h** |

**PROBLEME : CatBoost 5-fold = 35h >> 12h Kaggle.** Options :
- A) 3 kernels separes (1 par modele) — revient au plan V9 original
- B) 2 kernels : XGB+LGB ensemble (9h), CB seul (reste pas assez = TIMEOUT)
- C) Exclure CB du stack (XGB+LGB seulement, 6 features meta au lieu de 9)
- D) Reduire CB a 1 fold (pas OOF, juste test predictions)

**Recommandation : C** — exclure CB du stack. CB est 0.0089 derriere LGB en logloss
et structurellement limite (oblivious trees). XGB+LGB suffisent pour le meta-learner
(6 features, V8 Stack_MLP_cal utilisait deja principalement XGB+LGB).
1 kernel, ~9h, rentre dans le budget Kaggle.

### 3.3 Meta-learner (local, pas Kaggle)

Entraine LOCALEMENT sur les OOF predictions du kernel B :
- Input : X_meta = (n, 6) = XGB(W,D,L) + LGB(W,D,L), y = target
- Meta-learner : MLP(hidden=16, max_iter=500, early_stopping=True)
- Post-hoc : temperature scaling ou isotonic
- Quality gates T1-T12 sur la sortie finale

---

## 4. Criteres de selection champion

| Critere | Poids | Justification |
|---------|-------|---------------|
| ECE draw | **#1** | P(draw) = 45% variance E[score] pour le CE |
| draw_bias | **#2** | Biais systematique sur draw = compositions biaisees |
| test log_loss | #3 | Qualite globale des probas |
| test RPS | #4 | Ordinal-aware (Constantinou 2012) |
| E[score] MAE | #5 | Metrique directe du CE |

**Candidats :**
1. LGB V9 single (0.5619)
2. XGB V9 single (0.5622)
3. Stack XGB+LGB (kernel B meta-learner)
4. AG best_single (kernel A)
5. AG stacked ensemble (kernel A) — deployable en batch

**Seuil** : stack ou AG doit battre best single sur ECE draw ET draw_bias.
Si non → champion = LGB V9 (simplest deployable).

---

## 5. ISO Compliance

| Norme | Exigence | Implementation |
|-------|----------|---------------|
| ISO 42001 | Model card, tracabilite | metadata.json par kernel |
| ISO 5259 | Data lineage | Memes parquets V9 (hash trace) |
| ISO 25059 | Quality gates | T1-T12 via kaggle_quality_gates.py |
| ISO 24029 | Robustesse | Apres champion selection |
| ISO 24027 | Fairness | Apres champion selection |
| ISO 42005 | Impact assessment | Champion guide vraies decisions |
| ISO 5055 | Code quality | <300 lignes/fichier |

---

## 6. Multi-equipe et batch (validation architecture)

Le training V9 EST compatible multi-equipe :
- ML predit per-board (joueur x adversaire x contexte)
- Features modulaires : joueur (portable) + equipe (team-specific) + board
- `player_team_elo_gap` recalcule dynamiquement pour toute affectation hypothetique
- Multi-equipe = contrainte CE, pas ML (MODEL_SPECS §1)

Architecture batch :
- ML predictions cachees dans MongoDB (par ronde, par club)
- CE solver on-demand (<2s) avec contraintes du capitaine
- Changement joueur dispo / mode strategie → re-run CE seul, pas ML
- AG stacked deployable car batch nocturne (pas de contrainte latence)

---

## 7. Fichiers a creer/modifier

| Fichier | Action | Contenu |
|---------|--------|---------|
| `scripts/cloud/train_autogluon_v9.py` | Creer | Kernel AG benchmark |
| `scripts/cloud/train_oof_stack.py` | Creer | Kernel OOF XGB+LGB 5-fold |
| `scripts/cloud/train_final_autogluon.py` | Creer | Entry point AG |
| `scripts/cloud/train_final_oof.py` | Creer | Entry point OOF |
| `scripts/cloud/kernel-metadata-autogluon-v9.json` | Creer | Metadata AG |
| `scripts/cloud/kernel-metadata-oof-stack.json` | Creer | Metadata OOF |
| `scripts/cloud/upload_all_data.py` | Modifier | Whitelist nouveaux cloud modules |
| `config/hyperparameters.yaml` | Pas de modif | Memes params V9 |

---

## 8. Corrections post-implementation (2026-04-15, session runtime)

### 8.1 Bugs trouves et corriges

| # | Bug | Impact | Fix | Commit |
|---|-----|--------|-----|--------|
| 1 | `use_bag_holdout` manquant | AG crash v1 | Ajout `use_bag_holdout=True` | c7b3f60 |
| 2 | `ag_args` au lieu de `ag_args_fit` | 110 warnings, memoire non protegee | `ag_args_fit={"max_memory_usage_ratio": 1.5}` | c7c4bdd |
| 3 | Prefixe `ag.` sur la cle | Cle ignoree silencieusement | `max_memory_usage_ratio` sans prefixe (doc AG 1.5) | c7c4bdd |
| 4 | `num_gpus=0` (CPU only) | NN_TORCH/FASTAI skipped (OOM 89%) | `num_gpus=1` + `enable_gpu: true` (T4) | 39dfe1c |
| 5 | `time_limit=36000` (CPU 12h) | Depasse budget GPU 9h | `time_limit=28800` (8h + 1h marge) | 39dfe1c |
| 6 | Elo baseline fixe +20 | Diverge de baselines.py (+8.5 a +32.4) | `compute_elo_baseline()` dynamique | pending |
| 7 | `_save_predictions` positional `.iloc` | Class swap silencieux | Label-based `[[0,1,2]].values` | ae233bc |
| 8 | `leaderboard()` recoit `test_raw` | Schema mismatch | `test_ag` prepare avec target | ae233bc |
| 9 | Elo features `.values` sans `.reindex()` | Row misalignment forfeits | `.reindex(X.index).values` | ae233bc |
| 10 | draw_lookup leakage (OOF kernel) | Lookup sur combined au lieu de train | `build_draw_rate_lookup(train_raw)` | ae233bc |
| 11 | `test_preds_acc /= N_FOLDS` fixe | Probas invalides si fold fail | Track `test_fold_counts` per model | ae233bc |

### 8.2 Configuration finale AG kernel v4 (verifiee WebFetch doc AG 1.5)

| Param | Valeur | Source doc AG 1.5 |
|-------|--------|-------------------|
| `train_data` | DataFrame avec "target" | pd.DataFrame |
| `tuning_data` | valid_ag | Holdout pour scoring |
| `use_bag_holdout` | `True` | Requis tuning_data + bag |
| `presets` | `"best_quality"` | 110 model configs zeroshot |
| `time_limit` | `28800` | 8h (session GPU 9h max) |
| `num_bag_folds` | `5` | int >= 2 |
| `num_stack_levels` | `1` | V8 postmortem L2 overfit |
| `dynamic_stacking` | `False` | Force stack levels |
| `calibrate` | `True` | Temperature scaling post-hoc |
| `num_gpus` | `1` | T4 GPU pour NN_TORCH/FASTAI |
| `ag_args_fit` | `{"max_memory_usage_ratio": 1.5}` | SANS prefixe "ag." |
| `verbosity` | `2` | Standard logging |

### 8.3 State-of-the-art findings (WebSearch 2026-04-15)

**Composition predictive echecs :**
- AUCUN systeme publie ne fait ce qu'ALICE fait. Innovation pure.
- Litterature sports ML (Hubacek 2019, arXiv:2309.14807) : features relatives > absolues.
- Board allocation strategies (FIDE, Chess.com) : heuristiques manuelles seulement.

**AutoGluon benchmark :**
- Nature Scientific Reports 2025 : AG = meilleur AutoML overall (16 outils benchmarkes).
- AG paper (Erickson 2020) : multi-layer stacking + bagging = reduction variance.

**Calibration multiclass GBMs tabulaires (SOTA 2025) :**
- Temperature scaling (Guo 2017) : standard, competitif pour GBMs.
- Dirichlet calibration (Kull 2019 NeurIPS) : meilleur multiclass general.
- Isotonic per-class (Niculescu-Mizil 2005) : reference sklearn.
- GETS ICLR 2025 : **GNN only**, PAS applicable aux GBMs tabulaires.
- Top-versus-All (Le Coz 2024) : reformule multiclass en binaire — non teste.
- **Notre implementation (Temp + Iso + Dirichlet) = CONFORME SOTA 2025.**

**Elo + residual learning :**
- Standard confirme : NBA, EPL, UCL (2024-2025) utilisent Elo + ML corrections.
- Per-model alpha (ADR-008, 590 configs) : innovation ALICE.

**OR-Tools allocation :**
- Standard industrie constraint programming (Google).
- ML + combinatorial optimization : domaine actif NeurIPS 2024.

### 8.4 Elo proba features — implementation correcte

`_compute_elo_proba_features` utilise `compute_elo_baseline()` de `scripts/baselines.py` :
- Dynamic white advantage : +8.5 (Elo<1200) a +32.4 (Elo>2400) — donnees FFE verifiees.
- `draw_rate_lookup` : 45 cellules (elo_band x diff_band), construit sur train_raw UNIQUEMENT.
- Meme formule que les init_scores V9, sous forme de features au lieu de base_margin.

### 8.5 GPU T4 decision

NN_TORCH/FASTAI necessitent GPU pour 1.1M rows (OOM sur CPU 31GB).
T4 (sm_75) compatible PyTorch 2.9+cu128 (image Kaggle v168).
Session GPU = 9h max, quota 30h/semaine. Budget kernel = 8h + 1h marge.
Tree models (GBM, CAT, XGB, RF, XT) auto-detectent CPU meme avec num_gpus=1.
