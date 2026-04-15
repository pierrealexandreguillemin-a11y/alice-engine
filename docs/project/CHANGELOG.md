# Changelog - ALICE Engine

> **Format**: [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/)
> **Versioning**: [Semantic Versioning](https://semver.org/lang/fr/)

---

## [Unreleased] — Champion selection + OOF stacking

### Prochaine etape
- Selection champion (XGB vs LGB)
- OOF stacking (Steps 7-9 du plan V9)

---

## [0.9.1] - 2026-04-15 — V9 Training Final v4 COMPLETE (T1-T12)

### Added
- **`kaggle_quality_gates.py`** : module T1-T12 complet avec audit logging (237 lignes)
- **T7/T8** : assertions NaN/Inf + sum-to-1 dans `evaluate_on_test` (crash immediat)
- **T9** : check >5 features importance>0 (np.ndarray)
- **T10** : train-test gap via 50K subsample + calibration (GoogleML #37)
- **T11/T12** : ECE par classe + dual RPS/logloss logges pour audit trail

### Fixed
- **XGBoost `EarlyStopping(save_best=True)`** callback — remplace `early_stopping_rounds` parametre (retourne LAST pas BEST)
- Verbose=False dans boucle selection calibrateur, verbose=True pour gate authoritaire

### Resultats V9 Training Final (3 modeles, T1-T12 ALL PASS)

| Metrique | XGBoost | LightGBM | CatBoost | V8 best |
|----------|---------|----------|----------|---------|
| test log_loss | 0.5622 | **0.5619** | 0.5708 | 0.5660 |
| test RPS | 0.0887 | 0.0887 | 0.0895 | — |
| ECE draw | 0.0129 | 0.0145 | 0.0123 | 0.0156 |
| draw_bias | 0.0109 | 0.0136 | 0.0095 | 0.0146 |
| T10 gap | 0.0478 | 0.0148 | 0.0312 | — |
| T9 features | 197 | 193 | 198 | — |
| best iter | 6172 | 8623 | 14572 | 86500 |
| temps | 1h11m | 50m | 6h49m | — |

---

## [0.9.0] - 2026-04-13 — V9 Training Final v1-v3

### Added
- **Per-model alpha** (ADR-008) : XGB=0.5, LGB=0.1, CB=0.3 (590 configs)
- **Dirichlet calibration** (Kull 2019 NeurIPS) : Dir-L2, 12 params, triple comparaison
- **V9 guard** : RuntimeError si code stale detecte (eta<0.01 ou rsm<0.5)
- **Dataset propagation wait** : 120s sleep apres upload Kaggle
- **`_ALICE_KEYS`** : filtrage params custom avant constructeurs ML
- 3 entry points Training Final (`train_final_{xgboost,lightgbm,catboost}.py`)
- 3 kernel-metadata Training Final

### Fixed
- SHAP TreeExplainer : `get_booster()` pour XGBWrapper (crash v1)
- `init_score_alpha` passe aux constructeurs ML (crash CatBoost v2)
- Dataset Kaggle non propage (code V8 execute au lieu de V9, v1)
- XGBoost et CatBoost en cours de training

---

## [0.8.0] - 2026-04-12 — V9 HP Search COMPLET

### Added
- **590 configs testees** sur 13 kernels Kaggle (Optuna + Grid + Gaps + Tier 2)
- `config/MODEL_SPECS.md` : source de verite per-model (architecture × alpha)
- `docs/project/V9_HP_SEARCH_RESULTS.md` : donnees completes HP search
- ADR-008 (alpha per-model), ADR-009 (recent season HP search), ADR-010 (ISO local)
- Grid v2 CatBoost : l2_leaf_reg dans search space (gain 0.038 vs Grid v1)
- Gap-filling R1+R2 : 55 combos, decouverte alpha=0.1 LGB, depth=6 XGB
- Tier 2 draw calibration : bynode+gamma+mds XGB (draw_bias -22%)
- `config/hyperparameters.yaml` V9 (version 9.0.0)

---

## [0.7.0] - 2026-04-06 — V8 Milestone COMPLET

### Added
- **3 modeles converges ALL PASS** (15 quality gates chacun)
  - XGBoost v5 resume : test 0.566, 86K rounds, champion V8
  - CatBoost v6 : test 0.575, 37K rounds
  - LightGBM v7 : test 0.572, 16K rounds
- SHAP consensus 3 modeles (TreeSHAP 20K subsample)
- Temperature + isotonic calibration
- 4-kernel architecture (ADR-003) : 1 modele par kernel
- Checkpoints per-library (CatBoost snapshot, XGB TrainingCheckPoint, LGB callback)

### Fixed
- `resultat_blanc=2.0` = victoire jeunes FFE, pas forfait (62K rows)
- CatBoost `PredictionValuesChange` artefact (166/177=0) → ShapValues
- CatBoost `rsm` GPU incompatible → CPU obligatoire
- XGBoost `xgb.train()` retourne LAST pas best → `EarlyStopping(save_best=True)`
- 61 features 100% NaN sur valid/test (split temporel bug)
- Dynamic white advantage (+8.5 a +32.4, pas +35 fixe)

---

## [0.6.0] - 2026-03-21 — V8 MultiClass + Feature Engineering

### Added
- **219 features** : joueur, club, equipe, H2H, draw, differentiels, contexte
- Target 3-class : loss=0, draw=1, win=2
- Residual learning avec Elo baseline (init_scores)
- FE kernel Kaggle (`alice-fe-v8`)
- `scripts/features/` : pipeline, differentials, draw_priors, helpers
- Quality gates F1-F12 / T1-T12

---

## [0.5.0] - 2026-03-17 — Kaggle Cloud Training

### Added
- Cloud training sur Kaggle CPU (12h/session, illimite)
- `scripts/cloud/upload_all_data.py` : upload dataset alice-code
- `scripts/cloud/train_kaggle.py` : training kernel multi-model
- Data refresh pipeline (sync FFE → parse → validate → features)

---

## [0.4.0] - 2026-03-10 — Regles FFE + ISO

### Added
- `scripts/ffe_rules_features.py` : regles FFE (brule, noyau, mute, zone_enjeu)
- 66 tests unitaires
- Pipeline ISO 14 normes (hooks pre-commit/push)
- FastAPI stubs (endpoints /health, /predict)
- ADR-001 a ADR-007

---

## [0.1.0] - 2026-01-03 — Initial Release

### Added
- Structure projet, parsing FFE, premiers modeles ML
- 1,736,490 echiquiers, 35,320 joueurs
- CatBoost AUC 0.7527 (premiere evaluation)

---

## [0.1.0] - 2026-01-03

### Added
- **Initial release**
- Structure de base du projet
- README.md avec instructions
- pyproject.toml (Ruff, MyPy, Pytest, Bandit)
- requirements.txt et requirements-dev.txt
- .gitignore adapte Python/ML
- Lien symbolique vers dataset FFE (2.4 GB)

### Documentation
- ANALYSE_INITIALE_ALICE.md
- CDC_ALICE.md (cahier des charges)
- CONTEXTE_DATASET_FFE.md
- CONTEXTE_INTEGRATION.md
- API_CONTRACT.md
- DEPLOIEMENT_RENDER.md
- PYTHON-HOOKS-SETUP.md

---

## Roadmap

### [1.0.0] - Production
- [x] V8 : 3 modeles converges, quality gates ALL PASS
- [x] V9 HP search : 590 configs, params optimaux trouves
- [~] V9 Training Final (en cours)
- [ ] V9 OOF stacking + meta-learner
- [ ] Cablage API FastAPI → services (inference, CE)
- [ ] ALI (prediction adverse, Phase 3)
- [ ] CE multi-equipe (OR-Tools, Phase 4)
- [ ] Deploiement Oracle VM (24GB ARM)

Plan detaille : `docs/superpowers/specs/2026-04-07-optuna-v9-pipeline-design.md`

---

*Derniere mise a jour: 13 Avril 2026*
