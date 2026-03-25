# Alice-Engine - Guide Claude

## BUT DU PROJET

Alice Engine est un **système de recommandation de composition d'équipe** pour les interclubs FFE.
Satellite de chess-app (SaaS clubs d'échecs). Déployé sur Render (FastAPI).

**Pipeline complet :**
1. **ALI** (Adversarial Lineup Inference) : prédire la composition probable de l'adversaire
2. **Modèle ML** : prédire **P(win), P(draw), P(loss)** pour chaque affectation joueur→échiquier (MultiClass 3-way)
3. **CE** (Composition Engine) : optimiser E[score] = P(win) + 0.5×P(draw) sous contraintes FFE (100pts, noyau, mutés)
4. **API** : POST /api/v1/predict retourne composition recommandée + alternatives + score attendu

**IMPORTANT ML** : Le modèle DOIT produire 3 probabilités (win/draw/loss), PAS binaire.
Le CE attend `(win_probability, draw_probability, loss_probability)` — voir `services/composer.py:36-39`.
Les nulles = 12.6% des parties, varient de 4.9% (Elo<1200) à 45.8% (Elo>2400). Ignorer perd ~0.5 pts/match.

**Contraintes métier** : les compositions sont soumises simultanément sur place (A02 Art. 3.6.a) — le capitaine ne connaît PAS la composition adverse.

## État actuel (mars 2026)

| Couche | Statut | Fichiers |
|--------|--------|----------|
| API FastAPI + schemas | COMPLET | `app/api/routes.py` (STUBS), `schemas.py` |
| ComposerService (CE) | FONCTIONNEL | `services/composer.py` |
| InferenceService (ALI) | PARTIEL (fallback Elo) | `services/inference.py` |
| DataLoader (MongoDB) | FONCTIONNEL | `services/data_loader.py` |
| ML Training | FONCTIONNEL | `scripts/cloud/`, `scripts/kaggle_*.py` |
| Data Refresh | FONCTIONNEL | `make refresh-data` |
| **Câblage routes→services** | **MANQUANT** | routes.py retourne des zéros |
| **Chargement modèle ML** | **MANQUANT** | modèle entraîné mais pas chargé |
| **ALI prédiction adverse** | **MANQUANT** | generate_scenarios() vide |
| **Optimisation OR-Tools** | **MANQUANT** | get_alternatives() vide |

## Données Source (FFE)

**Dataset HuggingFace** : https://huggingface.co/datasets/Pierrax/ffe-history (public)

| Fichier | Contenu | Taille |
|---------|---------|--------|
| `parsed/echiquiers.parquet` | Echiquiers parsés (saison, ronde, elo, resultat) | ~35 MB |
| `parsed/joueurs.parquet` | Joueurs FFE (NrFFE, nom, elo, categorie, club) | ~3 MB |
| `compositions/` | HTML brut interclubs 2002-2026 (85,492 fichiers) | ~2.2 GB |
| `players_v2/` | HTML joueurs par club (83,930 joueurs) | ~150 MB |
| `clubs/clubs_index.json` | Index des 992 clubs FFE | ~200 KB |

**Scraping** : repo `C:\Dev\ffe_scrapper` (GitHub: `ffe-history`)
- `scrape.py --refresh-season 2026` : refresh compositions saison en cours
- `scrape_all_players.py --refresh` : refresh joueurs (Elos, nouveaux licencies)
- `discover.py --season 2026` : decouvrir nouveaux groupes

**Données locales** : `dataset_alice/` (symlink) et `data/echiquiers.parquet`, `data/joueurs.parquet`

```python
# Charger depuis HuggingFace
from datasets import load_dataset
ds = load_dataset("Pierrax/ffe-history", data_files="parsed/echiquiers.parquet")

# Ou push un modèle entraîné
from huggingface_hub import HfApi
api = HfApi()
api.upload_file("models/model.pkl", repo_id="Pierrax/alice-engine", repo_type="model")
```

**Compte HF** : Pierrax (token dans `~/.cache/huggingface/token`)

---

## RÈGLES ABSOLUES (ISO Compliance)

**TOUJOURS APPLIQUER - AUCUNE EXCEPTION:**

### ISO 5055 - Limites de code
- **Maximum 300 lignes par fichier** (inclut docstrings)
- **Maximum 50 lignes par fonction**
- **Complexité cyclomatique ≤ B** (Xenon)
- **Si fichier > 300 lignes → REFACTORER en modules SRP**

### ISO 5055 - SRP (Single Responsibility)
- 1 fichier = 1 responsabilité unique
- Préfixer les noms: `types.py`, `thresholds.py`, `metrics.py`, `report.py`
- Créer un `__init__.py` pour re-exporter

### ISO 27034 - Secure Coding
- Pydantic pour TOUTE validation d'entrée
- Jamais de secrets en clair (utiliser env vars)
- Input sanitization obligatoire

### ISO 29119 - Tests
- Docstring structuré avec: Document ID, Version, Tests count
- Fixtures réutilisables
- Tests groupés par classe thématique

### Démarche de rigueur (OBLIGATOIRE)
- **Raisonnement profond** : comprendre le scope métier AVANT de coder (Alice = P(victoire) pour CE, pas classification)
- **WebSearch si doute** : vérifier la doc officielle de CHAQUE API/outil avant utilisation. Ne JAMAIS assumer.
- **Audit domaine AVANT push** : auditer features/approche contre standards ML échecs avant push Kaggle. Ne JAMAIS utiliser l'importance d'un modèle raté pour sélectionner des features.
- **Checklist pré-déploiement** : avant tout push cloud (Kaggle, HF Hub), vérifier :
  - [ ] Dépendances dans l'env cible ?
  - [ ] Paths/montages corrects ?
  - [ ] Credentials/secrets configurés ?
  - [ ] Hardware compatible (GPU/CUDA/PyTorch) ?
  - [ ] Dataset uploadé avec les bons fichiers ?
  - [ ] Kernel slug versionné ?
  - [ ] Ordre des opérations (init_scores AVANT feature subset) ?
  - [ ] Features justifiées par logique domaine (pas modèle raté) ?
- **Post-mortem** : diagnostiquer CHAQUE échec avant de relancer. Documenter dans `docs/postmortem/`.
- **Pas de mensonge** : "je ne sais pas" > affirmation fausse. Toujours.

## Setup DevOps

Hooks pre-commit Python équivalents aux hooks Husky de `C:\Dev\chess-app`.

### Conformité ISO

**Normes Générales:**
- ISO 25010 (Qualité système)
- ISO 27001 (Sécurité)
- ISO 27034 (Secure coding)
- ISO 5055 (Qualité code) - **MAX 300 LIGNES/FICHIER**
- ISO 42010 (Architecture)
- ISO 29119 (Tests)
- ISO 15289 (Documentation)

**Normes ML/AI (ALICE Engine):**
- ISO/IEC 42001:2023 - AI Management System (Model Card, Traçabilité)
- ISO/IEC 23894:2023 - AI Risk Management
- ISO/IEC 5259:2024 - Data Quality for ML (Lineage, Validation)
- ISO/IEC 25059:2023 - AI Quality Model
- ISO/IEC 24029 - Neural Network Robustness
- ISO/IEC TR 24027 - Bias in AI

Voir `docs/iso/ISO_STANDARDS_REFERENCE.md` pour details et mapping fichiers -> normes.

## Avant chaque modification

1. **Vérifier les lignes**: `wc -l fichier.py` - si > 300 → refactorer
2. **Vérifier la complexité**: `radon cc fichier.py` - si C ou pire → refactorer
3. **Appliquer SRP**: 1 module = 1 responsabilité

## Fichiers clés

```
.pre-commit-config.yaml    # Hooks pre-commit/commit-msg/pre-push
pyproject.toml             # Config Ruff, MyPy, Bandit, Pytest, Commitizen
requirements-dev.txt       # Dépendances dev
Makefile                   # Commandes (make help)
```

## Scripts personnalisés

```
scripts/
├── generate_graphs.py      # Graphs SVG imports/deps (pydeps)
├── generate_ml_graphs.py   # Graphs SVG ML pipeline/inference/system (graphviz)
├── update_iso_docs.py      # Génère docs/iso/IMPLEMENTATION_STATUS.md
├── analyze_architecture.py # Score santé architecture (coupling, cycles)
├── autogluon/              # Pipeline AutoGluon (ISO 42001)
│   ├── trainer.py          # Training pipeline avec MLflow
│   ├── run_training.py     # Runner Phase 3 (<50 lignes)
│   ├── iso_robustness.py   # Validation ISO 24029
│   ├── iso_fairness.py     # Validation ISO 24027
│   └── iso_model_card.py   # Génération Model Card ISO 42001
├── comparison/             # Tests statistiques (ISO 24029)
│   ├── mcnemar_test.py     # Test McNemar 5x2cv
│   └── run_mcnemar.py      # Runner Phase 4.4 (<50 lignes)
├── cloud/                  # Training cloud Kaggle (Phase B2)
│   ├── train_kaggle.py           # CatBoost/XGBoost/LightGBM Kaggle
│   ├── train_autogluon_kaggle.py # AutoGluon ensemble (T4x2 GPU)
│   ├── autogluon_diagnostics.py  # ISO 24029/24027 diagnostics
│   ├── autogluon_model_card.py   # ISO 42001 Model Card builder
│   ├── promote_model.py          # Promotion locale ISO
│   ├── upload_all_data.py        # Upload data+code → Kaggle Dataset
│   ├── kernel-metadata.json      # Config Kaggle CatBoost kernel
│   └── kernel-metadata-autogluon.json # Config Kaggle AutoGluon kernel
├── kaggle_trainers.py      # ML training logic (CatBoost/XGBoost/LightGBM)
├── kaggle_metrics.py       # Multiclass metrics + predict_with_init + quality gates
├── kaggle_artifacts.py     # Model persistence + HF Hub push
├── kaggle_diagnostics.py   # ISO diagnostics (ROC, calibration, learning curves)
├── baselines.py            # Elo baseline + compute_elo_init_scores (residual learning)
├── sync_data/              # Sync données FFE (Phase B1)
│   ├── freshness.py        # Vérification fraîcheur données
│   ├── symlink.py          # Gestion symlink/junction Windows
│   ├── huggingface.py      # Pull/push HuggingFace Hub
│   └── types.py            # Pydantic models (SyncConfig, etc.)
├── parse_dataset/          # Parsing HTML → Parquet
│   └── __main__.py         # Entry point + validation ISO 5259
└── reports/                # Rapports ISO
    └── generate_iso25059.py # Rapport final ISO 25059 (<50 lignes)
```

## Hooks Claude Code

```
.claude/
├── hooks/
│   └── pre_check.py        # PreToolUse guards (ISO 5055, 27034)
└── settings.json           # Configuration hooks
```

## Hooks actifs

### Pre-commit
- Gitleaks (secrets P0-CRITICAL)
- Ruff (lint + format)
- MyPy (types)
- Bandit (sécurité)

### Commit-msg
- Commitizen (conventional commits)

### Pre-push
- Pytest + coverage >80%
- Xenon (complexité)
- pip-audit (vulnérabilités)
- `analyze_architecture.py`
- `generate_graphs.py`
- `update_iso_docs.py`

## Artefacts générés

```
graphs/dependencies.svg
reports/complexity/index.html
reports/architecture-health.json
docs/iso/IMPLEMENTATION_STATUS.md
```

## Commandes

```bash
make venv          # Créer venv isolé (.venv/)
make install       # Installer dépendances (dans .venv/)
make hooks         # Installer git hooks
make quality       # Lint + Format + Typecheck + Security
make test-cov      # Tests + coverage
make all           # Validation complète
make graphs        # Générer graphs SVG
make iso-docs      # MAJ documentation ISO
make sync          # Sync données depuis ffe_scrapper
make refresh-data  # Sync + parse + validate ISO 5259 + features (pipeline complet)
```

### Training cloud (Kaggle)
```bash
python -m scripts.cloud.upload_all_data    # Upload data+code → Kaggle Dataset

# CatBoost/XGBoost/LightGBM (kernel-metadata.json)
cp scripts/cloud/kernel-metadata.json scripts/cloud/kernel-metadata.json.bak
kaggle kernels push -p scripts/cloud/
kaggle kernels status pguillemin/alice-training

# AutoGluon (kernel-metadata-autogluon.json → T4x2 GPU)
cp scripts/cloud/kernel-metadata-autogluon.json scripts/cloud/kernel-metadata.json
kaggle kernels push -p scripts/cloud/
git checkout -- scripts/cloud/kernel-metadata.json
kaggle kernels status pguillemin/alice-autogluon-v1

python -m scripts.cloud.promote_model --version v20260318_120000  # Promotion ISO locale
```

**IMPORTANT Kaggle :**
- Datasets montés à `/kaggle/input/datasets/{user}/{slug}/` (PAS `/kaggle/input/{slug}/`)
- **kernel_sources montés à `/kaggle/input/notebooks/{user}/{slug}/`** (PAS `/kaggle/input/{slug}/`)
- AutoGluon pas pré-installé — pip install au runtime dans le script
- GPU T4x2 (sm_75) compatible CUDA 12.8 — P100 (sm_60) INCOMPATIBLE PyTorch mais OK tree models
- Toujours re-uploader `alice-code` dataset AVANT push kernel si fichiers modifiés
- **cudf RALENTIT le feature engineering** (groupby-heavy) — désactivé dans fe_kaggle.py
- **Secrets impossibles en batch push** (Kaggle API issue #582) — HF push échoue silencieusement
- **`--accelerator NvidiaTeslaT4`** obligatoire dans la commande push (enable_gpu donne P100 par défaut)

**V8 Architecture 2-kernel :**
```
# Kernel 1: Feature Engineering (P100 CPU, ~1h)
cp scripts/cloud/kernel-metadata-fe.json scripts/cloud/kernel-metadata.json
kaggle kernels push -p scripts/cloud/
# → Output: features/train.parquet, valid.parquet, test.parquet

# Kernel 2: Training MultiClass (T4 GPU, ~30min)
cp scripts/cloud/kernel-metadata-train.json scripts/cloud/kernel-metadata.json
kaggle kernels push -p scripts/cloud/ --accelerator NvidiaTeslaT4
# → Input: kernel_sources pguillemin/alice-fe-v8
# → Output: CatBoost/XGBoost/LightGBM MultiClass models + diagnostics
```

## Documentation

- `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md` - Roadmap 5 phases → prod
- `docs/superpowers/plans/2026-03-23-residual-learning-phase1.md` - Plan Phase 1 residual (Tasks 1-3 DONE, 4-6 SUPERSEDED)
- `docs/superpowers/plans/2026-03-25-shap-feature-validation.md` - Plan Phase 1b SHAP + calibration (ACTIF)
- `docs/postmortem/2026-03-22-training-v8-divergence.md` - Postmortem training V8
- `docs/architecture/ADR-002-inference-feature-construction.md` - Feature store decision
- `docs/bilan-v8-fe-complete.md` - Bilan FE V8 (196 cols, artefacts)
- `docs/iso/AI_DEVELOPMENT_DISCLOSURE.md` - LLM co-authorship (ISO 42001)
- `docs/iso/ISO_STANDARDS_REFERENCE.md` - Normes ISO applicables
- `docs/superpowers/specs/2026-03-21-multiclass-v8-design.md` - Spec V8 MultiClass
- `docs/superpowers/specs/2026-03-17-data-refresh-pipeline-design.md` - Spec data refresh
- `docs/superpowers/specs/2026-03-18-kaggle-cloud-training-design.md` - Spec Kaggle training
- `docs/PYTHON-HOOKS-SETUP.md` - Setup complet avec correspondances chess-app

## Contraintes Training

- **Séquentiel obligatoire** : 15.4 GB RAM, train set ~7 GB → un modèle à la fois avec `gc.collect()` entre chaque
- `scripts/training/parallel.py` est déjà séquentiel malgré son nom (refactoré en jan 2026)
- **Split temporel** (pas k-fold random) : les règles FFE évoluent au fil des saisons (scoring 2pts jeunes U12+, catégories d'âge, seuils Elo E/N/F), k-fold mélangerait des ères réglementaires différentes
- **@TODO Rolling Window** : comparer training 2012-2026 vs 2002-2026
  - 2012+ : features réglementaires plus cohérentes (scoring, catégories modernes)
  - 2002+ : historique long utile pour profiling clubs, comportements récurrents, H2H
  - Test : entraîner sur 2012+ d'abord, comparer AUC vs modèle actuel (2002+)
- **Training cloud Kaggle CatBoost V7** (OBSOLÈTE, 2026-03-19) : GPU P100, 29 GB RAM
  - AUC=0.8276 INVALIDE (leakage score_dom/ext + target bug: 2.0=victoire jeunes traitée comme forfait)
  - Corrigé localement (commit 05b19a7) : leakage fixé, eval_metric=Logloss, hyperparams améliorés
  - **Remplacé par V8 MultiClass** (en cours)
- **Training cloud Kaggle AutoGluon** (EN COURS, 2026-03-20) : T4x2 GPU, AutoGluon 1.5.0
  - best_quality preset, 6h time limit, 5-fold bagging, 2-stack levels
  - 123 features (3 dropped par AutoGluon), tuning_data=valid, use_bag_holdout=True
  - **NOTE** : P100 incompatible CUDA 12.8/PyTorch 2.9 → TOUJOURS utiliser T4x2
  - **NOTE** : NN_TORCH doit être en CPU (num_gpus=0) tant que P100 est assigné

## V8 MultiClass 3-way (EN COURS, 2026-03-21)

**Remplace V7 binaire** (4 bugs critiques + 8 bugs logique features)

### Décisions validées
- **Target** : loss=0, draw=1, win=2. TARGET_MAP = {0.0:0, 0.5:1, 1.0:2, 2.0:2}. resultat_blanc=2.0 = victoire jeunes FFE (J02 §4.1: 2pts éch. non-U10, 62K parties), mapped to win
- **Forfeits** : identifiés par `type_resultat` (forfait_blanc 43K, forfait_noir 42K, double_forfait 3K, non_joue 209K), PAS par resultat_blanc. Postmortem: `docs/postmortem/2026-03-25-resultat-blanc-2.0-bug.md`
- **Loss** : CatBoost `MultiClass`, XGBoost `multi:softprob`, LightGBM `multiclass`
- **Eval** : MultiClass log loss + RPS (Ranked Probability Score, standard ordinal)
- **Sortie** : P(win), P(draw), P(loss) → CE calcule E[score] = P(win) + 0.5×P(draw)
- **Calibration** : Isotonic par classe + renormalisation. Pas de class weights (dégrade calibration)
- **Quality gate** : log loss < baselines, E[score] MAE < Elo baseline, RPS < baselines, ECE < 0.05, calibration P(draw) ±2%

### Bugs logique corrigés dans features
1. `clutch_factor` : remplacer `|score_dom-score_ext|<=1` par `zone_enjeu IN (montee,danger)`
2. `score_blancs/noirs` : séparer home/away (color confondant avec domicile)
3. Features joueur : stratifier par type_competition (national 20.9% draws ≠ régional 9.6%)
4. Features joueur : rolling 3 saisons au lieu de global career
5. Forfaits : filter by `type_resultat` (not resultat_blanc). resultat_blanc=2.0 = victoire jeunes, recode as win

### Nouvelles features (draw + club-level)
- **8 draw features** (8 cols) : avg_elo, elo_proximity, draw_rate_prior, draw_rate_joueur×2, draw_rate_h2h, draw_rate_equipe×2
- **8 club/vases features** (16 cols) : joueur_promu/relegue, player_team_elo_gap, stabilite_effectif, elo_moyen_evolution, team_rank_in_club, reinforcement_rate, club_nb_teams
- **Toutes features rolling** (données antérieures), stratifiées par niveau de compétition

### Spec complète
- `docs/superpowers/specs/2026-03-21-multiclass-v8-design.md`
- Mémoire : `memory/project_multiclass_v8_design.md` (COMPLÈTE — lire en priorité)

### V8 Training Findings (2026-03-22→25) — CRITIQUE
- **v1-v3 échoués** (path, divergence, hyperparams) — postmortem dans `docs/postmortem/`
- **v5 : PREMIÈRE VICTOIRE** — CatBoost 0.886, LightGBM 0.885 < Elo baseline 0.92
- **v10 : MEILLEUR** — LightGBM 0.877 (test), gate 8/9 (E[score] régression isotonic)
- **v11 : ANNULÉ** — temperature scaling non vérifié
- **DÉCOUVERTE v10** : 166/177 features à importance 0 = **artefact CatBoost PredictionValuesChange**
  - XGBoost utilise **109 features**, LightGBM **50 features** (même données)
  - Root cause : CatBoost manque `rsm` (feature subsampling) — oblivious trees depth=4
  - CatBoost SHAP natif (`type='ShapValues'`) résout le problème
- **Residual learning** : `compute_elo_init_scores()` → `Pool(baseline=)` / `base_margin` / `init_score`
- **Eval cohérente** : `predict_with_init()` pour CatBoost/XGBoost/LightGBM (audit C2)
- **Quality gate** : 9 conditions, condition 9 = `mean_p_draw > 1%` (pas recall_draw)
- **Plan actuel** : `docs/superpowers/plans/2026-03-25-shap-feature-validation.md`
- **NE JAMAIS entraîner sans residual learning** quand une baseline forte existe (Elo en échecs)
- **NE JAMAIS utiliser PredictionValuesChange seul** — comparer importance cross-modèles + SHAP
- **NE JAMAIS sélectionner features par importance d'un modèle raté** — utiliser logique domaine
- **TOUJOURS calculer init_scores AVANT le filtrage features** (blanc_elo/noir_elo nécessaires)
- **TOUJOURS ajouter `rsm=0.3-0.5`** pour CatBoost avec >50 features
- **CONTAMINATION DATA (2026-03-25)** : resultat_blanc=2.0 (62K victoires jeunes) exclu à tort + 295K vrais forfeits inclus. Tous v1-v13 entraînés sur données contaminées. Fix: filter `type_resultat`, recode 2.0→win. Postmortem: `docs/postmortem/2026-03-25-resultat-blanc-2.0-bug.md`

### Inference REQUIERT init_scores (C1 — Phase 2)
- Les modèles entraînés avec residual learning ont besoin des init_scores à l'inférence
- `draw_rate_lookup.parquet` sauvé comme artefact (45 cells)
- L'inference service doit : compute_elo_baseline → compute_elo_init_scores → predict_with_init

### Lacunes versioning ISO 5259/42001 (@TODO Phase 2/5)
- Pas de lien commit git ↔ version Kaggle dataset ↔ version kernel
- Pas de hash du dataset uploadé (upload_all_data ne log pas le hash)
- Artefacts training (reports/) non versionnés — local seulement
- DVC recommandé (docs/devops/ML_MODEL_VERSIONING_STANDARDS.md) mais non implémenté

## V9 CE multi-équipe (@TODO après V8)

> Prérequis : V8 ML terminé (prédictions P(W/D/L) fiables par board dans tous contextes)

**Le vrai problème métier :** Un club a N équipes jouant le MÊME week-end. Le capitaine distribue
~30-50 joueurs entre toutes les équipes sous contraintes FFE (noyau, mutés, 100pts).

### Architecture cible
```
ML model (V8) → matrice P(W/D/L) pour CHAQUE combinaison joueur×board×équipe
  ↓
CE multi-équipe (V9) → allocation optimale joueurs×équipes
  ↓
Objectif modulable par l'utilisateur
```

### Modes de stratégie (V9)
| Mode | Objectif | Usage |
|------|----------|-------|
| **Agressif** | Max E[score] équipe prioritaire, seuil min pour les autres | Montée/titre |
| **Conservateur** | Max min(E[score]) toutes équipes (maximin) | Club loisir |
| **Tactique ronde** | Max P(victoire match) pour équipe en zone enjeu | Maintien |
| **Saison** | Min variance inter-équipes sur la saison | Cohérence |
| **Risk-adjusted** | Max E[score] - λ×Var[score] | P(draw) vs P(win) critique ici |

### Contraintes FFE multi-équipe
- Chaque joueur dans UNE SEULE équipe par week-end
- Noyau (A02 3.7.f) : verrouillé après 1er match dans une équipe
- Max 3 mutés par équipe par saison (A02 3.7.g)
- Joueur peut descendre (renforcer) mais pas monter sans conditions
- Ordre Elo par équipe (100pts A02 3.6.e)

## Architecture Prod (validée 2026-03-23)

```
Vercel (chess-app Next.js) → fetch() HTTPS → Oracle VM (FastAPI + ML, 24GB ARM)
  - Modèle CatBoost chargé en RAM au startup depuis HF Hub
  - Feature store (parquets pré-calculés) sur disque local
  - Inference ~10ms, pas de cold start
  - Oracle Always Free: 4 OCPUs ARM, 24 GB RAM, 200 GB disk
```

Spec complète : `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md`

## @TODO - Phase C : Pipeline CI automatisé

> Prérequis : Phase B (sync + validation ISO 5259) complétée

- [ ] GitHub Action : détection nouvelles données → re-training automatique
- [ ] Push automatique des modèles entraînés sur HuggingFace (`Pierrax/alice-engine`)
- [ ] Drift monitoring intégré (PSI thresholds, feature drift detection)
- [ ] Alertes sur dégradation AUC / fairness (seuils ISO 24027/24029)
- [ ] Rapport ISO 25059 auto-généré à chaque re-training
- [ ] Scheduled refresh : `scrape.py --refresh-season` + `sync_data.py` + `make train` en cron
