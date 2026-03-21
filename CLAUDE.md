# Alice-Engine - Guide Claude

## BUT DU PROJET

Alice Engine est un **système de recommandation de composition d'équipe** pour les interclubs FFE.
Satellite de chess-app (SaaS clubs d'échecs). Déployé sur Render (FastAPI).

**Pipeline complet :**
1. **ALI** (Adversarial Lineup Inference) : prédire la composition probable de l'adversaire
2. **Modèle ML** : prédire P(victoire blanc) pour chaque affectation joueur→échiquier
3. **CE** (Composition Engine) : optimiser la composition sous contraintes FFE (100pts, noyau, mutés)
4. **API** : POST /api/v1/predict retourne composition recommandée + alternatives + score attendu

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
- **Checklist pré-déploiement** : avant tout push cloud (Kaggle, HF Hub), vérifier :
  - [ ] Dépendances dans l'env cible ?
  - [ ] Paths/montages corrects ?
  - [ ] Credentials/secrets configurés ?
  - [ ] Hardware compatible (GPU/CUDA/PyTorch) ?
  - [ ] Dataset uploadé avec les bons fichiers ?
  - [ ] Kernel slug versionné ?
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
├── generate_graphs.py      # Graphs SVG architecture (pydeps)
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
├── kaggle_artifacts.py     # Model persistence + HF Hub push
├── kaggle_diagnostics.py   # ISO diagnostics (ROC, calibration, learning curves)
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
- AutoGluon pas pré-installé — pip install au runtime dans le script
- GPU T4x2 (sm_75) compatible CUDA 12.8 — P100 (sm_60) INCOMPATIBLE
- Toujours re-uploader `alice-code` dataset AVANT push kernel si fichiers modifiés

## Documentation

- `docs/PYTHON-HOOKS-SETUP.md` - Setup complet avec correspondances chess-app
- `docs/iso/ISO_STANDARDS_REFERENCE.md` - Normes ISO applicables
- `docs/superpowers/specs/2026-03-17-data-refresh-pipeline-design.md` - Spec data refresh
- `docs/superpowers/specs/2026-03-18-kaggle-cloud-training-design.md` - Spec Kaggle training
- `docs/superpowers/plans/2026-03-17-data-refresh-pipeline.md` - Plan data refresh
- `docs/superpowers/plans/2026-03-18-kaggle-cloud-training.md` - Plan Kaggle training

## Contraintes Training

- **Séquentiel obligatoire** : 15.4 GB RAM, train set ~7 GB → un modèle à la fois avec `gc.collect()` entre chaque
- `scripts/training/parallel.py` est déjà séquentiel malgré son nom (refactoré en jan 2026)
- **Split temporel** (pas k-fold random) : les règles FFE évoluent au fil des saisons (scoring 2pts jeunes U12+, catégories d'âge, seuils Elo E/N/F), k-fold mélangerait des ères réglementaires différentes
- **@TODO Rolling Window** : comparer training 2012-2026 vs 2002-2026
  - 2012+ : features réglementaires plus cohérentes (scoring, catégories modernes)
  - 2002+ : historique long utile pour profiling clubs, comportements récurrents, H2H
  - Test : entraîner sur 2012+ d'abord, comparer AUC vs modèle actuel (2002+)
- **Training cloud Kaggle CatBoost** (FAIT, 2026-03-19) : GPU P100, 29 GB RAM, pipeline SRP 4 modules
  - CatBoost AUC=0.8276 (GPU P100), XGBoost AUC=0.7600 (GPU cuda), LightGBM AUC=0.7292 (CPU)
  - 147 features, 1.14M rows, quality gate PASSED, models on HF Hub `Pierrax/alice-engine`
- **Training cloud Kaggle AutoGluon** (EN COURS, 2026-03-20) : T4x2 GPU, AutoGluon 1.5.0
  - best_quality preset, 6h time limit, 5-fold bagging, 2-stack levels
  - 123 features (3 dropped par AutoGluon), tuning_data=valid, use_bag_holdout=True
  - **NOTE** : P100 incompatible CUDA 12.8/PyTorch 2.9 → TOUJOURS utiliser T4x2
  - **NOTE** : NN_TORCH doit être en CPU (num_gpus=0) tant que P100 est assigné

## @TODO - Phase C : Pipeline CI automatisé

> Prérequis : Phase B (sync + validation ISO 5259) complétée

- [ ] GitHub Action : détection nouvelles données → re-training automatique
- [ ] Push automatique des modèles entraînés sur HuggingFace (`Pierrax/alice-engine`)
- [ ] Drift monitoring intégré (PSI thresholds, feature drift detection)
- [ ] Alertes sur dégradation AUC / fairness (seuils ISO 24027/24029)
- [ ] Rapport ISO 25059 auto-généré à chaque re-training
- [ ] Scheduled refresh : `scrape.py --refresh-season` + `sync_data.py` + `make train` en cron
