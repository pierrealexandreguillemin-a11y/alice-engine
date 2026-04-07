# ALICE - Adversarial Lineup Inference & Composition Engine

> *"J'ai fait les compos avec Alice"*

Module predictif pour l'application chess-app. Predit les compositions adverses et optimise les compositions d'equipes pour les interclubs d'echecs FFE.

## Composants

| Composant | Acronyme | Fonction | Statut |
|-----------|----------|----------|--------|
| **ML Training** | — | Entrainement XGBoost / CatBoost / LightGBM | V9 Optuna en cours |
| **Adversarial Lineup Inference** | ALI | Predire la composition adverse | Phase 3 (a venir) |
| **Composition Engine** | CE | Optimiser sa propre composition | Phase 4 (a venir) |
| **API FastAPI** | — | Endpoints REST (stubs) | Complet |

## Pipeline ML

```
FFE Dataset (HuggingFace: Pierrax/ffe-history)
    |
    v
[make refresh-data] -> Parse HTML, extract features (196-220 cols)
    |
    v
[Elo baseline] -> init_scores (residual learning)
    |
    v
[Optuna V9] -> XGBoost / CatBoost / LightGBM hyperparameter search
    |
    v
[Quality Gates] -> 15 conditions (log_loss, RPS, E[score], ECE, draw bias)
    |
    v
[SHAP] -> Feature importance, explicabilite
```

## Quick Start

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
git clone https://github.com/pierrealexandreguillemin-a11y/alice-engine.git
cd alice-engine
python -m venv venv
# Windows: .\venv\Scripts\Activate
# Unix: source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Editer : MongoDB URI, API_KEY
```

### Lancement

```bash
# Serveur API
uvicorn app.main:app --reload --port 8000

# Validation complete
make all

# Refresh donnees
make refresh-data

# Tests + coverage
make test-cov
```

## Stack

| Couche | Technologie |
|--------|-------------|
| API | FastAPI, Uvicorn, Pydantic |
| ML Training | CatBoost, XGBoost, LightGBM |
| Hyperparams | Optuna (SQLite storage) |
| Explicabilite | SHAP (TreeSHAP) |
| Baseline | Elo residual learning |
| Features | 196-220 colonnes (rolling, stratifiees, differentiels) |
| Qualite | Ruff, MyPy, Bandit, Gitleaks, Xenon |
| Tests | pytest (89 tests ISO) |
| Hooks | Husky (pre-commit, commit-msg, pre-push) |

## Architecture (ISO 42010 - SRP)

```
alice-engine/
├── app/                        # Controller layer (FastAPI)
│   ├── main.py                 # Entry point, CORS, rate limiting
│   ├── config.py               # Pydantic Settings (ISO 27001)
│   └── api/
│       ├── routes.py           # Endpoints (stubs)
│       └── schemas.py          # Pydantic validation
│
├── services/                   # Service layer
│   ├── inference.py            # ALI - Prediction adverse
│   ├── composer.py             # CE - Optimisation composition
│   └── data_loader.py          # MongoDB + Parquet
│
├── scripts/                    # ML pipeline
│   ├── cloud/                  # Kaggle kernels (V9 Optuna)
│   ├── features/               # Feature engineering
│   ├── training/               # Optuna objectives
│   ├── baselines.py            # Elo baseline + init_scores
│   ├── kaggle_trainers.py      # Training logic
│   └── kaggle_metrics.py       # Metrics + predict_with_init
│
├── config/hyperparameters.yaml # ML hyperparams + Optuna search spaces
├── docs/                       # 65+ documents (specs, plans, postmortems, ISO)
├── dataset_alice/              # Symlink -> ffe_data_backup (2.4 GB)
├── Makefile                    # make help for all commands
└── pyproject.toml              # Ruff, MyPy, Bandit, Pytest, Commitizen
```

## Dataset

Disponible sur HuggingFace : `Pierrax/ffe-history`

| Metrique | Valeur |
|----------|--------|
| Taille | 2.4 GB (HTML brut) |
| Echiquiers | ~750,000 |
| Joueurs | ~55,000 |
| Saisons | 25 (2002-2026) |

Le dataset local est via symlink (non versionne). Voir `docs/requirements/CONTEXTE_DATASET_FFE.md`.

## Standards ISO (14 normes)

| Norme | Focus | Implementation |
|-------|-------|----------------|
| **ISO 5055** | Qualite code | Ruff, MyPy, Xenon (max 300 lignes, complexite B) |
| **ISO 27001** | Secrets | `.env`, Gitleaks, Bandit |
| **ISO 29119** | Tests | pytest, 89 tests, coverage |
| **ISO 42001** | Gouvernance IA | Model Card, SHAP, tracabilite |
| **ISO 42010** | Architecture | SRP, layers separes, ADR |
| **ISO 25010** | Qualite systeme | Quality gates (15 conditions) |

Reference complete : `docs/iso/ISO_STANDARDS_REFERENCE.md`

## Deploiement cible

```
Vercel (chess-app) -> HTTPS -> Oracle VM (FastAPI + ML, 24GB ARM)
```

Spec : `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md`

## License

Proprietaire - Tous droits reserves

---

*Avril 2026 — V9 Optuna en cours*
