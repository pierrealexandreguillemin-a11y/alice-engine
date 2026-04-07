# Project Structure

## Key Files
```
.pre-commit-config.yaml    # Hooks pre-commit/commit-msg/pre-push
pyproject.toml             # Ruff, MyPy, Bandit, Pytest, Commitizen
config/hyperparameters.yaml # ML hyperparams + Optuna search spaces
Makefile                   # make help for all commands
```

## Scripts
```
scripts/
├── cloud/                  # Kaggle kernels
│   ├── train_kaggle.py     # V8 training (CatBoost/XGBoost/LightGBM)
│   ├── optuna_kaggle.py    # V9 Optuna objectives + main
│   ├── optuna_{xgboost,catboost,lightgbm}.py  # Entry points
│   ├── upload_all_data.py  # Upload data+code → Kaggle
│   └── kernel-metadata-*.json  # Kernel configs
├── kaggle_trainers.py      # Training logic + prepare_features
├── kaggle_metrics.py       # Metrics + predict_with_init
├── baselines.py            # Elo baseline + init_scores
├── features/               # Feature engineering pipeline
│   ├── pipeline.py         # extract_all_features, merge_all_features
│   ├── differentials.py    # compute_differentials (FTI anti-skew)
│   └── draw_priors.py      # Draw rate lookups
├── ensemble/stacking.py    # Stacking OOF (needs multiclass update)
└── training/optuna_*.py    # Optuna core (needs V9 update)
```

## App (FastAPI)
```
app/
├── main.py                 # Lifespan, CORS, rate limiting
├── config.py               # Pydantic Settings
├── api/routes.py           # Endpoints (stubs)
└── api/schemas.py          # Pydantic models
services/
├── composer.py             # CE (Composition Engine)
├── inference.py            # ALI (Adversarial Lineup Inference)
└── data_loader.py          # MongoDB + Parquet
```

## Hooks
- Pre-commit: Gitleaks, Ruff, MyPy, Bandit
- Commit-msg: Commitizen (conventional commits)
- Pre-push: Pytest+coverage, Xenon, pip-audit, architecture, graphs, ISO docs
- Claude Code: `.claude/hooks/pre_check.py` (ISO 5055/27034 guards)

## V9 CE multi-equipe (TODO apres V9 Optuna)
Un club a N equipes le MEME week-end. CE optimise allocation joueurs×equipes.
Modes : agressif, conservateur, tactique, saison, risk-adjusted.
Contraintes FFE : noyau, mutes, ordre Elo, 1 joueur = 1 equipe.
Spec : `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md`
