# V9 Optuna Pipeline Implementation Plan (Steps 1-3: Optuna Kernels)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt existing Optuna infrastructure to multiclass V8 with residual learning, create Kaggle kernel entry points and metadata, and push the first Optuna kernel (XGBoost canary).

**Architecture:** New `scripts/cloud/optuna_kaggle.py` orchestrates Optuna on Kaggle using SQLite storage, pruning callbacks, and init_scores with alpha in the search space. Reuses `train_kaggle.py` patterns (feature loading, init_scores, NaN audit). One entry point per model (`optuna_xgboost.py`, etc.). Existing `optuna_objectives.py` is REPLACED (binary→multiclass, categorical→continuous suggest, sklearn wrappers→native API).

**Tech Stack:** Optuna >= 4.6.0, optuna-integration (separate package since v4.0), XGBoost native API (xgb.train), CatBoost (CatBoostClassifier), LightGBM (lgb.train), SQLite, pandas, numpy.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/cloud/optuna_kaggle.py` | Create | Optuna orchestration for Kaggle (objectives + main) |
| `scripts/cloud/optuna_xgboost.py` | Create | Entry point: XGBoost Optuna kernel |
| `scripts/cloud/optuna_catboost.py` | Create | Entry point: CatBoost Optuna kernel |
| `scripts/cloud/optuna_lightgbm.py` | Create | Entry point: LightGBM Optuna kernel |
| `scripts/cloud/kernel-metadata-optuna-xgboost.json` | Create | Kaggle kernel metadata |
| `scripts/cloud/kernel-metadata-optuna-catboost.json` | Create | Kaggle kernel metadata |
| `scripts/cloud/kernel-metadata-optuna-lightgbm.json` | Create | Kaggle kernel metadata |
| `config/hyperparameters.yaml` | Modify | Update optuna search spaces for V9 multiclass |
| `tests/training_optuna/test_objectives_v9.py` | Create | Tests for multiclass objectives |

## Key Dependencies

- `scripts/cloud/train_kaggle.py` — `_setup_kaggle_imports()`, `_load_features()` (reuse)
- `scripts/kaggle_trainers.py` — `prepare_features()`, `default_hyperparameters()` (reuse)
- `scripts/baselines.py` — `compute_init_scores_from_features()` (reuse)
- `scripts/features/draw_priors.py` — `build_draw_rate_lookup()` (reuse)
- `scripts/kaggle_metrics.py` — `predict_with_init()` (reuse for eval)

---

### Task 1: Update hyperparameters.yaml with V9 Optuna search spaces

**Files:**
- Modify: `config/hyperparameters.yaml` (lines 215-254, optuna section)

- [ ] **Step 1: Read current optuna section**

Read `config/hyperparameters.yaml` lines 215-254 to see existing search spaces.

- [ ] **Step 2: Replace optuna section with V9 multiclass search spaces**

Replace the entire `optuna:` section (lines 215-254) with:

```yaml
optuna:
  n_trials: 100                 # Target trials (timeout may cut short)
  timeout: 36000                # 11h (1h margin on 12h Kaggle)

  # V9 MULTICLASS SEARCH SPACES (audited against fabricant docs 2026-04-07)
  # Source: docs/superpowers/specs/2026-04-07-optuna-v9-pipeline-design.md §3

  # Shared across all models
  shared:
    init_score_alpha:
      low: 0.3
      high: 0.8
      # Residual prior strength. Custom param (Guo 2017 temperature scaling).

  # XGBoost — https://xgboost.readthedocs.io/en/stable/parameter.html
  xgboost_search_space:
    max_depth: {low: 3, high: 8}                       # int, default=6
    eta: {low: 0.001, high: 0.1, log: true}            # float, default=0.3
    reg_lambda: {low: 0.1, high: 20.0, log: true}      # float, default=1
    reg_alpha: {low: 0.001, high: 1.0, log: true}      # float, default=0
    subsample: {low: 0.5, high: 1.0}                   # float, default=1
    colsample_bytree: {low: 0.3, high: 1.0}            # float, default=1
    min_child_weight: {low: 20, high: 200}              # int, default=1
    # Fixed: n_estimators=50000, early_stopping=200, objective=multi:softprob

  # CatBoost — https://catboost.ai/docs/en/concepts/parameter-tuning
  catboost_search_space:
    depth: {low: 4, high: 10}                           # int, "optimal 4-10"
    learning_rate: {low: 0.001, high: 0.1, log: true}   # float, default=0.03
    l2_leaf_reg: {low: 0.1, high: 20.0, log: true}      # float, default=3
    rsm: {low: 0.2, high: 0.8}                          # float, MANDATORY >50 features
    random_strength: {low: 0.5, high: 5.0}              # float, default=1
    min_data_in_leaf: {low: 20, high: 200}              # int
    # Fixed: iterations=50000, early_stopping=200, loss=MultiClass, task_type=CPU

  # LightGBM — https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
  lightgbm_search_space:
    max_depth: {low: 3, high: 8}                        # int, default=-1
    num_leaves: {low: 7, high: 63}                      # int, CLAMP: min(val, 2^depth-1)
    learning_rate: {low: 0.001, high: 0.1, log: true}   # float, default=0.1
    reg_lambda: {low: 0.1, high: 20.0, log: true}       # float, default=0
    feature_fraction: {low: 0.3, high: 1.0}             # float
    bagging_fraction: {low: 0.5, high: 1.0}             # float, requires bagging_freq=1
    min_child_samples: {low: 20, high: 200}             # int, default=20
    # Fixed: n_estimators=50000, early_stopping=200, objective=multiclass, bagging_freq=1
```

- [ ] **Step 3: Run YAML validation**

Run: `python -c "import yaml; yaml.safe_load(open('config/hyperparameters.yaml'))"`
Expected: No error

- [ ] **Step 4: Commit**

```bash
git add config/hyperparameters.yaml
git commit -m "feat(config): V9 Optuna search spaces — multiclass + init_score_alpha"
```

---

### Task 2: Create multiclass Optuna objectives + tests

**Files:**
- Create: `scripts/cloud/optuna_kaggle.py`
- Create: `tests/training_optuna/test_objectives_v9.py`

- [ ] **Step 1: Write failing tests**

Create `tests/training_optuna/test_objectives_v9.py`:

```python
"""Tests for V9 multiclass Optuna objectives — ISO 29119.

Document ID: ALICE-TEST-OPTUNA-V9
Version: 1.0.0
Tests count: 6
"""

import numpy as np
import pandas as pd
import pytest


def _make_data(n: int = 500, n_features: int = 10, seed: int = 42):
    """Create synthetic multiclass data matching Alice schema."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, n_features), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(rng.choice([0, 1, 2], size=n, p=[0.42, 0.13, 0.45]))
    init_scores = rng.randn(n, 3) * 0.5  # centered log-odds
    return X, y, init_scores


class TestXGBoostObjective:
    """Tests for XGBoost multiclass Optuna objective."""

    def test_returns_finite_logloss(self) -> None:
        import optuna
        from scripts.cloud.optuna_kaggle import create_xgboost_objective_v9

        X_train, y_train, init_train = _make_data(300, seed=0)
        X_valid, y_valid, init_valid = _make_data(100, seed=1)
        config = {"optuna": {"xgboost_search_space": {}}}

        objective = create_xgboost_objective_v9(
            X_train, y_train, init_train, X_valid, y_valid, init_valid, config
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=1, timeout=120)
        assert study.best_value < 2.0  # less than log(3)~1.099 naive
        assert np.isfinite(study.best_value)

    def test_alpha_in_params(self) -> None:
        import optuna
        from scripts.cloud.optuna_kaggle import create_xgboost_objective_v9

        X_train, y_train, init_train = _make_data(300, seed=0)
        X_valid, y_valid, init_valid = _make_data(100, seed=1)
        config = {"optuna": {"xgboost_search_space": {}}}

        objective = create_xgboost_objective_v9(
            X_train, y_train, init_train, X_valid, y_valid, init_valid, config
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=1, timeout=120)
        assert "init_score_alpha" in study.best_params
        assert 0.3 <= study.best_params["init_score_alpha"] <= 0.8


class TestCatBoostObjective:
    """Tests for CatBoost multiclass Optuna objective."""

    def test_returns_finite_logloss(self) -> None:
        import optuna
        from scripts.cloud.optuna_kaggle import create_catboost_objective_v9

        X_train, y_train, init_train = _make_data(300, seed=0)
        X_valid, y_valid, init_valid = _make_data(100, seed=1)
        config = {"optuna": {"catboost_search_space": {}}}

        objective = create_catboost_objective_v9(
            X_train, y_train, init_train, X_valid, y_valid, init_valid, config
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=1, timeout=120)
        assert study.best_value < 2.0
        assert np.isfinite(study.best_value)


class TestLightGBMObjective:
    """Tests for LightGBM multiclass Optuna objective."""

    def test_returns_finite_logloss(self) -> None:
        import optuna
        from scripts.cloud.optuna_kaggle import create_lightgbm_objective_v9

        X_train, y_train, init_train = _make_data(300, seed=0)
        X_valid, y_valid, init_valid = _make_data(100, seed=1)
        config = {"optuna": {"lightgbm_search_space": {}}}

        objective = create_lightgbm_objective_v9(
            X_train, y_train, init_train, X_valid, y_valid, init_valid, config
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=1, timeout=120)
        assert study.best_value < 2.0
        assert np.isfinite(study.best_value)

    def test_num_leaves_clamped(self) -> None:
        import optuna
        from scripts.cloud.optuna_kaggle import create_lightgbm_objective_v9

        X_train, y_train, init_train = _make_data(300, seed=0)
        X_valid, y_valid, init_valid = _make_data(100, seed=1)
        config = {"optuna": {"lightgbm_search_space": {}}}

        objective = create_lightgbm_objective_v9(
            X_train, y_train, init_train, X_valid, y_valid, init_valid, config
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=3, timeout=120)
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                depth = trial.params["max_depth"]
                leaves = trial.params["num_leaves"]
                assert leaves <= 2**depth - 1, f"num_leaves={leaves} > 2^{depth}-1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/training_optuna/test_objectives_v9.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.cloud.optuna_kaggle'`

- [ ] **Step 3: Create `scripts/cloud/optuna_kaggle.py`**

```python
"""Optuna V9 multiclass hyperparameter tuning — Kaggle CPU kernel.

Document ID: ALICE-OPTUNA-V9
Version: 1.0.0

Bayesian optimization with TPESampler on multiclass (loss/draw/win)
with residual learning (init_score_alpha in search space).

ISO Compliance:
- ISO/IEC 42001:2023 — AI Management (traceability via SQLite study)
- ISO/IEC 25059:2023 — AI Quality (baseline comparison gate)
- ISO/IEC 5055:2021 — Code Quality (SRP, <300 lines)

References:
- Spec: docs/superpowers/specs/2026-04-07-optuna-v9-pipeline-design.md
- XGBoost: https://xgboost.readthedocs.io/en/stable/parameter.html
- CatBoost: https://catboost.ai/docs/en/concepts/parameter-tuning
- LightGBM: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ELO_BASELINE_LOGLOSS = 0.9766  # Quality gate: must beat this


def _get_search_param(
    trial: Any, name: str, space: dict, default_low: float, default_high: float,
    param_type: str = "float", log: bool = False,
) -> Any:
    """Suggest a parameter from the search space config or defaults."""
    cfg = space.get(name, {})
    low = cfg.get("low", default_low)
    high = cfg.get("high", default_high)
    use_log = cfg.get("log", log)
    if param_type == "int":
        return trial.suggest_int(name, int(low), int(high))
    return trial.suggest_float(name, low, high, log=use_log)


def create_xgboost_objective_v9(
    X_train: pd.DataFrame, y_train: pd.Series, init_train: np.ndarray,
    X_valid: pd.DataFrame, y_valid: pd.Series, init_valid: np.ndarray,
    config: dict,
) -> Any:
    """Create XGBoost multiclass Optuna objective with residual learning."""
    import xgboost as xgb

    space = config.get("optuna", {}).get("xgboost_search_space", {})
    shared = config.get("optuna", {}).get("shared", {})
    alpha_cfg = shared.get("init_score_alpha", {})

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    def objective(trial: Any) -> float:
        alpha = trial.suggest_float(
            "init_score_alpha", alpha_cfg.get("low", 0.3), alpha_cfg.get("high", 0.8),
        )
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "max_depth": _get_search_param(trial, "max_depth", space, 3, 8, "int"),
            "eta": _get_search_param(trial, "eta", space, 0.001, 0.1, "float", log=True),
            "lambda": _get_search_param(trial, "reg_lambda", space, 0.1, 20.0, log=True),
            "alpha": _get_search_param(trial, "reg_alpha", space, 0.001, 1.0, log=True),
            "subsample": _get_search_param(trial, "subsample", space, 0.5, 1.0),
            "colsample_bytree": _get_search_param(trial, "colsample_bytree", space, 0.3, 1.0),
            "min_child_weight": _get_search_param(trial, "min_child_weight", space, 20, 200, "int"),
            "tree_method": "hist",
            "seed": 42,
            "nthread": int(os.environ.get("ALICE_NTHREAD", "4")),
            "verbosity": 0,
        }

        dtrain.set_base_margin((init_train * alpha).ravel())
        dvalid.set_base_margin((init_valid * alpha).ravel())

        try:
            from optuna_integration import XGBoostPruningCallback
            pruning_cb = XGBoostPruningCallback(trial, "valid-mlogloss")
            callbacks = [
                xgb.callback.EarlyStopping(rounds=200, save_best=True),
                pruning_cb,
            ]
        except ImportError:
            callbacks = [xgb.callback.EarlyStopping(rounds=200, save_best=True)]

        booster = xgb.train(
            params, dtrain, num_boost_round=50000,
            evals=[(dvalid, "valid")],
            callbacks=callbacks,
            verbose_eval=False,
        )
        return booster.best_score

    return objective


def create_catboost_objective_v9(
    X_train: pd.DataFrame, y_train: pd.Series, init_train: np.ndarray,
    X_valid: pd.DataFrame, y_valid: pd.Series, init_valid: np.ndarray,
    config: dict,
) -> Any:
    """Create CatBoost multiclass Optuna objective with residual learning."""
    from catboost import CatBoostClassifier, Pool

    space = config.get("optuna", {}).get("catboost_search_space", {})
    shared = config.get("optuna", {}).get("shared", {})
    alpha_cfg = shared.get("init_score_alpha", {})

    def objective(trial: Any) -> float:
        alpha = trial.suggest_float(
            "init_score_alpha", alpha_cfg.get("low", 0.3), alpha_cfg.get("high", 0.8),
        )
        params = {
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "iterations": 50000,
            "depth": _get_search_param(trial, "depth", space, 4, 10, "int"),
            "learning_rate": _get_search_param(
                trial, "learning_rate", space, 0.001, 0.1, log=True,
            ),
            "l2_leaf_reg": _get_search_param(
                trial, "l2_leaf_reg", space, 0.1, 20.0, log=True,
            ),
            "rsm": _get_search_param(trial, "rsm", space, 0.2, 0.8),
            "random_strength": _get_search_param(
                trial, "random_strength", space, 0.5, 5.0,
            ),
            "min_data_in_leaf": _get_search_param(
                trial, "min_data_in_leaf", space, 20, 200, "int",
            ),
            "task_type": "CPU",
            "random_seed": 42,
            "verbose": 0,
            "early_stopping_rounds": 200,
        }

        train_pool = Pool(X_train, y_train, baseline=(init_train * alpha))
        valid_pool = Pool(X_valid, y_valid, baseline=(init_valid * alpha))

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool)

        return model.get_best_score()["validation"]["MultiClass"]

    return objective


def create_lightgbm_objective_v9(
    X_train: pd.DataFrame, y_train: pd.Series, init_train: np.ndarray,
    X_valid: pd.DataFrame, y_valid: pd.Series, init_valid: np.ndarray,
    config: dict,
) -> Any:
    """Create LightGBM multiclass Optuna objective with residual learning."""
    import lightgbm as lgb

    space = config.get("optuna", {}).get("lightgbm_search_space", {})
    shared = config.get("optuna", {}).get("shared", {})
    alpha_cfg = shared.get("init_score_alpha", {})

    def objective(trial: Any) -> float:
        alpha = trial.suggest_float(
            "init_score_alpha", alpha_cfg.get("low", 0.3), alpha_cfg.get("high", 0.8),
        )
        max_depth = _get_search_param(trial, "max_depth", space, 3, 8, "int")
        raw_leaves = _get_search_param(trial, "num_leaves", space, 7, 63, "int")
        num_leaves = min(raw_leaves, 2**max_depth - 1)  # CLAMP per LightGBM docs

        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "learning_rate": _get_search_param(
                trial, "learning_rate", space, 0.001, 0.1, log=True,
            ),
            "reg_lambda": _get_search_param(
                trial, "reg_lambda", space, 0.1, 20.0, log=True,
            ),
            "feature_fraction": _get_search_param(
                trial, "feature_fraction", space, 0.3, 1.0,
            ),
            "bagging_fraction": _get_search_param(
                trial, "bagging_fraction", space, 0.5, 1.0,
            ),
            "bagging_freq": 1,
            "min_child_samples": _get_search_param(
                trial, "min_child_samples", space, 20, 200, "int",
            ),
            "seed": 42,
            "verbose": -1,
            "n_jobs": int(os.environ.get("ALICE_NTHREAD", "4")),
        }

        dtrain = lgb.Dataset(X_train, label=y_train,
                             init_score=(init_train * alpha).ravel())
        dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain,
                             init_score=(init_valid * alpha).ravel())

        try:
            from optuna_integration import LightGBMPruningCallback
            pruning_cb = LightGBMPruningCallback(trial, metric="multi_logloss")
            callbacks = [lgb.early_stopping(200), lgb.log_evaluation(0), pruning_cb]
        except ImportError:
            callbacks = [lgb.early_stopping(200), lgb.log_evaluation(0)]

        booster = lgb.train(
            params, dtrain, num_boost_round=50000,
            valid_sets=[dvalid], valid_names=["valid"],
            callbacks=callbacks,
        )
        return booster.best_score["valid"]["multi_logloss"]

    return objective


def main() -> None:
    """Run Optuna study for the model specified by ALICE_MODEL env var."""
    import optuna
    import yaml

    from scripts.baselines import compute_init_scores_from_features
    from scripts.cloud.train_kaggle import _load_features, _setup_kaggle_imports
    from scripts.features.draw_priors import build_draw_rate_lookup
    from scripts.kaggle_trainers import prepare_features

    _setup_kaggle_imports()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_name = os.environ.get("ALICE_MODEL", "xgboost")
    out_dir = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = Path("config/hyperparameters.yaml")
    if not config_path.exists():
        # Kaggle: config in alice-code dataset
        for p in [Path("/kaggle/input/alice-code/config/hyperparameters.yaml"),
                   Path("/kaggle/input/datasets/pguillemin/alice-code/config/hyperparameters.yaml")]:
            if p.exists():
                config_path = p
                break
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info("Config loaded from %s", config_path)

    # Load features
    train, valid, test, features_dir = _load_features()
    X_train, y_train, X_valid, y_valid, X_test, y_test, encoders = prepare_features(
        train, valid, test
    )
    logger.info("Features: train=%s, valid=%s", X_train.shape, X_valid.shape)

    # Compute init_scores (ONCE, alpha applied per trial)
    draw_lookup = build_draw_rate_lookup(train)
    init_train = compute_init_scores_from_features(X_train, draw_lookup)
    init_valid = compute_init_scores_from_features(X_valid, draw_lookup)
    logger.info("Init scores: train=%s, valid=%s", init_train.shape, init_valid.shape)

    # Create objective
    objective_factory = {
        "xgboost": create_xgboost_objective_v9,
        "catboost": create_catboost_objective_v9,
        "lightgbm": create_lightgbm_objective_v9,
    }
    if model_name not in objective_factory:
        msg = f"Unknown model: {model_name}. Expected: {list(objective_factory)}"
        raise ValueError(msg)

    objective = objective_factory[model_name](
        X_train, y_train, init_train, X_valid, y_valid, init_valid, config
    )

    # Create study (SQLite for resume)
    db_path = out_dir / f"optuna_{model_name}.db"
    study = optuna.create_study(
        study_name=f"alice_{model_name}_v9",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    optuna_config = config.get("optuna", {})
    n_trials = optuna_config.get("n_trials", 100)
    timeout = optuna_config.get("timeout", 36000)

    logger.info("Starting Optuna: model=%s, n_trials=%d, timeout=%ds", model_name, n_trials, timeout)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    elapsed = time.time() - t0

    # Log results
    logger.info("Optuna complete: %d trials in %.0fs", len(study.trials), elapsed)
    logger.info("Best logloss: %.5f", study.best_value)
    logger.info("Best params: %s", study.best_params)

    # Quality gate
    gate_pass = study.best_value < ELO_BASELINE_LOGLOSS
    logger.info("GATE: best_logloss %.5f < Elo %.5f = %s",
                study.best_value, ELO_BASELINE_LOGLOSS, "PASS" if gate_pass else "FAIL")

    # Save artifacts
    best = {
        "model": model_name,
        "best_logloss": study.best_value,
        "best_params": study.best_params,
        "n_trials_completed": len([t for t in study.trials
                                    if t.state.name == "COMPLETE"]),
        "n_trials_pruned": len([t for t in study.trials
                                 if t.state.name == "PRUNED"]),
        "total_time_s": round(elapsed, 1),
        "gate_pass": gate_pass,
        "elo_baseline_logloss": ELO_BASELINE_LOGLOSS,
    }
    best_path = out_dir / f"best_params_{model_name}.json"
    best_path.write_text(json.dumps(best, indent=2))
    logger.info("Saved: %s", best_path)

    # Trial history CSV
    rows = []
    for t in study.trials:
        row = {"trial": t.number, "value": t.value, "state": t.state.name,
               "duration_s": (t.datetime_complete - t.datetime_start).total_seconds()
               if t.datetime_complete and t.datetime_start else None}
        row.update(t.params)
        rows.append(row)
    history = pd.DataFrame(rows)
    history_path = out_dir / f"trial_history_{model_name}.csv"
    history.to_csv(history_path, index=False)
    logger.info("Saved: %s", history_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run ruff format**

Run: `ruff format scripts/cloud/optuna_kaggle.py tests/training_optuna/test_objectives_v9.py`

- [ ] **Step 5: Run tests**

Run: `pytest tests/training_optuna/test_objectives_v9.py -v --timeout=300`
Expected: 6 passed (each trial trains a small model, may take 10-30s each)

- [ ] **Step 6: Commit**

```bash
git add scripts/cloud/optuna_kaggle.py tests/training_optuna/test_objectives_v9.py
git commit -m "feat(ml): V9 Optuna multiclass objectives with init_score_alpha"
```

---

### Task 3: Create entry points and kernel metadata

**Files:**
- Create: `scripts/cloud/optuna_xgboost.py`
- Create: `scripts/cloud/optuna_catboost.py`
- Create: `scripts/cloud/optuna_lightgbm.py`
- Create: `scripts/cloud/kernel-metadata-optuna-xgboost.json`
- Create: `scripts/cloud/kernel-metadata-optuna-catboost.json`
- Create: `scripts/cloud/kernel-metadata-optuna-lightgbm.json`

- [ ] **Step 1: Create XGBoost entry point**

Create `scripts/cloud/optuna_xgboost.py`:

```python
"""Entry point: XGBoost Optuna V9 (CPU, 12h). Canary — push first."""

import os
import sys
import subprocess
from pathlib import Path

os.environ["ALICE_MODEL"] = "xgboost"

# sys.path BEFORE any alice import (CLAUDE.md mandatory)
for p in [
    Path("/kaggle/input/alice-code"),
    Path("/kaggle/input/datasets/pguillemin/alice-code"),
]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

# Install optuna (not pre-installed, optuna-integration separate since v4.0)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "optuna", "optuna-integration"]
)

from scripts.cloud.optuna_kaggle import main  # noqa: E402

main()
```

- [ ] **Step 2: Create CatBoost entry point**

Create `scripts/cloud/optuna_catboost.py`:

```python
"""Entry point: CatBoost Optuna V9 (CPU, 12h)."""

import os
import sys
import subprocess
from pathlib import Path

os.environ["ALICE_MODEL"] = "catboost"

for p in [
    Path("/kaggle/input/alice-code"),
    Path("/kaggle/input/datasets/pguillemin/alice-code"),
]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "optuna", "optuna-integration"]
)

from scripts.cloud.optuna_kaggle import main  # noqa: E402

main()
```

- [ ] **Step 3: Create LightGBM entry point**

Create `scripts/cloud/optuna_lightgbm.py`:

```python
"""Entry point: LightGBM Optuna V9 (CPU, 12h)."""

import os
import sys
import subprocess
from pathlib import Path

os.environ["ALICE_MODEL"] = "lightgbm"

for p in [
    Path("/kaggle/input/alice-code"),
    Path("/kaggle/input/datasets/pguillemin/alice-code"),
]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "optuna", "optuna-integration"]
)

from scripts.cloud.optuna_kaggle import main  # noqa: E402

main()
```

- [ ] **Step 4: Create kernel metadata — XGBoost**

Create `scripts/cloud/kernel-metadata-optuna-xgboost.json`:

```json
{
  "id": "pguillemin/alice-optuna-xgboost",
  "title": "alice-optuna-xgboost",
  "code_file": "optuna_xgboost.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": ["pguillemin/alice-code"],
  "competition_sources": [],
  "kernel_sources": ["pguillemin/alice-fe-v8"]
}
```

- [ ] **Step 5: Create kernel metadata — CatBoost**

Create `scripts/cloud/kernel-metadata-optuna-catboost.json`:

```json
{
  "id": "pguillemin/alice-optuna-catboost",
  "title": "alice-optuna-catboost",
  "code_file": "optuna_catboost.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": ["pguillemin/alice-code"],
  "competition_sources": [],
  "kernel_sources": ["pguillemin/alice-fe-v8"]
}
```

- [ ] **Step 6: Create kernel metadata — LightGBM**

Create `scripts/cloud/kernel-metadata-optuna-lightgbm.json`:

```json
{
  "id": "pguillemin/alice-optuna-lightgbm",
  "title": "alice-optuna-lightgbm",
  "code_file": "optuna_lightgbm.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": ["pguillemin/alice-code"],
  "competition_sources": [],
  "kernel_sources": ["pguillemin/alice-fe-v8"]
}
```

- [ ] **Step 7: Ruff format all entry points**

Run: `ruff format scripts/cloud/optuna_xgboost.py scripts/cloud/optuna_catboost.py scripts/cloud/optuna_lightgbm.py`

- [ ] **Step 8: Commit**

```bash
git add scripts/cloud/optuna_xgboost.py scripts/cloud/optuna_catboost.py scripts/cloud/optuna_lightgbm.py
git add scripts/cloud/kernel-metadata-optuna-xgboost.json scripts/cloud/kernel-metadata-optuna-catboost.json scripts/cloud/kernel-metadata-optuna-lightgbm.json
git commit -m "feat(cloud): Optuna V9 entry points + kernel metadata (CPU, 3 models)"
```

---

### Task 4: Upload dataset + push XGBoost canary kernel

**Files:**
- No new files — deployment task

- [ ] **Step 1: Upload alice-code dataset with new files**

Run: `python -m scripts.cloud.upload_all_data`

Verify: `kaggle datasets files pguillemin/alice-code --csv | head -5`
Check that `scripts/cloud/optuna_kaggle.py` is in the upload.

- [ ] **Step 2: Copy XGBoost Optuna metadata**

```bash
cp scripts/cloud/kernel-metadata-optuna-xgboost.json scripts/cloud/kernel-metadata.json
```

- [ ] **Step 3: Pre-flight checklist (kaggle-deployment skill)**

| Check | Command | Expected |
|-------|---------|----------|
| Dataset contains optuna_kaggle.py | `kaggle datasets files pguillemin/alice-code` | Present |
| FE kernel output exists | `kaggle kernels status pguillemin/alice-fe-v8` | COMPLETE |
| No running kernel on same slug | `kaggle kernels status pguillemin/alice-optuna-xgboost` | Not found (first push) |
| enable_gpu=false | Check kernel-metadata.json | false |
| enable_internet=true | Check kernel-metadata.json | true |

- [ ] **Step 4: Push XGBoost canary**

```bash
kaggle kernels push -p scripts/cloud/
git checkout -- scripts/cloud/kernel-metadata.json
```

Log: push time, expected kernel version (v1), git commit hash.

- [ ] **Step 5: Monitor trial 1 benchmark**

```bash
kaggle kernels status pguillemin/alice-optuna-xgboost
# Wait for first trial log line, note time per trial
```

**STOP HERE.** Wait for XGBoost Optuna kernel to complete (or at least
10 trials) before pushing CatBoost/LightGBM. Trial 1 time = benchmark
for the other models.

---

### Task 5: Push CatBoost and LightGBM kernels (after XGBoost benchmark)

**Files:**
- No new files — deployment task

**Prerequisite:** XGBoost Optuna kernel completed with GATE 1 PASS.

- [ ] **Step 1: Download XGBoost results**

```bash
kaggle kernels output pguillemin/alice-optuna-xgboost -p /tmp/optuna-xgb-output/
cat /tmp/optuna-xgb-output/best_params_xgboost.json
```

Verify: gate_pass=true, n_trials_completed >= 10.

- [ ] **Step 2: Check time per trial**

```bash
python -c "
import pandas as pd
h = pd.read_csv('/tmp/optuna-xgb-output/trial_history_xgboost.csv')
print(f'Trials: {len(h)}, mean time: {h.duration_s.mean():.0f}s, median: {h.duration_s.median():.0f}s')
print(f'Estimated 100 trials: {h.duration_s.mean() * 100 / 3600:.1f}h')
"
```

If estimated > 11h for CatBoost/LightGBM: reduce n_trials in config or accept partial study with SQLite resume.

- [ ] **Step 3: Push CatBoost kernel**

```bash
cp scripts/cloud/kernel-metadata-optuna-catboost.json scripts/cloud/kernel-metadata.json
kaggle kernels push -p scripts/cloud/
git checkout -- scripts/cloud/kernel-metadata.json
```

- [ ] **Step 4: Push LightGBM kernel**

```bash
cp scripts/cloud/kernel-metadata-optuna-lightgbm.json scripts/cloud/kernel-metadata.json
kaggle kernels push -p scripts/cloud/
git checkout -- scripts/cloud/kernel-metadata.json
```

CatBoost and LightGBM can run in parallel (independent kernels).

- [ ] **Step 5: Monitor and collect results**

Wait for both to complete. Download outputs:
```bash
kaggle kernels output pguillemin/alice-optuna-catboost -p /tmp/optuna-cb-output/
kaggle kernels output pguillemin/alice-optuna-lightgbm -p /tmp/optuna-lgb-output/
```

Verify gates: `gate_pass=true` for both.

**USER CHECKPOINT:** Present 3 best_params.json to user for review before Training Final.
