"""Tests for V9 multiclass Optuna objectives — ISO 29119.

Document ID: ALICE-TEST-OPTUNA-V9
Version: 1.0.0
Tests count: 6
"""

import numpy as np
import pandas as pd


def _make_data(n: int = 500, n_features: int = 10, seed: int = 42):
    """Create synthetic multiclass data matching Alice schema."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, n_features), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(rng.choice([0, 1, 2], size=n, p=[0.42, 0.13, 0.45]))
    init_scores = rng.randn(n, 3) * 0.5
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
        assert study.best_value < 2.0
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
