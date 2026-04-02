"""ML training constants — feature lists, labels, extensions (ISO 5055 SRP)."""

# fmt: off
CATEGORICAL_FEATURES = ["type_competition", "division", "ligue_code", "jour_semaine",
                        "zone_enjeu_dom", "zone_enjeu_ext"]
CATBOOST_CAT_FEATURES = ["type_competition", "division", "ligue_code", "blanc_titre",
                         "noir_titre", "jour_semaine", "zone_enjeu_dom", "zone_enjeu_ext"]
ADVANCED_CAT_FEATURES = [
    "win_trend_blanc", "win_trend_noir", "draw_trend_blanc", "draw_trend_noir",
    "couleur_preferee_blanc", "couleur_preferee_noir", "zone_enjeu_ext",
    "elo_trajectory_blanc", "elo_trajectory_noir",
    "pressure_type_blanc", "pressure_type_noir", "phase_saison", "regularite_blanc",
    "regularite_noir", "role_type_blanc", "role_type_noir",
    "data_quality_blanc", "data_quality_noir",  # computed by enrich_from_joueurs
    "elo_type_blanc", "elo_type_noir",          # computed by enrich_from_joueurs
    "categorie_blanc", "categorie_noir"]        # computed by enrich_from_joueurs
BOOL_FEATURES = [
    "joueur_fantome_blanc", "joueur_fantome_noir", "ffe_multi_equipe_blanc",
    "ffe_multi_equipe_noir", "est_dans_noyau_blanc", "est_dans_noyau_noir",
    "match_important", "renforce_fin_saison_dom", "renforce_fin_saison_ext",
    "joueur_promu_blanc", "joueur_promu_noir",      # V8 club_level (bool dtype)
    "joueur_relegue_blanc", "joueur_relegue_noir",  # V8 club_level (bool dtype)
    "h2h_exists"]                                    # advanced (bool dtype)
# fmt: on
LABEL_COLUMN = "resultat_blanc"
LEAKY_COLUMNS = {"score_dom", "score_ext"}  # match-score leakage (ISO 5259)
AUC_FLOOR = 0.70
MODEL_EXTENSIONS = {"CatBoost": ".cbm", "XGBoost": ".ubj", "LightGBM": ".txt"}


def default_hyperparameters() -> dict:
    """Optimal hyperparameters for Kaggle CPU training (ADR-003: all models CPU)."""
    # fmt: off
    return {
        "global": {
            "random_seed": 42,
            "early_stopping_rounds": 200,
            "eval_metric": "multi_logloss",
            # Init score shrink: reduce Elo prior dominance so features can express corrections.
            # Alpha < 1 = confidence in Elo prior (Ash & Adams 2020, NeurIPS). Not a hack.
            # v15: models converge in 89-133 iters → prior too strong → alpha=0.7
            "init_score_alpha": 0.7,
        },
        "catboost": {
            "iterations": 50000, "depth": 4, "border_count": 128,
            "learning_rate": 0.005, "l2_leaf_reg": 10, "min_data_in_leaf": 200,
            "random_strength": 3, "bagging_temperature": 1, "model_size_reg": 0.5,
            "rsm": 0.3,  # Feature subsampling — MANDATORY >50 features (v10 bug: 11/177 sans rsm)
            "thread_count": 4, "task_type": "CPU",  # rsm incompatible GPU (CatBoost: pairwise only)
            "use_best_model": True, "loss_function": "MultiClass",
            "random_seed": 42, "verbose": 500, "early_stopping_rounds": 200,
        },
        "xgboost": {
            "n_estimators": 50000, "max_depth": 4, "eta": 0.005,
            "objective": "multi:softprob", "num_class": 3,
            "lambda": 10.0, "alpha": 0.5, "min_child_weight": 50,
            "subsample": 0.7, "colsample_bytree": 0.5,
            "tree_method": "hist", "device": "cpu",  # CPU — no GPU needed for tree models
            "nthread": 4, "seed": 42,
            "early_stopping_rounds": 200, "verbosity": 1,
        },
        "lightgbm": {
            "n_estimators": 50000, "num_leaves": 15, "max_depth": 4,
            "learning_rate": 0.003, "objective": "multiclass", "num_class": 3,
            "reg_lambda": 10.0, "reg_alpha": 0.5, "min_child_samples": 200,
            "min_gain_to_split": 0.01, "subsample": 0.7, "colsample_bytree": 0.5,
            "n_jobs": 4, "random_state": 42,
            "early_stopping_rounds": 200, "verbose": -1,
        },
    }
    # fmt: on
