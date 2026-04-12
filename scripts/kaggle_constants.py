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
    """V9 Training Final hyperparameters (590 configs, 13 kernels — ADR-008/009/010).

    Per-model alpha (ADR-008): LGB=0.1, CB=0.3, XGB=0.5.
    Sources: config/MODEL_SPECS.md, docs/project/V9_HP_SEARCH_RESULTS.md.
    """
    # fmt: off
    return {
        "global": {
            "random_seed": 42,
            "early_stopping_rounds": 200,
            "eval_metric": "multi_logloss",
        },
        "catboost": {
            "init_score_alpha": 0.3,  # ADR-008: oblivious trees, moderate sensitivity (0.0024)
            "iterations": 50000, "depth": 5, "learning_rate": 0.05,
            "l2_leaf_reg": 8.0, "min_data_in_leaf": 200, "random_strength": 2.0,
            "rsm": 0.7,  # Grid v2: 0.7>0.45>0.3. MANDATORY >50 features
            # Tier 1 flags (CatBoost docs: golden features, leaf values)
            # score_function: GPU ONLY (catboost.ai/docs), removed for CPU training
            "border_count": 1024,              # more candidate splits (default CPU=254)
            "leaf_estimation_iterations": 3,   # better leaf values (default=auto)
            "thread_count": 4, "task_type": "CPU",  # rsm incompatible GPU
            "use_best_model": True, "loss_function": "MultiClass",
            "random_seed": 42, "verbose": 500, "early_stopping_rounds": 200,
        },
        "xgboost": {
            "init_score_alpha": 0.5,  # ADR-008: depth-wise, quasi-indifferent (0.001)
            "n_estimators": 50000, "max_depth": 6, "eta": 0.05,
            "objective": "multi:softprob", "num_class": 3,
            "lambda": 4.0, "alpha": 0.01, "min_child_weight": 50,
            "subsample": 0.8, "colsample_bytree": 1.0,
            # Tier 2 draw calibration: same logloss, draw_bias -22%, ECE draw -17%
            "colsample_bynode": 0.7,
            "gamma": 1.0,        # prevents noise splits
            "max_delta_step": 1,  # limits leaf outputs, helps draw imbalance (13.7%)
            "tree_method": "hist", "device": "cpu",
            "nthread": 4, "seed": 42,
            "early_stopping_rounds": 200, "verbosity": 1,
        },
        "lightgbm": {
            "init_score_alpha": 0.1,  # ADR-008: leaf-wise/GOSS, high sensitivity (0.054)
            "n_estimators": 50000, "num_leaves": 15, "max_depth": 8,
            "learning_rate": 0.05, "objective": "multiclass", "num_class": 3,
            "reg_lambda": 4.0, "min_child_samples": 275,
            "subsample": 0.8, "subsample_freq": 1,  # freq>0 activates bagging
            "colsample_bytree": 1.0,  # feature_fraction=1.0
            # Tier 2 confirmed: min_gain_to_split=0, lambda_l1=0 (defaults optimal)
            "n_jobs": 4, "random_state": 42,
            "early_stopping_rounds": 200, "verbose": -1,
        },
    }
    # fmt: on
