"""ML training constants — feature lists, labels, extensions (ISO 5055 SRP)."""

# fmt: off
CATEGORICAL_FEATURES = ["type_competition", "division", "ligue_code", "jour_semaine"]
CATBOOST_CAT_FEATURES = ["type_competition", "division", "ligue_code", "blanc_titre",
                         "noir_titre", "jour_semaine", "zone_enjeu_dom", "zone_enjeu_ext"]
ADVANCED_CAT_FEATURES = [
    "win_trend_blanc", "win_trend_noir", "draw_trend_blanc", "draw_trend_noir",
    "couleur_preferee_blanc", "couleur_preferee_noir", "zone_enjeu_ext",
    "elo_trajectory_blanc", "elo_trajectory_noir",
    "pressure_type_blanc", "pressure_type_noir", "phase_saison", "regularite_blanc",
    "regularite_noir", "role_type_blanc", "role_type_noir"]
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
