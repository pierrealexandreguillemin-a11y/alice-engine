# V8 Training Campaign Findings (2026-03-22 → 2026-04-06)

Historique complet de la campagne V8 MultiClass. Consulter pour contexte
sur les decisions prises et bugs rencontres.

## Decisions validees V8
- **Target** : loss=0, draw=1, win=2. TARGET_MAP = {0.0:0, 0.5:1, 1.0:2, 2.0:2}
- **resultat_blanc=2.0** = victoire jeunes FFE (J02 §4.1), mapped to win (PAS forfait)
- **Forfeits** : identifies par `type_resultat`, PAS par resultat_blanc
- **Loss** : CatBoost `MultiClass`, XGBoost `multi:softprob`, LightGBM `multiclass`
- **Eval** : MultiClass log loss + RPS (ordinal) + E[score] MAE
- **Calibration** : Isotonic par classe + renormalisation. Pas de class weights
- **Quality gate** : 15 conditions (log_loss, RPS, E[score], ECE, draw bias)

## Bugs logique corriges dans features
1. clutch_factor : zone_enjeu IN (montee,danger) au lieu de |score_dom-score_ext|<=1
2. score_blancs/noirs : separer home/away
3. Features joueur : stratifier par type_competition
4. Features joueur : rolling 3 saisons
5. Forfaits : filter by type_resultat, recode 2.0->win

## Nouvelles features V8
- 8 draw features (8 cols) : avg_elo, elo_proximity, draw rates
- 8 club/vases features (16 cols) : promu/relegue, team gaps, stabilite
- Toutes rolling, stratifiees par niveau de competition

## Chronologie training
- v1-v3 : echec (path, divergence, hyperparams)
- v5 : premiere victoire (CatBoost 0.886 < Elo 0.92)
- v10 : LightGBM 0.877, decouverte artefact CatBoost importance
- v15 : first clean data (contamination fix, dynamic white advantage, rsm, SHAP)
- v18 : first ALL-PASS (XGBoost 0.574, 15/15 gates)
- Resume v5 : XGBoost CONVERGE 86.5K rounds, test 0.566
- CatBoost v6 : test 0.575, ALL PASS
- LightGBM v7 : test 0.572, ALL PASS

## Decouvertes cles
- 166/177 features importance=0 CatBoost = artefact PredictionValuesChange (pas signal)
- CatBoost rsm=0.3 OBLIGATOIRE >50 features (oblivious trees)
- CatBoost rsm incompatible GPU (pairwise only)
- resultat_blanc=2.0 contamination (62K victoires jeunes exclues a tort)
- Dynamic white advantage +8.5 a +32.4 (pas +35 fixe)
- LightGBM 65K model text = 3h22m startup (pas de format binaire, issue #372)
- CatBoost init_model + Pool(baseline=) = erreur fatale
- CatBoost snapshot exige MEMES params
- LightGBM init_model n_estimators = ADDITIONAL
- TreeSHAP 231K x 85K = heures. Subsample 20K = 26min.
- xgb.train() retourne LAST pas best → EarlyStopping(save_best=True) obligatoire

## Init Score Alpha
- alpha=0.7 dans config global (override ALICE_INIT_ALPHA)
- Temperature scaling sur init logits (Guo 2017)
- V9 Optuna : alpha in [0.3, 0.8] search space joint
- Inference REQUIERT init_scores * alpha + draw_rate_lookup.parquet

## Lacunes versioning (TODO Phase 5)
- Pas de lien commit git <-> version Kaggle dataset <-> version kernel
- Artefacts training non versiones (local seulement)
- DVC recommande mais non implemente
