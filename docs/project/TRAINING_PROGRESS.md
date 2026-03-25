# ALICE Training Progress - Quality Records

> **Document Type**: Quality Records (QR) - ISO 15289
> **Version**: 1.3.0
> **Date creation**: 4 Janvier 2026
> **Derniere MAJ**: 8 Janvier 2026
> **Responsable**: Claude Code / Pierre
> **Methodologie**: [METHODOLOGIE_ML_TRAINING.md](./METHODOLOGIE_ML_TRAINING.md)

---

## 1. Objectif du document

Suivi de l'avancement du pipeline d'entrainement ALICE conforme ISO 15289.
Ce document trace chaque phase, son statut, et les artefacts produits.

**Audience**: Developpeurs, Data Scientists, Auditeurs qualite

---

## 2. Vue d'ensemble des phases

| Phase | Description | Statut | Duree reelle | Artefacts |
|-------|-------------|--------|--------------|-----------|
| 0 | Preparation | ✅ Complete | - | requirements.txt |
| 1 | Parsing HTML | ✅ Complete | 48 min | echiquiers.parquet, joueurs.parquet |
| 2 | Feature Engineering V7 | ✅ Complete | 2 min | 8 fichiers features/*.parquet |
| 3 | Split temporel | ✅ Complete | inclus Phase 2 | train/valid/test.parquet |
| 4 | Evaluation ML V7 | ✅ Complete (OBSOLÈTE V8) | 5 min | AUC 0.75 (binaire, leakage) |
| 5 | **V8 Feature Engineering** | ✅ Complete | 74 min | **196 cols, Kaggle FE kernel** |
| 6 | **V8 Residual Learning** | ⚠️ BLOQUÉ (calibration) | 11 versions | **Bat Elo (0.887) mais gate 8/9** |
| 7 | Calibration conforme | 🔄 EN COURS | - | Temperature scaling à valider |
| 8 | Feature Store + API | 🔄 A faire | - | Wiring model→CE |
| 9 | ALI (Adversarial Lineup) | 🔄 A faire | - | generate_scenarios() |
| 10 | CE V9 Multi-équipe | 🔄 A faire | - | OR-Tools solver |
| 11 | Deploy (Oracle VM) | 🔄 A faire | - | HTTPS + monitoring |

---

## 2bis. Gaps Critiques Identifies (8 Janvier 2026)

> **Audit**: Analyse conformite standards industrie + ISO ML
> **Documentation complete**: [METHODOLOGIE_ML_TRAINING.md](./METHODOLOGIE_ML_TRAINING.md)

### Gaps Bloquants Production

| ID | Gap | Severite | Fichier | Effort |
|----|-----|----------|---------|--------|
| **GAP-001** | Pas de script d'entrainement | CRITIQUE | evaluate_models.py | 2j |
| **GAP-002** | Data leakage features | CRITIQUE | feature_engineering.py:588 | 1j |
| **GAP-003** | Pas d'experiment tracking | CRITIQUE | N/A | 2j |
| **GAP-004** | Hyperparametres hardcodes | HAUTE | evaluate_models.py:143 | 1j |
| **GAP-005** | Inference service stub | CRITIQUE | inference.py:54 | 1j |

### Detail GAP-002: Data Leakage

```python
# PROBLEME ACTUEL (feature_engineering.py:588-598)
club_reliability = extract_club_reliability(df)  # TOUT le dataset!
# PUIS split temporel = features utilisent donnees futures

# CORRECTION REQUISE
train, valid, test = temporal_split(df)  # Split D'ABORD
train_features = compute_features(train)  # Features PAR split
```

### Conformite Standards

| Standard | Actuel | Cible |
|----------|--------|-------|
| MLOps Maturity | Level 0 (manuel) | Level 1 |
| ISO/IEC 42001 | 40% | 80% |
| ISO/IEC 5259 (Data Quality) | 50% | 90% |
| Reproducibility | Tier Bronze | Tier Silver |

### Plan Correctif

| Phase | Taches | Effort | Deadline |
|-------|--------|--------|----------|
| **Phase 1** | Fix critiques (GAP-001 to 005) | 7j | Semaine 2 |
| **Phase 2** | MLOps (Optuna, CV, metrics) | 5j | Semaine 3 |
| **Phase 3** | Production (monitoring, CI/CD) | 4j | Semaine 4 |

---

## 3. Detail par phase

### Phase 0 : Preparation ✅

**Date**: 3 Janvier 2026

| Tache | Statut | Notes |
|-------|--------|-------|
| Dataset copie | ✅ | C:/Dev/Alice-Engine/dataset_alice/ |
| Python 3.13 | ✅ | Installe |
| Dependencies ML | ✅ | catboost, xgboost, lightgbm |

### Phase 1 : Parsing HTML → Parquet ✅

**Date**: 3 Janvier 2026
**Script**: `scripts/parse_dataset.py`
**Documentation**: `docs/project/BILAN_PARSING.md`

| Metrique | Valeur |
|----------|--------|
| Fichiers HTML | 85,672 |
| Groupes parses | 13,935 |
| Echiquiers extraits | 1,736,490 |
| Joueurs | 66,208 |
| Duree | 48 minutes |
| Taille sortie | 37.2 MB |

**Artefacts**:
- `data/echiquiers.parquet` (34.2 MB)
- `data/joueurs.parquet` (3.0 MB)

### Phase 2 : Feature Engineering ✅

**Date**: 4 Janvier 2026
**Script**: `scripts/feature_engineering.py`

| Feature | Lignes | Description |
|---------|--------|-------------|
| club_reliability | 28,162 | taux_forfait, fiabilite_score |
| player_reliability | 131,550 | taux_presence, joueur_fantome |
| player_monthly | 128,257 | dispo_mois_1..12 |
| player_form | 65,344 | forme_recente (5 matchs) |
| player_board | 130,294 | echiquier_moyen |

**Artefacts**: `data/features/*.parquet`

### Phase 3 : Split temporel ✅

**Date**: 4 Janvier 2026

| Set | Saisons | Echiquiers | % |
|-----|---------|------------|---|
| Train | 2002-2022 | 1,139,819 | 81% |
| Valid | 2023 | 70,647 | 5% |
| Test | 2024-2026 | 197,843 | 14% |

**Artefacts**: `data/features/train.parquet`, `valid.parquet`, `test.parquet`

### Phase 4 : Evaluation ML ✅

**Date**: 4 Janvier 2026
**Script**: `scripts/evaluate_models.py`
**Documentation**: `docs/project/ML_EVALUATION_RESULTS.md`

| Modele | AUC (test) | Accuracy | Train (s) | Statut |
|--------|-----------|----------|-----------|--------|
| **CatBoost** | **0.7527** | **68.30%** | 292.9 | Retenu |
| LightGBM | 0.7506 | 68.22% | 8.5 | Backup |
| XGBoost | 0.7384 | 67.44% | 10.0 | Backup |

**Decision**: CatBoost retenu (+1.4% AUC vs XGBoost)

**Artefacts**: `data/ml_evaluation_results.csv`

#### Interpretation des resultats (MAJ 8 Janvier 2026)

| Metrique | Valeur | Interpretation |
|----------|--------|----------------|
| AUC 0.7527 | "Bon" | Echelle: 0.5=hasard, 0.7=acceptable, 0.8=tres bon |
| Accuracy 68% | Acceptable | 32% erreurs = 1 prediction sur 3 fausse |
| Ecart vs LightGBM | +0.21% | Faible, quasi ex-aequo |
| Ecart vs XGBoost | +1.43% | Significatif |

**Limites identifiees**:
- AUC < 0.80 (cible) → hyperparameter tuning necessaire
- Dataset ancien (2002-2022) → patterns potentiellement obsoletes
- Ecart faible vs LightGBM → choix justifie par categories natives uniquement

**Actions correctives**: Phase 5 (Optuna) + features supplementaires

### Phase 5 : Hyperparameter Tuning 🔄

**Statut**: A faire
**Outil prevu**: Optuna

| Hyperparametre | Plage | Actuel |
|----------------|-------|--------|
| depth | [4, 6, 8, 10] | 6 |
| learning_rate | [0.01, 0.05, 0.1] | 0.05 |
| iterations | [500, 1000, 2000] | 500 |
| l2_leaf_reg | [1, 3, 5, 10] | 3 |

### Phase 6 : Entrainement final 🔄

**Statut**: A faire

| Tache | Statut |
|-------|--------|
| Entrainer CatBoost avec best params | 🔄 |
| Sauvegarder modele (.cbm) | 🔄 |
| Feature importance | 🔄 |
| Validation croisee | 🔄 |

### Phase 7 : Deploiement 🔄

**Statut**: A faire

| Tache | Statut |
|-------|--------|
| API FastAPI | 🔄 |
| Endpoint /predict | 🔄 |
| Tests integration | 🔄 |
| Documentation API | 🔄 |

---

## 4. Integration regles FFE dans ML (Phase 4 bis) 🔄

> **Statut**: A faire
> **Priorite**: Haute - Impact attendu sur AUC: +5-10%
> **Documentation**: `docs/requirements/REGLES_FFE_ALICE.md`
> **Implementation**: `scripts/ffe_rules_features.py`

### 4.1 Dataset de regles disponible

Les regles FFE sont **documentees et implementees** mais **non integrees** dans l'entrainement ML.

| Composant | Fichier | Lignes | Statut |
|-----------|---------|--------|--------|
| Documentation | `REGLES_FFE_ALICE.md` | 1,153 | ✅ Complet |
| Implementation | `ffe_rules_features.py` | 845 | ✅ Complet |
| Tests | `test_ffe_rules_features.py` | 442 | ✅ 66 tests |
| **Integration ML** | `feature_engineering.py` | - | ⚠️ **A faire** |

### 4.2 Features reglementaires disponibles

```python
# Dans ffe_rules_features.py - PRETS A UTILISER
FeaturesReglementaires(TypedDict):
    joueur_brule: bool           # Joueur grille dans equipe superieure
    matchs_avant_brulage: int    # 0-3 matchs restants avant brulage
    est_dans_noyau: bool         # Fait partie du noyau equipe
    pct_noyau_equipe: float      # % noyau actuel de l'equipe
    joueur_mute: bool            # Transfert d'un autre club
    zone_enjeu_equipe: str       # montee/danger/mi_tableau/descente
```

### 4.3 Plan d'integration ML

#### Etape 1: Calcul features sur dataset historique

```python
# A ajouter dans feature_engineering.py
def extract_ffe_regulatory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque ligne (match joue):
    1. Construire historique_brulage depuis matchs precedents
    2. Construire historique_noyau depuis matchs precedents
    3. Calculer features reglementaires pour chaque joueur
    """
    from scripts.ffe_rules_features import (
        est_brule,
        matchs_avant_brulage,
        get_noyau,
        calculer_pct_noyau,
        calculer_zone_enjeu,
    )
    # Implementation...
```

#### Etape 2: Features par competition

| Type Competition | Features specifiques |
|------------------|---------------------|
| A02 (National) | `joueur_brule`, `pct_noyau`, `nb_mutes`, `quota_nat` |
| F01 (Feminin) | `joueur_brule` (seuil=1), `joueuse_fr_presente` |
| C01 (Coupe) | Aucune contrainte specifique |
| C03 (Loubatiere) | `tous_elo_sous_1800` |
| C04 (Parite) | `parite_2h_2f`, `elo_total_equipe` |
| J02 (Jeunes) | `joueur_brule` (seuil=4), ordre par age |
| REG/DEP | `noyau_2_absolu` |

#### Etape 3: Reentrainement avec features regles

| Feature | Type | Impact attendu |
|---------|------|----------------|
| `joueur_brule` | bool | **Eleve** - Exclut joueurs non-eligibles |
| `matchs_avant_brulage` | int 0-3 | **Moyen** - Probabilite utilisation strategique |
| `pct_noyau_equipe` | float | **Moyen** - Contrainte composition |
| `zone_enjeu` | cat | **Eleve** - Motivation equipe |
| `joueur_mute` | bool | **Faible** - Rare (<5% joueurs) |

### 4.4 Hypothese d'amelioration

```
AUC actuel:     0.7527 (sans features regles)
AUC cible:      0.80+  (avec features regles + tuning)

Justification:
- Un joueur brule NE PEUT PAS jouer → prediction certaine
- Zone enjeu "montee" → meilleurs joueurs alignes
- Zone enjeu "descente" → renforcement ou abandon
- Noyau insuffisant → composition contrainte
```

### 4.5 Validation de l'integration

| Test | Critere | Methode |
|------|---------|---------|
| Coherence | Features non-nulles | `assert df['joueur_brule'].notna().all()` |
| Distribution | % brules ~10-15% | Statistiques descriptives |
| Correlation | `joueur_brule` ↔ non-selection | Test chi2 |
| Impact AUC | Delta significatif | A/B test avec/sans features |

---

## 5. Features complementaires (Phase 2 bis)

### 5.1 Objectif equipe par saison

**Concept**: Chaque equipe a un objectif de fin de saison qui influence les compositions.

| Objectif | Description | Impact sur compo |
|----------|-------------|------------------|
| Titre/Montee | Viser 1ere ou 2e place | Align meilleurs joueurs |
| Maintien | Eviter descente | Renforcement fin saison |
| Mi-tableau | Pas d'enjeu fort | Rotation, repos joueurs |
| Descente probable | Situation critique | Desespoir ou abandon |

**Features a extraire**:
- `objectif_equipe`: calcule par position au classement
- `ecart_objectif`: distance a la zone danger/titre
- `pression_match`: enjeu selon classement et adversaire
- `renforcement_saison`: detection joueurs transferes intra-club

### 5.2 Effet vases communiquants

**Concept**: Dans un club multi-equipes, les joueurs peuvent migrer entre equipes.
Un renforcement d'une equipe = affaiblissement d'une autre.

**Features**:
- `joueur_promu`: joueur monte d'une equipe inferieure
- `joueur_relegue`: joueur descend d'une equipe superieure
- `stabilite_effectif`: % joueurs identiques vs saison N-1
- `elo_moyen_evolution`: evolution Elo equipe sur la saison

---

## 6. Decisions techniques (ADR)

### ADR-001: Choix CatBoost

**Date**: 4 Janvier 2026
**Statut**: Accepte

**Contexte**: Besoin de choisir un modele gradient boosting pour ALI.

**Decision**: CatBoost retenu.

**Raisons**:
1. Meilleur AUC (+1.4% vs XGBoost)
2. Gestion native categories (division, ligue, titre)
3. Moins de tuning requis
4. Inference rapide (<1ms/prediction)

**Consequences**:
- Temps train plus long (5 min vs 10s)
- Dependance catboost>=1.2

### ADR-002: Split temporel

**Date**: 4 Janvier 2026
**Statut**: Accepte

**Decision**: Split 2002-2022 / 2023 / 2024-2026.

**Raisons**:
1. Eviter data leakage temporel
2. Tester sur donnees "futures"
3. Validation sur saison complete (2023)

---

## 7. Conformite ISO

| Norme | Application | Statut |
|-------|-------------|--------|
| ISO 15289 | Structure document QR | ✅ |
| ISO 25010 | Fiabilite, Maintenabilite | ✅ |
| ISO 25012 | Qualite donnees (Elo=0, forfaits) | ✅ |
| ISO 29119 | Tests (split temporel) | ✅ |
| ISO 12207 | Cycle de vie (phases tracees) | ✅ |

---

## 8. Historique des modifications

| Version | Date | Auteur | Modifications |
|---------|------|--------|---------------|
| 1.0.0 | 2026-01-04 | Claude Code | Creation initiale |
| 1.1.0 | 2026-01-08 | Claude Code | Ajout interpretation Phase 4, limites performances |
| 1.2.0 | 2026-01-08 | Claude Code | Ajout section 4 "Integration regles FFE dans ML" |
| 1.3.0 | 2026-01-08 | Claude Code | Ajout section 2bis "Gaps Critiques", lien methodologie |
| 2.0.0 | 2026-03-25 | Claude Code | V8 MultiClass complet — résultats v3→v11, residual learning |

---

## 9. V8 MultiClass 3-way — Residual Learning (mars 2026)

### 9.1 Contexte

V7 (janvier) était binaire avec 4 bugs critiques (leakage, target, calibration, architecture).
V8 remplace par multiclass W/D/L (loss=0, draw=1, win=2) + residual learning sur Elo.

**Spec** : `docs/superpowers/specs/2026-03-21-multiclass-v8-design.md`
**Postmortem** : `docs/postmortem/2026-03-22-training-v8-divergence.md`
**Roadmap** : `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md`

### 9.2 Feature Engineering V8 — COMPLET

**Kernel** : `pguillemin/alice-fe-v8` (Kaggle, COMPLETE)
**Date** : 2026-03-21

| Métrique | Valeur |
|----------|--------|
| Colonnes | 196 (177 numériques après encodage) |
| Train rows | 1,090,150 (⚠️ excluait 2.0=victoires jeunes à tort, voir §9.11) |
| Valid rows | 67,824 |
| Test rows | 221,807 |
| Durée FE | 74 min (P100 CPU) |

Catégories : match context (12), player strength (10), player form W/D/L (20),
draw priors (8), presence (16), pressure (6), H2H (4), standings (16),
club behavior (16), vases communiquants (16), FFE regulatory (20),
Elo trajectory (4), composition strategy (8).

**166/177 features ont importance 0** dans v3 (sans residual). Signal concentré dans
~11 features : diff_elo, elo_proximity, win/draw rates, expected_score_recent, home advantage.

### 9.3 Training — Residual Learning

**Principe** : compute_elo_init_scores() → log-odds centrés → Pool(baseline=) / DMatrix.set_base_margin() / fit(init_score=). Les modèles apprennent les CORRECTIONS à l'Elo, pas la prédiction from-scratch.

**API bugs contournés (vérifiés contre GitHub issues) :**
- CatBoost #1554 : predict_proba(Pool(baseline=)) non-normalisé → raw+softmax
- XGBoost #5288 : XGBClassifier.fit(base_margin=) cassé multiclass → native xgb.train()+DMatrix
- LightGBM #1978 : predict() ne supporte pas init_score → raw_score+softmax

### 9.4 Résultats Kaggle (11 versions)

| Version | Date | Features | Init | LR | CB best | XGB best | LGB best | Gate | Commit |
|---------|------|----------|------|----|---------|----------|----------|------|--------|
| v1 | 03-22 | 177 | Non | 0.03 | — | — | — | ERROR (path) | affcb73 |
| v2 | 03-22 | 177 | Non | 0.03 | 0.940 | diverge | 0.967 | FAIL (ll) | 179a027 |
| v3 | 03-22 | 177 | Non | 0.03 | 0.926 | ~0.98 | 0.935 | FAIL (ll) | 1d0289d |
| v4 | 03-23 | 177 | Oui | 0.03 | — | — | — | ERROR (cache) | b114d1c |
| v5 | 03-23 | 177 | Oui | 0.03 | **0.886** | cassé | **0.885** | FAIL (draw_bias) | b114d1c |
| v6 | 03-23 | 11 | Bugué | 0.03 | 0.906 | cassé | 0.934 | FAIL (ece) | bf0392e |
| v7 | 03-23 | — | — | — | — | — | — | Stoppé | — |
| v8 | 03-23 | 13 | Oui | 0.03 | 0.886 | cassé | 0.888 | FAIL (draw_bias) | 574e0d1 |
| v9 | 03-24 | 177 | Oui | 0.005 | 0.888 | **0.889** | 0.887 | FAIL (draw_bias) | 0fbfd1f |
| v10 | 03-24 | 177 | Oui | 0.005 | 0.888 | 0.889 | 0.887 | FAIL (es_mae) RÉGRESSION | 6e1fb00 |
| v11 | 03-25 | 177 | Oui | 0.005 | — | — | — | Annulé | 39e4d75 |

### 9.5 Quality Gate 9 conditions — état par version

| Cond | Description | v9 (raw) | v10 (isotonic) |
|------|-------------|----------|----------------|
| 1 | log_loss < naive | PASS | PASS |
| 2 | log_loss < Elo | PASS | PASS |
| 3 | RPS < naive | PASS | PASS |
| 4 | RPS < Elo | PASS | PASS |
| 5 | Brier < naive | PASS | PASS |
| 6 | E[score] MAE < Elo | PASS | **FAIL** (régression) |
| 7 | ECE < 0.05 per class | Non évalué sur test calibré | PASS |
| 8 | draw_calibration_bias < 2% | **FAIL** | PASS |
| 9 | mean_p_draw > 1% | PASS | PASS |

### 9.6 Problème ouvert : calibration

**Isotonic per-class + renormalization** (v10) corrige draw_bias mais dégrade E[score].
Confirmé par ICML 2025 (arXiv:2512.09054) : "re-normalization to sum to one might compromise calibration."

**Temperature scaling** (v11, annulé) : 1 paramètre T, softmax(logits/T), préserve ratios.
Non vérifié sur données ALICE.

**Littérature :**
- Walsh & Joshi 2024 : calibration > accuracy pour decision-making (ROI +34.69%)
- Ramezani & Dinh 2025 (arXiv:2505.02170) : FPL predict-then-optimize = architecture ALICE
- Guo et al. 2017 : temperature scaling pour GBMs
- ICML 2025 : NA-FIR (normalization-aware isotonic) = state-of-the-art multiclass

**Statut** : BLOQUÉ. Le modèle bat l'Elo mais la calibration qui préserve E[score]
et corrige draw_bias n'est pas encore validée.

### 9.7 Acquis techniques (réutilisables)

| Acquis | Fichier | Tests |
|--------|---------|-------|
| compute_elo_init_scores() | scripts/baselines.py | 3 tests |
| compute_init_scores_from_features() | scripts/baselines.py | — |
| predict_with_init() per-library | scripts/kaggle_metrics.py | 4 tests |
| XGBWrapper (Booster→sklearn) | scripts/kaggle_metrics.py | — |
| Temperature scaling + isotonic | scripts/kaggle_diagnostics.py | — |
| Quality gate 9 conditions | scripts/kaggle_metrics.py | 7 tests |
| ML architecture SVGs (3) | scripts/generate_ml_graphs.py | — |
| Hyperparams YAML synced | config/hyperparameters.yaml | 2 tests |

### 9.8 Découverte v10 : feature importance artifact (2026-03-25)

**166/177 features à importance 0 = artefact CatBoost `PredictionValuesChange`.**

| Modèle | Méthode importance | Features non-zero | Biaisée ? |
|--------|-------------------|-------------------|-----------|
| CatBoost | PredictionValuesChange | **11** | OUI — oblivious trees + residual |
| XGBoost | Gain | **109** | Partiel — mais bien plus fiable |
| LightGBM | Split count | **50** | Partiel — conservateur |

Root cause CatBoost : pas de `rsm` (feature subsampling). Oblivious trees depth=4 =
4 splits/arbre, toujours les mêmes features. Fix : `rsm=0.3` + SHAP natif.

### 9.9 Plan Phase 1b : SHAP + Calibration

Plan : `docs/superpowers/plans/2026-03-25-shap-feature-validation.md`

1. SHAP natif CatBoost + permutation importance sur les 3 modèles v10
2. Feature validation par concordance cross-modèles
3. CatBoost `rsm=0.3` fix + retrain
4. Temperature scaling (Guo 2017) remplace isotonic renorm
5. Quality gate 9/9 → push HF Hub

### 9.11 Data Contamination Finding (2026-03-25)

**Découverte** : `resultat_blanc=2.0` était traité comme "forfait" et exclu de tout (target + features) depuis la spec V8. En réalité :

- **2.0 = victoire jeunes FFE** (62K parties réelles) — J02 §4.1 : victoire = 2 pts de partie sur éch. non-U10
- Les vrais forfeits sont dans la colonne `type_resultat` : forfait_blanc (43K), forfait_noir (42K), double_forfait (3K), non_joue (209K) — tous ont resultat_blanc=0.0 ou 1.0, PAS 2.0
- **295K forfeits sont INCLUS** dans le training comme résultats réels (resultat_blanc 0.0/1.0)
- Le parser est correct (echiquiers.parquet fidèle au HTML). Le problème est l'interprétation downstream

**Impact** :
- Tous les modèles v1-v13 sont entraînés sur données contaminées (62K wins exclus + 295K forfeits inclus)
- Les résultats SHAP de v13 ne sont PAS fiables
- Les métriques draw_rate, win_rate, etc. dans les features sont biaisées
- Le FE kernel + training kernel doivent être re-run après le fix

**Fix requis** :
- `scripts/features/helpers.py` : filter par `type_resultat` au lieu de `resultat_blanc==2.0`
- `scripts/kaggle_trainers.py` : TARGET_MAP = {0.0:0, 0.5:1, 1.0:2, 2.0:2}
- Re-run FE kernel puis training kernel

**Postmortem** : `docs/postmortem/2026-03-25-resultat-blanc-2.0-bug.md`

### 9.10 Lacunes identifiées

| Lacune | Sévérité | Action |
|--------|----------|--------|
| Pas de tracking commit↔dataset↔kernel | ~~HAUTE~~ | ✅ FAIT (commit f691e04) — SHA-256 + JSONL log |
| Artefacts training non versionnés (local) | HAUTE | DVC ou HF Hub systématique |
| IMPLEMENTATION_STATUS.md obsolète (jan 2026) | MOYENNE | Mettre à jour |
| ISO_COMPLIANCE_TODOS.md dit 100% (faux pour V8) | ~~HAUTE~~ | ✅ CORRIGÉ (2026-03-25) |
| Outputs Kaggle v9/v10 perdus (pas sauvés) | ~~CRITIQUE~~ | ✅ v10 récupéré via web UI (42 MB) |
| Feature importance SHAP manquante | HAUTE | Task 1 plan SHAP |
| CatBoost `rsm` manquant | HAUTE | Task 3 plan SHAP |
| Calibration (temperature scaling) | HAUTE | Task 4 plan SHAP |

---

*Document genere selon ISO 15289 - Quality Records*
