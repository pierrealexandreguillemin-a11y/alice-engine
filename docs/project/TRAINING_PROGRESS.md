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
| 5 | **V8 Feature Engineering** | ✅ Complete | 74 min | **201 cols (FE v4), Kaggle FE kernel** |
| 6 | **V8 Residual Learning** | ✅ Complete | 18 versions | **XGBoost v18: 15/15 gates PASS** |
| 7 | Calibration conforme | ✅ Complete | — | Temperature T=0.928, ECE<2% |
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
| **3.0.0** | **2026-03-30** | **Claude Code** | **v18 XGBoost 15/15 PASS — first all-gate pass, -34% vs Elo** |

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
| v12-v14 | — | — | — | — | — | — | — | Skipped (data contamination fix) | — |
| v15 | 03-26 | 201 | Oui α=0.7 | 0.005 | — | — | — | Clean data (running) | 56a58e7 |
| v16 | 03-26 | 201 | Oui α=0.7 | 0.005 | 0.869 | — | — | Prior trop fort (89 iters) | — |
| v17 | 03-29 | 201 | Oui α=0.7 | 0.005 | 0.574 | 0.549 | 0.564 | TIMEOUT (3 modèles, 0 sauvegardés) | — |
| **v18** | **03-30** | **201** | **Oui α=0.7** | **0.005** | **—** | **0.574** | **—** | **15/15 PASS** | **4d481fd** |

### 9.5 Quality Gate — état par version

| Gate | Description | v9 (raw) | v10 (isotonic) | **v18 (temperature)** |
|------|-------------|----------|----------------|-----------------------|
| T1 | log_loss < Elo | PASS | PASS | **PASS** (0.574) |
| T2 | RPS < Elo | PASS | PASS | **PASS** (0.090) |
| T3 | E[score] MAE < Elo | PASS | **FAIL** | **PASS** (0.250) |
| T4 | ECE < 0.05 per class | — | PASS | **PASS** (1.0-1.6%) |
| T5 | draw_bias < ±2% | **FAIL** | PASS | **PASS** (+1.6%) |
| T6 | mean_p_draw > 1% | PASS | PASS | **PASS** (14.2%) |
| T7-T8 | NaN/Inf + sum=1 | PASS | PASS | **PASS** |
| T9 | >5 features gain>0 | — | — | **PASS** (197) |
| T10-T12 | Surfit + diag + report | — | — | **PASS** |
| **Total** | | 7/9 | 8/9 | **15/15** |

### 9.6 Calibration — RÉSOLU (v18)

**Temperature scaling T=0.928** validé sur XGBoost v18.

Historique :
- v10 : Isotonic per-class + renormalization dégradait E[score] (ICML 2025 arXiv:2512.09054)
- v18 : Temperature scaling (Guo 2017) — 1 paramètre T, softmax(logits/T), préserve ratios
- Résultat : ECE < 2% toutes classes, draw_bias +1.6%, E[score] MAE 0.250 < Elo 0.372

**T=0.928 (mild)** confirme que le residual learning α=0.7 produit un modèle déjà bien calibré.
La temperature ne fait qu'un ajustement fin.

**Littérature confirmée :**
- Guo et al. 2017 : temperature scaling = simple + efficace pour GBMs ✅
- Walsh & Joshi 2024 : calibration > accuracy pour decision-making ✅ (Alice = predict-then-optimize)
- ICML 2025 : isotonic renorm problématique ✅ (confirmé par nos résultats v10)

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

### 9.12 v15: First Clean Data Training (2026-03-26)

**Date** : 2026-03-26
**Status** : Running on Kaggle T4

First training on clean data after fixing the resultat_blanc=2.0 contamination.
All previous versions (v1-v13) trained on contaminated data (295K forfeits included, 62K wins excluded).

**Fixes combined in v15 :**

| Fix | Commit | Impact |
|-----|--------|--------|
| Data contamination (2.0 bug) | 56a58e7 | 295K forfeits removed, 62K wins restored. FE v2 verified |
| Dynamic white advantage | cc8f2db | +35 replaced by Elo-level lookup (+8.5 to +32.4), verified 1.44M FFE games |
| CatBoost rsm=0.3 | 378b97a | Feature subsampling mandatory >50 features. rsm incompatible GPU → CPU forced |
| SHAP integrated | — | CatBoost native SHAP + manual permutation in training kernel (no separate SHAP kernel) |
| Dual calibration | 37ad4ec | Temperature scaling vs isotonic compared in same kernel, winner picked by quality gate |

**FE v2 verification (post data fix) :**
- 49K resultat_blanc=2.0 included (victoires jeunes)
- 0 forfeits in training data
- Feature changes: echiquier_moyen -0.42, k_coefficient +0.33

**Residual tuning options for v16+ (if needed) :**
- Shrink init_scores (multiply by 0.5-0.8)
- Increase lr/depth to compensate
- Partial init (Elo init for W/L only, flat prior for D)

### 9.13 v18: XGBoost — FIRST ALL-PASS (2026-03-30)

**Date** : 2026-03-30
**Status** : COMPLETE — 15/15 quality gates PASS
**Kernel** : `pguillemin/alice-eval-xgboost` (eval kernel, checkpoint-based)
**Commit** : 4d481fd

**Architecture** : XGBoost trained in `alice-train-xgboost` (50K rounds, early_stopping=200),
checkpoint uploaded as dataset `alice-xgboost-checkpoint`, evaluated in separate eval kernel.

**Metrics (test set, n=231,532) :**

| Metric | XGBoost v18 | Elo baseline | Naive baseline | Δ vs Elo |
|--------|-------------|--------------|----------------|----------|
| log_loss | **0.5742** | 0.8751 | 0.9839 | **-34.4%** |
| RPS | **0.0901** | 0.1388 | 0.1638 | **-35.2%** |
| E[score] MAE | **0.2499** | 0.3721 | — | **-32.8%** |
| Brier | **0.3454** | — | 0.6015 | — |
| Accuracy | 74.4% | — | — | — |
| F1-macro | 69.6% | — | — | — |

**Calibration (temperature scaling T=0.928) :**

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| ECE loss | 1.03% | < 5% | PASS |
| ECE draw | 1.64% | < 5% | PASS |
| ECE win | 1.04% | < 5% | PASS |
| Draw bias | +1.64% | < ±2% | PASS |
| mean_p_draw | 14.2% | > 1% | PASS |
| Recall draw | 54.4% | — | Info |
| Recall loss | 75.8% | — | Info |
| Recall win | 78.5% | — | Info |

**Features** : 197/201 with gain > 0 (XGBoost utilise quasi toutes les features).

**Key findings :**
- Temperature scaling mild (T=0.928) — model already well-calibrated from residual learning
- Residual α=0.7 + split fix + data contamination fix + differential features = massive improvement
- v10 best was LightGBM 0.877, v18 XGBoost 0.574 = **-34.5% log_loss improvement**
- RPS 0.090 = 2.3× better than best football ML models (~0.206)
- 74.4% accuracy on 3-class is misleading — log_loss/RPS/E[score] MAE are the real KPIs

**Quality gates T1-T12 + T1b/T2b/T3b :**

| Gate | Check | Value | Threshold | Status |
|------|-------|-------|-----------|--------|
| T1 | log_loss < Elo | 0.574 | < 0.875 | PASS |
| T1b | log_loss < Naive | 0.574 | < 0.984 | PASS |
| T2 | RPS < Elo | 0.090 | < 0.139 | PASS |
| T2b | RPS < Naive | 0.090 | < 0.164 | PASS |
| T3 | E[score] MAE < Elo | 0.250 | < 0.372 | PASS |
| T3b | Brier < Naive | 0.345 | < 0.601 | PASS |
| T4 | ECE < 0.05 all | 1.0-1.6% | < 5% | PASS |
| T5 | Draw bias < ±2% | +1.6% | < ±2% | PASS |
| T6 | mean_p_draw > 1% | 14.2% | > 1% | PASS |
| T7 | No NaN/Inf | OK | 0 | PASS |
| T8 | Probas sum=1 | OK | < 1e-6 | PASS |
| T9 | >5 features gain>0 | 197 | > 5 | PASS |
| T10 | Train-test gap < 0.05 | OK | < 0.05 | PASS |
| T11 | Reliability diagram | Visual | Near diagonal | PASS |
| T12 | Report RPS+ll | Both | — | PASS |

### 9.14 Resume XGBoost — Continuation Training (2026-03-31 → 04-01)

**Objectif** : continuer le training XGBoost au-delà de 50K rounds (v18).

| Version | Date | Checkpoint | eta | Rounds | val_mlogloss | Status |
|---------|------|-----------|-----|--------|-------------|--------|
| Resume v1 (brouillon) | 03-31 | 50K | 0.005? | +50K=100K | 0.5143 | COMPLETE — pas ISO (manque SHAP, diagnostics) |
| Resume v2 (ISO) | 03-31 | 50K | 0.01 | +35K=85K | 0.5127 | TIMEOUT — permutation 4h avant gates |
| Resume v3 | 03-31 | 50K | 0.01 | +35K=85K | 0.5127 | TIMEOUT — TreeSHAP 231K rows |
| Resume v4 | 03-31 | 50K | 0.01 | +35K=85K | 0.5127 | TIMEOUT — TreeSHAP 231K rows (avec checkpoints) |
| **Resume v5** | **04-01** | **85K** | **0.005** | **+1.7K=86.5K** | **0.5126** | **COMPLETE — ALL GATES PASS** |

**Resume v5 Results (2026-04-01) :**

| Metric | v18 (50K) | v5 Resume (86K) | Delta |
|--------|-----------|-----------------|-------|
| test_log_loss | 0.574 | **0.566** | -1.4% |
| test_rps | 0.090 | **0.089** | -1.1% |
| test_es_mae | 0.250 | **0.247** | -1.2% |
| test_accuracy | — | 0.746 | — |
| test_f1_macro | — | 0.699 | — |
| Temperature T | 0.928 | **0.971** | closer to 1.0 |
| Features active | 197/201 | **197/197** | all active |
| ECE (loss/draw/win) | — | 0.011/0.016/0.009 | all < 0.05 |
| P(draw) bias | — | 1.46% | within ±2% |
| Recall (loss/draw/win) | — | 76%/55%/79% | — |
| Total time | — | 4.7h | — |

**TreeSHAP Top 5** : `draw_rate_home_dom` (1.34), `saison` (0.71), `draw_rate_noir` (0.70), `draw_rate_blanc` (0.67), `win_rate_home_dom` (0.45)

**Artefacts** : `reports/v8_xgboost_v5_resume/` (model .ubj, calibrators, SHAP, predictions, diagnostics)

**Findings :**
- v2-v4 : 3 timeouts consécutifs. Root cause : compute post-training non budgété
  - Permutation 197×5×17s = 4h39m (v2)
  - TreeSHAP 231K×85K trees = ~5h (v3, v4)
  - Fix : subsample 20K, benchmark 26min/85K trees
- `xgb.train()` retourne last iteration pas best → `EarlyStopping(save_best=True)`
- `reshape(N,-1,3)` scramblait axes SHAP → auto-detect `(N,C,F+1)` vs `(N,F+1,C)`
- eta=0.01 early-stop à 85K → eta=0.005 pour finer steps → +1.7K rounds seulement
- **Le modèle est à son optimum** : 0.51269 → 0.51255 (Δ=0.00014 en 1.7K rounds)
- HF Hub push failed (Kaggle Secrets connection error) — manual push needed

**Conclusion** : XGBoost CONVERGÉ à 86K rounds, val=0.5126, test=0.566. All gates PASS.

### 9.15 CatBoost v3 — Single-Model (2026-04-02→03)

| Metric | Valid | Test | vs Elo |
|--------|-------|------|--------|
| log_loss | 0.5468 | **0.5895** | -39.6% |
| RPS | 0.0864 | 0.0919 | — |
| E[score] MAE | 0.2483 | 0.2553 | — |
| Temperature | — | T=0.935 | — |
| Rounds | 50K (no early stop) | — | 8h11m |

SHAP top: draw_rate_home_dom (1.28), draw_rate_noir (0.63), draw_rate_blanc (0.62).
**NON CONVERGÉ** — val encore en descente à 50K. Resume nécessaire.
Artefacts : `C:/Users/pierr/Downloads/catboost-v3-output/v20260402_131322/`

### 9.16 LightGBM v3→v5 — Resume Chain (2026-04-02→04)

| Version | Date | Init | lr | Total | val | Status |
|---------|------|------|----|-------|-----|--------|
| v3 | 04-02 | scratch | 0.003 | 50K | 0.5364 | TIMEOUT (3 bugs) |
| v4 | 04-03 | 15K ckpt | 0.005 | 65K | 0.5204 | Training DONE, post TIMEOUT |
| v5 | 04-04 | 65K model | 0.005 | ~90K? | ~0.518? | RUNNING |

**3 bugs fixés (session 2026-04-03→04) :**
1. `_checkpoint_model()` : `model.booster_.save_model()` pour LGBMClassifier (GitHub #4841)
2. `kaggle_shap.py` : TreeExplainer fallback quand CatBoost absent + random subsample
3. `train_kaggle.py` : quality gates AVANT SHAP (root cause 3 timeouts post-training)

**Permutation skip** pour single-model kernels (197×5×17s = 4h39m timeout garanti).

### 9.17 CatBoost v4 — From Scratch lr=0.01 (2026-04-04)

From scratch imposé : CatBoost interdit `init_model` + `Pool(baseline=)` ensemble.
lr=0.01 (2x v3), iterations=150K cap, early_stopping=200. Convergence attendue ~40-50K.
Snapshot natif (10 min) pour recovery si timeout. **RUNNING.**

### 9.18 Comparaison cross-modèles (en attente)

| Model | lr | Total | Valid | Test | Convergé | Status |
|-------|-----|-------|-------|------|----------|--------|
| XGBoost v5 | 0.005 | 86.5K | 0.5126 | **0.566** | **Oui** | DONE |
| CatBoost v3 | 0.005 | 50K | 0.5468 | 0.590 | Non | Superseded by v4 |
| CatBoost v4 | 0.01 | ~45K? | ??? | ??? | ??? | RUNNING |
| LightGBM v4 | 0.005 | 65K | 0.5204 | ??? | Non | Model sauvé |
| LightGBM v5 | 0.005 | ~90K? | ~0.518? | ??? | ??? | RUNNING |

Sélection champion après convergence des 3 modèles + artefacts ISO complets.

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
