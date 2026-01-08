# ALICE Training Progress - Quality Records

> **Document Type**: Quality Records (QR) - ISO 15289
> **Version**: 1.2.0
> **Date creation**: 4 Janvier 2026
> **Derniere MAJ**: 8 Janvier 2026
> **Responsable**: Claude Code / Pierre

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
| 2 | Feature Engineering | ✅ Complete | 2 min | 8 fichiers features/*.parquet |
| 3 | Split temporel | ✅ Complete | inclus Phase 2 | train/valid/test.parquet |
| 4 | Evaluation ML | ✅ Complete | 5 min | ml_evaluation_results.csv |
| 5 | Hyperparameter Tuning | 🔄 A faire | - | - |
| 6 | Entrainement final | 🔄 A faire | - | models/*.cbm |
| 7 | Deploiement | 🔄 A faire | - | API FastAPI |

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

---

*Document genere selon ISO 15289 - Quality Records*
