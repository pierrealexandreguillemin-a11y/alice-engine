# Differential Features — Design Spec

<!--
LLM CONTEXT: Ce spec ajoute ~30 features differentielles et d'interaction au pipeline FE
existant (V8). Objectif: debloquer le quality gate es_mae en donnant au modele des features
qui corrigent l'Elo au niveau board, basees sur la litterature multisports.

Prerequis: lire docs/requirements/FEATURE_DOMAIN_LOGIC.md AVANT ce spec.
Companion: FEATURE_SPECIFICATION.md (types, plages), FEATURE_DOMAIN_LOGIC.md (logique metier)
-->

> **Status**: DESIGN
> **Date**: 27 Mars 2026
> **Branch**: `feat/v8-multiclass-features`
> **Prerequis**: v16 training complete (alpha=0.7, gate 8/9, es_mae bloque)
> **Reference**: `docs/requirements/FEATURE_DOMAIN_LOGIC.md` §3bis, §5, §6, §7

---

## Sommaire

1. [Probleme et objectif](#1-probleme-et-objectif)
2. [Features ajoutees](#2-features-ajoutees)
3. [Architecture module](#3-architecture-module)
4. [Integration pipeline](#4-integration-pipeline)
5. [Fix features orphelines](#5-fix-features-orphelines)
6. [Tests](#6-tests)
7. [Deploiement Kaggle](#7-deploiement-kaggle)
8. [Criteres de succes](#8-criteres-de-succes)
9. [References](#9-references)

---

## 1. Probleme et objectif

### 1.1 Diagnostic v16

196 features, 17 validated, 116 mortes. Gate 8/9 : `es_mae >= elo` (gap ~0.001).
Les features mortes sont individuelles (_blanc/_noir, _dom/_ext) et match-level.
Le modele depth=4 gaspille 2 splits sur 4 pour soustraire deux features individuelles.

### 1.2 Litterature

La litterature multisports (NBA PMC11265715, Hubacek 2019, arXiv:2309.14807)
utilise des DIFFERENTIELS comme features primaires : "features for both teams
are subtracted from each other" → 1 split au lieu de 2.

### 1.3 Objectif

Ajouter ~30 features differentielles + interactions au pipeline FE existant.
Pas de refonte : un module `differentials.py` appele en fin de chaine.

**Critere de succes** : quality gate 9/9 (es_mae < elo baseline).

---

## 2. Features ajoutees

### 2.1 Differentiels joueur — 8 features (P0)

Chacune = feature_blanc - feature_noir. Capture le MATCHUP en 1 split.

| Feature | Calcul | Colonnes source | Reference |
|---|---|---|---|
| `diff_form` | expected_score_recent_blanc - expected_score_recent_noir | recent_form.py | PMC11265715 |
| `diff_win_rate_recent` | win_rate_recent_blanc - win_rate_recent_noir | recent_form.py | Hubacek 2019 |
| `diff_draw_rate` | draw_rate_blanc - draw_rate_noir | draw_priors.py | ALICE domain |
| `diff_draw_rate_recent` | draw_rate_recent_blanc - draw_rate_recent_noir | recent_form.py | ALICE domain |
| `diff_win_rate_normal` | win_rate_normal_blanc - win_rate_normal_noir | pressure.py | arXiv:2309.14807 |
| `diff_clutch` | clutch_win_blanc - clutch_win_noir | pressure.py | NBA clutch stats |
| `diff_momentum` | momentum_blanc - momentum_noir | elo_trajectory.py | Glicko-2 |
| `diff_derniere_presence` | derniere_presence_blanc - derniere_presence_noir | ali/presence.py | arXiv:2410.21484 |

### 2.2 Differentiels equipe — 6 features (P0)

Chacune = feature_dom - feature_ext. Signal match-level utilisable en 1 split.

| Feature | Calcul | Colonnes source | Reference |
|---|---|---|---|
| `diff_position` | position_dom - position_ext | standings.py | arXiv:2309.14807 |
| `diff_points_cumules` | points_cumules_dom - points_cumules_ext | standings.py | Hubacek 2019 |
| `diff_profondeur` | profondeur_effectif_dom - profondeur_effectif_ext | club_behavior.py | soccer bench value |
| `diff_stabilite` | noyau_stable_dom - noyau_stable_ext | club_behavior.py | arXiv:1912.11762 |
| `diff_win_rate_home` | win_rate_home_dom - win_rate_home_ext | club_behavior.py | PMC8656876 |
| `diff_draw_rate_home` | draw_rate_home_dom - draw_rate_home_ext | club_behavior.py | ALICE domain |

### 2.3 Interactions board × match — 6 features (P1)

Connectent le contexte match (constant 8 boards) au joueur specifique.
Signature unique d'ALICE vs soccer (1 match = 8 boards = 8 joueurs differents).

| Feature | Calcul | Signal metier |
|---|---|---|
| `form_in_danger` | diff_form × zone_danger_dom | Forme relative sous pression equipe |
| `color_match` | (couleur_effective == couleur_preferee_blanc).astype(int) | Joueur a sa couleur FFE preferee |
| `decalage_important` | decalage_position_blanc × match_important | Placement strategique en match cle |
| `marge100_decale` | club_utilise_marge_100_dom × abs(decalage_position_blanc) | Strategie deliberee du capitaine |
| `flex_decale` | flexibilite_echiquier_blanc × abs(decalage_position_blanc) | Joueur flexible vs specialiste deplace |
| `promu_vs_strong` | joueur_promu_blanc × clip(-diff_elo, 0, 800) / 400 | Renforce face a adversaire fort |

**Calcul `color_match`** :
```python
# Convention FFE: echiquiers impairs = blancs pour equipe dom
is_odd_board = (df["echiquier"] % 2 == 1)
blanc_plays_white = (df["est_domicile_blanc"] == 1) == is_odd_board
# True si le joueur blanc a les pieces blanches sur ce board
pref = df.get("couleur_preferee_blanc", pd.Series("neutre", index=df.index))
color_match = ((pref == "blanc") & blanc_plays_white) | ((pref == "noir") & ~blanc_plays_white)
```

### 2.4 Features nouvelles — 4 features (P1)

| Feature | Calcul | Source |
|---|---|---|
| `elo_uncertainty` | k_coefficient_blanc + k_coefficient_noir | Glicko-2 (Glickman 2001) |
| `k_asymmetry` | k_coefficient_blanc - k_coefficient_noir | Bayesian Elo (arXiv:2512.18013) |
| `zone_danger_dom` | (zone_enjeu_dom == "danger").astype(int) | One-hot pour interactions |
| `zone_montee_dom` | (zone_enjeu_dom == "montee").astype(int) | One-hot pour interactions |

**Note k_coefficient** : si pas encore calcule dans le FE (statut "A implementer"
dans FEATURE_SPECIFICATION.md), utiliser un proxy :
```python
k_proxy = 40 if categorie in ("U08".."U18") else 20
# Ou: k_proxy = 40 if elo < 2300 and age < 18 else (10 if elo >= 2400 else 20)
```

### 2.5 Comptage total

| Tier | Features | Priorite |
|---|---|---|
| Diff joueur | 8 | P0 |
| Diff equipe | 6 | P0 |
| Interactions | 6 | P1 |
| Nouvelles | 4 | P1 |
| **Total** | **24** | |

Plus fix orphelines (section 5) : nettoyage config + urgence_proxy + adversaire_niveau.
Les features individuelles existantes sont CONSERVEES (le CE en a besoin).

---

## 3. Architecture module

### 3.1 Fichier : `scripts/features/differentials.py`

```python
"""Differential and interaction features for matchup-level prediction.

Transforms individual features (blanc/noir, dom/ext) into relative features
as recommended by multisport ML literature (PMC11265715, Hubacek 2019).

Stateless, vectorized, usable in batch (FE pipeline) and online (inference).
Training-serving skew prevention: same function called in both contexts.

Document ID: ALICE-DIFFERENTIALS
Version: 1.0.0
ISO: 5055 (SRP, <300 lines), 5259 (no leakage), 42001 (traceable)

References:
- PMC11265715: NBA XGBoost, "features subtracted from each other"
- Hubacek 2019: Soccer Prediction Challenge winner
- Hopsworks FTI: training-serving skew prevention
"""

def compute_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """Add differential + interaction features. Pure, no state."""
```

### 3.2 Contrat

- **Input** : DataFrame avec colonnes individuelles (blanc/noir, dom/ext)
- **Output** : meme DataFrame + ~24 colonnes supplementaires
- **Pas de side-effect** : pur, sans etat, vectorise
- **NaN** : diff(NaN, x) = NaN (comportement pandas, OK pour arbres)
- **1 ligne ou 1M** : fonctionne identiquement (batch et inference)
- **Colonnes manquantes** : si une colonne source n'existe pas, skip le diff (pas d'erreur).
  Permet d'utiliser le module meme si certaines features amont ne sont pas calculees.

### 3.3 Estimation taille

~250 lignes max (ISO 5055) :
- Docstring + imports : ~30 lignes
- _player_differentials : ~40 lignes (8 soustractions + guards)
- _team_differentials : ~35 lignes (6 soustractions + guards)
- _board_match_interactions : ~60 lignes (6 interactions, color_match complexe)
- _new_features : ~30 lignes (k_coeff, zone dummies)
- compute_differentials : ~15 lignes (orchestration)
- Helper _safe_diff : ~10 lignes (NaN-safe soustraction avec guard colonnes)
- Tests inline docstring : ~30 lignes

---

## 4. Integration pipeline

### 4.1 FE batch (pipeline.py / fe_kaggle.py)

Appel en DERNIERE etape, apres toutes les features individuelles :

```python
# scripts/features/pipeline.py — fin de merge_all_features()
from scripts.features.differentials import compute_differentials
df = compute_differentials(df)
```

```python
# scripts/cloud/fe_kaggle.py — apres compute features, avant sauvegarde parquet
from scripts.features.differentials import compute_differentials
df_train = compute_differentials(df_train)
df_valid = compute_differentials(df_valid)
df_test = compute_differentials(df_test)
```

### 4.2 Inference online (services/inference.py)

Meme appel, meme module — zero skew :

```python
# services/inference.py — apres assemblage features pour un board
from scripts.features.differentials import compute_differentials
df_board = compute_differentials(df_board)  # 1 ligne
```

### 4.3 Training (train_kaggle.py)

**Aucun changement.** Les features numeriques sont auto-incluses par
`select_dtypes(include=["int64","float64"...])` dans `prepare_features()`.

---

## 5. Fix features orphelines (audit 2026-03-27)

### 5.0 Contexte : flow d'encodage

`_encode_categoricals()` transforme TOUTES les colonnes de CATEGORICAL + CATBOOST_CAT
+ ADVANCED_CAT en entiers (LabelEncoder) AVANT `select_dtypes()`. Donc les features
string listees dans ADVANCED_CAT NE SONT PAS droppees — elles sont label-encodees
puis gardees. Les vraies orphelines sont les features LISTEES MAIS JAMAIS COMPUTEES,
et les modules JAMAIS APPELES.

### 5.1 Fantomes dans kaggle_constants.py (P0 — nettoyer)

Features dans ADVANCED_CAT_FEATURES mais JAMAIS COMPUTEES par le FE pipeline.
L'encodeur les skip silencieusement (`if col not in train.columns: continue`).

| Feature | Dans ADVANCED_CAT | Computee | Action |
|---|---|---|---|
| `data_quality_blanc/noir` | Oui | **NON** | Retirer de la liste |
| `elo_type_blanc/noir` | Oui | **NON** | Retirer (redondant avec Elo brut) |
| `categorie_blanc/noir` | Oui | **NON** | Retirer OU implementer (P2: k_coefficient proxy) |

**Action** : nettoyer ADVANCED_CAT pour ne garder que les colonnes reellement computees.

### 5.2 Asymetrie zone_enjeu (P0 — corriger)

`zone_enjeu_dom` est dans CATBOOST_CAT_FEATURES (encodage natif CatBoost).
`zone_enjeu_ext` est dans ADVANCED_CAT_FEATURES (LabelEncoder generique).
→ Encodage DIFFERENT pour dom et ext de la meme feature.

**Action** : ajouter `zone_enjeu_ext` dans CATBOOST_CAT_FEATURES pour coherence.

### 5.3 Modules existants jamais appeles (P1)

| Module | Fonction | Features | Action |
|---|---|---|---|
| `extract_adversaire_niveau()` | pipeline_extended.py | adversaire_niveau_dom/ext | Brancher dans pipeline (P1) |
| `extract_temporal_features()` | pipeline_extended.py | phase_saison, ronde_normalisee | Verifier si ronde_normalisee arrive par autre chemin |
| `ce/scenarios.py` | calculate_scenario_features() | urgence_score | P2 — proxy urgence dans differentials.py |
| `ce/urgency.py` | calculate_urgency_features() | montee_possible | P2 — CE only pour l'instant |
| `ce/transferability.py` | calculate_transferability() | transfer_score | HORS SCOPE — CE V9 |

### 5.4 couleur_preferee : subsumee par color_match

`couleur_preferee_blanc/noir` est dans ADVANCED_CAT → label-encodee → dans le modele.
MAIS elle est DEAD en v16 (permutation 0, SHAP 0) car elle est DECONTEXTUALISEE :
savoir qu'un joueur "prefere les blancs" sans savoir s'il A les blancs sur ce board
est inutile.

**Action** : `color_match` dans differentials.py subsume couleur_preferee.
`couleur_preferee` reste dans ADVANCED_CAT (pas de regression) mais `color_match`
est le signal qui compte.

### 5.5 urgence_score proxy (P1)

`ce/scenarios.py` calcule `urgence_score` [0,1] mais n'est pas branche.
Plutot que wirer le module CE dans le pipeline ML (melange responsabilites),
on cree un proxy dans differentials.py :

```python
# Proxy urgence : zone critique × avancement saison
# zone_danger ou zone_montee EN FIN de saison = pression reelle
# Meilleur que match_important (81% = 1, inutile)
urgence_proxy = (zone_danger_dom | zone_montee_dom) * ronde_normalisee
```

Ce proxy capture 80% du signal de urgence_score sans dependance au module CE.

### 5.6 Recap actions par priorite

| Priorite | Action | Fichier | Lignes |
|---|---|---|---|
| P0 | Retirer fantomes ADVANCED_CAT (data_quality, elo_type, categorie) | kaggle_constants.py | ~3 |
| P0 | Ajouter zone_enjeu_ext dans CATBOOST_CAT | kaggle_constants.py | ~1 |
| P1 | Brancher extract_adversaire_niveau() | pipeline.py ou fe_kaggle.py | ~5 |
| P1 | Ajouter urgence_proxy dans differentials.py | differentials.py | ~5 |
| P1 | Verifier ronde_normalisee presence | pipeline.py | ~2 |
| P2 | Implementer k_coefficient (FIDE 8.3.3) → elo_uncertainty | nouveau module | ~50 |

---

## 6. Tests

### 6.1 Unit tests : `tests/features/test_differentials.py`

```python
class TestPlayerDifferentials:
    """8 tests: un par differential joueur."""
    def test_diff_form_basic(self):
        # ESR_blanc=0.7, ESR_noir=0.3 → diff_form=0.4
    def test_diff_form_nan(self):
        # ESR_blanc=NaN → diff_form=NaN
    def test_diff_form_equal(self):
        # ESR_blanc=ESR_noir=0.5 → diff_form=0.0

class TestTeamDifferentials:
    """6 tests: un par differential equipe."""

class TestInteractions:
    """6 tests: un par interaction."""
    def test_color_match_dom_odd_board_pref_blanc(self):
        # est_domicile=1, echiquier=1 (impair), pref=blanc → True
    def test_color_match_ext_odd_board_pref_blanc(self):
        # est_domicile=0, echiquier=1 → blanc joue noir → pref=blanc → False
    def test_form_in_danger_not_danger(self):
        # zone != danger → form_in_danger = 0

class TestMissingColumns:
    """Le module ne crash pas si une colonne source manque."""
    def test_missing_momentum(self):
        # pas de momentum_blanc → diff_momentum absent, pas d'erreur

class TestBatchVsSingle:
    """Meme resultat sur 1 ligne et N lignes."""
```

### 6.2 Integration test

Verifier que le parquet de sortie contient les nouvelles colonnes :
```python
def test_fe_pipeline_has_differentials():
    # Run pipeline on sample data
    # Assert "diff_form" in df.columns
    # Assert "color_match" in df.columns
```

---

## 7. Deploiement Kaggle

### 7.1 Sequence

1. Coder `differentials.py` + tests locaux
2. Ajouter appel dans `fe_kaggle.py`
3. `upload_all_data` (dataset contient differentials.py)
4. Push FE kernel → genere parquets avec ~26 colonnes de plus
5. Push Training kernel v17 → modele voit nouvelles features automatiquement

### 7.2 Risque

- **FE kernel plus long** : ~26 colonnes x 1.1M lignes = ~100 MB supplementaires
  dans le parquet. Impact negligeable (parquet compresse bien les floats).
- **Colonnes source manquantes** : le module skip les diff si colonne absente.
  Pas de crash, mais moins de features. Verifier dans le log FE que toutes
  les sources existent.

### 7.3 Compatibilite

- **init_score_alpha=0.7** : reste actif (les differentiels s'ajoutent, pas de conflit)
- **rsm=0.3** : les differentiels sont des features supplementaires pour CatBoost
- **Calibration** : temperature scaling s'applique apres, pas de changement

---

## 8. Criteres de succes

### 8.1 Gate

| Condition | Seuil | Verification |
|---|---|---|
| es_mae < elo baseline | es_mae < elo.es_mae | quality gate condition 6 |
| Gate 9/9 | toutes conditions | quality gate complete |

### 8.2 Feature signal

| Metrique | Attendu |
|---|---|
| diff_form dans top 5 permutation | #1 ou #2 (c'est LE differentiel) |
| diff_position permutation > 0 | Signal match-level debloquer |
| color_match permutation > 0 | Signal unique ALICE |
| best_iter > 300 | Plus d'iterations = features exploitees |

### 8.3 Non-regression

| Metrique | Contrainte |
|---|---|
| log_loss test | <= v16 (0.870) |
| mean_p_draw | > 1% (gate 9) |
| Features individuelles existantes | toujours presentes dans le parquet |

---

## 9. References

### Litterature multisports
- **PMC11265715 (NBA 2024)** : XGBoost + SHAP, "features subtracted", AUC 0.982
- **Hubacek et al. 2019** : Soccer Prediction Challenge, CatBoost + pi-ratings, RPS=0.1925
- **arXiv:2309.14807 (2024)** : 205 features soccer, form/standings differenced
- **arXiv:1912.11762 (2019)** : Review 100+ papers, "richer features > algorithm choice"
- **arXiv:2410.21484 (2024)** : Systematic review sports betting ML, opponent-adjusted
- **PMC8656876 (2021)** : Rugby home advantage, individual player splits

### Echecs
- **Pawnalyze 2022** : LightGBM bat Elo, "expected score from Elo not enough"
- **Glickman 2001** : Glicko-2, rating deviation = uncertainty

### Architecture
- **Hopsworks FTI** : Feature/Training/Inference pipelines, training-serving skew prevention
- **ISO 5055** : SRP, <300 lignes
- **ISO 5259** : Data quality, no leakage
- **ISO 42001** : Tracabilite features
