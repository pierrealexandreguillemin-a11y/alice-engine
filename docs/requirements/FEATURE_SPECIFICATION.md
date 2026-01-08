# Specification des Features ML - ALICE Engine

> **Document Type**: System Requirements (SyRS) - ISO 15289
> **Version**: 1.1.0
> **Date**: 8 Janvier 2026
> **Norme ISO 5259**: Data Quality for ML
> **Source**: Chapitre 6.1 FIDE (Oct 2025), Reglements FFE 2025-2026

---

## 0. Statut Implementation (ISO 15289)

| Feature | Module | Status | Tests |
|---------|--------|--------|-------|
| **ELO (blanc/noir/diff)** | echiquiers.parquet | ✅ Implementé | ✅ |
| **Type ELO (F/N/E)** | joueurs.parquet | ✅ Implementé | ✅ |
| **Fiabilite club** | feature_engineering.py | ✅ Implementé | ⚠️ |
| **Fiabilite joueur** | feature_engineering.py | ✅ Implementé | ⚠️ |
| **Forme recente** | feature_engineering.py | ✅ Implementé | ⚠️ |
| **Position echiquier** | feature_engineering.py | ✅ Implementé | ⚠️ |
| **Performance couleur** | feature_engineering.py | ✅ Implementé | ✅ 14 tests |
| **Classement reel** | feature_engineering.py | ✅ Implementé | ✅ 14 tests |
| **Zone enjeu reelle** | feature_engineering.py | ✅ Implementé | ✅ 14 tests |
| **Regles FFE (brulage/noyau)** | ffe_rules_features.py | ✅ Implementé | ✅ 66 tests |
| **Presence joueur (ALI)** | - | ❌ Planifie | - |
| **Comportement club (ALI)** | - | ❌ Planifie | - |
| **Scenario fin saison (CE)** | - | ❌ Planifie | - |

**Legende**: ✅ = OK | ⚠️ = Tests a ajouter | ❌ = Non implemente

---

## 1. Vue d'ensemble

Ce document definit formellement les features utilisees pour l'entrainement
du modele ALICE, leurs plages de valeurs, et leur justification metier.

**Objectif prediction** : Resultat d'une partie d'echecs (Victoire Blanc / Nulle / Victoire Noir)

---

## 2. Features Principales

### 2.1 ELO - Force des joueurs

| Feature | Type | Plage | Source |
|---------|------|-------|--------|
| `blanc_elo` | int | 1000-3000 | FFE/FIDE |
| `noir_elo` | int | 1000-3000 | FFE/FIDE |
| `diff_elo` | int | -2000 to +2000 | Calcule |

**Justification metier** :
- L'ELO est le predicteur principal de force relative
- Systeme FIDE : echelle logistique, intervalle 200 pts = 1 categorie
- Difference ELO → probabilite de victoire (voir Section 4)

**Plages validees** (ISO 5259) :
```python
ELO_MIN = 1000   # Debutant classe
ELO_MAX = 3000   # Elite mondiale (~2850 Magnus Carlsen)
ELO_WARNING_LOW = 1100   # Alerte si >10% sous ce seuil
ELO_WARNING_HIGH = 2700  # Alerte si >5% au-dessus
```

**Contrainte coherence** :
```python
assert diff_elo == blanc_elo - noir_elo
```

### 2.2 Type d'ELO

| Feature | Type | Valeurs | Signification |
|---------|------|---------|---------------|
| `blanc_elo_type` | cat | F, N, E | Type classement |
| `noir_elo_type` | cat | F, N, E | Type classement |

| Code | Signification | Fiabilite |
|------|---------------|-----------|
| **F** | FIDE (international) | Haute |
| **N** | National FFE | Moyenne |
| **E** | Estime (debutants) | Basse |

**Impact prediction** :
- ELO type E = moins fiable, plus de variance attendue
- Joueurs jeunes souvent en type E (progression rapide)

---

## 3. Categories d'Age FFE

### 3.1 Mapping officiel

| Code Legacy | Code FFE | Nom | Age | K FIDE |
|-------------|----------|-----|-----|--------|
| PpoM/PpoF | U08 | Petits Poussins | < 8 ans | 40 |
| PouM/PouF | U10 | Poussins | 8-9 ans | 40 |
| PupM/PupF | U12 | Pupilles | 10-11 ans | 40 |
| BenM/BenF | U14 | Benjamins | 12-13 ans | 40 |
| MinM/MinF | U16 | Minimes | 14-15 ans | 40 |
| CadM/CadF | U18 | Cadets | 16-17 ans | 40 |
| JunM/JunF | U20 | Juniors | 18-19 ans | 20* |
| SenM/SenF | Sen/X20 | Seniors | 20-49 ans | 20/10** |
| SepM/SepF | S50/X50 | Seniors Plus | 50-64 ans | 20/10 |
| VetM/VetF | S65/X65 | Veterans | 65+ ans | 20/10 |

*K=40 jusqu'a fin annee 18 ans si ELO < 2300
**K=10 si ELO a deja atteint 2400

### 3.2 Impact sur la prediction

**Progression naturelle** (FIDE 8.3.3) :
- Jeunes (< 18 ans) : K=40 → progression/regression rapide
- Adultes : K=20 → stabilite
- Elite (>2400) : K=10 → tres stable

**Feature derivee recommandee** :
```python
def get_k_coefficient(age: int, elo: int, parties_jouees: int) -> int:
    """Coefficient K selon regles FIDE 8.3.3."""
    # Nouveau joueur < 30 parties
    if parties_jouees < 30:
        return 40

    # Jeune < 18 ans et ELO < 2300
    if age < 18 and elo < 2300:
        return 40

    # A deja atteint 2400
    if elo >= 2400:  # Note: reste K=10 meme si redescend
        return 10

    return 20
```

**Impact matchup age** :
- Jeune vs Veteran : le jeune progresse, le veteran regresse
- Meme ELO mais trajectoires opposees → feature potentielle

---

## 4. Tables de Conversion FIDE

### 4.1 Difference ELO → Probabilite (8.1.2)

| Diff ELO | P(High) | P(Low) |
|----------|---------|--------|
| 0-3 | 0.50 | 0.50 |
| 4-10 | 0.51 | 0.49 |
| 11-17 | 0.52 | 0.48 |
| 18-25 | 0.53 | 0.47 |
| 26-32 | 0.54 | 0.46 |
| 33-39 | 0.55 | 0.45 |
| 40-46 | 0.56 | 0.44 |
| 47-53 | 0.57 | 0.43 |
| 54-61 | 0.58 | 0.42 |
| 62-68 | 0.59 | 0.41 |
| 69-76 | 0.60 | 0.40 |
| 77-83 | 0.61 | 0.39 |
| 84-91 | 0.62 | 0.38 |
| 92-98 | 0.63 | 0.37 |
| 99-106 | 0.64 | 0.36 |
| 107-113 | 0.65 | 0.35 |
| 114-121 | 0.66 | 0.34 |
| 122-129 | 0.67 | 0.33 |
| 130-137 | 0.68 | 0.32 |
| 138-145 | 0.69 | 0.31 |
| 146-153 | 0.70 | 0.30 |
| ... | ... | ... |
| >735 | 1.00 | 0.00 |

**Note FIDE 8.3.1** : Pour joueurs < 2650, diff > 400 pts comptee comme 400.

### 4.2 Feature derivee : Probabilite theorique

```python
def get_expected_score(diff_elo: int) -> float:
    """Calcule le score attendu selon table FIDE 8.1.2."""
    # Plafond FIDE pour ELO < 2650
    diff = min(abs(diff_elo), 400)

    # Approximation logistique (formule FIDE)
    expected = 1 / (1 + 10 ** (-diff / 400))

    if diff_elo < 0:
        expected = 1 - expected

    return expected
```

---

## 5. Features Contextuelles

### 5.1 Competition

| Feature | Type | Valeurs | Source |
|---------|------|---------|--------|
| `type_competition` | cat | A02, F01, C01, C03, C04, J02, J03, REG, DEP | FFE |
| `division` | cat | Top16, N1, N2, N3, N4, REG, DEP | FFE |
| `ronde` | int | 1-11 | Match |

**Justification** : Enjeu different selon competition/division.

### 5.2 Enjeu (CORRIGÉ - Position réelle)

**✅ IMPLÉMENTÉ** dans `feature_engineering.py:calculate_standings()`

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| `position` | int | Classement réel calculé depuis scores | ✅ |
| `points_cumules` | int | Points accumulés (2 victoire, 1 nul) | ✅ |
| `zone_enjeu` | cat | montee, danger, mi_tableau | ✅ |
| `ecart_premier` | int | Points du 1er - points equipe | ✅ |
| `ecart_dernier` | int | Points equipe - points du dernier | ✅ |
| `nb_equipes` | int | Nombre d'equipes dans le groupe | ✅ |

**Calcul position** (ISO 5259 - depuis données réelles):
```python
# Points Interclubs FFE:
# - Victoire match (score > adversaire): 2 pts
# - Nul match (score = adversaire): 1 pt
# - Défaite match (score < adversaire): 0 pt

standings = calculate_standings(df)  # Position réelle par ronde
```

### 5.3 Performance par Couleur

**✅ IMPLÉMENTÉ** dans `feature_engineering.py:calculate_color_performance()`

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| `score_blancs` | float | Score moyen avec blancs (0-1) | ✅ |
| `score_noirs` | float | Score moyen avec noirs (0-1) | ✅ |
| `nb_blancs` | int | Nombre de parties avec blancs | ✅ |
| `nb_noirs` | int | Nombre de parties avec noirs | ✅ |
| `avantage_blancs` | float | score_blancs - score_noirs | ✅ |
| `couleur_preferee` | cat | 'blanc', 'noir', 'neutre' | ✅ |

**Convention échecs interclubs**:
- Échiquiers impairs (1, 3, 5, 7) = Blancs pour équipe domicile
- Échiquiers pairs (2, 4, 6, 8) = Noirs pour équipe domicile

**Seuil préférence**: |avantage_blancs| > 0.05 = significatif

---

## 6. Features Reglementaires FFE

### 6.1 Joueur brule (A02 Art. 3.7.c)

| Feature | Type | Description |
|---------|------|-------------|
| `joueur_brule` | bool | A joue 3+ fois en equipe superieure |
| `matchs_avant_brulage` | int | 0-3 matchs restants |

### 6.2 Noyau (A02 Art. 3.7.f)

| Feature | Type | Description |
|---------|------|-------------|
| `est_dans_noyau` | bool | A deja joue pour cette equipe |
| `pct_noyau_equipe` | float | % composition dans noyau |

### 6.3 Disponibilite

| Feature | Type | Description |
|---------|------|-------------|
| `nb_matchs_joues_saison` | int | Compteur saison |
| `peut_jouer_ronde_n` | bool | matchs < numero_ronde |

---

## 7. Distribution Attendue (ISO 5259)

### 7.1 ELO

| Percentile | Valeur attendue |
|------------|-----------------|
| 5% | ~1200 |
| 25% | ~1400 |
| 50% | ~1550 |
| 75% | ~1750 |
| 95% | ~2100 |
| 99% | ~2400 |

### 7.2 Categories d'age (dataset FFE)

| Categorie | % Dataset |
|-----------|-----------|
| U08-U14 (Jeunes) | ~51% |
| U16-U20 (Ados) | ~8% |
| X20-X65 (Adultes) | ~41% |

---

## 8. Validation Schema (Implementation)

```python
# scripts/model_registry.py

REQUIRED_TRAIN_COLUMNS = {"resultat_blanc", "blanc_elo", "noir_elo"}
REQUIRED_NUMERIC_COLUMNS = {"blanc_elo", "noir_elo", "diff_elo"}

# Plages FFE
ELO_MIN = 1000
ELO_MAX = 3000
ELO_WARNING_LOW = 1100
ELO_WARNING_HIGH = 2700

def validate_dataframe_schema(df, validate_elo_ranges=True):
    """Valide schema et plages FFE (ISO 5259)."""
    # ... voir implementation complete dans model_registry.py
```

---

## 9. Features Joueur - Prediction Presence (ALI)

### 9.1 Historique Joueur

| Feature | Type | Description | Usage |
|---------|------|-------------|-------|
| `taux_presence_saison` | float | % rondes jouees cette saison | Fiabilite |
| `taux_presence_global` | float | % rondes jouees sur 3 saisons | Tendance |
| `derniere_presence` | int | Nb rondes depuis dernier match | Disponibilite |
| `nb_matchs_saison` | int | Matchs joues cette saison | Regle < ronde N |
| `rondes_manquees_consecutives` | int | Serie d'absences | Blessure/indispo ? |

### 9.2 Pattern Joueur

| Feature | Type | Description |
|---------|------|-------------|
| `titulaire_regulier` | bool | Joue >70% des rondes |
| `joueur_occasionnel` | bool | Joue 30-70% des rondes |
| `remplacant` | bool | Joue <30% des rondes |
| `position_preferee` | int | Echiquier modal (1-8) |
| `variance_position` | float | Ecart-type position jouee |

---

## 10. Features Club/Equipe - Comportement (ALI)

### 10.1 Stabilite Effectif

| Feature | Type | Description |
|---------|------|-------------|
| `nb_joueurs_utilises_saison` | int | Joueurs differents alignes |
| `rotation_effectif` | float | Turnover moyen par ronde |
| `noyau_stable` | int | Joueurs presents >80% |
| `profondeur_effectif` | int | Joueurs disponibles total |

### 10.2 Pattern Composition

| Feature | Type | Description |
|---------|------|-------------|
| `compo_type_domicile` | cat | Pattern typique a domicile |
| `compo_type_exterieur` | cat | Pattern typique a l'exterieur |
| `renforce_fin_saison` | bool | Tendance a renforcer R7-R9 |
| `preserve_titulaires` | bool | Fait tourner pour eviter brulage |

---

## 11. Features Contextuelles - Scenario Fin Saison (CE)

### 11.1 Enjeu Sportif

| Feature | Type | Description | Impact prediction |
|---------|------|-------------|-------------------|
| `zone_enjeu` | cat | montee/danger/mi_tableau | Intensite |
| `ecart_montee_pts` | int | Distance du 1er | Motivation |
| `ecart_descente_pts` | int | Marge sur relegation | Pression |
| `matchs_restants` | int | Rondes restantes | Urgence |
| `montee_mathematique` | bool | Peut encore monter ? | Abandon possible |
| `maintien_assure` | bool | Ne peut plus descendre ? | Relachement |

### 11.2 Scenarios Fin de Saison

| Scenario | Description | Comportement attendu |
|----------|-------------|----------------------|
| `course_titre` | 1er-2e, ecart < 3 pts | Composition maximale |
| `course_montee` | 3e-4e, barrage possible | Renforcement progressif |
| `confort` | Mi-tableau, rien a jouer | Rotation, repos titulaires |
| `danger` | Zone rouge, -3 pts du maintien | Tous les titulaires |
| `condamne` | Descente mathematique | Demobilisation possible |

### 11.3 Features Derivees

```python
def calculer_scenario_saison(
    position: int,
    points: int,
    pts_premier: int,
    pts_releguable: int,
    matchs_restants: int,
) -> str:
    """Determine le scenario de fin de saison."""
    pts_max = points + matchs_restants * 3

    # Montee impossible
    if pts_max < pts_premier:
        if points <= pts_releguable:
            return "condamne"
        return "confort"

    # En course
    if position <= 2:
        return "course_titre"
    if position <= 4:
        return "course_montee"

    # Danger
    ecart_releguable = points - pts_releguable
    if ecart_releguable <= 3:
        return "danger"

    return "confort"
```

---

## 12. Features Temporelles - Periode Saison

### 12.1 Cycle Saison

| Feature | Type | Valeurs | Description |
|---------|------|---------|-------------|
| `phase_saison` | cat | debut/milieu/fin | Position dans saison |
| `ronde_normalisee` | float | 0-1 | ronde / total_rondes |
| `mois` | int | 9-5 | Septembre a Mai |

### 12.2 Pattern Saisonnier

| Phase | Rondes | Comportement typique |
|-------|--------|----------------------|
| **Debut** | R1-R3 | Exploration, test effectif |
| **Milieu** | R4-R6 | Stabilisation, rotation |
| **Fin** | R7-R9 | Enjeu max, titulaires |

### 12.3 Calendrier

| Feature | Type | Description |
|---------|------|-------------|
| `domicile` | bool | Match a domicile |
| `adversaire_niveau` | int | Classement adverse actuel |
| `match_derby` | bool | Meme ville/region |
| `match_important` | bool | Top 4 ou lutte maintien |

---

## 13. Matrice Features par Module ALICE

| Module | Features principales | Objectif |
|--------|---------------------|----------|
| **ALI** (Inference) | Presence joueur, pattern club, calendrier | Predire QUI jouera |
| **CE** (Composition) | ELO, enjeu, scenario fin saison | Optimiser matchups |

### 13.1 Pipeline ALI

```
Entrees:
├── club_adverse_id
├── ronde
└── historique_compositions[]

Features extraites:
├── Par joueur adverse:
│   ├── taux_presence
│   ├── derniere_presence
│   └── position_preferee
├── Par equipe:
│   ├── stabilite_effectif
│   └── pattern_composition
└── Contexte:
    ├── enjeu_sportif
    ├── phase_saison
    └── match_domicile

Sortie:
└── P(joueur_i present) pour chaque joueur
```

### 13.2 Pipeline CE

```
Entrees:
├── joueurs_disponibles[]
├── contraintes_ffe{}
└── prediction_adverse (depuis ALI)

Features:
├── diff_elo par echiquier
├── score_attendu FIDE
├── scenario_fin_saison
└── enjeu_match

Sortie:
└── composition_optimale + score_attendu
```

---

## 14. Impact Inconnu (A Investiguer)

### 14.1 Age sur prediction

**Hypothese** : A ELO egal, un jeune (K=40) a plus de variance qu'un adulte (K=20).

**Questions ouvertes** :
- La categorie d'age ameliore-t-elle la prediction ?
- Faut-il une feature `trajectoire_elo` (montee/descente recente) ?
- L'age relatif (jeune vs veteran) a-t-il un impact psychologique ?

**Action** : Experimentation A/B avec/sans feature age.

### 14.2 Forme recente

**Hypothese** : L'ELO mensuel (mis a jour chaque 1er du mois) capture la forme.

**Limitation** : ELO = indicateur retarde (1 mois de latence).

---

## 15. Tracabilite (ISO 5259)

### 15.1 Sources

| Donnee | Source | MAJ |
|--------|--------|-----|
| ELO FIDE | ratings.fide.com | Mensuelle |
| ELO FFE | echecs.asso.fr | Mensuelle |
| Resultats | Papi FFE | Apres ronde |
| Regles | Chapitre 6.1 FIDE | Annuelle |
| Categories | Reglements FFE | Annuelle |

### 15.2 Historique

| Version | Date | Changement |
|---------|------|------------|
| 1.0.0 | 2026-01-08 | Creation initiale |

---

*Document genere le 8 Janvier 2026*
*Conforme ISO/IEC 5259 (Data Quality for ML)*
*Source: FIDE Chapitre 6.1 (Oct 2025) + FFE 2025-2026*
