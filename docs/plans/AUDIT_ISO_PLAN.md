# AUDIT ISO - PLAN D'IMPLEMENTATION

**Version:** 1.0
**Date:** 2026-01-09
**Auteur:** ALICE Engine Team
**Normes ISO:** 5055, 29119, 5259, 15289, 25010

---

## Vue d'ensemble

Ce plan detaille les actions correctives et d'amelioration pour assurer la conformite ISO du projet Alice-Engine.

| Priorite | Categorie | Nb Actions | Effort estime |
|----------|-----------|------------|---------------|
| P0 | Corrections critiques | 3 | 3h |
| P1 | Tests ISO 29119 | 4 | 16h |
| P2 | Features ALI/CE | 4 | 12h |
| P2 | Documentation | 2 | 2h |

---

## P0 - CORRECTIONS CRITIQUES

### P0-1: SUPPRIMER fatigue.py (Non pertinent pour echecs)

**Objectif ISO:** ISO 5259 (Data Quality for ML) - Features doivent etre pertinentes au domaine

**Justification:**
- La fatigue physique (jours de repos) est pertinente pour sports physiques (football, basket)
- Aux echecs, la fatigue est cognitive, non physique
- Un joueur peut jouer 2 jours consecutifs sans impact sur performance echiquienne
- Feature source de bruit plutot que signal

**Fichiers:**
- `scripts/features/advanced/fatigue.py` (SUPPRIMER)
- `scripts/features/advanced/__init__.py` (retirer import/export)
- `scripts/features/pipeline.py` (retirer import/usage)
- `tests/test_feature_engineering.py` (retirer tests fatigue)

**Criteres de succes:**
- [ ] `fatigue.py` supprime
- [ ] `pytest tests/` passe sans erreur
- [ ] `ruff check scripts/features/` sans erreur d'import

---

### P0-2: SUPPRIMER home_away.py (Confusion conceptuelle)

**Objectif ISO:** ISO 5259 (Data Quality) - Coherence conceptuelle domaine

**Justification:**
- "Domicile/Exterieur" est un concept de sports physiques (stade)
- Aux echecs interclubs, pas de notion de lieu physique affectant performance
- Le concept pertinent est "Preference couleur" (blanc/noir)
- Le fichier `color_perf.py` existe deja et fait le bon travail

**Note:** Le fichier `scripts/features/performance/color_perf.py` existe deja et fournit la feature correcte:
- `calculate_color_performance()` - Score par couleur blanc/noir
- `couleur_preferee`: 'blanc', 'noir', 'neutre', 'donnees_insuffisantes'
- Conforme ISO 5259 (pas de fillna artificiel)

**Fichiers:**
- `scripts/features/advanced/home_away.py` (SUPPRIMER)
- `scripts/features/advanced/__init__.py` (retirer import/export)
- `scripts/features/pipeline.py` (retirer import/usage)

**Criteres de succes:**
- [ ] `home_away.py` supprime
- [ ] `color_perf.py` utilise comme feature officielle
- [ ] Pipeline utilise `calculate_color_performance`

---

## P1 - TESTS ISO 29119

### P1-1: Tests train_models_parallel.py (0% -> 50%)

**Objectif ISO:** ISO 29119 (Software Testing) - Couverture fonctions critiques

**Tests a ajouter:**
- `TestRunTraining`: pipeline complet avec donnees mock
- `TestLoadDatasets`: chargement reussi/echec
- `TestEvaluateOnTest`: mise a jour metrics

**Criteres de succes:**
- [ ] Coverage `train_models_parallel.py` >= 50%
- [ ] Tests integration avec fixtures parquet
- [ ] Mock MLflow pour eviter side effects

---

### P1-2: Tests services/inference.py (38% -> 70%)

**Objectif ISO:** ISO 29119 + ISO 42001 (AI Testing)

**Tests a ajouter:**
- `TestInferenceService`: init, load_model, predict_lineup
- Tests fallback (modele non charge)
- Tests edge cases (vide, None)

**Criteres de succes:**
- [ ] Coverage `inference.py` >= 70%
- [ ] Tests PlayerProbability dataclass

---

### P1-3: Tests services/data_loader.py (13% -> 70%)

**Objectif ISO:** ISO 29119 + ISO 5259 (Data Quality Testing)

**Tests a ajouter:**
- `TestDataLoaderInit`: initialisation
- `TestDataLoaderConnect`: connexion MongoDB (mock)
- `TestDataLoaderQueries`: requetes async
- `TestDataLoaderTrainingData`: chargement parquet

**Criteres de succes:**
- [ ] Coverage `data_loader.py` >= 70%
- [ ] Tests async avec pytest-asyncio
- [ ] Mocks MongoDB pour isolation

---

### P1-4: Tests advanced features (h2h, pressure, elo_trajectory)

**Objectif ISO:** ISO 29119 - Tests edge cases features ML

**Tests supplementaires:**
- `TestHeadToHeadEdgeCases`: meme joueur, ordre alphabetique
- `TestPressurePerformanceEdgeCases`: seuil ronde >= 7
- `TestEloTrajectoryEdgeCases`: seuil +-50 pts

---

## P2 - FEATURES ALI/CE

### P2-1: Scripts ALI - presence.py

**Fichier:** `scripts/features/ali/presence.py`

Features:
- `taux_presence_saison`: % rondes jouees sur saison
- `derniere_presence`: nb rondes depuis derniere apparition
- `regularite`: 'regulier', 'occasionnel', 'rare'

---

### P2-2: Scripts ALI - patterns.py

**Fichier:** `scripts/features/ali/patterns.py`

Features:
- `role_type`: 'titulaire', 'remplacant', 'polyvalent'
- `echiquier_prefere`: int ou None
- `flexibilite_echiquier`: float

---

### P2-3: Scripts CE - scenarios.py

**Fichier:** `scripts/features/ce/scenarios.py`

Features:
- `scenario`: 'course_titre', 'danger', 'condamne', 'mi_tableau'
- `urgence_score`: float [0, 1]

---

### P2-4: Scripts CE - urgency.py

**Fichier:** `scripts/features/ce/urgency.py`

Features:
- `montee_possible`: bool
- `maintien_assure`: bool
- `urgence_level`: 'critique', 'haute', 'normale', 'aucune'

---

## P2 - DOCUMENTATION

### P2-5: Documenter seuils pressure.py

Justifier dans docstring:
- Seuil `ronde >= 7`: Format interclubs FFE 7-11 rondes
- Seuil `ecart <= 1`: Match ou chaque partie compte

### P2-6: Documenter seuils elo_trajectory.py

Justifier dans docstring:
- Seuil `+-50 pts`: ~1 categorie FFE
- Reference: FIDE Handbook Elo Rating System

---

## VERIFICATION FINALE

**Commandes de validation:**

```bash
# Qualite code
make quality

# Tests avec coverage
pytest --cov=scripts --cov=services --cov-report=term-missing

# Architecture
python scripts/analyze_architecture.py
```

**Criteres globaux DoD:**
- [ ] Coverage globale >= 55% (cible 80%)
- [ ] Aucun fichier > 300 lignes
- [ ] Tous tests passent
- [ ] Ruff 0 erreurs
- [ ] MyPy 0 erreurs critiques

---

*Document conforme ISO 15289 (Documentation)*
