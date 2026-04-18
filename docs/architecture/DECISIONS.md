# Architecture Decision Records (ADR)

> **Norme**: ISO 42010 - Architecture Decisions
> **Format**: MADR (Markdown Any Decision Records)

---

## ADR-001: CatBoost vs XGBoost pour le modele ML

**Date**: 3 Janvier 2026
**Statut**: Accepte

### Contexte
Choix du framework ML pour la prediction des compositions adverses.

### Decision
Utiliser **CatBoost** comme framework principal.

### Raisons
1. Gestion native des features categoriques (licence, division, ligue_code)
2. Inference 30-60x plus rapide que XGBoost
3. Moins de tuning requis (bon out-of-the-box)
4. Meilleure gestion des valeurs manquantes

### Consequences
- XGBoost garde en fallback
- Format modele: `.cbm` (CatBoost Model)
- Dependance: `catboost>=1.2.7`

---

## ADR-002: FastAPI vs Flask pour l'API

**Date**: 3 Janvier 2026
**Statut**: Accepte

### Contexte
Choix du framework web Python.

### Decision
Utiliser **FastAPI**.

### Raisons
1. Validation automatique avec Pydantic v2
2. Documentation OpenAPI auto-generee
3. Support async natif (Motor MongoDB)
4. Performance superieure a Flask

### Consequences
- Schemas Pydantic obligatoires
- Typage strict
- Dependance: `fastapi>=0.109.0`

---

## ADR-003: OR-Tools pour l'optimisation

**Date**: 3 Janvier 2026
**Statut**: Accepte

### Contexte
Choix du solver pour le Composition Engine (CE).

### Decision
Utiliser **Google OR-Tools**.

### Raisons
1. Solver CP-SAT performant
2. Adapte aux problemes d'assignment
3. Gratuit et open-source
4. Meilleur que PuLP pour notre cas

### Consequences
- Contraintes FFE modelisables
- Dependance: `ortools>=9.8`

---

## ADR-004: Architecture 3 couches SRP

**Date**: 3 Janvier 2026
**Statut**: Accepte

### Contexte
Organisation du code Python.

### Decision
Architecture en 3 couches: Controller → Service → Repository.

### Raisons
1. Separation des responsabilites (ISO 42010)
2. Testabilite (services purs)
3. Coherence avec chess-app backend

### Consequences
- `app/api/` = Controllers
- `services/` = Logique metier
- `services/data_loader.py` = Repository

---

## ADR-005: Render vs Vercel pour le deploiement

**Date**: 3 Janvier 2026
**Statut**: Accepte

### Contexte
Choix de la plateforme de deploiement (budget 0€).

### Decision
Utiliser **Render** (free tier).

### Raisons
1. Support Python natif (pas de serverless)
2. Long-running processes supportes
3. Blueprint YAML pour IaC
4. Region Frankfurt (proche France)

### Consequences
- Cold start apres 15 min inactivite
- Solution keep-alive requise
- 750h/mois partagees

### Alternatives evaluees
- Vercel: Timeout 10s, pas adapte ML
- Koyeb: 1 service gratuit seulement
- Railway: Plus de free tier

---

## ADR-006: MongoDB partage avec chess-app

**Date**: 3 Janvier 2026
**Statut**: Accepte

### Contexte
Acces aux donnees joueurs/clubs.

### Decision
Reutiliser le cluster MongoDB Atlas de chess-app.

### Raisons
1. Donnees deja presentes (joueurs, clubs)
2. Pas de duplication
3. Cout 0€
4. Lecture seule pour ALICE

### Consequences
- Meme connexion string
- Pas de collections propres a ALICE
- Dependance a chess-app pour les donnees

---

## ADR-007: Layered + SRP vs Domain-Driven Design (DDD)

**Date**: 8 Janvier 2026
**Statut**: Accepte

### Contexte

Choix du paradigme architectural pour structurer le code d'Alice-Engine.
Deux approches principales considerees:

1. **Layered Architecture + SRP** (actuel)
   - Controller → Service → Repository
   - Single Responsibility Principle par couche
   - Simple, explicite

2. **Domain-Driven Design (DDD)**
   - Bounded Contexts, Aggregates, Entities, Value Objects
   - Ubiquitous Language
   - Architecture hexagonale/onion

### Decision

Conserver **Layered Architecture + SRP** et ne pas adopter DDD.

### Raisons

#### 1. Complexite du domaine insuffisante pour DDD

| Critere | Alice-Engine | Seuil DDD |
|---------|--------------|-----------|
| Bounded Contexts | 1 (Composition) | 3+ |
| Regles metier | ~10 regles FFE | 50+ |
| Workflows | Lineaire (predict→optimize) | Multiples, branches |
| Equipe | 1-2 devs | 5+ devs |

#### 2. DDD serait over-engineering

Le domaine Alice-Engine est **algorithmique**, pas **metier complexe**:
- ALI: Inference ML (CatBoost) → calcul statistique
- CE: Optimisation (OR-Tools) → probleme mathematique
- Regles FFE: ~10 regles, pas de processus metier

DDD brille pour: banque, assurance, e-commerce, ERP.
DDD est excessif pour: API ML, microservices simples, CRUD.

#### 3. Cout cognitif non justifie

| Element DDD | Cout | Benefice Alice |
|-------------|------|----------------|
| Bounded Contexts | Eleve | Nul (1 seul contexte) |
| Aggregates | Moyen | Faible |
| Domain Events | Eleve | Nul (pas d'evenements) |
| Ubiquitous Language | Moyen | Deja present (termes FFE) |
| Repository pattern | Faible | Deja implemente |

#### 4. Layered suffit pour les besoins actuels

```
app/api/routes.py      # Controller: HTTP validation
services/inference.py  # Service: Logique ML pure
services/composer.py   # Service: Logique optimisation pure
services/data_loader.py # Repository: I/O MongoDB/Parquet
```

- **Testabilite**: Services sans I/O, facilement mockables
- **Lisibilite**: Flux lineaire, pas d'indirection
- **Maintenance**: 1 dev peut comprendre tout le code

### Quand reconsiderer DDD

Adopter DDD si Alice-Engine evolue vers:

1. **Multi-domaines**: Gestion licences, calendrier, paiements, notifications
2. **Equipe elargie**: 5+ developpeurs necessitant bounded contexts
3. **Regles metier complexes**: Workflows avec etats, branches, rollbacks
4. **Event-driven**: Besoin de Domain Events, CQRS, Event Sourcing

Indicateurs de bascule:
- Plus de 3 Bounded Contexts identifies
- Plus de 50 regles metier
- Plus de 5 developpeurs
- Couplage fort entre modules

### Consequences

#### Positif
- Code simple et explicite
- Onboarding rapide (< 1 jour)
- Pas de framework DDD a maitriser
- Performance: pas d'indirection inutile

#### Negatif
- Si le domaine se complexifie, refactoring necessaire
- Moins de patterns "enterprise-ready"

#### Neutre
- ISO 42010 respecte (documentation architecture)
- SRP respecte (separation des couches)
- Testabilite equivalente

### Alternatives evaluees

| Approche | Verdict | Raison |
|----------|---------|--------|
| DDD complet | Rejete | Over-engineering |
| DDD-lite (tactique only) | Rejete | Benefice insuffisant |
| Hexagonal | Rejete | Ports/Adapters excessifs |
| Clean Architecture | Rejete | 4 couches = trop pour le projet |
| **Layered + SRP** | **Accepte** | Equilibre complexite/benefice |

### References

- ISO 42010: Architecture Description
- Martin Fowler: "Is Design Dead?" (simplicite vs patterns)
- Eric Evans: "DDD - Tackling Complexity" (quand utiliser DDD)
- YAGNI: You Aren't Gonna Need It

---

*Derniere mise a jour: 8 Janvier 2026*

---

## ADR-008: init_score_alpha per-model (V9)

**Date**: 11 Avril 2026
**Statut**: Accepte
**Detail**: `docs/architecture/ADR-008-alpha-per-model.md`

### Decision
alpha DOIT etre tune independamment par modele. LightGBM (leaf-wise) = 50x plus
sensible que XGBoost (depth-wise). NE JAMAIS appliquer alpha uniforme.

### Consequences
- LightGBM alpha=0.4, XGBoost alpha=0.5, CatBoost TBD
- config/MODEL_SPECS.md = source de verite per-model

---

## ADR-009: HP search on recent season (V9)

**Date**: 11 Avril 2026
**Statut**: Accepte

### Contexte
HP search sur full dataset (1.1M) = 2-19 trials/10h. Trop lent pour CatBoost.

### Decision
HP search sur saison=2022 uniquement (~62K train, 71K valid).
Training Final utilise le dataset complet (1.1M) avec les best params.

### Raisons
1. AUTOMATA (NeurIPS 2022): subset HP search = HP quality comparable, 3-30x speedup
2. ISO 5259-2: donnees recentes = meilleure representativite pour le deploiement
3. Valide empiriquement: Grid 200K et Optuna 1.1M trouvent memes directions
4. Budget: 100+ trials/10h au lieu de 2-19

### Consequences
- ALICE_HP_MIN_SEASON=2022 (env var override possible)
- Grid et Optuna sur MEMES donnees pour comparaison directe
- Resultats absolus (logloss) non comparables entre HP search et Training Final

---

## ADR-010: ISO validation locale vs cloud (V9)

**Date**: 11 Avril 2026
**Statut**: Accepte

### Decision
- **Kaggle kernels**: NaN audit (5 lignes, bloquant) + quality gates (logloss/RPS/ECE)
- **Local**: Pandera validation complète, robustness (ISO 24029), fairness (ISO 24027)
- Pandera non installe sur Kaggle (overhead pip install + fragilite)

### Consequences
- scripts/cloud/ n'importe PAS schemas/training_validation.py
- Le NaN audit inline remplit le role essentiel
- Phase 2 (promote_model.py) fait la validation ISO complete localement

---

## ADR-011: AutoGluon elimine du pipeline ALICE (YAGNI)

**Date**: 16 Avril 2026
**Statut**: Accepte
**Postmortem**: `docs/postmortem/2026-04-16-autogluon-v9-time-allocation-failure.md`
**Postmortem**: `docs/postmortem/2026-03-21-autogluon-kaggle-postmortem.md`

### Contexte

AutoGluon a ete evalue 2 fois sur ALICE :
- **2026-03-21** : 7 tentatives, 6 echecs (postmortem). Run degrade, sous-performant.
- **2026-04-15** : V9 benchmark best_quality, 2×T4 GPU, 7h. Resultat : test logloss 0.5716
  vs V9 LGB single 0.5619. Pire sur TOUTES les metriques. Processus OOM-killed.

Total investi : ~14h GPU (quota limite 30h/semaine), 10+ pushes, 3 sessions.

### Decision

**AutoGluon est ELIMINE du pipeline ALICE.** Pas de re-run, pas de reconfiguration.
Les modules `scripts/autogluon/` restent dans le repo (code ISO historique) mais ne sont
plus dans le pipeline actif. Le pipeline ML est : Optuna HP search → Training Final
(XGB/LGB/CB) → OOF stack → champion selection.

### Raisons (structurelles, pas conjoncturelles)

1. **Pas de residual learning.** AG ne supporte pas `base_margin` / `init_score`.
   ALICE utilise l'Elo comme baseline (92% du signal). Sans residual, AG reduit le
   probleme de 0.92→0.57 a 0.92→0.57 — mais nos modeles font 0.92→0.56 AVEC residual.
   L'ecart n'est pas du au tuning mais a l'architecture d'apprentissage.

2. **Calibration incompatible avec le CE.** Le CE d'ALICE requiert P(draw) calibree
   (45% de la variance E[score], cf. project_v9_ce_multi_equipe). AG V9 : ECE_draw=0.0209,
   draw_bias=-0.0137 (sous-estime les draws). V9 LGB : ECE_draw=0.0145, draw_bias=0.0136.
   AG `calibrate=True` fait une calibration globale, pas la temperature scaling per-class
   (Guo 2017) requise par ALICE.

3. **Budget temps incalculable.** AG best_quality sur 1.2M rows genere 110 configs avec
   allocation temporelle non deterministe. FASTAI a pris 4503s, LightGBMXT 10257s,
   CatBoost 167s, XGBoost 190s. Les GBMs (nos meilleurs modeles) ont recu 1.3% du budget.
   Impossible a planifier dans un kernel Kaggle 12h.

4. **Memoire insuffisante.** RF, ExtraTrees (×4 L1, ×4 L2) tous OOM-skipped sur 32GB.
   AG estime 16GB requis par modele. Le dataset ALICE (1.2M × 204 features) est trop gros
   pour les modeles non-GBM en bag mode.

5. **Aucune valeur ajoutee.** AG best single (LightGBM_BAG_L2, val 0.5121) produit un test
   logloss 0.5716 — pire que notre LGB V9 single (0.5619). Le stacking AG n'apporte rien
   car la diversite est absente (90.5% du poids sur un seul LGB).

### Alternatives evaluees

| Approche | Verdict | Raison |
|----------|---------|--------|
| AG best_quality (actuel) | Elimine | 5 raisons ci-dessus |
| AG GBM-only (hyperparameters={'GBM': {}, 'CAT': {}, 'XGB': {}}) | Rejete | Residual learning toujours absent. Revient a faire Optuna sans init_scores. |
| AG avec features Elo en input | Teste (V9) | Ne remplace pas le residual — feature input ≠ logit baseline |
| AG comme meta-learner sur OOF | Rejete | MLP/LogReg local suffit, pas besoin d'AG pour 6 inputs |

---

## ADR-012: Validation FFE autonome — regles simplifiees Phase 2, Flat-Six Phase 4

**Date**: 18 Avril 2026
**Statut**: Accepte

### Contexte

ALICE doit produire des compositions CONFORMES aux regles FFE. Deux sources de
validation existent :

1. **Chess-app Flat-Six** : moteur complet, 30 validateurs TypeScript, 22 fichiers
   JSON de regles, 7 types de competition. Production-grade. Mais chess-app est en
   impasse de dev (60K lignes frontend a auditer). ALICE ne peut pas en dependre.

2. **REGLES_FFE_ALICE.md** : doc exhaustive dans le projet ALICE avec la matrice des
   7 competitions, les configs Python (`ReglesCompetition(TypedDict)`), et la fonction
   `valider_composition()` parametree.

### Decision

**Phase 2 : ALICE embarque les regles FFE autonomes** via `services/ffe_rules.py`.
Implementation basee sur la config `ReglesCompetition` de REGLES_FFE_ALICE.md §7.7
(matrice complete). Pre-filtre joueurs + post-check composition.

**Phase 4 : Port Flat-Six en Python** si ALICE doit supporter toutes les competitions
avec la meme exhaustivite que chess-app. Ou integration API si chess-app reprend.

### Regles implementees Phase 2 (interclubs adultes N1-N6)

| # | Regle | Article | Parametrable par competition |
|---|-------|---------|------------------------------|
| 1 | Ordre Elo 100pts | A02 3.6.e | `ordre_elo_obligatoire: bool` |
| 2 | 1 joueur = 1 equipe | physique | non parametrable |
| 3 | Joueur brule | A02 3.7.c | `seuil_brulage: int` (A02=3, F01=1, J02=4) |
| 4 | Meme groupe | A02 3.7.d | non parametrable |
| 5 | Match count | A02 3.7.e | `max_parties_saison: int` |
| 6 | Noyau 50% | A02 3.7.f | `noyau: int, noyau_type: str` |
| 7 | Max mutes | A02 3.7.g | `max_mutes: int` (N1-N3=3, N4=1) |
| 8 | Quota etrangers | A02 3.7.h | `min_fr_eu: int` |
| 9 | FR genre obligatoire | A02 3.7.i | N1/N2 : 1M FR + 1F FR |
| 10 | Elo max N4 | A02 3.7.j | `elo_max: int` (N4=2400) |
| 11 | Team size | A02 3.7.a | `taille_equipe: int` |

### Limitations Phase 2 (DOCUMENTEES)

| Regle NON implementee | Raison | Risque | Remede |
|----------------------|--------|--------|--------|
| Ordre par age (Jeunes J02) | Pas de date_naissance fiable dans parquets | Compo jeunes invalide | Phase 3 : scraper dates naissance FFE |
| Categories age par board (J02) | Idem | Contrainte age non verifiee | Phase 3 |
| Double partie ech 7-8 (J02) | Logique specifique | Comptage incorrect | Phase 3 |
| Playoff eligibility (A02 3.7.k) | Besoin historique national complet | Joueur ineligible non detecte | Phase 4 : historique saison complete |
| Team strength order (A02 3.7.b) | Warning seulement, pas bloquant | Aucun (warning) | Phase 4 si pertinent |
| Arbitre (A02 3.7) | Non bloquant + data manquante | Aucun | Hors scope ALICE |
| Home-grown (A02 2.1) | Non bloquant + data manquante | Aucun | Hors scope |
| Coupe Loubatiere (C03) | Elo max 1800 — config dispo dans ReglesCompetition | Ajout trivial si necessaire | Ajouter config quand demande |
| Coupe Parite (C04) | 2H+2F + elo_total — config dispo | Idem | Idem |
| Coupe France (C01) | Ordre libre — config dispo | Idem | Idem |
| Scolaire (J03) | Appartenance etablissement pas club | Hors cible ALICE | Hors scope |

### Donnees requises pour les 11 regles

| Donnee | Source | Statut |
|--------|--------|--------|
| Elo joueur | parquets FFE (scrappes) | DISPONIBLE |
| Historique matchs (brulage, match count) | parquets FFE | DISPONIBLE |
| Noyau (joueurs ayant joue) | parquets FFE | DISPONIBLE |
| Statut mute | joueurs.parquet | DISPONIBLE |
| Nationalite | joueurs.parquet (scrapper) | PARTIEL — a verifier |
| Sexe | joueurs.parquet (scrapper) | PARTIEL — a verifier |
| Division/competition | calendrier FFE | DISPONIBLE |

### Consequences

#### Positif
- ALICE 100% autonome — fonctionne sans chess-app
- 11 regles bloquantes = compositions legales pour interclubs adultes
- Config parametrable par competition (ReglesCompetition TypedDict)
- Meme source de verite que chess-app (reglements FFE 2025-26)

#### Negatif
- Duplication partielle avec Flat-Six (risque divergence)
- Jeunes (J02) non supporte Phase 2 (besoin dates naissance)
- Coupes non supportees Phase 2 (configs dispo, pas implementees)

#### Mitigation divergence
- ReglesCompetition = source de verite ALICE, derivee des PDF FFE
- Chess-app Flat-Six = source de verite chess-app, derivee des memes PDF
- Les deux DOIVENT produire les memes resultats — test de regression croisse
  quand integration chess-app se fait (Phase 4)

### Alternatives evaluees

| Approche | Verdict | Raison |
|----------|---------|--------|
| Dependre de chess-app Flat-Six | Rejete | Chess-app en impasse (60K lignes), ALICE bloquee |
| Porter 30 validateurs TS en Python | Rejete Phase 2 | Effort disproportionne, coupes/scolaire hors cible |
| 8 regles simplifiees | Insuffisant | Manque team size, elo max, FR genre = compos illegales |
| **11 regles parametrees (actuel)** | **Accepte** | Couvre interclubs adultes N1-N6, extensible coupes |

---

### Consequences

#### Code
- `scripts/autogluon/` : conserve pour historique ISO, mais HORS pipeline actif
- `scripts/cloud/train_autogluon_v9.py` : archive, pas de v5
- `scripts/cloud/train_autogluon_kaggle.py` : archive
- `config/hyperparameters.yaml` section autogluon : conservee, non utilisee
- `config/MODEL_SPECS.md` section AutoGluon : maj avec statut ELIMINE
- `scripts/cloud/upload_all_data.py` : AG modules toujours uploades (pas de regression)

#### Pipeline
- Champion selection = 4 candidats (LGB, XGB, CB, Stack XGB+LGB), pas 5
- Phase 2 (API/CE) ne depend pas d'AG
- ISO validation locale (`scripts/autogluon/iso_*.py`) : historique, remplacee par
  validation integree dans le pipeline V9

#### Lecons (feedback memories)
- `feedback_time_budget_kernels` : CALCULER avant d'ecrire
- `feedback_websearch_api_before_code` : WebFetch doc param par param
- `feedback_no_lies` : ne pas promettre AG > hand-tuned sans donnees
