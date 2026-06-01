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

---

## ADR-018: ALIDataCache historical state reconstruction depuis echiquiers.parquet

**Date**: 10 Mai 2026
**Statut**: Accepte (D8 Phase 3.5 STRICT scope §1.3 N≥200)

### Contexte

D8 audit (Phase 3.5 STRICT, spec `2026-04-30-d8-fairness-robustness-design.md`)
exige N≥200 matches multi-saisons cross-year pour validation ML champion. Tentative
initiale (saisons 2021-2024 × division N3) a échoué empiriquement à 0 matches
sur saisons antérieures.

**Root cause** : `ALIDataCache.team_to_club` et `joueurs_by_club` sont chargés
depuis `joueurs.parquet` qui contient UNIQUEMENT le snapshot CURRENT (scrape FFE
le plus récent). Pour saison antérieure, équipes/clubs/rosters ont muté
(départs, dissolutions, transferts), `team_to_club.get(equipe_dom)` retourne
None → enumerate_candidates exclut le match.

### Decision

Ajouter `classmethod ALIDataCache.from_parquets_at_saison(joueurs_path,
echiquiers_path, saison)` qui reconstruit l'état historique du cache à partir
des matches eux-mêmes (`echiquiers.parquet`).

### Reasons

1. `echiquiers.parquet` contient pour CHAQUE match toutes les infos requises :
   `equipe_dom`, `equipe_ext`, `nr_blanc/noir`, `nom`, `prenom`, `elo_blanc/noir`,
   `equipe` (club_id).
2. Reconstruct historique = groupby(equipe, nr_ffe) → roster réel à saison X.
3. Permet audit D8 cross-year SOTA (spec §1.3 N≥200) sans porter joueurs.parquet
   à un format temporal versioning lourd.
4. `from_parquets()` API CURRENT préservée (aucun impact app/main.py FastAPI).

### Limitations reconnues (R-ALI-05 NEW)

| Champ historique | Reconstructible ? | Workaround |
|------------------|-------------------|------------|
| Elo (per saison) | ✅ via `elo_blanc/noir` | median(elos) across rondes |
| nr_ffe, nom, prenom | ✅ stable FFE | direct |
| Club d'équipe | ✅ via `equipe` | direct |
| Catégorie (Sen/Vet/etc) | ⚠️ peut être nullable historique | défaut "Sen" |
| Genre | ⚠️ peut être nullable | défaut "M" |
| `mute` status | ❌ non capturé historique | défaut False — **R-ALI-05** |
| `licence_active` | ❌ non historique | défaut True (a joué donc actif) |
| `age_min/max` | ❌ non historique | None |

`RuleEngine.A02 §3.7.f noyau` (mutes) non validable empiriquement historique
→ tag violations potentielles dans D8_FAILURE_ANALYSIS_LOG.md.

### Consequences

- **Code** : `services/ali/cache.py` ADD classmethod ~80L. Tests 5 NEW dans
  `tests/services/test_cache.py`.
- **Production app** : `app/main.py::lifespan` continue d'utiliser
  `from_parquets()` (current state). Pas de régression FastAPI.
- **D8 pipeline** : `BacktestHarness.setup(historical_saison=int|None)` route
  vers la bonne factory. Default None = current behavior.
- **Audit cross-year** : R-ALI-04 (drift FFE rules + roster turnover) validable
  empiriquement Phase B (saisons 2021-2023).
- **Phase 4+ extension** : reconstruction historique extensible à J02 jeunes
  (D3) et Coupes (D4) en lisant les divisions adaptées.

### Sources

- ISO/IEC TR 24029-2:2024 §6.6 robustness under distribution shift cross-year
- Tran et al. 2022 NeurIPS — distribution shifts ML
- Spec D8 §1.3 (`2026-04-30-d8-fairness-robustness-design.md`)
- Plan implémentation `2026-05-10-d8-historical-multidiv-implementation.md`

---

## ADR-020: D8 match identity `groupe` + conformal `support_max` fix

**Date**: 11 Mai 2026
**Statut**: Accepté (user explicit 2026-05-11 "SOTA ML + ISO conforme")
**Detail**: `docs/architecture/adr/ADR-020-d8-groupe-filter-and-conformal-support-fix.md`

### Décision

3 fixes structurels D8 Phase A 2024 :

1. **Match identity étendue** : `MatchCandidate.groupe` propagé partout (Top 16
   saison 2024 = 4 groupes Groupe A/B + Poule Haute/Basse séquentiels). Sans
   `groupe`, `_select_match_rows` mélange Phase 1 + Phase 2 → invariant FFE
   trip → matches skipped.
2. **Conformal `support_max`** : `conformal_set_size_mean` clip [0, K=team_size]
   au lieu de [0, 1.0]. Pour E[score] ∈ [0, 8], le clip [0,1] saturait
   `set_size_mean=1.0` artificiellement. Gate G_ROB_07 non-discriminant.
3. **`ALICE_MAX_MATCHES` env var** : RunnerConfig.max_matches paramétrable via
   env (default 50 préservé). Wrappers Phase A bump à 200 pour buffer post-filter
   conformal N≥31.

### Conséquences

- 6 fichiers source + 5 wrappers Phase A + 3 fichiers tests modifiés.
- 49 tests scope ciblé PASS + 9 ground_truth (slow) PASS + 3 test_runner E2E PASS.
- Backward compat préservé (defaults safe : groupe="", support_max=1.0, max_matches=50).
- Phase A v2 outputs (3 COMPLETE N1/N2/N3) **à invalider** : `set_size_mean=1.0`
  saturé artificiellement, Phase A v3 re-push requis.

### Sources SOTA

- Vovk 2024 §2.3 split conformal
- Angelopoulos & Bates 2023 §4.2 efficiency
- ISO 5259 data lineage, ISO 24029 §5.3 robustness, ISO 27034 input validation

---

## ADR-023: Frontière de responsabilité ALICE ↔ chess-app + contrat de données live

**Date**: 1 Juin 2026
**Statut**: Accepté (décision déléguée par owner, harnais ISO 42010 + SOTA-ML)
**Detail**: `docs/architecture/adr/ADR-023-alice-chessapp-responsibility-boundary-live-data-contract.md`

### Décision principale

Frontière par **cycle de vie de la donnée** : chess-app = source de vérité
**opérationnelle live** (scraping FFE, rosters/calendrier/équipes engagés, multiTeam,
composition tenant) ; ALICE = **cerveau ML + offline feature store historique**
(entraînement + backtest). ALICE est un **service d'inférence stateless** : chess-app
lui envoie une **requête auto-suffisante** (roster FFE-ids + Elo + `simultaneous_teams`
+ contexte match) ; ALICE enrichit par clé d'entité depuis son store historique et
répond. **ALICE ne scrape jamais, ne lit jamais la base chess-app, ne rappelle jamais.**

Évite le shared-database anti-pattern (microservices.io) + le training-serving skew
(AWS ML Lens MLREL-07). Re-scope T4 : `build_clubs_teams.py` offline (parquet ALICE),
pas `sync_clubs_teams.py` (REST chess-app). N'impacte pas ADR-013 (règles statiques).

Dette tracée (Phase 5 intégration) : `D-2026-06-01-live-data-integration-contract`
(le "tuyau" live n'existe pas encore — ALICE ne lit que les parquets figés aujourd'hui ;
disponibilité roster adverse chez chess-app non vérifiée) + `D-2026-06-01-historical-store-refresh`.

---

## ADR-022: D8 Phase A acceptance verdict — Phase 4a ALI conditional retained (D-P3-19 empirical confirmation)

**Date**: 16 Mai 2026
**Statut**: Accepté (user explicit 2026-05-16, pivot diagnostique guidé ISO+SOTA ML)
**Detail**: `docs/architecture/adr/ADR-022-d8-phase-a-acceptance-verdict-ali-conditional-phase-4a.md`

### Décision principale

Phase 4a (ALI joint conditionnel CE-adverse miroir Approche A SOTA, déjà validée
2026-04-28) RETENUE comme entry strategy. 10/13 FAIL D8 Phase A = confirmation
empirique quantifiée D-P3-19 / R-ALI-06 (T22 post-mortem 2026-04-28).

### Verdict aggregator empirique 2026-05-16

- **6/19 PASS, 13 FAIL, 0 INCONCLUSIVE** sur 492 matches Phase A (Top 16 v4 + N1-N4 v3).
- Cross-niveau gap stress Elo ×20 : Top 16 = 0.335 vs N4 = 0.017 @ 1% noise.
- ECE_ali uniforme cross-strata Elo (Q1 n=39 = 0.468, Q2 n=38 = 0.467) → calibration ALI mauvaise par construction (sample pool club total ignore A02 §3.7.b).

### Décisions dérivées

1. **Phase 3.5b cleanup mineur** (non-bloquant Phase 4a entry, tracé) :
   - G_ROB_07 threshold `set_size_max ≤ 3.0` absolu → `set_size_mean / support_max ≤ 0.50` ratio (Angelopoulos 2023 §4.2)
   - Kernel `scripts/d8/dro.py` agrégation `min over per-match` → `percentile(5%)` (Sinha 2018 §4)
   - Refactor `_fairness_metrics` consommer `r["breakdowns"]` au lieu de proxy `_by_ronde`
2. **Phase 3.6 retraining adversarial** : NON RETENUE pré-emptive (patch symptomatique), PLANIFIÉE contingency post-Phase 4a si robustness insuffisante.
3. **Aggregator patch ISO 5055 SRP** (cette session RÉSOLUE) : `aggregate.py main()` argparse `--mode {saison,phase-a}` + branche `load_audit_reports` ADR-019. 29 tests PASS (17 saison + 12 phase-a).

### Pourquoi Phase 4a (pas Phase 3.6)

Madry 2018 PGD + Goodfellow 2015 augmentation rendrait MLP plus robuste à small noise local Elo, **mais ne corrige pas le bug structurel sampling pool naïf** (cause profonde D-P3-19 manque A02 §3.7.b conditionnement multi-équipes). Logique SOTA = corriger architecture (Phase 4a) avant patcher symptômes (Phase 3.6 contingency).

### Conséquences

- 5 fichiers source touchés (aggregate.py + types.py + nouveau test file + acceptance report + ADR-022).
- 29 tests aggregator PASS (saison + phase-a modes).
- 11/13 FAIL → phase résolution prévue (Phase 4a), 3/13 FAIL → Phase 3.5b cleanup.
- 7 dettes traceables (no silent debt) : 2 RÉSOLUES + 4 OPEN Phase 3.5b/5/contingency.

### Sources SOTA

- Mehrabi 2021 §4.1 (fairness max_gap)
- Pleiss 2017 §4 (calibration ECE per-group)
- Goodfellow 2015 ε=0.01 robustness
- Tran 2022 §3.4 roster turnover
- ISO 24029-2 (robustness), 24027 (fairness), 42001 (lifecycle), 42005 (impact), 23894 (risk)

### Liens

- ADR-016 (Phase 4a Approche A SOTA decision)
- ADR-019 (Phase A multi-divisions)
- ADR-020 (groupe filter + conformal support)
- ADR-021 (Top 16 rondes_default)
- D-P3-19 source of truth : `memory/project_debt_current.md`
- Acceptance report détaillé : `reports/d8/phase_a/2026-05-16-acceptance.md`

---

## ADR-021: D8 `DIVISION_RONDES_DEFAULT` mapping (Top 16 full 7 rondes)

**Date**: 14 Mai 2026
**Statut**: Accepté (user explicit 2026-05-14 "démarche ISO+SOTA guide solution")
**Detail**: `docs/architecture/adr/ADR-021-d8-rondes-default-division-specific.md`

### Décision

Introduction d'un mapping `DIVISION_RONDES_DEFAULT` dans `scripts/d8/run.py`
override division-specific consulté AVANT le fallback saison-based pour
`rondes_default`. Top 16 = `(1,2,3,4,5,6,7)` complet (régulière 1-7 + finale 1-4
agrégées). N1-N4 absents du mapping → fallback `(5,7,9,11)` préservé.

### Root cause Phase A v3 Top 16 ERROR (2026-05-12, log Kaggle 0-byte)

`rondes_default = (5, 7, 9, 11) if saison >= 2022` calibré pour Nationale 1-4
format championnat fin-saison. Top 16 format élite (7 rondes régulière +
4 rondes finale) capture seulement 16/88 candidates → `len(per_match) < 31` →
`raise RuntimeError(msg)` uncaught (ValueError catch run.py:333 ne match pas) →
kernel ERROR Kaggle worker, stderr non-flushed (log .log 0-byte).

### Conséquences

- 1 fichier source modifié (`scripts/d8/run.py` +14 LoC mapping + 4 LoC branche).
- 1 fichier tests créé (`tests/d8/test_run.py` 3 tests).
- Top 16 v4 attendu : **88 candidates** → conformal robuste.
- Backward compat strict (N1-N4 inchangés, mapping additif).
- Phase A v3 outputs N1-N4 valides (rondes default 5,7,9,11 OK pour Nationale).
- Mapping extensible futurs formats FFE (Top 12, Coupes, ...) sans toucher
  `_run_backtest`.

### Sources SOTA

- Vovk 2024 §2.3 split-conformal minimum N=30
- Lei et al. 2014/2018 minimum calibration set 30+
- ISO 5055 SRP single-source-of-truth, ISO 42010 ADR, ISO 27034 input validation

### Liens

- ADR-020 (préalable structurel groupe propagation) — ADR-021 complète sur
  dimension temporelle (rondes) après ADR-020 dimension identitaire (groupe).

---

## ADR-019: D8 audit pivot multi-divisions × multi-saisons (rejection trade-off "saison 2024 N3 seule")

**Date**: 10 Mai 2026
**Statut**: Accepte (user explicit 2026-05-10 "j'exige produit livrable SOTA pour commercialisation")

### Contexte

Premier diagnostic D8 sur saison 2024 N3 seule a livré N=38 matches valides
(<200 spec §1.3). Trade-off paresseux initialement proposé : "accepter N=38 saison
2024 N3 seule, tracer dette multi-saisons Phase 5+". User a rejeté ("trade-off
ou paresse !") et exigé résolution propre.

### Decision

Pivot architecture D8 vers **5 divisions × saisons disponibles** :

- **Phase A** (atteinte spec §1.3 garantie) : saison 2024 × {Top 16, N1, N2, N3, N4}
  - 5 kernels CPU Kaggle
  - Total candidates ~2700 → N>>200 garanti
- **Phase B** (optional post-V1, validation drift R-ALI-04) : saisons 2021-2023
  × 5 divisions, sur état historique reconstruit (ADR-018)
  - 15 kernels supplémentaires (optional)

### Reasons

1. Spec §1.3 N≥200 atteinte sans relâcher
2. ISO 24027 §6 `by_niveau` breakdown enrichi : 5 niveaux Elo distincts
   (Top 16 ≈ 2300+, N1 ≈ 2100-2300, N2 ≈ 1900-2100, N3 ≈ 1700-1900,
   N4 ≈ 1500-1700) → validation champion cross-niveau pour SaaS multi-tenant
3. R-ALI-01 (PRIVATE rules unverifiable cross-niveau N1 stricter) quantifiable
   empiriquement
4. Naming FFE stable saison 2024 (pas de problème "Nationale III" vs "Nationale 3")
5. CPU illimité Kaggle → 5 kernels parallèles = ~12h wallclock max, vs GPU 30h
   quota/sem bottleneck

### Alternatives rejetées

| Alternative | Verdict | Raison |
|-------------|---------|--------|
| Saison 2024 N3 seule (N=38) | Rejeté | Spec §1.3 non atteinte, R-ALI-01 / R-ALI-04 non quantifiés |
| GPU acceleration cuML/Triton | Rejeté | Per-match batch 8 boards trop petit, overhead transfer > gain. Quota GPU 30h/sem bottleneck. Port LGB/XGB/CB/MLP = effort substantiel + risque regression D-P3-13. |
| Multi-saisons seules (sans multi-div) | Insuffisant | Reconstruction historique = R-ALI-05 NEW, gain by_niveau perdu |
| **Multi-divisions saison 2024 + Phase B optional** | **Accepte** | Phase A garantit spec §1.3, Phase B post-V1 valide drift cross-year |

### Consequences

- **Code** : `scripts/d8/run.py` lit `ALICE_DIVISION` env var. `D8Lineage` extend
  avec field `division`. `aggregate.py` fuse keyed par `(saison, division)`.
- **Wrappers** : 5 wrappers Phase A `run_2024_{top16,n1,n2,n3,n4}.py` +
  kernel-metadata associés.
- **Aggregator** : `D8_FINDINGS.md` template ajoute sections by_niveau (Phase A)
  + by_saison (Phase B). Phase A skip by_saison si saison unique.
- **Tests** : ~10 nouveaux tests parametrisés (saison, division).
- **ADR-016** (ALI multi-équipes joint conditionné Phase 4a) : non impacté.

### Sources

- Spec D8 §1.3 (N≥200), §3.5 (by_niveau breakdown ISO 24027 §6.1)
- Pappalardo 2019 sport prediction baseline (per-level evaluation SOTA)
- Mehrabi 2021 §4.1 (max_gap fairness across groups)
- WebSearch Kaggle limites 2026 (CPU illimité, GPU 30h/sem)
- NVIDIA cuML/FIL benchmark (rejetée car batch trop petit)
