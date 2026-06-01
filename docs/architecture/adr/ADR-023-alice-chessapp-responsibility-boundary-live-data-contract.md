# ADR-023 : Frontière de responsabilité ALICE ↔ chess-app + contrat de données live

**Date** : 2026-06-01
**Status** : ACCEPTED
**Décideur** : Claude (ingénieur ML senior), décision déléguée par le owner (user
"je ne sais que ce que je veux, le comment c'est toi" — 2026-06-01). Harnais de
décision : ISO 42010 (architecture) + SOTA-ML + ISO 5259/25010/27001.
**Context** : Phase 4a T4 (re-scope) — la prémisse "ALICE synchronise un référentiel
clubs/équipes depuis l'API REST chess-app" s'est révélée invalide au traçage du code.

## Contexte

Deux constats factuels (vérifiés dans le code, 2026-06-01) ont déclenché cette décision :

1. **chess-app est multi-tenant et tenant-scoped.** Toute requête clubs/équipes/joueurs
   est filtrée `{ clubId }` (`Team.find({ clubId })`, `clubsReadRoutes` exige
   `authenticate` + `requireClubId`). chess-app ne connaît **que ses clubs abonnés**,
   pas le référentiel FFE global (~4000 clubs). Il **ne peut pas** fournir les équipes
   d'un club adverse arbitraire.

2. **ALICE n'a aucune capacité d'acquisition de données live.** Au lifespan
   (`app/main.py`), ALICE charge **uniquement** ses deux parquets historiques figés
   (`joueurs.parquet`, `echiquiers.parquet`, source HuggingFace `Pierrax/ffe-history`,
   instantané mars 2024). `PlayerPoolLoader` → `ALIDataCache.lookup_club` lit **dans
   ce cahier figé**. ALICE n'a ni scraping, ni feed live, ni mise à jour des adhérents/
   adversaires/équipes/clubs. Elle ne sait raisonner que sur les clubs/joueurs présents
   dans l'historique (≤ 2024).

Conséquence : la tâche T4 (Q4 → "pattern ADR-013 : sync REST chess-app → JSON vendoré")
était invalide pour **deux** raisons — (a) chess-app ne possède pas la donnée globale,
(b) faire dépendre les tests/build d'ALICE d'un chess-app vivant est un couplage fort.

Le sujet sous-jacent — **qui acquiert quelle donnée, et comment elle circule entre
chess-app et ALICE** — n'avait jamais été spécifié. Cet ADR le pose.

## Décision

### Frontière de responsabilité (par cycle de vie de la donnée, pas par entité)

| Domaine de donnée | Propriétaire (source de vérité) | Justification SOTA/ISO |
|---|---|---|
| **Données opérationnelles vivantes** (saison courante : rosters, calendrier, clubs/équipes engagés, adversaires, mutations, mutés) | **chess-app** | Lui seul scrape la FFE en continu (`scraping/ffeCheerioScraper`) + possède la feature `multiTeam` (détection équipes simultanées) + la composition manuelle tenant. Single source of truth opérationnelle. |
| **Données historiques** (entraînement + backtest, ≤ saison N-1) | **ALICE** | Territoire ML, hors-ligne. ALICE possède l'historique FFE complet (HuggingFace) que chess-app n'a pas. = **offline feature store**. |
| **Modèles ML entraînés** (MLP champion, preference model, calibrateurs) | **ALICE** | Cœur ML. |
| **Règles FFE normatives** (statiques, A02) | **chess-app → vendoré ALICE** (ADR-013, INCHANGÉ) | Statiques, normatives, rarement modifiées : un snapshot JSON vendoré est valide (≠ donnée live). |

### Contrat d'intégration : requête API auto-suffisante (synchrone, sens unique)

ALICE est un **service d'inférence ML stateless**. Au moment d'un vrai match :

```
chess-app  ──(POST /compose : roster FFE-ids + Elo courant +     ──►  ALICE
              simultaneous_teams[{name, division, board_count}] +
              contexte match {division, ronde, date})
                                                                  ◄── (prédiction + compo recommandée)
```

- **chess-app envoie dans le payload TOUTE la donnée opérationnelle** du match (faits
  vivants : qui est dispo, Elo courant, équipes adverses simultanées du week-end via
  sa feature `multiTeam`, division → nombre d'échiquiers).
- **ALICE enrichit** ces entités (clé = FFE ID) avec ses **features historiques**
  calculées depuis son offline store (forme glissante, h2h, draw priors…), exactement
  comme à l'entraînement → **évite le training-serving skew** (AWS ML Lens MLREL-07).
- **ALICE n'appelle jamais chess-app en retour, ne lit jamais sa base, ne scrape jamais.**
  Couplage lâche par contrat API → évite le **shared-database anti-pattern**
  (microservices.io database-per-service).
- **Cold-start** : un joueur absent de l'historique d'ALICE → features dégradées en
  baseline Elo-only (déjà le comportement, cf. D1 résolu : fallback Elo).

### Conséquence directe sur T4 (re-scope)

- **Path PROD** : aucune donnée clubs/équipes à acquérir par ALICE — elle reçoit
  `simultaneous_teams` dans la requête (déjà prévu par T7 + consommé par `adverse_ce.py`
  qui lit `team.board_count`). Zéro sync, zéro JSON snapshot.
- **Path OFFLINE (backtest/tests/Kaggle)** : pas de caller → ALICE reconstruit les
  équipes simultanées historiques depuis **son propre parquet** (autonome, FFE-wide).
  T4 devient `scripts/build_clubs_teams.py` (fixture offline dérivé du parquet), **pas**
  `sync_clubs_teams.py` (REST chess-app). Aucune dépendance à chess-app pour les tests.

## Conséquences

**Positives :**
- Single source of truth par domaine ; zéro duplication de scraping (DRY).
- ALICE testable/buildable sans chess-app vivant (couplage lâche).
- Requête auto-suffisante → pas de training-serving skew, pas de shared-DB.
- T4 fortement réduit (plus de client REST/auth/staleness-vs-chess-app).
- Frontière claire = maintenabilité (ISO 25010).

**Négatives / dette ouverte (tracée, non enterrée) :**
- **Le "tuyau" live n'existe pas encore en code.** Aujourd'hui ALICE ne lit que les
  parquets figés → elle marche pour le **backtest historique**, pas pour un vrai match
  de la saison courante. L'implémentation du contrat (chess-app construit le payload
  enrichi + ALICE consomme + enrichit) = **chantier Phase 5 (intégration)**. Debt
  `D-2026-06-01-live-data-integration-contract`.
- **Disponibilité du roster adverse chez chess-app : NON VÉRIFIÉE.** chess-app a-t-il
  déjà le roster des clubs *adverses* (pas seulement du tenant) ? À investiguer en
  Phase 5. Si non, c'est à chess-app de le scraper (pas à ALICE). Debt idem.
- **Fraîcheur de l'offline store d'ALICE** : le parquet historique vieillit. Refresh =
  job batch hors-ligne (re-pull FFE history), **jamais** du scraping dans le path de
  service. Qui/quand → Phase 5. Debt `D-2026-06-01-historical-store-refresh`.

## Alternatives rejetées

- **ALICE scrape la FFE elle-même** : duplique le scraper de chess-app → 2 sources qui
  divergent, double maintenance de code fragile (pages FFE), ambiguïté "qui a raison ?".
  Anti-pattern (viole single-source-of-truth/DRY). Rejeté.
- **ALICE lit directement la base MongoDB de chess-app** : shared-database anti-pattern
  (couplage caché, schema lock-in). Rejeté au profit du contrat API.
- **Snapshot statique clubs/équipes vendoré (réapplication ADR-013 à la donnée live)** :
  invalide — chess-app tenant-scoped + donnée live obsolète en jours. C'est l'erreur de
  Q4 que cet ADR corrige. (ADR-013 reste valide pour les **règles** statiques.)
- **Étendre ALICE en système de gestion de données** : casse le rôle "cerise ML" voulu
  par le owner + crée l'overlap de responsabilité. Rejeté.

## Sources SOTA

- Microservices.io — *Database per service* / *Shared Database anti-pattern*
  (https://microservices.io/patterns/data/database-per-service.html) : data sovereignty,
  API/event communication, single source of truth par domaine.
- AWS Well-Architected ML Lens — **MLREL-07** *Ensure feature consistency across training
  and inference* : training-serving skew mitigation via feature consistency.
- Feature store literature (Feast, Aerospike) : offline store (training/batch) vs online
  store (serving), enrichment par clé d'entité — modèle de l'offline parquet d'ALICE.
- Sculley et al. 2015 "Hidden Technical Debt in ML Systems" (NeurIPS) : boundary erosion,
  data dependencies — justifie la frontière nette de responsabilité.

## Liens

- **Révise** : Q4 du spec Phase 4a + le framing T4 (spec §3.3, plan §T4).
- **N'impacte pas** : ADR-013 (vendoring des **règles** JSON reste valide).
- **Complète** : ADR-016 (Phase 4a Approche A) — la "détection équipes simultanées via
  canonical mapping chess-app" devient "via payload (prod) / parquet historique (offline)".
- **Phase 5** : ADR-022 §entry strategy — le contrat live est un prérequis prod (Pre-V1).
- Debts : `D-2026-06-01-live-data-integration-contract`,
  `D-2026-06-01-historical-store-refresh` (memory/project_debt_current.md).
