# ADR-023 : Frontière de responsabilité ALICE ↔ chess-app + contrat d'intégration (API composition)

**Date** : 2026-06-01
**Status** : ACCEPTED (validé par owner 2026-06-01 après exploration des deux codebases)
**Décideur** : Claude (ingénieur ML senior), décision déléguée par le owner. Harnais :
ISO 42010 (architecture) + ISO 27001 (sécurité/isolation) + ISO 25010 (maintenabilité) +
standards microservices + SOTA-ML (training-serving skew).
**Supersedes** : `docs/requirements/CDC_ALICE.md` §4.3 (dimension "ALICE lit MongoDB chess-app
en lecture seule") — VOIR §"Décision" pourquoi cette partie du CDC est abandonnée.
**N'impacte pas** : ADR-013 (vendoring des **règles FFE statiques** — domaine différent).

---

## ⚠️ Note de lecture pour Claude+1 / fresh-Claude

Cet ADR n'est PAS qu'un verdict : c'est le compte-rendu d'une investigation qui a corrigé
une erreur. La V1 de cet ADR (commit `f46094a`) affirmait "pas de base partagée" SANS avoir
lu le contrat pré-existant `CDC_ALICE.md §4`, et a été marquée "Accepted" trop vite. Le owner
a (à juste titre) exigé : *"ce n'est pas parce que c'est prévu/que ça existe que c'est adapté —
j'ai peut-être fait de la merde avant ; évalue contre les standards industrie."* Cette V2 est
le résultat de cette évaluation. **Si tu reprends ce sujet : re-pose les questions de §7, ne
prends pas cette décision pour acquise — re-challenge-la avec les données du moment.**

---

## 1. Contexte — pourquoi cet ADR existe

ALICE-Engine (`C:\Dev\Alice-Engine`, Python/FastAPI + ML) et chess-app (`C:\Dev\chess-app`,
TS/Fastify + MongoDB + React, SaaS multi-tenant de gestion de clubs d'échecs FFE) sont **deux
projets fortement couplés** : ALICE est la "cerise ML" sur le "gâteau" chess-app (CDC_ALICE.md
§1 : ALICE = "composant satellite" + "cœur différenciant du SaaS"). Le déclencheur de cet ADR
fut Phase 4a T4, dont la prémisse ("ALICE synchronise un référentiel clubs/équipes depuis
chess-app") s'est révélée invalide au traçage.

### 1.1 Ce qui existe RÉELLEMENT (vérifié dans le code, 2026-06-01)

**Côté chess-app** :
- **Tenant-scoped strict** : toute requête clubs/teams/players filtrée `{ clubId }`
  (`Team.find({ clubId })`, routes exigent `authenticate` + `requireClubId`). Isolation
  multi-tenant par JWT. → chess-app ne connaît **que ses clubs abonnés**, PAS le référentiel
  FFE global (~4000 clubs).
- **Scrape la FFE** (`features/scraping/ffeCheerioScraper` + `FFEPlayerScraper` +
  `FFECalendarScraper` + `FFEGroupDetailsScraper`) → stocke en MongoDB (db "chess-app").
- **Données adverses** : `Match.away` stocke les adversaires en `{ name, elo }` bruts depuis
  les résultats scrapés (`clubPlayer: false`), **pas** comme records `Player`. Route
  `GET /opponent/:clubId` existe → chess-app **peut scraper le roster d'un club adverse à la demande**.
- **Compose déjà localement SANS ML** : `flat-six/` (TeamComposer cascade/balanced/reinforce)
  + `features/multiTeam/` (conflicts = équipes simultanées, optimize). Il **sait déjà** "club X
  aligne équipes A,B,C ce week-end".
- **Intention d'intégration ALICE explicite mais NON implémentée** : `Match.ts` TODO P2 :
  *"Alimenter route `/_predict-opponent` avec données enrichies"* + *"Ne PAS inventer de joueurs
  adverses dans la DB"*. `Match.away.player` est optionnel, prêt à recevoir des prédictions.
  **Aucun** client HTTP vers ALICE, **aucun** `ALICE_URL`, **aucune** route `/_predict-opponent`
  réelle à ce jour.
- Déployé (Render, MongoDB Atlas). Auth JWT + httpOnly cookies + clubId.

**Côté ALICE** :
- **Aucun scraping, aucune acquisition live.** Au lifespan (`app/main.py`), charge UNIQUEMENT
  les parquets historiques figés (`joueurs.parquet` + `echiquiers.parquet`, HuggingFace
  `Pierrax/ffe-history`, instantané mars 2024). `pool_loader.lookup_club` lit ce cahier figé.
- **Configurée pour lire MongoDB chess-app en lecture seule, MAIS dormant** : `config.py`
  (`mongodb_database = "chess-app"`, "# MongoDB (lecture seule)"), `DataLoader.get_club_players`
  / `get_opponent_history` existent — mais `DataLoader` **n'est PAS instancié** dans `main.py`
  (seul l'audit logger ouvre Mongo). L'inférence tourne 100% sur parquet. `mongodb_uri` vide
  par défaut.
- **CDC_ALICE.md §4 (contrat pré-existant)** : ALICE = satellite ; chess-app appelle ALICE en
  REST (`POST /predict`) ; tableau §4.3 : "Données joueurs → ALICE = **lecture seule MongoDB**".

### 1.2 Le problème central

ALICE ne peut PAS être data-autonome pour la donnée vivante (pas de scraping, et on ne veut
pas dupliquer le scraper de chess-app). chess-app est le seul à scraper la FFE. **Donc la donnée
opérationnelle DOIT circuler de chess-app vers ALICE.** La question : **comment ?**

---

## 2. Options évaluées (contre standards industrie)

| Option | Description | Verdict |
|---|---|---|
| **A — Base MongoDB partagée (lecture seule)** | ALICE lit directement la db "chess-app" (= ce que CDC §4 prévoit) | ❌ **Rejeté** (anti-pattern, voir §3) |
| **B — API composition (requête auto-suffisante)** | chess-app met TOUTES les données du match dans l'appel REST ; ALICE stateless | ✅ **Retenu** |
| **C — API query (callback)** | ALICE rappelle chess-app en REST pour chercher la donnée | ❌ Rejeté (chatty + auth service + dépendance retour) |
| **D — Event-driven / réplication** | chess-app publie des events, ALICE maintient sa copie | ❌ Over-engineering pour 2 services solo |

---

## 3. Décision

**ALICE est un service d'inférence ML stateless. L'intégration se fait par API composition :
chess-app (orchestrateur) rassemble les données opérationnelles du match et les envoie dans
l'appel REST `/compose` (ou `/predict`). ALICE ne lit JAMAIS la base de chess-app.**

### 3.1 Frontière de responsabilité (par cycle de vie de la donnée)

| Domaine | Propriétaire | Justification |
|---|---|---|
| Données opérationnelles **vivantes** (rosters, calendrier, équipes engagées, adversaires, mutés) | **chess-app** | Seul scraper FFE + multiTeam + compo tenant. Source de vérité opérationnelle. INTOUCHÉ. |
| Données **historiques** (entraînement + backtest) | **ALICE** | Territoire ML, hors-ligne. Offline feature store (parquet HuggingFace) que chess-app n'a pas. |
| Modèles ML | **ALICE** | Cœur ML. |
| Règles FFE normatives statiques | chess-app → vendoré ALICE (ADR-013, INCHANGÉ) | Statiques ≠ donnée live ; snapshot vendoré valide. |

### 3.2 Contrat (sens unique, auto-suffisant)

```
chess-app (orchestrateur, owner données live)        ALICE (inférence ML stateless)
  rassemble : available_players (tenant) +
  opponent roster (via son scraper /opponent/:clubId) +
  simultaneous_teams[{name,division,board_count}] +
  match context {division, ronde, date, opponent_club_id}
        ──────── POST /compose (payload auto-suffisant) ────────►
                                                          enrichit par clé FFE-id depuis
                                                          son offline store historique
                                                          (features ML — même calcul qu'à
                                                          l'entraînement → pas de skew)
        ◄──────── prédiction adverse + compo recommandée ───────
  populate Match.away (predictions, sans créer de Player) + UI
```

- **ALICE ne touche jamais la base chess-app**, ne rappelle jamais, ne scrape jamais.
- **ALICE garde son offline store** (parquet) UNIQUEMENT pour ses features ML historiques
  (clé = FFE-id). Cold-start (joueur absent de l'historique) → baseline Elo-only.

### 3.3 Pourquoi B et pas A (la partie du CDC qu'on abandonne)

L'option A (lecture base partagée du CDC §4.3) est rejetée pour **3 raisons concrètes,
pas du dogme** :

1. **🔴 Isolation multi-tenant / ISO 27001 (décisif)** : chess-app filtre chaque requête par
   `clubId` (JWT). Si ALICE tape la base directement, elle **court-circuite cette barrière** et
   doit ré-implémenter parfaitement le scoping par tenant — une erreur = **fuite de données
   entre clubs clients**. En SaaS, rédhibitoire. Avec B, l'isolation est imposée à la source :
   chess-app n'envoie que les données que le tenant a le droit de voir.
2. **Couplage de schéma / ISO 25010** : schéma Mongoose (TS) re-modélisé en Python → un
   renommage de champ côté chess-app casse ALICE en silence. *Shared-database anti-pattern*
   (microservices.io : "communicate via APIs, not the database ; schema = implementation detail,
   not a stable interface").
3. **Double source de vérité de fait** + drift d'hypothèses entre 2 langages sur un schéma.

KISS check : B n'est PAS plus complexe — la base partagée *paraît* simple mais cache la synchro
de schéma + la ré-implémentation de l'isolation. chess-app assemble déjà ces données pour sa
propre UI → les envoyer dans la requête est quasi gratuit. CDC §4 a **raison sur le REST
trigger**, **tort sur la lecture base partagée**.

### 3.4 Conséquence sur Phase 4a T4

- **PROD** : ALICE n'acquiert rien — `simultaneous_teams` + roster adverse arrivent dans le
  payload (T7 + `adverse_ce.py::team.board_count` déjà prêts).
- **OFFLINE (backtest/Kaggle/tests)** : pas de caller → ALICE dérive les équipes simultanées
  historiques depuis son propre parquet. T4 = `scripts/build_clubs_teams.py` (offline fixture),
  PAS `sync_clubs_teams.py` (REST chess-app).

---

## 4. Trous d'implémentation (concrets, des DEUX côtés — pas "Phase 5 flou")

Le contrat est DÉCIDÉ ; l'implémentation manque des deux côtés. Inventaire précis :

**Côté chess-app** (le owner a confirmé que les solutions peuvent toucher chess-app) :
- [ ] Client `getAlicePrediction(clubId, opponentClubId, round, availablePlayers, ...)` (CDC §4.2 le prévoit, non créé). Fichier `backend/services/aliceClient.ts`.
- [ ] Route `/_predict-opponent` (TODO P2 de `Match.ts`, non créée) : orchestration fetch match → assemble payload → POST ALICE → populate `Match.away`.
- [ ] Assemblage du payload : roster adverse via `GET /opponent/:clubId` (scraper existant) + `simultaneous_teams` via `multiTeam.conflicts` (existant).
- [ ] Types TS du contrat (mapping de la réponse ALICE) + validation Zod.
- [ ] UI : afficher la prédiction adverse + confiance + reasoning (pour que le capitaine valide).

**Côté ALICE** :
- [ ] `ComposeRequest` étendu : recevoir roster + `simultaneous_teams` + contexte (amorcé T7).
- [ ] `pool_loader` : enrichir par FFE-id depuis l'offline store (entité fournie par payload)
      au lieu de `lookup_club` figé. Cold-start = Elo baseline.
- [ ] **Débrancher / NE PAS brancher** `DataLoader` MongoDB dans le path d'inférence
      (cohérence ADR-023 : pas de lecture base partagée). Garder Mongo uniquement pour l'audit log.
- [ ] (Optionnel) Auth du caller : valider que la requête vient de chess-app (clé partagée / JWT).

**Phase** : ces trous sont la **Phase 5 (intégration)** MAIS le **design est figé ici** (ADR-023),
donc ce n'est plus "à concevoir plus tard", c'est "à implémenter selon ce contrat". Debts :
`D-2026-06-01-live-data-integration-contract` + `D-2026-06-01-historical-store-refresh`.

---

## 5. Conséquences

**Positives** : single source of truth par domaine (DRY) ; isolation tenant préservée (ISO 27001) ;
contrat API stable/versionnable (ISO 25010) ; ALICE testable sans chess-app ; pas de
training-serving skew (features calculées pareil train/serve depuis l'offline store) ;
T4 réduit ; frontière nette = maintenabilité.

**Négatives / honnêtes** :
- Le contrat n'est PAS implémenté (les 2 listes de §4). ALICE marche aujourd'hui pour le
  **backtest historique** (parquet figé), pas pour un vrai match de la saison courante.
- Payloads plus gros (roster + roster adverse) qu'un simple ID — acceptable (quelques dizaines
  de joueurs), à mesurer si problème.
- **Non vérifié** : la qualité/complétude du roster adverse que chess-app peut scraper via
  `/opponent/:clubId` (à valider Phase 5).
- Le CDC_ALICE.md doit être annoté (§4.3 superseded) — fait dans le même commit.

---

## 6. Alternatives rejetées (détail)

- **A — Lecture base MongoDB partagée** (CDC §4.3) : anti-pattern + risque isolation tenant
  (§3.3). Rejeté. C'est la déviation principale de cet ADR vs le CDC.
- **C — API query (ALICE rappelle chess-app)** : ajoute une dépendance retour + auth service
  account + chattiness. Inférieur à mettre la donnée dans la requête initiale. Rejeté.
- **D — Event-driven / réplication de données** : ALICE maintiendrait sa copie via events.
  Over-engineering pour 2 services solo (le owner a explicitement flaggé l'over-engineering).
  Rejeté — à reconsidérer SEULEMENT si volume/latence l'imposent en Phase 5+.
- **ALICE scrape la FFE elle-même** : duplique le scraper chess-app → 2 sources qui divergent,
  double maintenance fragile. Rejeté.

---

## 7. Questions que Claude+1 / fresh-Claude DOIT re-poser (ne pas prendre cet ADR pour acquis)

1. **Le payload est-il trop gros en pratique ?** Mesurer la taille réelle (roster tenant +
   roster adverse + simultaneous_teams). Si > quelques 100 Ko récurrents → reconsidérer C ou
   un cache. (Aujourd'hui : hypothèse "acceptable", non mesurée.)
2. **chess-app sait-il vraiment fournir un roster adverse fiable** via `/opponent/:clubId` pour
   un match À VENIR (pas seulement post-match) ? À vérifier dans le code de scraping chess-app.
3. **L'enrichissement par FFE-id depuis l'offline store** couvre-t-il assez de joueurs (taux de
   cold-start) ? Mesurer sur une saison récente.
4. **Faut-il une auth caller→ALICE** dès maintenant (clé partagée) ou est-ce acceptable de
   différer (ALICE derrière le réseau privé Oracle VM) ? Trancher selon le déploiement réel.
5. **La fraîcheur de l'offline store** (parquet mars 2024) : à quelle cadence le rafraîchir, et
   par quel job batch ? (debt `D-2026-06-01-historical-store-refresh`.)
6. **Y a-t-il une donnée que SEULE la base partagée donnerait** efficacement (ex: historique
   compositions tenant volumineux) et qui justifierait une exception read-only ciblée ? Si oui,
   documenter l'exception, ne pas rouvrir A en bloc.

---

## 8. Sources

- Microservices.io — *Database per service* / *Shared Database anti-pattern*
  (https://microservices.io/patterns/data/database-per-service.html).
- AWS Well-Architected ML Lens — **MLREL-07** *Ensure feature consistency across training and
  inference* (training-serving skew).
- Feature store literature (Feast, Aerospike) — offline (training) vs online (serving) store,
  enrichment par clé d'entité.
- Sculley et al. 2015 "Hidden Technical Debt in ML Systems" (NeurIPS) — boundary erosion,
  data dependencies, CACE.
- ISO 27001 (isolation/sécurité multi-tenant), ISO 25010 (modularité/maintenabilité),
  ISO 42010 (architecture/ADR).

## 9. Liens

- **Supersedes** : `docs/requirements/CDC_ALICE.md` §4.3 (dimension lecture base partagée).
- **N'impacte pas** : ADR-013 (règles FFE statiques).
- **Complète** : ADR-016 (Phase 4a Approche A), ADR-022 (acceptance verdict / entry strategy).
- **Révise** : spec Phase 4a Q4 + §3.3 + T4 ; plan Phase 4a T4.
- **Debts** : `D-2026-06-01-live-data-integration-contract`, `D-2026-06-01-historical-store-refresh`
  (`memory/project_debt_current.md`).
