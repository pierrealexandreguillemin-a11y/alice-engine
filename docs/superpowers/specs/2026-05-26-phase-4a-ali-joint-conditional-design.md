# Phase 4a — ALI Joint Conditionnel Multi-Équipes via CE-Adverse Miroir SOTA — Design Spec

**Document ID** : ALICE-SPEC-PHASE-4A-ALI-JOINT-CONDITIONAL
**Version** : 1.0.0
**Status** : DRAFT (brainstorming Q1-Q10 validé 2026-05-16 + 2026-05-25/26)
**Author** : Pierre Alexandre Guillemin + Claude Opus 4.7
**Date** : 2026-05-26
**Phase** : 4a (NEW upstream Phase 4b CE-user OR-Tools)
**Supersedes** : ADR-016 status `Proposed` (sera Accepted post-acceptance D8)
**Standards** : ISO 5055 (architecture), ISO 5259 (lineage), ISO 24027 (fairness),
ISO 24029 (robustness), ISO 25010 (quality system), ISO 25059 (quality model AI),
ISO 27001 (security), ISO 27034 (input validation), ISO 29119 (testing),
ISO 42001 (AI management), ISO 42005 (AI impact), ISO 42010 (architecture),
ISO 15289 (life cycle), ISO 23894 (AI risk)

---

## 0. Doctrine / priorité utilisateur (2026-05-26)

**Qualité > Autonomie > Vitesse**. Self-review per task obligatoire pour éviter
paresse et dette cachée. Spec = dette cachée → propagation explicite dans
`memory/project_debt_current.md` à la création.

---

## 1. Contexte

### 1.1 Origine — D-P3-19 / R-ALI-06

ADR-022 acceptance verdict (2026-05-16) sur D8 Phase A 5/5 Kaggle COMPLETE
(492 matches) confirme **11/13 FAIL → Phase 4a (D-P3-19 empirical confirmation)** :

- `by_pool_size max_gap = 0.284` (small Q1 0.74 vs xlarge Q4 0.46) — ALI sample
  dans pool club total ⇒ sur-représentation top Elo en équipe basse niveau.
- `recall_per_group < 0.85` toutes dims — gates absolus P3G07-P3G11 FAIL.
- 117 clubs alignent 2-4 équipes simultanément en N3 ronde 5 saison 2024 sans
  conditionnement sur l'allocation jointe des autres équipes du club.

### 1.2 Mécanisme structural

A02 §3.7.b texte officiel : "Force des équipes : la Commission Technique Fédérale
(la Ligue pour la N4) compose les groupes en **numérotant les équipes appartenant
à un même Club par ordre de force décroissante**."

Conséquence : pool joueurs club → team_1 (force max) capte rangs Elo 1-8, team_2
capte rangs 9-16, team_N capte rangs (N-1)×8+1 à N×8. ALI Phase 3 marginal viole
ce conditionnement → biais structurel recall.

### 1.3 Cible Phase 4a

Pour chaque équipe cible `target_team` d'un club adverse :
1. Détecter les équipes simultanées du même club via canonical mapping chess-app.
2. Solveur OR-Tools CP-SAT miroir Top-Down (Q5) sample allocation team_1 → team_N
   sous contraintes A02 §3.7.b/c/d/f.
3. Échantillonnage ancestral Bayésien hiérarchique (Pearl 1988) : pool target_team
   = pool club total moins joueurs alloués aux équipes supérieures.
4. 20 scénarios joints par équipe cible (10 TopK + 10 MC LHS+antithetic Phase 3).

### 1.4 Acceptance criteria (per ADR-022 §gates Phase 4a)

| Métrique | Phase 3 baseline | Phase 4a target | Source |
|---|---|---|---|
| Recall | 0.57 | ≥ 0.65 | ADR-022 §gate |
| Jaccard | 0.39 | ≥ 0.50 | ADR-022 §gate |
| Brier | 0.29 | ≤ 0.22 | ADR-022 §gate |
| McNemar n_disc | 3 | ≥ 25 | Power α=0.05 |

---

## 2. Décisions clés (brainstorming Q1-Q10 validé)

| Q | Décision | Justification SOTA |
|---|----------|-------------------|
| Q1 | OR-Tools location = **self-contained `services/ali/adverse_ce.py`** | Anti-premature-abstraction (Sandi Metz Squint Test + Beck Rule of Three). Phase 4b CE-user pas encore brainstormé → shared primitives Phase 4a contraindrait design 4b. Refactor extract primitives @ Phase 4b kickoff (~3-5j). |
| Q2 | CE-adverse objective = **MAP + diversified preference data SOTA** | User explicit "go sota or go fuck yourself" doctrine. Fit `P(player → team_rank \| Elo, history)` sur echiquiers.parquet. Diversification Hamming/K-best Yannakakis 1990. |
| Q3 | §3.7.f noyau côté adverse = **data DIRECTE depuis echiquiers.parquet** | REGLES_FFE_ALICE.md §2.5+§5.1 : noyau = joueurs ayant DÉJÀ JOUÉ pour équipe cette saison. `get_noyau()` existant. Pas de proxy. |
| Q4 | Multi-team detection = **Hybrid JSON vendored + Makefile refresh** | Pattern ADR-013 strict (chess-app source canonique). `config/clubs_teams_<season>.json` vendored. `make sync-clubs-teams` target appelle chess-app REST + update JSON + recompute SHA + commit. Reproducibility ISO 5259/42001. Kaggle internet OK mais reproducibilité prime. |
| Q5 | Sim ordering = **Top-down ancestral sampling** (A02 §3.7.b mirror) | Échantillonnage hiérarchique Bayésien (Pearl 1988). Match A02 §3.7.b texte. Fits SLA /compose <2s. APPROXIMATION assumant top-down (majorité clubs), strategic sacrifice patterns minoritaires testés empirique D8. **@TODO post-MVP** : Option B joint OR-Tools si A.recall < 0.65 (Phase 4c contingency). |
| Q6 | Cache CE-adverse = **SQLite TTL 7j + Redis Phase 5 migration** | Phase 4a MVP : SQLite stdlib zero-deps `data/cache/ce_adverse.sqlite` key=`(saison, opponent_club_id, ronde_date, target_team, CODE_SHA)`. Phase 5 SaaS multi-tenant → migration Redis. CODE_SHA invalidation auto deploy. |
| Q7 | Fallback infeasible = **complete_or_nothing strict** | UNSAT → RuntimeError /compose 500 (data integrity diagnostic). TIMEOUT → RuntimeError /compose 503 (capacity). Aucune régression silencieuse Phase 3. Aligne `feedback_complete_or_nothing.md` CRITIQUE. |
| Q8 | Backward-compat = **BC préservée + deprecation post-D8** | `generator.generate(simultaneous_teams=None)` route Phase 3 path inchangé (safety net rollback Pre-V1 audit). Après D8 acceptance + 1-2 sem prod stable → deprecate path Phase 3. SOTA ML staged rollout (Sculley 2015, Polyzotis 2018). |
| Q9 | Validation = **Local pilot N=70 N3 → Kaggle D8 Phase A 5/5** | Step 1 : pilot 70 matches N3 local (1-2h CPU, fast iter, early gate recall ≥0.50). Step 2 : D8 Phase A 492 matches sur 5 kernels Kaggle (3-4h). Step 3 : aggregator phase-a verdict acceptance. ROI fail-fast local. |
| Q10 | Décomposition = **12 tasks T1-T12 atomiques** | Quality gates SOTA ML + ISO 14 normes + DoD measurable + self-review per task pour éviter paresse et dette cachée. Split fin si task > 3j. Priorité qualité > autonomie > vitesse. |

---

## 3. Architecture

### 3.1 Vue d'ensemble

```
chess-app (downstream)                         ALICE Engine (upstream)
─────────────────────                          ────────────────────────
                                               ┌──────────────────────┐
POST /compose {                                │  app/api/routes.py   │
  user_club_id,                                │     /compose         │
  target_team,                ───────────────► │                      │
  opponent_club_id,                            │  Pydantic schema     │
  round_date, saison, ronde,                   │  (Q4 payload)        │
  simultaneous_teams: [...]   (Q4 prod path)   └─────────┬────────────┘
}                                                        │
                                              ┌──────────▼────────────┐
                                              │ generator.generate(   │
                                              │  simultaneous_teams,  │
                                              │  target_team, ... )   │
                                              │  (Q8 BC: None→Phase3) │
                                              └──────────┬────────────┘
                                                         │
                                              ┌──────────▼────────────┐
                                              │ adverse_ce.py         │
                                              │ OR-Tools CP-SAT       │
                                              │ Top-down team_1..N    │
                                              │ A02 §3.7.b/c/d/f      │
                                              │ (Q1 self-contained)   │
                                              └──────────┬────────────┘
                                                         │
                                              ┌──────────▼────────────┐
                                              │ preference_model.py   │
                                              │ MAP P(player→         │
                                              │  team_rank|Elo,hist)  │
                                              │ (Q2 SOTA)             │
                                              └──────────┬────────────┘
                                                         │
                                              ┌──────────▼────────────┐
                                              │ diversification.py    │
                                              │ Hamming K-best        │
                                              │ Yannakakis 1990       │
                                              │ (Q2 diversification)  │
                                              └──────────┬────────────┘
                                                         │
                                              ┌──────────▼────────────┐
                                              │ pool_loader.py        │
                                              │ exclude_players       │
                                              │ (Q5 top-down excl)    │
                                              └──────────┬────────────┘
                                                         │
                                              ┌──────────▼────────────┐
                                              │ ce_adverse_cache.py   │
                                              │ SQLite TTL 7j         │
                                              │ (Q6)                  │
                                              └───────────────────────┘
```

### 3.2 Fichiers à créer / modifier

**NEW fichiers** (5) :
- `services/ali/adverse_ce.py` — OR-Tools CP-SAT solveur miroir self-contained
- `services/ali/preference_model.py` — MAP P(player→team_rank | Elo, history)
- `services/ali/diversification.py` — Hamming K-best Yannakakis 1990
- `services/ali/ce_adverse_cache.py` — SQLite TTL 7j cache
- `scripts/sync_clubs_teams.py` — Makefile target sync chess-app → vendored JSON

**MODIFIED fichiers** (4) :
- `services/ali/generator.py` — ajout `simultaneous_teams`, `target_team` params + BC
- `services/ali/pool_loader.py` — ajout `exclude_players` param + ordering top-down
- `app/api/routes.py` — accept `simultaneous_teams` payload + Pydantic schema
- `app/api/schemas.py` — ComposeRequest schema extension

**VENDORED config** (1) :
- `config/clubs_teams_2024.json` — chess-app teams snapshot (ADR-013 pattern)

---

## 4. Implementation tasks T1-T12

> **Méta-règle** : chaque task a (a) DoD measurable, (b) quality gates ISO/F/T,
> (c) self-review checkpoint pre-merge, (d) blast radius limité. Si task > 3j
> wall → split fin obligatoire.

### T1 — `services/ali/adverse_ce.py` skeleton + OR-Tools CP-SAT (3-5j)

**Goal** : NEW solveur OR-Tools CP-SAT self-contained pour CE-adverse miroir.
Variables : `assign[player_idx, team_idx, board_idx] ∈ {0,1}`. Constraints :
A02 §3.7.b (force descending entre teams), §3.7.c (joueur brûlé ≤3 games sup),
§3.7.d (même groupe), §3.7.f (noyau 50%), board count per team, 1 player =
1 (team, board) assignment.

**DoD** :
- [ ] `services/ali/adverse_ce.py` ≤ 300 lignes (ISO 5055)
- [ ] Toutes fonctions ≤ 50 lignes (ISO 5055)
- [ ] Complexité cyclomatique radon ≤ B (ISO 5055)
- [ ] `tests/services/ali/test_adverse_ce.py` ≥ 15 cas (ISO 29119) :
  - 3 cas feasible 2-team / 3-team / 5-team
  - 3 cas UNSAT pool too small + noyau impossible + brûlé all up
  - 3 cas constraint coverage §3.7.b/c/d/f isolés
  - 3 cas determinism seed reproductibility (lineage_hash SHA-256)
  - 3 cas performance solve <500ms on N=10 fake matches
- [ ] Coverage ≥ 90% sur module (T1G01)
- [ ] Type annotations 100% strict (mypy --strict)
- [ ] Pydantic validation entrées (ISO 27034)
- [ ] Docstring structuré ID, Version, Count par fonction (ISO 29119)

**Quality gates** :
- F1 (ISO 5055 max lines), F2 (cyclomatic ≤ B), T1 (tests pass), T7 (coverage ≥ 80%)

**Self-review checkpoint pre-merge** :
- [ ] Re-read DoD checklist : all items checked ?
- [ ] Scan for `TODO`, `FIXME`, `XXX`, `HACK` keywords → escalate to debt or fix
- [ ] Scan for `pass` statements, `NotImplementedError` placeholders
- [ ] `radon cc services/ali/adverse_ce.py -nb` → no findings
- [ ] `wc -l services/ali/adverse_ce.py` → ≤ 300
- [ ] `mypy --strict services/ali/adverse_ce.py` → 0 errors
- [ ] `ruff check services/ali/adverse_ce.py` → 0 errors
- [ ] `pytest tests/services/ali/test_adverse_ce.py -v` → all PASS
- [ ] Manual trace : 1 solve N=3 teams hand-verified vs OR-Tools output

**ISO normes** : 5055 (architecture), 27034 (input validation), 29119 (testing),
42001 (lineage SHA-256), 42010 (ADR-016 reference)

**Split if needed** : T1.a (model variables + constraints def, 2j), T1.b (solver
call + result extraction, 1j), T1.c (tests + determinism, 1-2j)

---

### T2 — `services/ali/preference_model.py` MAP fit + Model Card (5j)

**Goal** : NEW preference model fit `P(player → team_rank | Elo, history)` sur
echiquiers.parquet historique. Sources SOTA : Bradley-Terry-Luce preference
learning (Hunter 2004), MAP estimation avec Laplace prior pour sparse data,
features par player : Elo, recency F2, streak F3, brûlé count history.

**DoD** :
- [ ] `services/ali/preference_model.py` ≤ 300 lignes (ISO 5055)
- [ ] Training script `scripts/train_preference_model.py` reproductible
- [ ] Model artifact `models/preference_model_<saison>.joblib` SHA-256 traceable
- [ ] Model Card `docs/iso/MODEL_CARD_PREFERENCE_<saison>.md` (ISO 42001) :
  - Sources data (echiquiers.parquet SHA + filter ranges)
  - Features list + provenance
  - Training hyperparams (Laplace prior alpha, optimizer)
  - Performance metrics (log-loss, ECE per division)
  - Limitations (sparse data clubs faible volume)
  - Lineage hash SHA-256 propagation
- [ ] `tests/services/ali/test_preference_model.py` ≥ 10 cas (ISO 29119) :
  - 3 cas fit on synthetic data (ground truth recover)
  - 3 cas determinism (same seed → same artifact SHA)
  - 2 cas serialization (joblib roundtrip)
  - 2 cas inference batch + single
- [ ] Coverage ≥ 90% (T1G01)
- [ ] Pandera schema validation echiquiers.parquet (ISO 5259)

**Quality gates** : F1, F2, T1, T7, T10 (Model Card ISO 42001)

**Self-review checkpoint pre-merge** :
- [ ] DoD checklist all items checked
- [ ] Model Card 8 sections complètes (no TBD)
- [ ] Lineage hash propagation tracé end-to-end (input SHA → model SHA → inference SHA)
- [ ] Sanity check : predict_proba sum to 1.0 per (player, club)
- [ ] Bias check : recall per gender (ISO 24027) — fail-fast if gap > 0.10
- [ ] Performance check : inference batch 1000 players < 100ms

**ISO normes** : 5055, 5259 (lineage SHA), 24027 (bias), 29119, 42001 (Model Card),
42005 (impact assessment)

**Split if needed** : T2.a (data loading + Pandera schema, 1j), T2.b (model fit +
serialization, 2j), T2.c (Model Card writing, 1j), T2.d (tests + bias check, 1j)

---

### T3 — `services/ali/diversification.py` Hamming K-best (2-3j)

**Goal** : NEW post-MAP diversification via Hamming distance K-best (Yannakakis
1990 OR-Tools K-best paths) ou Hahn-Murray 2024 diversified solutions CSP. Output :
10 scénarios distincts par équipe avec Hamming distance ≥ 3 entre tout pair.

**DoD** :
- [ ] `services/ali/diversification.py` ≤ 200 lignes
- [ ] Tests ≥ 8 cas : 3 trivial diversity, 3 hamming threshold edge, 2 determinism
- [ ] Coverage ≥ 90%
- [ ] Documented algorithm ref (Yannakakis 1990 ou Hahn-Murray 2024 citée)

**Quality gates** : F1, F2, T1, T7

**Self-review checkpoint** :
- [ ] DoD checklist
- [ ] Manual verify K=10 diversity on N=20 player pool
- [ ] No infinite loop on degenerate input (pool size < K)

**ISO normes** : 5055, 24029 (robustness — diversity = stress test), 29119

---

### T4 — `config/clubs_teams_2024.json` vendored + `make sync-clubs-teams` (2-3j)

**Goal** : ADR-013 pattern hybrid JSON vendored + Makefile refresh target.
chess-app REST API call → JSON dump → SHA-256 → commit.

**DoD** :
- [ ] `config/clubs_teams_2024.json` ≥ 200 clubs avec teams arrays
- [ ] `scripts/sync_clubs_teams.py` ≤ 200 lignes
- [ ] `Makefile` target `sync-clubs-teams` : `python scripts/sync_clubs_teams.py && git diff --stat`
- [ ] CI staleness check : age JSON > 30j → warning
- [ ] Tests ≥ 6 cas : schema validation, idempotence, error handling chess-app down
- [ ] README section in `config/README.md` documenting refresh procedure

**Quality gates** : F1, T1, T7, T9 (CI check)

**Self-review checkpoint** :
- [ ] DoD checklist
- [ ] Run `make sync-clubs-teams` twice → JSON unchanged (idempotence)
- [ ] Verify chess-app down → script exits with clear error (no silent fail)
- [ ] Verify SHA-256 of JSON logged at sync time

**ISO normes** : 5055, 5259 (data versioning), 15289 (lifecycle)

---

### T5 — Refactor `generator.py::generate(simultaneous_teams, target_team)` BC (2-3j)

**Goal** : Étendre signature + dispatch BC : `None` → Phase 3 path, sinon → Phase 4a
joint conditional path. Préserve toute la suite tests Phase 3.

**DoD** :
- [ ] Signature étendue : `simultaneous_teams: list[TeamSpec] | None = None, target_team: str | None = None`
- [ ] Code path dispatch : if `simultaneous_teams is None` → Phase 3 (existing logic), else Phase 4a
- [ ] Phase 4a orchestration : top-down loop team_1..N appel CE-adverse + exclude players
- [ ] Tests Phase 3 existants : 100% PASS sans modification
- [ ] Tests NEW Phase 4a path ≥ 8 cas : 2-team / 3-team / 5-team feasible, UNSAT, TIMEOUT, BC None
- [ ] Coverage ≥ 90% sur module (was 85%)

**Quality gates** : F1, F2, T1, T7, T9 (no regression Phase 3)

**Self-review checkpoint** :
- [ ] DoD checklist
- [ ] `pytest tests/services/ali/test_generator.py -v` → all PASS (Phase 3 inchangé)
- [ ] `pytest tests/services/ali/test_generator_phase4a.py -v` → NEW path PASS
- [ ] BC verified : 3 callers Phase 3 (api/routes, backtest, smoke) appellent sans `simultaneous_teams` → Phase 3 path
- [ ] Lineage hash propagation Phase 4a path tested

**ISO normes** : 5055, 5259, 29119, 42001 (lineage)

**Split** : T5.a (signature + dispatch BC, 1j), T5.b (orchestration top-down, 1j), T5.c (tests, 1j)

---

### T6 — Refactor `pool_loader.py::load_pool(exclude_players)` (2j)

**Goal** : Ajout param `exclude_players: set[str] | None = None`. Filter out
excluded players from returned pool. Maintain F7 survivor filter + overrides.

**DoD** :
- [ ] Signature étendue avec `exclude_players: set[str] | None = None`
- [ ] Filter logic : `if c.nr_ffe in exclude_players: continue` post F7
- [ ] Tests existants PASS (BC `None`)
- [ ] Tests NEW ≥ 4 cas : exclude empty, exclude 5 players, exclude all (empty pool), exclude with overrides

**Quality gates** : F1, T1, T7

**Self-review checkpoint** :
- [ ] DoD checklist
- [ ] BC None : pool size identical to before
- [ ] exclude set non-trivial : verify excluded players absent from result

**ISO normes** : 5055, 29119

---

### T7 — `app/api/routes.py::/compose` accept `simultaneous_teams` + Pydantic (2j)

**Goal** : Accept payload `simultaneous_teams: list[TeamSpec]` dans `/compose`.
Update Pydantic schema + Pass-through to generator.

**DoD** :
- [ ] Pydantic schema `ComposeRequest` étendue avec `simultaneous_teams: list[TeamSpec] | None = None`
- [ ] `TeamSpec` schema NEW : `team_name: str, division: str, target_team: bool = False, board_count: int`
- [ ] Route `/compose` parse + pass to `generator.generate(simultaneous_teams=...)`
- [ ] Tests E2E ≥ 6 cas : 
  - 2 cas payload sans `simultaneous_teams` (BC Phase 3)
  - 2 cas payload avec `simultaneous_teams` (Phase 4a)
  - 1 cas payload mal-formé (Pydantic 422)
  - 1 cas UNSAT propagation 500

**Quality gates** : F1, T1, T7, T11 (E2E integration)

**Self-review checkpoint** :
- [ ] DoD checklist
- [ ] OpenAPI spec auto-generated updated
- [ ] Test client (httpx) integration test PASS
- [ ] Pydantic 422 returns clear error message (debugging)

**ISO normes** : 5055, 27034 (input validation), 29119

---

### T8 — `services/ali/ce_adverse_cache.py` SQLite TTL 7j (2j)

**Goal** : SQLite persistent cache key=`(saison, opponent_club_id, ronde_date,
target_team, CODE_SHA)`. TTL 7j. Hit/miss metrics logged.

**DoD** :
- [ ] `services/ali/ce_adverse_cache.py` ≤ 200 lignes
- [ ] SQLite schema : `(key_hash TEXT PRIMARY KEY, payload BLOB, created_at INTEGER, ttl_sec INTEGER)`
- [ ] Methods : `get(key) → payload | None`, `set(key, payload)`, `evict_expired()`, `stats() → {hits, misses, size}`
- [ ] Tests ≥ 10 cas : set/get, TTL expiry, eviction, concurrent access, CODE_SHA invalidation, stats
- [ ] Coverage ≥ 90%
- [ ] Cache file path configurable via env var `ALICE_CACHE_DIR` (default `data/cache/`)

**Quality gates** : F1, F2, T1, T7

**Self-review checkpoint** :
- [ ] DoD checklist
- [ ] Bench : 1000 set/get < 1s (in-process SQLite)
- [ ] TTL expiry verified via time.sleep test
- [ ] Concurrent access (2 threads) no corruption

**ISO normes** : 5055, 27001 (no secrets in payload), 29119

---

### T9 — Local pilot N=70 N3 backtest (1-2j)

**Goal** : `scripts/backtest/pilot_phase4a.py` re-backtest N=70 N3 saison 2024
en mode Phase 4a joint conditional. Output rapport gates early : recall, Jaccard,
Brier, McNemar. Early gate : recall ≥ 0.50 (compared to baseline 0.57 Phase 3).

**DoD** :
- [ ] `scripts/backtest/pilot_phase4a.py` ≤ 250 lignes
- [ ] Output `reports/pilot_phase4a_<DATE>.md` :
  - Setup (saison, division, N matches, seed)
  - Metrics (recall, Jaccard, Brier, McNemar n_disc, p-value)
  - Per-club breakdown (top 10 fail clubs)
  - Decision : early gate PASS (≥0.50) → proceed Kaggle | FAIL (<0.50) → diagnostic + fix
- [ ] Runtime ≤ 2h CPU local
- [ ] Deterministic via seed (same run → same SHA report)

**Quality gates** : F1, T1, T9 (acceptance pilot)

**Self-review checkpoint** :
- [ ] DoD checklist
- [ ] Report contains all 4 metrics + per-club breakdown
- [ ] Lineage hash chain : data SHA → model SHA → predictions SHA → report SHA
- [ ] If recall < 0.50 → STOP + investigate before Kaggle (saves 3-4h compute)

**ISO normes** : 5055, 5259, 25059 (quality gates), 29119

---

### T10 — D8 Phase A 5/5 Kaggle re-run + aggregator phase-a (3-4j)

**Goal** : Re-deploy 5 kernels D8 Phase A (Top16, N1, N2, N3, N4) avec code Phase 4a.
Upload alice-code dataset, push kernels, monitor execution, fetch outputs, run
aggregator phase-a mode (ADR-022).

**DoD** :
- [ ] Skill `kernel-push` 9 étapes RESPECTED (no shortcut)
- [ ] 5/5 kernels COMPLETE
- [ ] Outputs téléchargés dans `outputs/d8/2024-phase4a/{top-16,nationale-1..4}/`
- [ ] Aggregator `python scripts/d8/run_aggregator.py --mode phase-a` produces `reports/d8/phase_a/<DATE>-phase4a-verdict.md`
- [ ] Verdict : 19 gates evaluated, PASS/FAIL counts logged
- [ ] Acceptance criteria Phase 4a per ADR-022 :
  - recall ≥ 0.65 PASS/FAIL
  - Jaccard ≥ 0.50 PASS/FAIL
  - Brier ≤ 0.22 PASS/FAIL
  - McNemar n_disc ≥ 25 PASS/FAIL

**Quality gates** : F1-F12 + T1-T12 ALL applicable, ISO 42001 lineage, ISO 25059 quality

**Self-review checkpoint** :
- [ ] DoD checklist
- [ ] 5/5 kernel outputs SHA-256 logged + commit
- [ ] No silent fallback Phase 3 in Phase 4a kernels (verify lineage_hash distinct)
- [ ] Aggregator output reproducible (run twice → same SHA)
- [ ] Push origin commit + CI green BEFORE acceptance report

**ISO normes** : ALL 14 (production gate complète)

**Split** : T10.a (deploy kernels, 1j), T10.b (monitor + fetch, 1j), T10.c (aggregator + push, 1-2j)

---

### T11 — Acceptance report + ADR-016 Accepted + D-P3-19 résolu (1-2j)

**Goal** : Si T10 acceptance PASS → finalize acceptance via report formel + update
ADRs + close debt.

**DoD** :
- [ ] `reports/phase_4a_acceptance.md` :
  - Executive summary (PASS/FAIL)
  - Metrics table (Phase 3 baseline vs Phase 4a actual vs target)
  - Per-division breakdown
  - Per-club breakdown top 20
  - Failure analysis (clubs FAIL recall < 0.65)
  - Decision rationale + ADR cross-refs
- [ ] ADR-016 status `Proposed → Accepted` via Edit on `docs/architecture/adr/ADR-016-*.md`
- [ ] D-P3-19 résolution dans `memory/project_debt_current.md` + `docs/project/DEBT_LEDGER.md` v1.0.4
- [ ] ADR-022 update §gates Phase 4a : actual values + verdict
- [ ] NEW ADR-023 : "Phase 4a acceptance — Phase 4b CE-user OR-Tools kickoff"
- [ ] Commit conventional `feat(ali): Phase 4a acceptance ADR-016 Accepted, D-P3-19 RÉSOLUE`

**Quality gates** : F1-F12 + T1-T12 ALL, ISO 42001 (acceptance traceability)

**Self-review checkpoint** :
- [ ] DoD checklist
- [ ] All ADR refs cross-linked correctly
- [ ] DEBT_LEDGER version incremented (v1.0.3 → v1.0.4)
- [ ] Commit message follows ADR-XXX convention
- [ ] Pre-push hook PASS (R-PRE-PUSH-01 ≤ 90s)
- [ ] CI run GREEN before declaring task complete

**ISO normes** : ALL 14

---

### T12 — Update `memory/project_debt_current.md` 3 NEW debts (1j)

**Goal** : Propager 3 NEW debts générées par Phase 4a brainstorming dans memory.
"Spec = dette cachée → propage en memory" doctrine 2026-05-26.

**DoD** :
- [ ] `memory/project_debt_current.md` updated avec 3 NEW entries :
  - `D-2026-05-26-phase4c-joint-ortools-escalation` : Si Phase 4a A.recall <0.65 → escalate Option B joint OR-Tools (Q5 contingency).
    - Phase cible : 4c (post-Phase 4b)
    - Reviewer : user
    - Trigger : D8 acceptance recall < 0.65 OR observation prod patterns sacrifice strategy
  - `D-2026-05-26-phase5-redis-migration` : SQLite → Redis migration pour SaaS multi-tenant.
    - Phase cible : 5 (SaaS multi-tenant)
    - Reviewer : Claude + user
    - Trigger : Phase 5 kickoff
  - `D-2026-05-26-deprecate-phase3-single-team-path` : Supprimer Phase 3 code path `simultaneous_teams=None` post D8 acceptance + 1-2 sem prod stable.
    - Phase cible : 4a+T (T+2 semaines post-acceptance)
    - Reviewer : Claude + user
    - Trigger : 14 jours post-acceptance avec 0 rollback déclenchée
- [ ] `docs/project/DEBT_LEDGER.md` v1.0.5 mirrors memory entries
- [ ] Commit `docs(debt): 3 NEW debts post-Phase 4a brainstorming`

**Quality gates** : F1 (≤300 lignes file), T11 (lifecycle documentation ISO 15289)

**Self-review checkpoint** :
- [ ] DoD checklist
- [ ] 3 NEW entries respectent format `D-YYYY-MM-DD-<tag>` (D-XX legacy + dated)
- [ ] Phase cible explicitement nommée pour chaque
- [ ] Trigger condition mesurable (pas vague "si bug")
- [ ] DEBT_LEDGER versioning aligné

**ISO normes** : 15289 (lifecycle), 42010 (ADR cross-ref)

---

## 5. Quality gates Phase 4a globaux (F1-F12 / T1-T12)

| Gate | Description | Verification |
|---|---|---|
| F1 | Max 300 lignes/fichier (ISO 5055) | `wc -l services/ali/*.py` |
| F2 | Cyclomatique ≤ B (ISO 5055) | `radon cc services/ali/ -nb` |
| F3 | SRP per module | Review naming + responsibility |
| F4 | Pydantic validation entrées (ISO 27034) | `mypy --strict` |
| F5 | Lineage hash SHA-256 propagation (ISO 5259) | Tests determinism |
| F6 | Model Card preference_model (ISO 42001) | `docs/iso/MODEL_CARD_*.md` |
| F7 | DVC versioning artefacts ML | `dvc.yaml` updated |
| F8 | Coverage ≥ 70% global (ISO 29119) | `pytest --cov` |
| F9 | Coverage ≥ 90% NEW modules | Module-level cov report |
| F10 | Pre-push <90s (R-PRE-PUSH-01) | `time .git/hooks/pre-push` |
| F11 | CI green sur HEAD | `gh run list --limit 1` |
| F12 | No silent fallback (complete_or_nothing) | Grep `try/except.*pass` |
| T1 | Tests unit + integration PASS | `pytest tests/ -v` |
| T2 | Tests determinism (lineage SHA) | `pytest tests/backtest/test_determinism.py` |
| T3 | Tests fairness (ISO 24027) | gender breakdown gap < 0.10 |
| T4 | Tests robustness (ISO 24029) | noise stress + DRO Wasserstein |
| T5 | Tests E2E API (httpx) | `pytest tests/test_compose_e2e.py` |
| T6 | Tests Pydantic schemas validation | `pytest tests/api/test_schemas.py` |
| T7 | Coverage targets met | Module-level + global |
| T8 | Mypy strict 0 errors | `mypy --strict services/ali/` |
| T9 | Ruff check 0 errors | `ruff check services/ali/ tests/` |
| T10 | Bandit security 0 high | `bandit -r services/ali/` |
| T11 | Pre-push hook PASS | `git push` (dry-run) |
| T12 | Aggregator phase-a verdict | `python scripts/d8/run_aggregator.py --mode phase-a` |

---

## 6. NEW debts à propager (T12)

| ID | Description | Phase cible | Reviewer | Trigger |
|---|---|---|---|---|
| D-2026-05-26-phase4c-joint-ortools-escalation | Escalate Option B joint OR-Tools si A.recall < 0.65 | Phase 4c | user | D8 acceptance OR prod observation |
| D-2026-05-26-phase5-redis-migration | SQLite → Redis pour SaaS multi-tenant | Phase 5 | Claude+user | Phase 5 kickoff |
| D-2026-05-26-deprecate-phase3-single-team-path | Supprimer Phase 3 code path BC | Phase 4a+T | Claude+user | D8 PASS + 14j prod stable |

---

## 7. Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Phase 4a A.recall < 0.65 (strategic sacrifice patterns) | Moyenne | Haute | Pilot N=70 N3 early gate (T9). Si FAIL → diagnose avant Kaggle. Contingency Phase 4c escalate B. |
| OR-Tools UNSAT clubs faible volume | Faible | Moyenne | complete_or_nothing strict (Q7). Diagnostic log. |
| SQLite cache concurrent access bug | Faible | Moyenne | Tests concurrent T8. Migration Redis Phase 5 si problème. |
| chess-app schema drift (clubs/teams) | Moyenne | Haute | Makefile sync target + CI staleness check (T4). |
| Lineage hash propagation broken | Faible | Critique | Tests determinism T1/T2/T5. |
| /compose SLA <2s violation | Moyenne | Moyenne | Bench load test pre-acceptance. Cache T8 mitigation. |
| Phase 3 BC path rot (D-2026-05-26-deprecate) | Faible | Faible | 14j prod observation gate avant cleanup. |

---

## 8. Rollback plan

Si Phase 4a déployé en prod et metrics dégradées (recall, Jaccard, Brier) :

1. **Hot rollback** : feature flag `ALICE_ENABLE_PHASE4A=false` env var → /compose
   ignore `simultaneous_teams` payload, route Phase 3 path (BC préservé T5).
2. **Cold rollback** : revert commit `feat(ali): Phase 4a acceptance` → `git revert
   <SHA>` + push origin + redeploy.
3. **Debt escalation** : si rollback déclenché → bloque D-2026-05-26-deprecate-phase3
   indefiniment, escalate diagnostic Phase 4a.

---

## 9. Sources SOTA

- **A02 FFE** : §3.6.e, §3.7.b, §3.7.c, §3.7.d, §3.7.f (rules officielles
  interclubs)
- **Pearl 1988** : "Probabilistic Reasoning in Intelligent Systems" — ancestral
  sampling Bayesian hierarchical
- **Yannakakis 1990** : K-best paths / diversified solutions CSP
- **Hahn & Murray 2024** : "Diversified solutions for constraint satisfaction"
- **Bradley & Terry 1952 / Hunter 2004** : Preference learning MAP
- **Sculley et al. 2015** : "Hidden Technical Debt in Machine Learning Systems"
  (NeurIPS) — ML deployment patterns
- **Polyzotis et al. 2018** : "Data lifecycle challenges in production ML"
- **Sandi Metz** : Squint Test (anti-premature-abstraction)
- **Kent Beck** : Rule of Three (refactor extraction)
- **Google OR-Tools** : CP-SAT solver documentation
- **ISO/IEC 5055:2021** : Code quality measures
- **ISO/IEC 5259** : Data quality + lineage
- **ISO/IEC 24027:2021** : AI Bias
- **ISO/IEC 24029-1:2021** : AI Robustness (also -2:2023 for methodology)
- **ISO/IEC 25059:2023** : Quality model for AI systems
- **ISO/IEC 29119** : Software testing
- **ISO/IEC 42001:2023** : AI Management System
- **ISO/IEC 42005:2024** : AI Impact Assessment
- **ISO/IEC 42010:2022** : Architecture description
- **ISO/IEC 23894:2023** : AI Risk Management

---

## 10. Cross-references

- [ADR-016](../../architecture/adr/ADR-016-ali-conditioned-multi-team-adverse-ce-mirror.md) — Approche A SOTA Phase 4a (Proposed → Accepted post T11)
- [ADR-022](../../architecture/adr/ADR-022-d8-phase-a-acceptance-verdict-ali-conditional-phase-4a.md) — D8 Phase A acceptance verdict + Phase 4a gates
- [ADR-013](../../architecture/adr/ADR-013-rule-engine-json.md) — JSON vendored pattern (référence Q4)
- [ADR-017](../../architecture/adr/ADR-017-wilcoxon-mcnemar-pivot.md) — Statistical test methodology
- `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md` §Phase 4a + 4b
- `docs/requirements/REGLES_FFE_ALICE.md` §2.5 + §5.1 (noyau dynamique)
- `docs/iso/AI_RISK_ASSESSMENT.md` §R-ALI-06
- `docs/iso/AI_RISK_REGISTER.md` §2.7 R-ALI-06
- `docs/iso/ALI_QUALITY_GATES_REPORT.md` §6.2 §7.5
- `memory/project_debt_current.md` D-P3-19 (à résoudre T11)
- `docs/project/DEBT_LEDGER.md` v1.0.3 (target v1.0.5 post T12)

---

## 11. Validation gate spec (self-review pre-user-review)

- [x] Placeholder scan : 0 `TBD`, 0 `TODO`, 0 `XXX`
- [x] Internal consistency : Q1-Q10 décisions ↔ T1-T12 tasks aligned
- [x] Architecture diagram ↔ Files NEW/MODIFIED list ↔ Tasks T1-T12
- [x] Scope check : single implementation plan, ~5-7 sem
- [x] Ambiguity check : DoD measurable per task, no "make it good"
- [x] ISO normes : 14/14 listed avec mapping task
- [x] Quality gates F1-F12/T1-T12 explicit per task
- [x] Self-review checkpoints explicit per task
- [x] Rollback plan documented
- [x] NEW debts identified + Phase cible explicit
- [x] Risk register with probability/impact/mitigation

---

**Generated** : 2026-05-26 via skill `superpowers:brainstorming` (Q1-Q10 validated)
**Next step** : User reviews this spec → `superpowers:writing-plans` skill invocation
