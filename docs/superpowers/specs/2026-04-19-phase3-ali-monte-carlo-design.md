# Phase 3 — ALI Monte Carlo Hybride SOTA : Spec de design

**Date** : 2026-04-19
**Status** : APPROVED (brainstorming validé section par section)
**Auteur** : Pierre + Claude (brainstorming collaboratif)
**Objectif** : ALICE (Adversarial Lineup Inference) produit **état-de-l'art** pour prédire la composition adverse et alimenter le Composition Engine.

> **Décisions clés (brainstorming 2026-04-18/19)** :
> - Q1 (scope) : **B** — Monte Carlo + data loader réel (résorbe dette Phase 2)
> - Q2 (méthode) : **C** — hybride Top-K déterministe + MC stochastique
> - Q3 (API) : **C** — club_id + overrides optionnels
> - Q4 (validation) : **B** (5 métriques) + D8 Phase 3.5 fairness/robustness
> - Q5 (règles) : **Z++** — RuleEngine JSON + CI lineage pocket-arbiter
> - SOTA intégration : **option B maximum** — F1/F2/F3/F5/F6/F7 intégrés dès Phase 3 (qualité > temps, éviter réécriture future)

---

## 1. Contexte et scope

### 1.1 Problème
ALICE sert `/compose` : recommander une composition multi-équipe pour un club FFE. Le pipeline est :
```
ALI (prédire adversaire) → ML (P(W/D/L) per board) → CE (optimiser E[score])
```
Phase 2 a wired le ML et le CE avec fallback (ALI = tri Elo 1 scénario, joueurs = stub Elo 1500). **Phase 3 résorbe cette dette et livre un ALI SOTA.**

### 1.2 Scope retenu (option B + audit final)
1. Monte Carlo hybride (10 TopK déterministes + 10 MC stochastiques pondérés)
2. Data loader réel sur `joueurs.parquet` + `echiquiers.parquet` (résorbe D1, D2, D5)
3. RuleEngine JSON-driven (remplace les 11 Python `ffe_rules.py`), vendor depuis `chess-app/flat-six/rules/`
4. Classification PUBLIC/PRIVATE des règles FFE (annotations locales ALICE)
5. **SOTA intégrés** : F1-copule gaussienne (F6), F2-recency decay, F3-streak autoregressive, F5-LHS/antithetic, F7-survivor filter
6. Backtest walk-forward + 10 quality gates T13-T22
7. **ScenarioExplainer** (ISO 42001 explainability) — top-3 facteurs contributifs par scénario
8. **Kill switch / graceful degradation** via feature flag `ALI_MODE={sota,fallback,hybrid}`
9. **ConfidenceLevel par ScenarioSet** basé sur volume historique (fairness petits clubs)
10. **Feedback loop utilisateur** via endpoint `POST /feedback/scenario` + storage MongoDB
11. **Normes ISO étendues à 19** : ajout 23053, 22989, 38507, 25012, 25024
12. **STRIDE threat model** + rate limiting renforcé pour /compose et overrides
13. **Capacity planning benchmark** + SLO documenté (p95 latence, RPS sustainable)
14. **Peer review gate** avant merge via skill `superpowers:requesting-code-review`
15. **Property-based testing** via Hypothesis sur RuleEngine + fuzzing JSON inputs
16. **Observability DIY custom 100%** (~440 lignes, 0 nouvelle dépendance externe) — Prometheus metrics, structured JSON logging, drift tracker weekly, alerting webhook, health check enrichi
17. **Multi-tenant-ready design** (sans implémentation full multi-tenant Phase 3) : API versioning `/v1/compose`, rate limiting per-`user_club_id`, `tenant_id` dans audit log — ambition SaaS multi-clubs France confirmée

### 1.3 Hors scope Phase 3 (traçabilité en dette)
- **D3** : jeunes J02 (Phase 3.5)
- **D4** : Coupes (Phase 3.5)
- **D8** : fairness/robustness breakdown ALI (Phase 3.5 STRICT bloquant Phase 4)
- **D9** : Adaptive Importance Sampling avec drift monitoring (Phase 5+, requiert data prod)
- **D10** : sync ALICE ↔ chess-app JSON automatique (Phase 3 si simple, sinon 3.5)
- **D11** : completeness audit PDF FFE → chess-app JSON (tâche NLP séparée, pocket-arbiter stale)
- **D13** : `zone_enjeu` consommé par ALI (Phase 4+, couplé CE OR-Tools)
- **D15** : conformal prediction bout-en-bout (Phase 4+, couplé CE multi-objectif)

---

## 2. Architecture haut niveau

```
POST /compose request
  ├── user_club_id + opponent_club_id + round_date + competition
  │
  ▼
ScenarioGenerator.generate(...)
  ├── 1. PlayerPoolLoader         → effectif éligible (F7 survivor filter)
  ├── 2. HistoryEnricher          → features ALI (F2 recency + F3 streak AR)
  ├── 3. RuleEngine.filter        → contraintes eligibility publiques
  ├── 4. VerifiabilityClassifier  → partition (public, private)
  ├── 5. CopulaJointSampler.fit   → copule gaussienne sur co-présence (F6)
  ├── 6. TopKEnumerator           → 10 lineups déterministes (mode)
  └── 7. MonteCarloSampler        → 10 lineups LHS/antithetic (queue, F5)
        ↓
      ScenarioSet (20 scenarios × weight, lineage_hash)

  ▼
Pour chaque scenario × board :
  features = feature_store.assemble(user, opponent, context)
  StackingInferenceService.predict_board(...)  # pipeline Phase 2 inchangé
  → PredictionResult(p_loss, p_draw, p_win, e_score)

  ▼
ComposerService (Phase 2 fallback tri Elo ; Phase 4 OR-Tools)
  → best_lineup en intégrant sur moyenne pondérée des 20 scenarios

  ▼
Response : ComposeResponse (+ metadata ISO : lineage_hash, rule_uuids, model_versions)
```

**Nouveaux packages** :
- `services/ffe/rule_engine.py` — RuleEngine générique JSON-driven
- `services/ali/` — package complet (10 modules, détail §4)

**Modules supprimés/remplacés** :
- `services/ffe_rules.py` (11 Python rules) → **supprimé**
- `services/inference.py::InferenceService` (legacy ALI stub) → **supprimé**
- `services/composer.py` → **audit Phase 3 début** : supprimé ou documenté

---

## 3. Classification PUBLIC / PRIVATE des règles FFE

Pour chaque règle chargée par RuleEngine, ALICE annote localement :
- **PUBLIC** : vérifiable via `joueurs.parquet` + `echiquiers.parquet` → appliquée comme contrainte dure au générateur
- **PRIVATE** : nécessite info club interne (noyau, stratégie) → supposée respectée par l'adversaire (survivor bias favorable : les compositions historiques ont déjà respecté leur noyau)

### Classification A02 (14 règles, scope Phase 3)

| Article | Règle | Classification |
|---------|-------|----------------|
| 3.7.a | team_size | PUBLIC |
| 3.7.b | force équipes | PRIVATE (décision CTF/Ligue) |
| 3.6.e | ordre Elo | PUBLIC |
| 3.7.c | brûlé (matchs équipe sup) | PUBLIC |
| 3.7.d | same_group | PUBLIC |
| 3.7.e | match count | PUBLIC |
| 3.2 | désignation titulaires | PRIVATE |
| 3.7.f | noyau | **PRIVATE** |
| 3.7.g | mutes ≤ 3 | PUBLIC |
| 3.7.h | foreign quota FR/UE ≥ 5 | PUBLIC |
| 3.7.i | FR gender N1/N2 | PUBLIC |
| 3.7.j | elo_max | PUBLIC |
| 3.7.k | inscriptions | PRIVATE |
| 3.7 | arbitrage | hors scope composition |

**Total A02 : 10 PUBLIC applicables au générateur, 4 PRIVATE supposées respectées.**

Stocké dans `config/ffe_rules/alice_verifiability.json` — **annotations locales ALICE** (indépendantes de chess-app upstream).

---

## 4. Design détaillé des composants

### 4.1 RuleEngine (`services/ffe/rule_engine.py`)
~250 lignes, ISO 5055. Format vendored depuis `chess-app/backend/flat-six/rules/`.

```python
@dataclass(frozen=True)
class Rule:
    uuid: str              # ISO 42001 (RFC4122)
    source_ref: str        # ISO 5259 lineage (PDF FFE)
    article: str
    texte: str
    conditions: dict
    effet: str
    priority: int

class RuleEngine:
    @classmethod
    def from_json_file(cls, path: Path) -> "RuleEngine"  # Pydantic validation
    def filter_candidates(self, pool, context) -> list
    def validate_lineup(self, lineup, context) -> list[RuleViolation]
    def lineage_hash(self) -> str  # SHA-256 JSONs chargés
```

Dispatcher par `effet` + Strategy handler par article. Audit MongoDB + JSONL fallback. Tests 1-par-UUID.

### 4.2 VerifiabilityClassifier (`services/ali/verifiability.py`)
~80 lignes. Charge `alice_verifiability.json`, partitionne `list[Rule] → (public, private)`.

### 4.3 PlayerPoolLoader (`services/ali/pool_loader.py`)
~120 lignes. Lookup `joueurs.parquet` par `club_id`, filter `licence_active at round_date` (**F7 survivor**), merge avec `player_overrides` si fournis.

### 4.4 HistoryEnricher (`services/ali/history.py`)
~180 lignes. Calcule pour chaque joueur du pool :
- **F2** : `taux_presence_effectif(j, date) = Σ_r λ^(age_r) × 1[j joue r] / Σ_r λ^(age_r)`, λ=0.9 tuné backtest
- **F3** : `played_lag1`, `played_lag2`, `played_lag3` (3 dernières rondes, booléen)
- **Distribution échiquier** : `p(board=k | j joue) = count(j, k) / count(j)`, smoothé Laplace α=1

### 4.5 CopulaJointSampler (`services/ali/joint_sampler.py`) — **F6 SOTA**
~200 lignes. Remplace le Gibbs simple (F1 original).

**Algorithme** :
1. Fit : rank correlations de Spearman sur co-présence historique (matrix N×N)
2. Transform : chaque marginale → N(0,1) via rangs empiriques
3. Sample : N(0, Σ) via Cholesky, inverse transform vers présence binaire via seuil basé sur `taux_presence_effectif`
4. Complexité : O(N log N) fit + O(N²) sample (vs O(N²) × iterations pour Gibbs)

**Extension temporelle (bonus F3)** : la copule peut modéliser corrélation (ronde × joueur), couvrant partiellement le streak autoregressive.

**Sources** : Sklar 1959, Genest & Favre 2007, Nelsen 2006.

### 4.6 TopKEnumerator (`services/ali/topk.py`)
~150 lignes. Branch-and-bound priorisé :
1. Pre-filter pool via `engine.filter_candidates` (règles PUBLIC eligibility)
2. Sort par `P(present, recency+streak) × P(board=k | present)`
3. Énumération greedy backtracking : position 1..K, meilleur candidat respectant ordre Elo + contraintes
4. Top-10 lineups distincts par joint_prob

Déterministe (ISO 29119 testable reproductible). Complexité O(k × pool × team_size).

### 4.7 MonteCarloSampler (`services/ali/monte_carlo.py`) — **F5 SOTA**
~180 lignes. 10 scénarios stochastiques via **Latin Hypercube Sampling + antithetic variates** :

**LHS** : stratifier l'espace [0,1]^N en 10 cellules × N dimensions, un tirage par cellule → couverture uniforme garantie.

**Antithetic pairs** : si `u` est tiré, ajouter automatiquement `1-u` → 5 paires = 10 scénarios négativement corrélés, réduit variance 30-50%.

Pipeline par scénario :
1. Sample uniform `u ∈ LHS × antithetic`
2. Inverse-transform via CopulaJointSampler → presence vector
3. Conditionner par `played_lag1-3` (F3 boost)
4. Select team_size + assign boards Elo desc
5. `engine.validate_lineup()` → reject/resample (max 50 retries)
6. Compute `joint_prob`

**Sources** : McKay 1979, Hammersley & Morton 1956, Owen 2013.

### 4.8 ScenarioGenerator (`services/ali/generator.py`)
~100 lignes. Orchestrateur :

```python
class ScenarioGenerator:
    def generate(self, opponent_club_id, round_date, context,
                 overrides=None, n_topk=10, n_mc=10) -> ScenarioSet:
        # 1. pool = PlayerPoolLoader.load(...)
        # 2. enriched = HistoryEnricher.enrich(pool, cache.history)
        # 3. copula = CopulaJointSampler.fit(enriched, context)
        # 4. eligible = engine.filter_candidates(enriched, context)
        # 5. public, private = classifier.partition_rules(engine.rules)
        # 6. topk = TopKEnumerator.enumerate(eligible, 10, engine, public)
        # 7. mc = MonteCarloSampler.sample(eligible, 10, copula, engine, public)
        # 8. merged = merge(topk, mc); normalized = normalize_weights(merged)
        # 9. validate(sum_weights=1.0, len=20, unique)
        # 10. return ScenarioSet(scenarios=normalized,
        #                       lineage_hash=SHA256(parquet_sigs + rules_sigs + λ + inputs))
```

### 4.9 ALIDataCache (`services/ali/cache.py`)
~150 lignes. Chargé au lifespan FastAPI, mémoire permanente :
- `joueurs_by_club: dict[str, DataFrame]` — index O(1)
- `history_by_player: dict[str, DataFrame]` — index O(1)
- `presence_features: DataFrame` (précalculé)
- `pattern_features: DataFrame` (précalculé)
- `parquet_sig_joueurs/echiquiers: str` — SHA-256
- Méthode `is_stale(max_age_days=7)` pour alerte

### 4.11 ScenarioExplainer (`services/ali/explainer.py`) — ISO 42001 explainability
~80 lignes. Pour chaque `Scenario`, expose les top-3 facteurs contributifs :
- `taux_presence_effectif` (F2)
- `streak_lag_boost` (F3)
- `board_preference_match` (echiquier_prefere)
- `correlation_partners` (F6 copule : joueurs co-présents historiquement)
- `rule_pressure` (règles PUBLIC dominantes)

```python
@dataclass(frozen=True)
class ScenarioExplanation:
    scenario_index: int
    joint_prob: float
    top_factors: tuple[FactorContribution, ...]  # triés par impact desc

class ScenarioExplainer:
    def explain(self, scenario: Scenario, copula: CopulaJointSampler,
                enriched_pool: pd.DataFrame) -> ScenarioExplanation: ...
    def explain_all(self, scenario_set: ScenarioSet, ...) -> list[ScenarioExplanation]: ...
```

Exposé dans `ComposeResponse.metadata.explanations`.

### 4.12 KillSwitch / feature flag (`services/ali/mode.py`)
~40 lignes. Pydantic setting `ALI_MODE: Literal["sota", "fallback", "hybrid"]`.
- `sota` : pipeline complet Phase 3 (défaut production)
- `fallback` : force Phase 2 fallback (tri Elo) — utilisation si ALI bug
- `hybrid` : SOTA mais si rejection_rate > 50% OU timeout > 1.5s, bascule auto fallback avec warning dans response

```python
class ALIModeManager:
    def should_use_sota(self, health: ALIHealth) -> bool: ...
    def get_fallback_reason(self) -> str | None: ...
```

Observable via `/health` endpoint. Feature flag change sans redéploiement (env var hot reload).

### 4.13 ConfidenceLevel (`services/ali/confidence.py`)
~60 lignes. Métadonnée fairness attachée à `ScenarioSet` :

```python
@dataclass(frozen=True)
class ConfidenceMetadata:
    n_historical_rounds: int      # volume data disponible
    coverage_ratio: float         # % joueurs pool avec ≥3 rondes historique
    confidence_tier: str          # "high" | "medium" | "low" (seuils tunés backtest)
    sample_warnings: tuple[str, ...]  # ex "club a <5 rondes cette saison"
```

Seuils tunés backtest :
- `high` : coverage ≥ 0.8 ET n_historical_rounds ≥ 20
- `medium` : coverage ≥ 0.5 ET n_historical_rounds ≥ 10
- `low` : sinon

API retourne ce tier → capitaine informé de la fiabilité.

### 4.15 Observability DIY custom (package `services/observability/`)
**100% custom Python, aucune dep externe nouvelle.** Intégré Phase 3.

**4.15.1 `metrics.py` (~100 lignes)** — Prometheus text format export
- `Counter`, `Histogram`, `Gauge` custom thread-safe
- Endpoint `GET /metrics` retourne text format Prometheus stable
- Metrics exposées : `compose_requests_total`, `compose_duration_seconds`, `scenario_generation_duration`, `mc_rejection_rate`, `copula_fit_duration`, `ali_mode_gauge` (sota=1/fallback=0/hybrid=0.5)

**4.15.2 `logging.py` (~50 lignes)** — Structured JSON logging
- `logging.Formatter` custom output JSON ligne par ligne
- Context binding via `contextvars` : `request_id`, `user_id`, `tenant_id` propagés automatiquement
- Rotation fichier JSONL (déjà en place pour audit log, étendu général)

**4.15.3 `drift_tracker.py` (~200 lignes)** — Weekly data/model drift
- Tracker pandas hebdomadaire : KL divergence features input + PSI (Population Stability Index) + Brier score per-week + rejection rate hebdo
- Seuils configurables (`config/observability.yaml`) : drift warning si KL > 0.1, drift alert si KL > 0.3
- Output : `reports/drift/weekly_YYYY-WW.parquet` + MongoDB collection `ali_drift_history`

**4.15.4 `alerting.py` (~50 lignes)** — Webhooks Slack/email
- `send_webhook(url, payload)` via httpx (déjà là)
- Template messages : "ALI drift critical", "Latency p95 breach", "Rejection rate >50%"
- Throttle : pas plus de 1 alerte/type/heure

**4.15.5 `health.py` (~40 lignes)** — Health check enrichi
- `/health` retourne : `cache_age_seconds`, `cache_sigs`, `model_versions`, `ali_mode`, `last_Brier`, `last_rejection_rate`, `lineage_hash_prefix`
- Status degradé si cache_age > 7 jours OU drift > threshold

**Pas de deps externes ajoutées**. httpx déjà présent (Phase 2), pandas/numpy déjà présents (core stack).

### 4.16 Multi-tenant-ready tweaks (Phase 3 design, implémentation Phase 5)
- **API versioning** : routes passent de `/compose` à `/v1/compose` (~2 lignes `routes.py`)
- **Rate limiting per-tenant** : config slowapi key function = `user_club_id` au lieu de IP (~1 ligne)
- **tenant_id dans audit log** : ajout field dans `AuditEntry` + log format (~2 lignes, mapping = `user_club_id` Phase 3)
- **Cache structure partitionnable** : déjà partitionné par `club_id` (no change)
- **user_club_id explicite** dans ComposeRequest : déjà Phase 2 (no change)

Pas d'auth multi-tenant réelle en Phase 3 (Phase 5). Juste les crochets.

### 4.14 FeedbackCollector (`services/ali/feedback.py`) — D17 intégré Phase 3
~100 lignes. Endpoint `POST /feedback/scenario` pour capitaine :

```python
class ScenarioFeedback(BaseModel):
    scenario_set_lineage_hash: str  # identifie la prédiction
    scenario_index: int
    feedback_type: Literal["unrealistic", "realistic", "missing_player", "wrong_board"]
    message: str | None = None

class FeedbackCollector:
    async def record(self, feedback: ScenarioFeedback, user_id: str) -> None:
        # Store MongoDB collection `ali_feedback` (ISO 27001 audit)
        # Non-bloquant, async write
```

Bootstrap pour futur D9 (AIS adaptatif Phase 5+) : accumule signal utilisateur dès Phase 3.

### 4.10 Structures de données (`services/ali/types.py`)
~100 lignes, dataclasses `frozen=True` :
- `PlayerCandidate`
- `BoardAssignment`
- `Lineup` (tuple de K BoardAssignments)
- `Scenario(lineup, joint_prob, weight, source: "topk"|"monte_carlo")`
- `ScenarioSet(scenarios, opponent_club_id, round_date, generated_at, lineage_hash)`
- `CompetitionContext(competition_code, ronde, niveau, ...)`

---

## 5. Data flow et wiring API

### 5.1 Lifespan FastAPI (extension Phase 2)
`app/main.py::lifespan` ajoute :
```python
app.state.ali_cache = ALIDataCache.load_from_parquets(...)
app.state.rule_engine = RuleEngine.from_json_file(settings.ffe_rules_dir / "a02.json")
app.state.verifiability_classifier = VerifiabilityClassifier.from_json_file(...)
app.state.scenario_generator = ScenarioGenerator(engine, classifier, cache)
```
Log startup : `N_joueurs, N_rules, lineage_hash_prefix, λ_recency, copula_dim`.

### 5.2 Route `/compose` (extension Phase 2)
`app/api/routes.py::compose_route` appelle `scenario_generator.generate(...)` → boucle inférence Phase 2 × 20 scénarios → CE sur moyenne pondérée.

**Budget latence** :
- ALI generate : ~200ms (copule fit + LHS/antithetic 20 scenarios, cache in-RAM)
- Inference 20×K boards : ~800ms
- CE fallback : ~20ms
- Audit async : ~0ms bloquant
- **Total : ~1.1s** ✓ budget <2s

### 5.3 Contrat API (enrichi)
`ComposeRequest` ajoute : `opponent_club_id`, `round_date`, `competition`, `overrides?: list`, `n_topk?`, `n_mc?`.
`ComposeResponse` ajoute : `metadata { lineage_hash, rule_uuids_applied, model_versions, n_scenarios, sota_components }`.

---

## 6. Validation : backtest + métriques

### 6.1 Walk-forward backtest
- **Train λ / tuning** : saisons 2021-2023
- **Hold-out** : saison 2024 (gates T13-T22 verts exigés avant merge)
- **Test final** : saison 2025 (jamais vue en dev)

### 6.2 Les 10 quality gates T13-T22

| ID | Métrique | Seuil |
|----|----------|-------|
| T13 | Top-K recall (union 20 scénarios) | ≥ 0.90 |
| T14 | Jaccard max scenario vs observed | ≥ 0.75 |
| T15 | Brier score P(présence) | ≤ 0.20 |
| T16 | ECE P(présence), 10 bins | ≤ 0.05 |
| T17 | MAE E[score] team_size=8 | ≤ 1.0 |
| T18 | `sum(weights) == 1.0` | ± 1e-4 |
| T19 | `len(scenarios) == 20` | strict |
| T20 | scénarios distincts | strict |
| T21 | MC rejection rate | ≤ 0.30 |
| T22 | ≥3 métriques T13-T17 améliorées vs baseline Elo, aucune régression > 10% | strict |

### 6.3 Tests (~80 tests incluant property-based + fuzzing)

**Tests classiques (~60)** : voir §5.4 brainstorming. Coverage cible `services/ali/` ≥ 75%.

**Property-based testing via Hypothesis (~15 nouveaux)** :
- `test_rule_engine_properties.py` : toute Lineup valide → aucune violation ; toute violation → rule_uuid résolvable
- `test_copula_properties.py` : samples ∈ [0,1]^N ; Σ marginales ≈ taux_presence empirique ; positive semi-definite Σ
- `test_scenario_set_properties.py` : sum(weights) == 1.0 invariant ; 20 scenarios distincts invariant ; lineage_hash reproductible

**Fuzzing JSON inputs (~5)** :
- `test_rule_engine_fuzz.py` : JSON malformés, UUIDs invalides, priorités négatives, `conditions` vides
- `test_api_fuzz.py` : overrides avec Elo extrêmes, team_size aberrants, caractères Unicode adversarial

### 6.4 Peer review gate (nouveau Phase 3)
Avant merge main : invocation skill `superpowers:requesting-code-review` sur le diff complet Phase 3. Review bloquant si ≥1 finding critique.

---

## 6bis. Security (STRIDE threat model)

Document complet : `docs/security/ALI_THREAT_MODEL.md` (livré Phase 3).

| Threat | Vecteur | Mitigation Phase 3 |
|--------|---------|--------------------|
| **Spoofing** | Faux `opponent_club_id` ou `user_id` | Auth JWT Phase 2 déjà là + validation Pydantic (club_id ∈ liste FFE connue) |
| **Tampering** | `overrides` malveillants (Elo 3000, faux NR_FFE) | Pydantic strict + validation contre `joueurs.parquet` (NR_FFE doit exister) |
| **Repudiation** | Capitaine nie avoir reçu prédiction | Audit log MongoDB + fallback JSONL (ISO 27001) — chaque /compose tracé avec lineage_hash |
| **Info disclosure** | Fuite stats privées joueurs | Données publiques FFE uniquement (confirmé user : pas de RGPD issue) |
| **DoS** | `n_mc=1000000` dans overrides | Rate limiting Phase 2 + hard cap `n_topk ∈ [1,20]`, `n_mc ∈ [0,50]` |
| **Elevation of privilege** | Accès endpoints admin via /compose | Séparation endpoints + auth scope (Phase 2 déjà) |

## 6ter. Capacity planning et SLO

### Targets (documentés dans `docs/operations/ALI_SLO.md`)
- **p50 latence /compose** : ≤ 800ms
- **p95 latence /compose** : ≤ 1500ms
- **p99 latence /compose** : ≤ 2000ms (budget dur)
- **RPS sustainable** : ≥ 3 req/s sur Oracle VM ARM 24GB (benchmark à mesurer)
- **Memory footprint** : ≤ 1.5 GB (parquets cached + models + overhead)
- **Cache warm-up** : ≤ 10s au lifespan startup

### Benchmark script
`scripts/benchmark/ali_benchmark.py` (~150 lignes) mesure p50/p95/p99 sur jeux de test simulés. CI check : si p95 > 1500ms → warning.

## 7. Conformité ISO (19/19 normes — étendues)

| Norme | Mécanisme |
|-------|-----------|
| **5055** | Tous modules < 300 lignes, SRP strict, xenon complexité ≤ B |
| **27001** | Audit log MongoDB + JSONL fallback (violations, compositions, predictions) |
| **27034** | Pydantic validation JSONs chargés + inputs API |
| **25010** | Budget latence <2s, rejection rates logged, circuit breaker |
| **29119** | Docstring structuré, 1-test-par-UUID règle, E2E, seed RNG reproductible |
| **42010** | **ADR-013** RuleEngine JSON ; **ADR-014** ALI MC hybride SOTA |
| **15289** | Diagramme séquence `docs/architecture/ALI_ARCHITECTURE.md`, MkDocs strict |
| **42001** | `docs/iso/ALI_MODEL_CARD.md` incl. SOTA citations + alternatives évaluées |
| **42005** | `docs/iso/AI_RISK_ASSESSMENT.md` extension ALI (risques F1/F3/F7 assumptions) |
| **23894** | `docs/iso/AI_RISK_REGISTER.md` R-ALI-01 à R-ALI-05 |
| **5259** | lineage_hash = SHA-256(parquets + rules + λ + copula_params + config) |
| **25059** | 10 quality gates T13-T22, `docs/iso/ALI_QUALITY_GATES_REPORT.md` |
| **24029** | Robustness test Phase 3 smoke (pool vide, tous mute) ; full en Phase 3.5 (D8) |
| **24027** | Fairness smoke Phase 3 (breakdown niveau cpt) + **ConfidenceLevel metadata** (§4.13) ; full en Phase 3.5 (D8) |
| **23053** | AI lifecycle framework : phases ALICE documentées dans `docs/iso/AI_LIFECYCLE.md` |
| **22989** | Terminology standardisée : glossaire AI dans `docs/iso/AI_GLOSSARY.md` |
| **38507** | AI governance org-level : `docs/iso/AI_GOVERNANCE.md` (rôles, responsabilités, review) |
| **25012** | Data quality model : Pandera schemas sur joueurs/echiquiers (déjà en pre-commit hooks) |
| **25024** | Data quality measurement : métriques completeness/accuracy dans `compute_data_lineage()` |

---

## 8. Positionnement SOTA

### 8.1 SOTA intégrés (Phase 3)

| Composant | Choix | Source littérature |
|-----------|-------|---------------------|
| Joint sampling | Copule gaussienne | Sklar 1959, Genest & Favre 2007, Nelsen 2006 |
| MC diversity | Latin Hypercube + antithetic | McKay 1979, Hammersley 1956, Owen 2013 |
| Recency | Exponential decay λ | Brown 1959, Silver 2012 (FiveThirtyEight) |
| Streak | Autoregressive lag 1-3 | Box & Jenkins 1970, Pappalardo 2019 |
| Survivor | Licence active filter | Brown/Goetzmann/Ross 1992 |
| Calibration (ML) | Temp scaling + Dirichlet | Guo 2017, Kull 2019 |
| Optim paradigm | Predict-then-Optimize | Elmachtoub & Grigas 2022 |
| Backtest | Walk-forward | Bergmeir 2018 |
| Rules | JSON declarative | LegalRuleML (W3C) |

### 8.2 Alternatives SOTA évaluées, non retenues Phase 3

| Alternative | Raison report |
|-------------|---------------|
| Decision-Focused Learning (DFL) | ADR : PtO validé pour explicabilité (Wilder 2019) |
| Conformal prediction bout-en-bout | D15 — requiert CE multi-objectif (Phase 4+) |
| zone_enjeu modulation ALI | D13 — couplé CE OR-Tools (Phase 4+) |
| Adaptive Importance Sampling prod | D9 — requiert feedback loop prod (Phase 5+) |
| Deep lineup prediction (RNN, Transformer) | overkill pour N~40 joueurs/club, pas de data pour deep learning |

Toutes listées explicitement dans `ALI_MODEL_CARD.md` (ISO 42001 exige justification SOTA).

---

## 9. Résorption dette post-Phase 3

| Dette | Statut Phase 3 |
|-------|----------------|
| D1 (Elo 1500 stub) | **Résolue** — PlayerPoolLoader depuis joueurs.parquet |
| D2 (ALI Elo 1 scénario) | **Résolue** — ScenarioGenerator 20 scénarios SOTA |
| D5 (composer.py legacy) | **Résolue** — audit kickoff, supprimé ou documenté |
| D10 (sync chess-app JSON) | **Résolue** — script `sync_ffe_rules.py` + CI drift alert |
| D12, D14, D16 | **Résolues** — remontées en scope Phase 3 (option B, SOTA) |
| D3, D4 | Restent Phase 3.5 (J02, Coupes) |
| D8 | Reste Phase 3.5 STRICT (fairness/robustness full) |
| D9, D11, D13, D15 | Restent phases ultérieures (dépendances structurelles) |

---

## 10. Definition of Done Phase 3

### Qualité code
- [ ] 10 quality gates T13-T22 verts sur hold-out saison 2024
- [ ] ≥ 3 métriques T13-T17 améliorées vs baseline Elo, aucune régression > 10%
- [ ] Coverage `services/ali/` ≥ 75% incl. property-based + fuzzing
- [ ] CI green (ruff, mypy, bandit, pytest, xenon)
- [ ] Capacity benchmark : p95 ≤ 1500ms, RPS ≥ 3

### Résorption dette
- [ ] D1, D2, D5, D10, D12, D14, D16 résorbées
- [ ] D17 (feedback loop) résorbée via §4.14

### Artefacts ISO (19 normes)
- [ ] ADR-013 (RuleEngine JSON), ADR-014 (ALI MC hybride SOTA), ADR-015 (extension 19 normes ISO)
- [ ] `docs/iso/ALI_MODEL_CARD.md` incl. SOTA citations + alternatives évaluées + data sources disclaimer
- [ ] `docs/iso/AI_RISK_ASSESSMENT.md` étendu ALI
- [ ] `docs/iso/AI_RISK_REGISTER.md` R-ALI-01..05
- [ ] `docs/iso/ALI_QUALITY_GATES_REPORT.md` hold-out 2024
- [ ] `docs/iso/ALI_DATA_LINEAGE.md`
- [ ] `docs/iso/AI_LIFECYCLE.md` (23053)
- [ ] `docs/iso/AI_GLOSSARY.md` (22989)
- [ ] `docs/iso/AI_GOVERNANCE.md` (38507)
- [ ] `docs/security/ALI_THREAT_MODEL.md` (STRIDE)
- [ ] `docs/operations/ALI_SLO.md`
- [ ] `docs/architecture/ALI_ARCHITECTURE.md` diagramme séquence à jour

### Fonctionnalités nouvelles integrees (audit final)
- [ ] ScenarioExplainer exposé via ComposeResponse
- [ ] KillSwitch `ALI_MODE` opérationnel (sota/fallback/hybrid)
- [ ] ConfidenceLevel metadata exposée
- [ ] FeedbackCollector endpoint `POST /feedback/scenario` actif
- [ ] Observability DIY complète (metrics, logging, drift tracker, alerting, health) — 0 dep nouvelle
- [ ] Multi-tenant-ready : `/v1/compose`, rate limit per-tenant, tenant_id audit

### Gates process
- [ ] MkDocs build strict pass
- [ ] Peer review skill `superpowers:requesting-code-review` exécuté, 0 finding critique
- [ ] Checkpoint user final avant merge

---

## 11. Prochaine étape

Invoquer skill `writing-plans` pour générer le plan d'implémentation détaillé (découpage en tasks TDD-first, dépendances, critères de done par task).
