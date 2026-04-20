# Phase 3 · Plan 3 — Validation : Backtest + Quality Gates + Model Card SOTA

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Tasks ordonnées avec checkbox, TDD strict.

**Goal:** Mesurer empiriquement la qualité ALI sur saisons FFE historiques via walk-forward backtest + 5 quality gates T13-T17 + comparaison baseline Elo (T22). Livrer Model Card SOTA + AI_RISK_REGISTER.

**Architecture:** Backtest harness `scripts/backtest/ali_backtest.py` walk-forward sur saisons 2021-2023 (train/tune) + hold-out 2024 (validation gates). Réutilise ScenarioGenerator + StackingInferenceService Plan 2 via wrapper `run_backtest_match()` direct (pas TestClient). Métriques aggregées par run, comparées baseline Elo.

**Tech Stack:** pytest, pandas, numpy, scipy.stats (ECE bins), matplotlib (ROC/calibration plots), Pandera (schema). Pas de nouvelle dep majeure.

**Scope:** 16 tasks (Pre-Task 0 + 15 P3-Task), ~3 semaines estim. Gates T13-T22 + smoke fairness/robustness ISO 24027/24029.

**Décisions actées (brainstorming 2026-04-20)** :
- D1 ✅ walk-forward strict (Bergmeir & Benítez 2012, ISO 25059)
- D2 ✅ baseline Elo legitimate (tri Elo joueurs observés pré-match)
- D3 ✅ pilote local 10 matches debug (P3-Task 1) + backtest full 2024 Kaggle (P3-Task 9)
- D4 ✅ smoke fairness/robustness OBLIGATOIRE Plan 3 Task 12 (ISO 24027/24029 minimum, full Plan 3.5)
- D5 ✅ Pre-Task 0 audit `_safe_predict` (DONE commit 5750112), wrapper `run_backtest_match()` direct (no TestClient overhead)

**Principes directeurs :**
- **Walk-forward strict** : pour match (S, R), data accessible = data < (S, R) only
- **Strict mode backtest** : `_safe_predict(strict=True)` → fail-fast sur défaillance ML
- **Sample size** : ≥ 100 matches hold-out 2024 pour significance statistique (Bergmeir)
- **Baseline reproductible** : tri Elo joueurs observés pré-match, 1 seul scenario
- **ISO 42001 traceability** : lineage_hash backtest run propagé dans rapports

---

## File Structure

**Créer :**
```
scripts/backtest/__init__.py
scripts/backtest/harness.py            # walk-forward orchestrator
scripts/backtest/ground_truth.py       # extract observed lineup from echiquiers.parquet
scripts/backtest/metrics.py            # T13-T17 implementations
scripts/backtest/baseline_elo.py       # baseline scenario generator (1 lineup tri Elo)
scripts/backtest/run_match.py          # run_backtest_match wrapper (réutilise Plan 2)
scripts/backtest/runner.py             # full walk-forward main
scripts/backtest/report.py             # aggregate metrics + comparison
scripts/backtest/kaggle_kernel.py      # Kaggle kernel adapter pour P3-Task 9
docs/iso/ALI_MODEL_CARD.md
docs/iso/AI_RISK_REGISTER.md           # extension ALI (R-ALI-01..05)
docs/iso/AI_RISK_ASSESSMENT.md         # extension ALI
docs/iso/ALI_QUALITY_GATES_REPORT.md
docs/iso/ALI_DATA_LINEAGE.md
scripts/verify_plan3_dod.sh
tests/backtest/__init__.py
tests/backtest/test_ground_truth.py
tests/backtest/test_metrics.py
tests/backtest/test_baseline_elo.py
tests/backtest/test_run_match.py
tests/backtest/test_harness.py
tests/test_phase3_plan3_smoke.py       # smoke pilote local 10 matches
```

**Modifier :**
```
docs/architecture/ALI_ARCHITECTURE.md  # update Plan 3 backtest section
CLAUDE.md                               # D8 Plan 3.5, gates Plan 3 résolus
```

---

## Pre-Task 0 : Audit `_safe_predict` strict mode ✅ DONE (commit 5750112)

`services/ali/aggregation.py::_safe_predict(strict=False)` ajouté. `ScenarioAggregationCtx.strict: bool = False` propagé. Tests aggregation 6/6 PASS. Backtest harness Plan 3 set `strict=True`.

## Pre-Task 1 : Corrections CRITIQUES ML inference end-to-end ✅ DONE

**Root causes détectées pendant P3-Task 1 pilote** (zero debt policy — 0 avancement sans fix) :

### C1 : FeatureStore stub (only 3 cols vs model 201 features)
- **Cause** : `FeatureStore` cherchait `joueur_features.parquet` inexistant. Retournait ~5 cols. Model entraîné sur 201 features → XGBoost rejetait silencieusement.
- **Fix** : nouveau script `scripts/build_feature_store.py` qui aggrège `data/features/train.parquet` (1.14M rows × 147 cols) en :
  - `data/feature_store/training_mean.parquet` (201-col fallback row matchant `LightGBM.txt::feature_names`)
  - `data/feature_store/joueur_features.parquet` (95851 joueurs × 21 canonical features)
- **FeatureStore v2.0** rewritten : `assemble()` retourne DataFrame 201-col matching model feature_names. Override blanc_*/noir_* depuis per-player aggregates + training_mean fallback pour unknowns.

### C2 : `_assemble_features` retournait ndarray (lost feature_names → XGBoost refuse)
- **Cause** : `.values` → ndarray → XGBoost Booster fail validation
- **Fix** : `services/ali/aggregation.py::_assemble_features` retourne DataFrame (preserve feature_names) + raise RuntimeError si `feature_store=None` (ISO 42001 fail-fast)

### C3 : MLP champion manquant sur HF Hub → stacking crash
- **Cause** : `mlp_meta_learner.joblib` absent de `Pierrax/alice-engine/v9/`. Seul calibrateur Dirichlet déployé.
- **Fix** : `scripts/serving/model_loader.py` détecte MLP absent → auto-fallback LGB+Dirichlet mode avec warning. Re-deploy MLP(32,16) scope Phase 1 futur (D-P3-13 à tracer).

### C4 : `_predict_fallback` utilisait `.predict_proba` mais Dirichlet est dict (W matrix + bias)
- **Cause** : `_predict_fallback` crashait sur `b.mlp_model.predict_proba(log_p)` (AttributeError : 'dict' object).
- **Fix** : `services/inference.py::_predict_fallback` détecte `isinstance(b.mlp_model, dict)` → appel helper `_apply_dirichlet` qui implémente `softmax(W @ log(p) + bias)`.

### C5 : harness.py avait fallback silent `self.feature_store = None`
- **Fix** : suppression du try/except silent. FeatureStore MUST load (raise si training_mean.parquet absent) — ISO 42001 fail-fast.

### Validation empirique end-to-end
- **ML inference fonctionne** : `PredictionResult(p_loss=0.47, p_draw=0.26, p_win=0.26)`, sum=1.0 sur match test (KRAMNIK vs TREGUBOV)
- **3/3 tests PASS** avec strict=True réel :
  - `tests/backtest/test_harness.py` 2/2 (services bootstrap + run_match complet)
  - `tests/test_phase3_plan3_smoke.py::test_pilot_10_matches_no_silent_fallback` 1/1 (10 matches, strict mode, pas de fallback silencieux)
- **6/6 tests** `tests/test_compose_ali_aggregation.py` avec mock FeatureStore pattern

### Tests et couverture post-fix
- aggregation module : 6/6 PASS
- harness + smoke : 3/3 PASS ML inference réel, feature store 201-col, latence < 5s/match
- pilote 10 matches : 10/10 strict=True, AUCUN fallback silent

---

## P3-Task 1 : Backtest harness + pilote feasibility 10 matches

**Files:**
- Create: `scripts/backtest/harness.py` (~150 lignes)
- Create: `scripts/backtest/run_match.py` (~80 lignes wrapper direct, no TestClient)
- Create: `tests/backtest/test_harness.py` (~5 tests)
- Create: `tests/test_phase3_plan3_smoke.py` (~3 tests pilote 10 matches)

**ISO:** 29119 (test integrity), 5259 (lineage_hash backtest run), 25010 (perf observable)

**Spec wrapper** :
```python
@dataclass
class BacktestMatchResult:
    saison: int
    ronde: int
    user_club: str
    opponent_club: str
    scenario_set: ScenarioSet
    aggregated_boards: list[AggregatedBoard]
    observed_lineup: list[str]   # ground truth (filled P3-Task 2)
    elapsed_ms: float
    lineage_hash: str

def run_backtest_match(
    user_club_id: str,
    opponent_club_id: str,
    saison: int,
    ronde: int,
    nb_rondes_total: int,
    division: str,
    cache: ALIDataCache,
    rule_engine: RuleEngine,
    classifier: VerifiabilityClassifier,
    pool_loader: PlayerPoolLoader,
    history_enricher: HistoryEnricher,
    inference: StackingInferenceService,
    feature_store: Any,
    user_lineup: list[dict],
    seed: int = 42,
) -> BacktestMatchResult:
    """Direct wrapper: generator + aggregation, no FastAPI overhead.
    
    Strict mode: ScenarioAggregationCtx(strict=True) → fail-fast sur défaillance.
    """
    ...
```

**Walk-forward isolation** : cache joueurs.parquet OK (state actuel), mais `cache.lookup_history(names)` doit filtrer `(saison, ronde) < (target_saison, target_ronde)` pour éviter leakage. Helper `_filter_temporal()` à ajouter.

**Pilote** : 10 matches 2024 random échantillonnés, vérifier :
- run_backtest_match termine sans raise
- ScenarioSet contient 20 scenarios
- AggregatedBoard probas dans [0,1] et somme=1
- Pas de fallback silent (strict=True, ML inference doit marcher)
- Latence par match < 5s (acceptable pour scope full)

---

## P3-Task 2 : Ground truth extraction

**Files:**
- Create: `scripts/backtest/ground_truth.py` (~100 lignes)
- Create: `tests/backtest/test_ground_truth.py` (~5 tests)

**ISO:** 5259 (data lineage), 24027 (no leakage)

```python
def extract_observed_lineup(
    cache: ALIDataCache,
    club_name: str,        # equipe_dom or equipe_ext (echiquiers.parquet)
    saison: int,
    ronde: int,
) -> list[ObservedPlayer]:
    """Extract real lineup from echiquiers.parquet for (club, saison, ronde).
    
    Returns sorted by board (échiquier 1..K). Raises if match absent ou incomplet.
    """
```

Tests :
- Match existant retourne lineup complet K joueurs
- Match absent raise
- Mapping club_id (joueurs.parquet) ↔ equipe_name (echiquiers.parquet) correct

---

## P3-Task 3 : Metric T13 Top-K recall

**Files:**
- Add to: `scripts/backtest/metrics.py` (~30 lignes)
- Add to: `tests/backtest/test_metrics.py` (~3 tests T13)

**ISO:** 25059

**Formule** : `T13 = |observed_players ∩ (∪_s scenario_s.players)| / |observed_players|`

Gate seuil **≥ 0.90** (au moins 90% des joueurs observés capturés dans union des 20 scenarios).

Tests :
- Tous joueurs présents dans union → T13 = 1.0
- Aucun joueur dans union → T13 = 0.0
- Mix → ratio correct

---

## P3-Task 4 : Metric T14 Jaccard max

**Files:**
- Add to: `scripts/backtest/metrics.py` (~30 lignes)
- Tests T14

**ISO:** 25059

**Formule** : `T14 = max_s |observed ∩ s_lineup| / |observed ∪ s_lineup|`

Gate seuil **≥ 0.75** (au moins 1 scenario proche de la réalité).

---

## P3-Task 5 : Metric T15 Brier score P(présence)

**Files:**
- Add to: `scripts/backtest/metrics.py` (~50 lignes)
- Tests T15

**ISO:** 25059, 24029 (calibration)

**Formule** : per-player, `p_presence = Σ_s scenario_s.weight × 1[player ∈ s_lineup]`. Comparer à observed (0/1).
`T15 = (1/N) Σ_p (p_presence_p - 1[p observed])²`

Gate seuil **≤ 0.20** (baseline Elo ~0.26 attendu).

---

## P3-Task 6 : Metric T16 ECE calibration 10-bins

**Files:**
- Add to: `scripts/backtest/metrics.py` (~70 lignes)
- Tests T16

**ISO:** 24029 (calibration robustness)

**Formule** : 10 bins de p_presence. ECE = `Σ_b |bucket_b|/N × |mean(p_b) - freq_observed_b|`

Gate seuil **≤ 0.05** (calibration acceptable).

---

## P3-Task 7 : Metric T17 E[score] MAE (ML inference loop)

**Files:**
- Add to: `scripts/backtest/metrics.py` (~80 lignes)
- Tests T17 sur 5-10 matches sample

**ISO:** 25059, 25010 (perf), 42001 (traceability ML inference)

**Formule** : `E[score_predicted] = Σ_k aggregated_board_k.e_score`. Comparer à observed score équipe (somme victoires + 0.5 × draws).
`T17 = MAE = (1/N_matches) Σ |E[score_pred] - score_observed|`

Gate seuil **≤ 1.0** (sur team_size=8, MAE < 1 point/match acceptable).

---

## P3-Task 8 : Baseline Elo legitimate (tri Elo)

**Files:**
- Create: `scripts/backtest/baseline_elo.py` (~80 lignes)
- Tests baseline (~3 tests)

**ISO:** 25059 (baseline comparison)

```python
def baseline_elo_scenario(
    pool: list[PlayerCandidate], context: CompetitionContext,
) -> ScenarioSet:
    """Baseline 1-scenario : tri Elo desc des joueurs observés pré-match.
    
    1 lineup déterministe, weight=1.0, source='baseline_elo'.
    """
```

Pour T22, mesurer T13-T17 sur baseline en parallèle de ALI SOTA. Comparaison tableau.

---

## P3-Task 9 : Backtest runner full walk-forward + Kaggle adapter

**Files:**
- Create: `scripts/backtest/runner.py` (~150 lignes)
- Create: `scripts/backtest/kaggle_kernel.py` (~80 lignes adapter)
- Create: `kaggle/kernel_metadata_backtest.json`

**ISO:** 25059 (production-grade backtest), 5259 (lineage)

Walk-forward strict :
- Saisons 2021-2023 = tuning (λ recency, n_topk, n_mc_pairs)
- Saison 2024 = hold-out validation (T13-T22)
- ≥ 100 matches 2024 hold-out (Bergmeir significance)

**Kaggle kernel** : push code + parquets + model bundle, run full 2024 hold-out, sauve résultats parquet artifact, rapatrie.

**Output** : `reports/backtest/2024_holdout_run_<lineage>.parquet` avec per-match métriques.

---

## P3-Task 10 : Model Card ALI SOTA (D-P3-10)

**Files:**
- Create: `docs/iso/ALI_MODEL_CARD.md`

**ISO:** 42001 (Model Card MANDATORY), Mitchell et al. 2019 (FAccT)

Sections OBLIGATOIRES :
- Model details (name, version, date, type, contact)
- Intended use (ALI predicts adversary lineup pour CE composition)
- Training data (rule-based + empirical, no ML training stricto sensu)
- Evaluation data (saison 2024 hold-out, ≥100 matches)
- **Quantitative analyses (T13-T22 results + breakdown fairness niveau compet)**
- **Ethical considerations** (FFE public data, no PII)
- **Limitations** (A02 scope only, noyau private, pool joueurs cache hebdo, etc.)
- **SOTA comparative audit (D-P3-10)** : tableau alternatives évaluées vs choix retenus pour F1/F5/F6/F2/F3/F7

---

## P3-Task 11 : AI_RISK_REGISTER.md (R-ALI-01..05)

**Files:**
- Modify/Create: `docs/iso/AI_RISK_REGISTER.md`

**ISO:** 23894 (AI risk management)

Risques identifiés :
- R-ALI-01 : biais MC conservateur si classifier mal calibré (mitigé Plan 2 D-P2-02)
- R-ALI-02 : F1 corrélation indépendance fausse → couvert par F6 copule
- R-ALI-03 : F7 survivor bias hypothèse implicite (membership joueurs.parquet)
- R-ALI-04 : silent fallback ML inference masque défaillance (mitigé Pre-Task 0)
- R-ALI-05 : seed=42 fixe → 2 calls identiques même résultat (idempotence observable, D-P2-04)

---

## P3-Task 12 : AI_RISK_ASSESSMENT.md ALI extension + smoke fairness/robustness OBLIGATOIRE

**Files:**
- Modify: `docs/iso/AI_RISK_ASSESSMENT.md`
- Create: tests/backtest/test_fairness_smoke.py (~5 tests breakdown)
- Create: tests/backtest/test_robustness_smoke.py (~3 tests stress Elo)

**ISO:** 42005 (impact assessment), 24027 (fairness MANDATORY ISO + Mitchell 2019), 24029 (robustness)

**Smoke fairness OBLIGATOIRE Plan 3** (NIST AI RMF "Measure" core function) :
- Breakdown T13-T17 par niveau competition (N1/N2/N3/N4)
- Breakdown par taille club (quartiles : Q1<10 joueurs, Q2 10-25, Q3 25-50, Q4 >50)
- Documentation findings dans Model Card §Quantitative analyses

**Smoke robustness OBLIGATOIRE Plan 3** :
- 1 stress test : perturber Elo joueurs ±50, mesurer variation Top-K recall (gate <10%)

**Plan 3.5 (D8 STRICT)** : full breakdown (incl. genre, croisé levels×sizes), full robustness suite (perturb features, adversarial, missing data).

---

## P3-Task 13 : ALI_QUALITY_GATES_REPORT.md hold-out 2024

**Files:**
- Create: `docs/iso/ALI_QUALITY_GATES_REPORT.md`

**ISO:** 25059 (quality gates report)

Tableau résultats T13-T22 sur hold-out 2024 :
- Cible vs réalisé
- Pass/Fail par gate
- Comparaison baseline Elo (T22)
- IC bootstrap pour chaque métrique
- N matches utilisés

---

## P3-Task 14 : ALI_DATA_LINEAGE.md end-to-end

**Files:**
- Create: `docs/iso/ALI_DATA_LINEAGE.md`

**ISO:** 5259 (data lineage)

Diagramme + tableau :
- PDFs FFE → docling pocket-arbiter → JSON chess-app → vendored config/ffe_rules → RuleEngine
- joueurs.parquet (scrape FFE hebdo) → ALIDataCache → PlayerPoolLoader → enrichi
- echiquiers.parquet (scrape FFE) → cache → HistoryEnricher F2/F3 → CopulaJointSampler
- Backtest run lineage_hash = SHA-256(parquet sigs + rules sig + saison + run params + run timestamp)

---

## P3-Task 15 : verify_plan3_dod.sh + peer review + merge

**Files:**
- Create: `scripts/verify_plan3_dod.sh`
- Update: `CLAUDE.md` (D8 Plan 3.5, gates Plan 3 résolus)

**ISO:** 29119, 5055 + tous P3G ci-dessous

### P3G — Plan 3 Quality Gates (16 gates)

| Gate | Norme | Check |
|------|-------|-------|
| P3G01 | 5055 | Files ≤ 300 lignes (services + scripts/backtest) |
| P3G02 | 5055 | xenon ≤ B |
| P3G03 | 5055 | mypy --strict |
| P3G04 | 5055 | ruff |
| P3G05 | 27001 | gitleaks |
| **P3G06** | 25059 (T13) | **Top-K recall ≥ 0.90** sur hold-out 2024 |
| **P3G07** | 25059 (T14) | **Jaccard max ≥ 0.75** |
| **P3G08** | 25059 (T15) | **Brier ≤ 0.20** |
| **P3G09** | 25059 (T16) | **ECE ≤ 0.05** |
| **P3G10** | 25059 (T17) | **E[score] MAE ≤ 1.0** |
| **P3G11** | 25059 (T22) | **≥3 métriques T13-T17 améliorées vs baseline Elo, aucune régression > 10%** |
| P3G12 | 24027 | Smoke fairness breakdown niveau competition documenté Model Card |
| P3G13 | 24029 | Smoke robustness perturb Elo ±50 < 10% variation |
| P3G14 | 42001 | ALI_MODEL_CARD.md committé incl. SOTA comparative audit |
| P3G15 | 23894 | AI_RISK_REGISTER R-ALI-01..05 |
| P3G16 | 5259 | lineage_hash backtest run présent dans rapport |

Peer review skill `superpowers:requesting-code-review` sur diff complet Plan 3. Merge fast-forward sur master si APPROVED.

---

## Definition of Done Plan 3

### Quality gates
- [ ] 16 P3G gates verts
- [ ] verify_plan3_dod.sh exit 0
- [ ] Backtest hold-out 2024 ≥ 100 matches (Bergmeir significance)
- [ ] T22 ≥ 3 métriques améliorées vs baseline Elo
- [ ] Smoke fairness + robustness documentés Model Card

### Résorption dette
- [ ] D-P3-10 résolu via Model Card SOTA comparative audit
- [ ] D8 partiel résolu via smoke fairness/robustness (full Plan 3.5)
- [ ] D-P2-02bis : si bias surface backtest, fix avant merge

### Artefacts
- [ ] ALI_MODEL_CARD.md (SOTA citations + alternatives évaluées + fairness breakdown)
- [ ] AI_RISK_REGISTER.md R-ALI-01..05
- [ ] AI_RISK_ASSESSMENT.md ALI extension
- [ ] ALI_QUALITY_GATES_REPORT.md hold-out 2024
- [ ] ALI_DATA_LINEAGE.md end-to-end
- [ ] backtest results parquet + Kaggle kernel

### Process
- [ ] Peer review skill `superpowers:requesting-code-review`, 0 finding critique
- [ ] Checkpoint user final avant merge
