# Phase 3 · Plan 3 — Validation : Backtest + Quality Gates + Model Card SOTA (V2)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> **V2 (2026-04-20)** : refonte complète après audit 8 dimensions (feedback_plan_8_dimensions_audit.md).

**Goal:** Mesurer empiriquement la qualité ALI SOTA sur saisons FFE historiques via walk-forward backtest rigoureux + 16 quality gates P3G01-P3G16 + comparaison baseline Elo (T22 McNemar). Livrer Model Card SOTA + AI_RISK_REGISTER + smoke fairness/robustness.

**Architecture:** Harness backtest `scripts/backtest/*` walk-forward strict sur saisons 2021-2023 (tuning) + hold-out 2024 (validation gates). Réutilise ScenarioGenerator + StackingInferenceService + FeatureStore Plan 1+2 via `run_backtest_match()` direct. Bootstrap CI (BCa) + McNemar T22. Réutilise `scripts/robustness/`, `scripts/monitoring/bias_tracker.py`, templates existants ISO_25059.

**Tech Stack:** pytest, pandas, numpy, scipy.stats (ECE, McNemar, bootstrap), matplotlib (reliability diagrams). Pandera pour schemas. DVC pour backtest artifacts.

**Scope:** 24 tasks + Pre-Task 0/1 DONE, ~4 semaines estim. 16 P3G gates + DoD per-task + peer review intermédiaires.

---

## Audit Matrix (8 dimensions — respect feedback_plan_8_dimensions_audit.md)

### Dim 1 : Spec Phase 3 mapping

| Spec § | Content | Plan 3 Task |
|--------|---------|-------------|
| §6.1 walk-forward | Rigorous temporal split | T1 harness + T10 runner |
| §6.2 T13-T17 | 5 quality gates | T3-T7 metrics |
| §6.3 ~60 tests + property + fuzzing | Tests completeness | T19 property-based Hypothesis |
| §7 19 normes ISO | Full compliance | T20-T24 docs ISO |
| §8 SOTA positionnement | Alternatives audit | T11 Model Card + D-P3-10 |
| §10 DoD | Definition of Done | **Per-task DoD** + T25 verify |

### Dim 2 : Normes ML génériques

| Standard | Task |
|----------|------|
| Walk-forward strict (Bergmeir 2012) | T1, T10 |
| Bootstrap CI BCa | T8 CI framework |
| McNemar test T22 | T9 statistical comparison |
| Stratified sampling fairness | T15 stratification protocol |
| Seed + lineage reproducibility | T1 (lineage_hash) |
| Sample size ≥ 100 matches | T10 (Bergmeir significance) |

### Dim 3 : Normes ML sports + compositions prédictives

| Standard | Task |
|----------|------|
| Brier skill score (vs baseline) | T6 |
| Reliability diagrams | T7 |
| Accuracy@K lineup prediction | T3 (Top-K) + T3b Accuracy@K formally |
| Multi-season validation CARMELO | T10 rolling multi-season |
| Pappalardo 2019 ≥ 1000 obs | T10 ≥ 100 matches 2024 + 200+ 2023 tuning |

### Dim 4 : Audit existant (réutilisation DRY)

| Module existant | Réutilisation Plan 3 |
|-----------------|---------------------|
| `scripts/robustness/` | T16 stress-test réutilise framework |
| `scripts/monitoring/bias_tracker.py` | T15 fairness réutilise helpers |
| `services/feature_store.py::check_quality_gates()` | T25 pattern P3G script |
| `reports/ISO_25059_TRAINING_REPORT_v20260117.md` | T22 template ALI_QUALITY_GATES_REPORT |
| `scripts/features/ali/presence.py` + `patterns.py` | T2 ground truth réutilise schemas |
| Plan 2 `services/ali/*` | T1 run_match wrapper |

### Dim 5 : Inventaire exhaustivité

| Task cachée | Ajoutée au plan |
|-------------|-----------------|
| Pandera schema inputs backtest | T2b |
| Tests déterminisme 2 runs | T18 |
| Edge cases (ronde 1, clubs petits) | T17 |
| Peer review intermédiaire (Tasks 1, 9, 15, 22) | T0-gate entre phases |
| DVC backtest results | T24 versioning |
| D5 composer.py legacy (différé Plan 2) | T23 audit + decision |
| D-P2-03 `_EXPECTED_SCENARIOS=20` | T23 refactor configurable |
| D-P2-04 seed=42 | T23 expose via ComposeRequest |
| D-P3-13 NOUVEAU MLP redeploy | T23 tracker |

### Dim 6 : DAG dépendances

```
Pre-Task 0 (DONE) ── Pre-Task 1 (DONE) ── Task 1 (DONE)
                                         │
Task 1 ──────────────────────────────────┼── Task 2 ground truth
                                         │   Task 2b Pandera schema
                                         │
Task 2 ─── Tasks 3-7 (metrics, //)       │
                                         │
Task 8 (Bootstrap CI) ←─ Tasks 3-7       │
Task 9 (McNemar T22) ←─ Tasks 3-7 + T8   │
                                         │
Task 10 (baseline) ←─ Task 2             │
Tasks 8,9 + T10 ─── Task 11 (Runner full 2024)
                                         │
GATE: Peer Review intermédiaire post-T11 │
                                         │
T11 ─── Tasks 12-16 (reliability, fairness, robustness, edges, determinism)
                                         │
T11 + T13 (fairness) + T14 (robust) ─── T17 (Model Card SOTA)
T17 + T18 (risk reg) + T19 (risk assess) ─── T22 (Gates Report)
                                         │
T22 ─── T23 (dette résorption) ─── T24 (DVC) ─── T25 (verify + review + merge)
                                         │
Peer Review Final ─── Merge master
```

### Dim 7 : Qualité per-task

- **DoD per-task** : chaque task a sa checklist acceptance criteria
- **Coverage target** : ≥ 80% per-task nouveau module
- **TDD first** : test → rouge → impl → vert → commit
- **Peer review intermédiaire** : skill `superpowers:requesting-code-review` à 3 jalons (T7, T11, T22)

### Dim 8 : DoD + P3G + code reviewer

- **16 P3G gates** + mapping per-task (ci-dessous)
- **verify_plan3_dod.sh** designé dès T25 (script tests tous P3G)
- **Code reviewer final** MANDATORY avant merge (T25)

---

## Pre-Task 0 ✅ DONE (commit 5750112) — `_safe_predict(strict=True)`

## Pre-Task 1 ✅ DONE (commit 7ac528e) — ML inference end-to-end
- C1 FeatureStore v2 201-col + build_feature_store.py
- C2 _assemble_features DataFrame preserve feature_names
- C3 MLP champion absent → auto-fallback LGB+Dirichlet
- C4 _apply_dirichlet helper (softmax W @ log(p) + bias)
- C5 harness fail-fast si feature_store absent

## Task 1 ✅ DONE (commit 8ca30c8) — backtest harness + pilote 10 matches

---

## NOUVELLES TASKS V2 (détail)

### T2. Ground truth extraction
**Files:** `scripts/backtest/ground_truth.py` + tests
**DoD:** extract_observed_lineup(cache, club_name, saison, ronde) retourne lineup observed, raise si absent. 5 tests incl. edge case (match absent, club inconnu).
**P3G:** contribue P3G06 (coverage)
**Dépend:** T1

### T2b. Pandera schema backtest inputs (NEW)
**Files:** `scripts/backtest/schemas.py` Pandera definitions
**DoD:** BacktestInputSchema (club_id, saison, ronde, division, team_size) validation + exceptions claires. Tests validation reject/accept.
**P3G:** P3G05 (input validation ISO 27034)

### T3. T13 Top-K recall
**DoD:** top_k_recall(observed, scenarios) formule correcte, gate ≥0.90, 3 tests.

### T3b. Accuracy@K formalisé (NEW sports SOTA)
**DoD:** accuracy_at_k(observed, top_k_scenario) = |obs ∩ top_scenario_k| / k. Gate ≥0.75 (SOTA sports lineup).

### T4. T14 Jaccard max
**DoD:** jaccard_max gate ≥0.75, 3 tests.

### T5. T15 Brier score
**DoD:** brier_presence gate ≤0.20.

### T6. T15b Brier skill score (NEW sports SOTA)
**DoD:** brier_skill_score = 1 - (Brier_model / Brier_baseline). Gate ≥0.05 (model > baseline). Pappalardo 2019.

### T7. T16 ECE 10-bins
**DoD:** ece_10bins gate ≤0.05.

### T7b. Reliability diagram (NEW FAccT SOTA)
**Files:** `scripts/backtest/plots.py` + `reports/reliability_diagram_<lineage>.png`
**DoD:** reliability_diagram(p_presence, observed) génère plot calibration. 3 tests plot.
**Peer review intermédiaire ici** (jalon 1 : metrics framework complet)

### T8. Bootstrap CI (BCa) framework (NEW)
**Files:** `scripts/backtest/bootstrap.py`
**DoD:** bootstrap_ci(values, n_resamples=1000, confidence=0.95, method='BCa'). IC à 95% pour toute métrique. 3 tests.
**P3G:** P3G11 (toutes metrics avec IC)

### T9. McNemar test T22 (NEW)
**Files:** `scripts/backtest/mcnemar.py`
**DoD:** mcnemar_test(preds_model, preds_baseline, observed) retourne stat + p-value. Gate p<0.05 pour ≥3 metrics améliorées. 3 tests.
**P3G:** P3G11 (significance statistique baseline comparison)

### T10. Baseline Elo legitimate
**Files:** `scripts/backtest/baseline_elo.py`
**DoD:** baseline 1-scenario tri Elo desc joueurs observés pré-match, 3 tests.

### T11. Runner full walk-forward + Kaggle adapter
**Files:** `scripts/backtest/runner.py` + `kaggle/kernel_backtest.py`
**DoD:** saisons 2021-2023 tuning + 2024 hold-out ≥100 matches (Bergmeir significance). Multi-season rolling CARMELO pattern. Output parquet lineage.
**Peer review intermédiaire ici** (jalon 2 : backtest core complet)

### T12. T17 E[score] MAE (ML inference loop)
**DoD:** mae sur team_size=8, gate ≤1.0. Sample 100 matches 2024.
**Dépend:** T11

### T13. Smoke fairness OBLIGATOIRE (ISO 24027)
**Files:** tests/backtest/test_fairness_smoke.py réutilise `scripts/monitoring/bias_tracker.py`
**DoD:** breakdown T13-T17 par niveau (N1/N2/N3/N4) + taille club (quartiles). Stratified sampling.

### T14. Smoke robustness OBLIGATOIRE (ISO 24029)
**Files:** tests/backtest/test_robustness_smoke.py réutilise `scripts/robustness/` framework
**DoD:** perturb Elo ±50 → variation Top-K <10%. 3 tests.

### T15. Fairness protocol stratifié
**Files:** `scripts/backtest/fairness.py` breakdown formalisé
**DoD:** StratifiedSampler(level, club_size) pour fair gates T13-T17.

### T16. Robustness stress-test suite
**Files:** `scripts/backtest/robustness.py` réutilise `scripts/robustness/`
**DoD:** 3 stress-tests : Elo perturb, feature noise, missing data. Gates.

### T17. Edge cases tests (NEW)
**Files:** tests/backtest/test_edge_cases.py
**DoD:** 5 tests : ronde 1 (no history), club <12 joueurs, ronde=dernière, saison incomplète, match avec forfaits.

### T18. Determinism tests (NEW)
**Files:** tests/backtest/test_determinism.py
**DoD:** 2 runs seed=42 identiques → mêmes metrics (bit-identiques). Hash check ScenarioSet.

### T19. Property-based tests Hypothesis (spec §6.3)
**Files:** tests/backtest/test_properties_hypothesis.py
**DoD:** 15 property tests sur metrics (T13-T17) via Hypothesis : symmetries, bounds, invariants.

### T20. Model Card ALI SOTA (Mitchell 2019 FAccT)
**Files:** `docs/iso/ALI_MODEL_CARD.md`
**DoD:** Sections Mitchell 2019 : Model details, Intended use, Eval data, Quantitative analyses (T13-T22 + breakdown fairness), Ethical, Limitations, **SOTA comparative audit (D-P3-10)** tableau alternatives évaluées.

### T21. AI_RISK_REGISTER R-ALI-01..05 (ISO 23894)
**Files:** `docs/iso/AI_RISK_REGISTER.md`
**DoD:** R-ALI-01..05 identifiés + mitigations ref + owner + status.

### T22. AI_RISK_ASSESSMENT + Quality Gates Report (ISO 42005 + 25059)
**Files:** `docs/iso/AI_RISK_ASSESSMENT.md` + `docs/iso/ALI_QUALITY_GATES_REPORT.md`
**DoD:** réutilise template `ISO_25059_TRAINING_REPORT_v20260117.md`. Tableau T13-T22 + IC bootstrap + N matches.
**Peer review intermédiaire ici** (jalon 3 : docs ISO complètes)

### T23. Dette résorption (NEW)
**DoD per-item:**
- D5 composer.py : audit + suppression OU documentation
- D-P2-03 `_EXPECTED_SCENARIOS` : configurable via ScenarioGenerator param
- D-P2-04 seed=42 : expose via ComposeRequest
- D-P3-13 NOUVEAU : tracer MLP redeploy

### T24. DVC versioning backtest artifacts (NEW) — DONE 2026-04-29
**Files:** `dvc.yaml` (2 stages : refit_mlp_champion + backtest_holdout_2024), `dvc.lock`, `.dvc/config`, `.dvcignore`, doc append `docs/devops/ML_MODEL_VERSIONING_STANDARDS.md`
**DoD:** reports/backtest/ali_holdout_2024.json tracked (cache: false, persist), models/cache/mlp_*.joblib + temperature_T DVC-cached avec md5 deps, DAG `refit → backtest`, `dvc status` reproducible, lineage commit ↔ artefacts via dvc.lock. **Limites assumées Phase 5** : pas de remote DVC distant + pas de couverture training Kaggle (CatBoost/XGB/LGB).

### T25. verify_plan3_dod.sh + peer review FINAL + merge — DONE 2026-04-29
**Files:** `scripts/verify_plan3_dod.sh` (270L), `docs/architecture/adr/ADR-016-ali-conditioned-multi-team-adverse-ce-mirror.md` (stub Proposed Phase 4a), `docs/project/DEBT_LEDGER.md` (versioned mirror), `docs/iso/AI_RISK_REGISTER.md` (R-ALI-06 ajouté §2.7), commit `9e51dd2` (JALON #3 fixes) + `652fac3` (dvc.lock resync), merge master fast-forward.
**DoD:** 16 P3G gates + 9 structural gates via script (PRIMARY = BSS+Wilcoxon, SECONDARY = absolus FAIL accepté D-P3-19). DoD verify exit 0 "Plan 3 V2 DoD : SATISFIED". JALON #3 peer review (skill `superpowers:requesting-code-review`) verdict "Ready to merge WITH FIXES" — 2 Important + 2 Minor traçabilité ISO appliqués. 0 finding critique. Merge fast-forward master OK (commit `652fac3`, 90 commits ahead origin).

---

## P3G Quality Gates mapping per-task

| Gate | ISO | Check | Task produit gate |
|------|-----|-------|-------------------|
| P3G01 | 5055 | Files ≤ 300 lignes | T1-T24 (tous) |
| P3G02 | 5055 | xenon ≤ B | idem |
| P3G03 | 5055 | mypy --strict | idem |
| P3G04 | 5055 | ruff clean | idem |
| P3G05 | 27001/27034 | gitleaks + Pandera | T2b |
| P3G06 | 29119 | coverage ≥80% per new module | T1-T24 |
| P3G07 | 25059 (T13) | Top-K recall ≥0.90 | T3 + T11 |
| P3G08 | 25059 (T14) | Jaccard max ≥0.75 | T4 + T11 |
| P3G09 | 25059 (T15) | Brier ≤0.20 + Brier skill ≥0.05 | T5, T6 + T11 |
| P3G10 | 25059 (T16) | ECE ≤0.05 + reliability diagram | T7, T7b + T11 |
| P3G11 | 25059 (T17+T22) | MAE ≤1.0 + McNemar p<0.05 | T9, T12 + T11 |
| P3G12 | 24027 | Smoke fairness breakdown | T13, T15 |
| P3G13 | 24029 | Smoke robustness perturb | T14, T16 |
| P3G14 | 42001 | ALI_MODEL_CARD.md incl. SOTA audit | T20 |
| P3G15 | 23894 | AI_RISK_REGISTER R-ALI-01..05 | T21 |
| P3G16 | 5259/42010 | Lineage + ADR Plan 3 | T24 + T22 |

---

## Definition of Done Plan 3 (global + per-task)

### Global
- [ ] 16 P3G gates verts via `verify_plan3_dod.sh`
- [ ] 3 peer reviews intermédiaires (après T7b, T11, T22) + 1 final (T25)
- [ ] Backtest hold-out 2024 ≥100 matches
- [ ] T22 ≥3 metrics améliorées baseline (McNemar p<0.05)
- [ ] Dettes D5, D-P2-03, D-P2-04, D-P3-10 résorbées

### Per-task (template appliqué T1-T25)
- [ ] Tests TDD 80%+ coverage
- [ ] Ruff + mypy strict + xenon B
- [ ] DoD acceptance criteria validés
- [ ] Commit conventionnel ISO 15289
- [ ] Documentation inline (docstring structuré)

---

## Références

- feedback_plan_8_dimensions_audit.md (mandat audit 8 dimensions)
- spec : docs/superpowers/specs/2026-04-19-phase3-ali-monte-carlo-design.md
- Bergmeir & Benítez 2012 (walk-forward)
- Mitchell 2019 FAccT (Model Card)
- Pappalardo 2019 (Brier skill score sports)
- ISO 25059 (quality model), ISO 24027/24029 (fairness/robustness), ISO 23894 (risk), ISO 42001/42005 (AI management/impact)
