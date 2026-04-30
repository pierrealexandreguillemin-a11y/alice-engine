# D8 — ALI Fairness/Robustness STRICT — Design Spec

**Document ID** : ALICE-SPEC-D8-FAIRNESS-ROBUSTNESS
**Version** : 1.0.0
**Status** : APPROVED (brainstorming validé Q1/Q2/Q3/Q4 + Architecture)
**Author** : Pierre Alexandre Guillemin + Claude Opus 4.7
**Date** : 2026-04-30
**Phase** : 3.5 STRICT (entry gate Phase 4a / D-P3-19 / R-ALI-06)
**Standards** : ISO 24027 (fairness), ISO 24029 (robustness), ISO 5259 (lineage),
ISO 29119 (testing), ISO 42001 (AI risk), ISO 25059 (quality model)

---

## 1. Contexte

### 1.1 Origine

Phase 3 (Plan 3 V2 COMPLETE 2026-04-29) a livré T13/T14 smoke fairness +
robustness à N=70 matches saison 2024 (`docs/iso/ALI_QUALITY_GATES_REPORT.md`
§4 §5). Findings T22 :

- `by_ronde max_gap = 0.152` ❌ FAIL gate P3G12 (≤0.15)
- `by_pool_size max_gap = 0.284` ❌ FAIL (small recall 0.74 vs xlarge 0.46)
- `recall_per_group < 0.85` toutes dims ❌ FAIL P3G12

**Conclusion T22** : sample size 70 limite la précision des breakdowns.
Phase 3.5 STRICT (D8) requiert N≥200 matches multi-saisons pour valider
rigoureusement. Bloquant Phase 4a (D-P3-19 / R-ALI-06 ALI multi-équipes
joint conditionné CE-adverse miroir).

### 1.2 Risques mitigés

- **R-ALI-01** PRIVATE rules unverifiable (4/14 A02 articles) — quantifié
  via breakdown `niveau` + `elo_strata`.
- **R-ALI-02** Pool too small réduit espace ALI vs observed — quantifié
  via breakdown `pool_size` (déjà T22, étendu N=280).
- **R-ALI-04** Drift FFE rules + roster turnover — quantifié via stress
  `roster_turnover` + DRO Wasserstein.

### 1.3 Cible

Audit complet ALI champion (MLP(32,16) 18f + temp scaling, ECE_draw 0.0016)
sur N≥200 matches multi-saisons (2021-2024) avec breakdown 7 dimensions +
robustesse multi-noise + conformal coverage + DRO Wasserstein + 19 gates
G-A SOTA strict. Output : Phase 4a entry gate go/no-go.

---

## 2. Décisions clés (brainstorming)

| Q | Décision | Justification |
|---|----------|---------------|
| Q1 saisons | **A** : 2021 + 2022 + 2023 + 2024 (4 saisons) | Multi-saisons SOTA distribution shift Tran 2022. N=280 expected. Drift FFE rules captable via dim `saison`. |
| Q2 dimensions breakdown | **B+** : 7 dims 1D | gender (protected ISO 24027 §6.1) + pool_size + ronde + saison + niveau + elo_strata_team + categorie_age 5 buckets. Préparation Phase 4+ J02/Coupes. Holstein 2024 : breadth > depth at low N. |
| Q3 robustness | **R-B+** : multi-noise + roster + conformal + DRO | ISO 24029 §6.5 + Tran 2022 + Vovk 2024 + Sinha 2018. PGD adversarial inapplicable (ALI MC sampler non-différentiable). |
| Q4 gates | **G-A** : 19 gates SOTA strict + case-by-case analysis on FAIL | Mehrabi 2021 + Hardt 2016 + Pleiss 2017 + Vovk 2024 + Sinha 2018 + EEOC 4/5 rule. Pas de relâchement aveugle. |
| Architecture | **Arch-2** : 4 kernels Kaggle parallèle + 1 aggregator | CPU illimité Kaggle, wallclock ~75 min. Resilience (1 fail = 1 saison à rejouer). |
| Autonomie | Claude exécute, monitor, debug Kaggle errors via /loop. STOP sur gate FAIL → user décide case-by-case. | User instruction explicite 2026-04-30. |

---

## 3. Stratification — 7 dimensions (B+)

### 3.1 Gender (ISO 24027 §6.1, protected RGPD/EEOC)

Source : `joueurs.parquet` colonne `genre` (M/F).
Buckets : 2.
Distribution attendue : ~75% M / ~25% F (FFE Interclubs 2024).
Niveau N stratum : F ~70 / M ~210 (déséquilibré mais ≥30 cf Pleiss 2017).

### 3.2 Pool size opponent (service-level)

Source : `len(echiquiers_by_player)` du club adversaire à la date du match.
Buckets quartiles : small (Q1) / medium (Q2) / large (Q3) / xlarge (Q4).
Niveau N stratum : ~70 / quartile.

### 3.3 Ronde (temporal phase)

Source : `MatchStats.ronde`.
Buckets : 1, 3, 5, 7, 9 (rondes paires non-incluses si pas de match dans dataset).
Niveau N stratum : ~56 / ronde.

### 3.4 Saison (cross-year shift, ISO 24029 §6.6)

Source : `MatchStats.saison`.
Buckets : 2021, 2022, 2023, 2024.
Niveau N stratum : ~70 / saison.

### 3.5 Niveau competition (R-ALI-01 capture)

Source : `echiquiers.parquet` colonne `niveau` (N1/N2/N3/N4/Régional).
Buckets : 5.
Distribution attendue : variable selon saison, mais ≥30/group attendu sur 4 saisons.
**Critical** : capture l'effet PRIVATE rules (N1 stricter).

### 3.6 Elo strata team (R-ALI-02 capture, ability bias)

Source : moyenne Elo des 8 joueurs du `equipe_dom` (user-side).
Buckets quartiles team mean Elo :
- Q1 : <1500
- Q2 : 1500–1700
- Q3 : 1700–1900
- Q4 : ≥1900
Niveau N stratum : ~70 / quartile.

### 3.7 Categorie age (préparation Phase 4+)

Source : `joueurs.parquet` colonne `categorie` (12 valeurs FFE) mappée
en 5 buckets via `scripts/parse_dataset/constants.py::CATEGORIES_AGE` :

| Bucket | Catégories FFE |
|--------|----------------|
| **U12** | PpoM, PpoF, PouM, PouF, PupM, PupF |
| **U18** | BenM, BenF, MinM, MinF, CadM, CadF |
| **U20** | JunM, JunF |
| **Sen** | SenM, SenF |
| **S50+** | SepM, SepF, VetM, VetF |

Note distribution Phase 3.5 : ~92% Sen sur Interclubs Open. Stratification
prépare Phase 4+ extension D3 (J02 Jeunes) + D4 (Coupes Loubatière S65).

---

## 4. Robustness perturbations (R-B+)

### 4.1 Multi-noise Elo (ISO 24029 §6.5)

```python
# scripts/d8/stress_elo.py
NOISE_LEVELS = [0.01, 0.03, 0.05, 0.07, 0.10]
for noise_pct in NOISE_LEVELS:
    perturbed_elos = perturb_elos(opp_pool_elos, noise_pct, seed=42)
    perturbed_recall = backtest_run(match, opp_elos=perturbed_elos)
    recall_drop = max(0.0, baseline_recall - perturbed_recall)
```

Source : Goodfellow 2015 ε-bounded, Madry 2018 PGD bounded.

### 4.2 Roster turnover (R-ALI-04)

```python
# scripts/d8/stress_roster.py
TURNOVER_RATES = [0.05, 0.10, 0.20]
for turnover_pct in TURNOVER_RATES:
    n_drop = int(turnover_pct * len(opp_pool))
    rng = np.random.default_rng(42)
    drop_idx = rng.choice(len(opp_pool), size=n_drop, replace=False)
    perturbed_pool = [p for i, p in enumerate(opp_pool) if i not in drop_idx]
    if len(perturbed_pool) < team_size:
        result.skipped = True; continue
    perturbed_recall = backtest_run(match, opp_pool=perturbed_pool)
```

Source : Tran 2022 §3.4 distribution shift.

### 4.3 Conformal prediction coverage (Vovk 2024)

```python
# scripts/d8/conformal.py
# Split conformal : 30 matches calibration, 250 test
calib_set, test_set = split(matches, calib_n=30, seed=42)
nonconf_scores = [|y_obs - y_pred| for match in calib_set]
quantile_90 = np.quantile(nonconf_scores, 0.90)

# Coverage measurement on test_set
for match in test_set:
    pred_set = {y : |y - y_pred| <= quantile_90}
    covered = (y_obs in pred_set)

coverage_rate = mean(covered)  # global
coverage_per_group = breakdown(covered_flags, dim_keys)
```

Source : Vovk 2024 §2.3 marginal coverage. Angelopoulos & Bates 2023 §4.

### 4.4 DRO Wasserstein worst-case (Sinha 2018)

```python
# scripts/d8/dro.py
EPSILONS = [0.05, 0.10]
N_PERTURBATIONS = 50
for eps in EPSILONS:
    worst_recall = +inf
    for perturb_id in range(N_PERTURBATIONS):
        # Perturbation = sample shift + scale within Wasserstein-2 ball
        rng = np.random.default_rng(seed=42 + perturb_id)
        shift = rng.uniform(-eps * mean_elo, eps * mean_elo)
        scale = rng.uniform(1 - eps, 1 + eps)
        perturbed_elos = ((elos - mean_elo) * scale + mean_elo + shift).clip(800, 2900)
        perturbed_recall = backtest_run(match, opp_elos=perturbed_elos)
        worst_recall = min(worst_recall, perturbed_recall)
    finding_pertinent = "shift={shift_max}+scale={scale_max} {context}"
```

Source : Sinha 2018 §4. Duchi & Namkoong 2021 §6.

---

## 5. 19 Gates G-A SOTA strict + case-by-case

### 5.1 Fairness gates (ISO 24027)

| Gate ID | Threshold | Métrique | Source |
|---------|-----------|----------|--------|
| G_FAIR_01 | max_gap_recall ≤ **0.10** toute dim | max-min recall | Mehrabi 2021 §4.1 |
| G_FAIR_02 | recall_per_group ≥ **0.85** toute dim | min recall | P3G07 - 5pts |
| G_FAIR_03 | demographic_parity_diff ≤ **0.10** (gender) | |P(positive\|F) - P(positive\|M)| | Hardt 2016 |
| G_FAIR_04 | equalized_odds_diff ≤ **0.10** (gender × outcome) | TPR + FPR diff | Hardt 2016 §3.2 |
| G_FAIR_05 | calibration_ECE ≤ **0.05** par groupe toute dim | per-group ECE | Pleiss 2017 §4 |
| G_FAIR_06 | multicalibration α ≤ **0.05** | Hébert-Johnson α | Hébert-Johnson 2018 |
| G_FAIR_07 | TPR_min/TPR_max ≥ **0.80** (4/5 rule) | EEOC ratio | EEOC §1607.4D + Feldman 2015 |
| G_FAIR_08 | brier_score_per_group ≤ **0.30** toute dim | per-group Brier | Brier 1950 + Pappalardo 2019 §3 |
| G_FAIR_09 | BSS_per_group ≥ **0.30** toute dim | per-group BSS | Pappalardo 2019 §3.4 |
| G_FAIR_10 | PSI ≤ **0.20** par dim cross-saisons | distribution shift index | Yurdakul 2020 |

### 5.2 Robustness gates (ISO 24029)

| Gate ID | Threshold | Métrique | Source |
|---------|-----------|----------|--------|
| G_ROB_01 | recall_drop ≤ **0.02** @ 1% Elo noise | absolute drop | Goodfellow 2015 ε=0.01 |
| G_ROB_02 | recall_drop ≤ **0.05** @ 5% Elo noise | absolute drop | Madry 2018 |
| G_ROB_03 | recall_drop ≤ **0.10** @ 10% Elo noise | absolute drop | Madry 2018 strict |
| G_ROB_04 | recall_drop ≤ **0.05** @ 5% roster turnover | absolute drop | Tran 2022 §3.4 |
| G_ROB_05 | recall_drop ≤ **0.15** @ 20% roster turnover | absolute drop | Recht 2019 §5 |
| G_ROB_06 | coverage_conformal_90 ≥ **0.90** strict | marginal coverage | Vovk 2024 §2.3 |
| G_ROB_07 | conformal_set_size ≤ **3.0** mean | efficiency | Angelopoulos 2023 §4.2 |
| G_ROB_08 | recall_worst_case_DRO ≥ **0.70** @ ε=0.05 | Wasserstein worst | Sinha 2018 §4 |
| G_ROB_09 | recall_worst_case_DRO ≥ **0.55** @ ε=0.10 | Wasserstein worst | Duchi 2021 §6 |

### 5.3 Case-by-case analysis policy on FAIL

Pour chaque gate FAIL, le rapport `D8_FAILURE_ANALYSIS_LOG.md` contient :

```markdown
## Gate G_FAIR_XX FAIL — analysis 2026-04-30

**Mesured** : <value>
**Threshold** : <threshold>
**Δ from threshold** : <delta>

### Gate validity
Le seuil <threshold> est-il approprié pour ALICE Engine
(domaine échecs + Interclubs FFE) ? Litterature source : <ref>.
- Argument validité : ...
- Argument inapplicabilité : ...

### Utilité métier
La gate mesure-t-elle un risque concret pour les utilisateurs ALICE ?
- Impact si métrique reste à <value> : ...
- Mitigation produit possible : ...

### Seuil recalibré (proposé)
- Si gate validity ✓ : threshold reste <threshold>, fix code
- Si validité ✗ : threshold proposé <new_threshold> avec justification

### Mitigations options (3 max)
1. <option>
2. <option>
3. <option>

### Décision user (à remplir)
[ ] Accepter mitigation N°1
[ ] Recalibrer seuil à <new_threshold>
[ ] Bloquer Phase 4a entry gate jusqu'à fix
```

**Pas de relâchement aveugle**. Décision user explicite par gate.

---

## 6. Architecture (Arch-2)

### 6.1 Vue d'ensemble

```
Dataset Kaggle alice-d8-input
  ├ data/joueurs.parquet
  ├ data/echiquiers.parquet (saisons 2021-2024)
  ├ artefacts/{mlp_meta_learner,temp_scaler,18f}.joblib
  ├ config/ffe_rules/a02.json
  └ alice-d8-code/scripts/d8/* + services/ali/*

  ↓ kaggle datasets create + alice-d8-code

5 kernels :
  d8-saison-2021 ┐
  d8-saison-2022 ├─ ALICE_SAISON env var, ~65 min CPU each, parallèle
  d8-saison-2023 │
  d8-saison-2024 ┘
       ↓ outputs → datasets {pierrax/d8-saison-{2021..24}}
  d8-aggregator (~10 min, depends on 4 saisons datasets)
       ↓ outputs → reports/d8/* via download_outputs.py

  ↓ DVC stage d8_audit (étend dvc.yaml T24)
```

### 6.2 Modules nouveaux

| Module | Lignes | Responsabilité |
|--------|--------|---------------|
| `scripts/d8/loader.py` | ~120 | Filter parquets saison + match-eligibility (ronde≥1, équipes valides, observed non-vide) |
| `scripts/d8/breakdowns.py` | ~220 | 7 fonctions `breakdown_by_*` + buckets configurés |
| `scripts/d8/calibration.py` | ~90 | Per-group ECE + multicalibration α (Hébert-Johnson 2018) |
| `scripts/d8/stress_elo.py` | ~110 | Multi-noise Elo [1/3/5/7/10]% |
| `scripts/d8/stress_roster.py` | ~90 | Roster drop [5/10/20]% |
| `scripts/d8/conformal.py` | ~180 | Split conformal CI 90% (Vovk 2024) + coverage breakdown |
| `scripts/d8/dro.py` | ~220 | Wasserstein-2 ε-ball worst-case grid 50 perturb |
| `scripts/d8/gates.py` | ~250 | 19 gates G-A + case-by-case logger |
| `scripts/d8/run.py` | ~150 | Orchestrate per-saison kernel |
| `scripts/d8/aggregate.py` | ~280 | Fuse 4 saisons + global gates + render markdown |
| `scripts/d8/upload_d8_dataset.py` | ~80 | Upload alice-d8-input dataset Kaggle |
| `scripts/d8/download_outputs.py` | ~50 | Download d8-aggregator output local |
| `scripts/d8/kernel-metadata-saison-{2021..2024}.json` | — | 4 configs |
| `scripts/d8/kernel-metadata-aggregator.json` | — | 1 config |

**Total** : ~1840 lignes Python + 5 JSON kernel configs.

### 6.3 Modules réutilisés

- `services/ali/*` — champion MLP stacking pipeline (production)
- `scripts/backtest/runner.py` — backtest harness existant
- `scripts/backtest/fairness.py` — `breakdown_by_key` + `max_gap` génériques
- `scripts/backtest/robustness.py` — `perturb_elos` + `compute_recall_drop`
- `scripts/parse_dataset/constants.py` — `CATEGORIES_AGE` mapping

---

## 7. Data flow

### 7.1 Input lineage

```
SHA-256 calculé au démarrage de chaque kernel saison :
  joueurs_sha256, echiquiers_sha256, mlp_artefact_sha256,
  temp_scaler_sha256, code_sha256 (git rev-parse HEAD à upload)
+ ali_seed=42 (default), ali_n_topk=10, ali_n_mc_pairs=5, ali_decay_lambda=0.9
+ kernel_id, kernel_version_kaggle, run_at_utc
```

Aggregator vérifie cohérence : tous SHA artefacts identiques cross-saisons
(ALI champion identique). Si mismatch → fail aggregator.

### 7.2 Pipeline per-saison

```
1. loader.load_match_eligible(saison) → list[MatchSpec]
2. for each match :
   a. ali_predict(match) → dict[player, P(presence)]
   b. ground_truth.extract_observed_lineup(match)
   c. metrics.compute_match_stats(predict, observed) → MatchStats
3. breakdowns.compute_all_7(match_stats_list) → dict[dim, dict[group, GroupStats]]
4. calibration.multicalibration_per_group()
5. stress_elo.run([0.01, 0.03, 0.05, 0.07, 0.10])
6. stress_roster.run([0.05, 0.10, 0.20])
7. conformal.split_calibrate(calib_n=30) + coverage_per_group()
8. dro.wasserstein_worst_case(epsilons=[0.05, 0.10], n_perturbations=50)
9. write d8_saison_{saison}.json
```

### 7.3 Output JSON schema (per saison)

```json
{
  "schema_version": "d8.v1",
  "saison": 2021,
  "n_matches": 70,
  "lineage": { ... },
  "per_match": [ {match_stats × 70} ],
  "breakdowns": {
    "by_gender": {…}, "by_pool_size": {…}, "by_ronde": {…},
    "by_saison": {…}, "by_niveau": {…}, "by_elo_strata": {…},
    "by_categorie_age": {…}
  },
  "multicalibration": { dim → {group → alpha_HJ_2018} },
  "stress_elo": { noise_pct → recall_drop },
  "stress_roster": { turnover_pct → recall_drop },
  "conformal": {
    "coverage_global": 0.91,
    "set_size_mean": 2.4,
    "coverage_per_group": { dim → {group → coverage} }
  },
  "dro_wasserstein": {
    "eps_0.05": { "recall_worst_case": 0.71, "perturbation_finding_pertinent": "..." },
    "eps_0.10": { "recall_worst_case": 0.56, "perturbation_finding_pertinent": "..." }
  }
}
```

### 7.4 Aggregator pipeline + outputs

```
inputs : 4 datasets pierrax/d8-saison-{2021..2024}
process :
  1. concat per_match → 280 matches
  2. recompute breakdowns + multicalibration + conformal sur N=280 global
  3. union stress_elo + stress_roster + DRO sur 4 saisons
  4. gates.evaluate_19_GA(report_280) → dict[gate_id, GateStatus]
  5. case-by-case : pour chaque FAIL → entry D8_FAILURE_ANALYSIS_LOG.md
  6. render markdown report

outputs :
  d8_full_report.json (schema d8-aggregator.v1)
  D8_FINDINGS.md (8-12 pages humain)
  D8_FAILURE_ANALYSIS_LOG.md (entries case-by-case)
  gates_19_status.json
```

### 7.5 Local download + DVC

```bash
python -m scripts.d8.download_outputs  # → reports/d8/*.json + *.md
dvc add reports/d8/d8_full_report.json
dvc commit -m "d8: phase 3.5 strict audit complete"
```

`dvc.yaml` étendu (stage T24) :

```yaml
stages:
  d8_audit:
    deps:
      - data/joueurs.parquet
      - data/echiquiers.parquet
      - artefacts/mlp_meta_learner.joblib
    outs:
      - reports/d8/d8_full_report.json
      - reports/d8/D8_FINDINGS.md
      - reports/d8/gates_19_status.json
    cmd: python -m scripts.d8.download_outputs
```

---

## 8. Error handling

| Cas | Comportement | Niveau fail |
|-----|--------------|-------------|
| Match invalide (ronde absente, équipe inconnue) | log warning, skip match. Fail saison si >5% skip. | local |
| ALI predict crash 1 match | log + skip, continue. Fail saison si >5% skip. | local |
| Lineage mismatch artefact SHA | RuntimeError fail-fast (ISO 24029, D-P3-12 leçon) | saison |
| Kernel timeout 12h | Status timeout, re-run (Phase 5 = vrai checkpoint resume) | kernel |
| OOM Kaggle 13GB | Stream chunks 20 matches | kernel |
| Aggregator missing 1/4 saison | Fail-fast, pas de rapport partiel (R-PRE-PUSH-01 leçon) | aggregator |
| Conformal calib_n<30 | Fail-fast, message clair | saison |
| DRO non-convergence | Log + flag `INCONCLUSIVE` (≠ FAIL) gate G_ROB_08/09 | local skip |
| Gate G-A FAIL | D8_FAILURE_ANALYSIS_LOG entry, **continue** 18 autres gates | global |

---

## 9. Testing strategy (ISO 29119)

```
tests/d8/
├── test_loader.py            (15 tests)
├── test_breakdowns.py        (40 tests : 7 fonctions × 5-6)
├── test_calibration.py       (20 tests)
├── test_stress_elo.py        (15 tests)
├── test_stress_roster.py     (12 tests)
├── test_conformal.py         (25 tests)
├── test_dro.py               (20 tests)
├── test_gates.py             (38 tests : 19 gates × 2 + case-by-case logger)
├── test_aggregate.py         (15 tests)
└── test_run_e2e_smoke.py     (5 tests : 1 saison subset N=5 matches en <30s)
```

**~205 tests**. Tous markés `slow` si chargent vrais parquets (R-PRE-PUSH-01 leçons).
Smoke E2E sur subset N=5 dummy → pre-push fast.

---

## 10. Lineage SHA-256 (ISO 5259)

Voir §7.1. Aggregator vérifie cohérence cross-saisons (artefacts ALI
identiques). Mismatch → fail.

---

## 11. Compute budget

| Étape | CPU/saison | Wallclock parallèle |
|-------|-----------|---------------------|
| Loader + eligibility | 2 min | 2 min |
| ALI predict 70 matches | 25 min | 25 min |
| Breakdowns 7 dims | 1 min | 1 min |
| Multicalibration HJ-2018 | 3 min | 3 min |
| Stress Elo × 5 noise | 12 min | 12 min |
| Stress roster × 3 turnover | 8 min | 8 min |
| Conformal split + coverage | 4 min | 4 min |
| DRO Wasserstein | 10 min | 10 min |
| **Total per saison** | **~65 min** | **~65 min** |
| Aggregator | — | ~10 min |
| **Total wallclock** | — | **~75 min** |

Budget 4h annoncé : marge sécurité large pour debug + reruns.

---

## 12. Autonomous execution (user instruction 2026-04-30)

Claude exécute en autonomie :
- Spec + plan + implementation
- Upload Kaggle datasets + push 5 kernels
- Monitor via /loop poll `kaggle kernels status`
- Diagnostic + fix sur fail Kaggle (ModuleNotFoundError, sys.path, env var,
  resolution-too-deep, OOM, dataset missing)
- Re-upload + re-run
- Aggregate outputs + render reports

Claude STOP autonomie sur :
- Gate G-A FAIL → ping user pour décision case-by-case
- Pattern d'erreur Kaggle inconnu (>3 itérations sans progrès)
- Push origin (jamais sans validation user)

User supervise via `/loop 5m d8-monitor` ou cadence dynamique.
Kill switch : `/pause` ou Ctrl-C.

---

## 13. Phase 4a entry gate

D8 **doit produire au moins** :
- `d8_full_report.json` schema d8-aggregator.v1 valide
- `gates_19_status.json` avec décisions PASS / FAIL / INCONCLUSIVE
- Si FAIL : `D8_FAILURE_ANALYSIS_LOG.md` entry par gate FAIL

ADR-016 (Proposed → Accepted) attendra :
- Soit toutes gates G-A PASS
- Soit user a décidé case-by-case sur tous FAIL (mitigation acceptée
  ou seuil recalibré documenté)

---

## 14. Cross-references

- `docs/architecture/adr/ADR-016-ali-conditioned-multi-team-adverse-ce-mirror.md`
- `docs/iso/AI_RISK_REGISTER.md` §R-ALI-01 / R-ALI-02 / R-ALI-04 / R-ALI-06
- `docs/iso/ALI_QUALITY_GATES_REPORT.md` §4 fairness §5 robustness
- `docs/iso/ALI_MODEL_CARD.md` §7 §8 (deferred D8)
- `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md` §Phase 3.5
- `memory/project_debt_current.md` §D8
- `docs/project/DEBT_LEDGER.md` §D8
- `dvc.yaml` (extend stage `d8_audit`)
- `scripts/backtest/{fairness,robustness}.py` (extend)
- `scripts/parse_dataset/constants.py::CATEGORIES_AGE` (5-bucket mapping)

---

## 15. Sources SOTA citées

| Source | Référence | Usage D8 |
|--------|-----------|----------|
| ISO/IEC TR 24027:2021 | Bias Detection in AI | §5.1 fairness gates |
| ISO/IEC TR 24029:2021 | Robustness of neural networks | §5.2 robustness gates |
| ISO/IEC TR 5259:2024 | Data quality + lineage | §7.1 SHA-256 |
| ISO/IEC 29119:2022 | Software testing | §9 testing strategy |
| Mehrabi et al. 2021 | "Survey on Bias and Fairness in ML" ACM CSUR | G_FAIR_01 max_gap≤0.10 |
| Hardt, Price, Srebro 2016 | "Equality of Opportunity" NeurIPS | G_FAIR_03/04 demographic parity + equalized odds |
| Pleiss et al. 2017 | "On Fairness and Calibration" NeurIPS | G_FAIR_05 ECE |
| Hébert-Johnson et al. 2018 | "Multicalibration" ICML | G_FAIR_06 multicalibration α |
| Feldman et al. 2015 | "Certifying and Removing Disparate Impact" KDD | G_FAIR_07 EEOC 4/5 rule |
| Brier 1950 | Verification of forecasts | G_FAIR_08 Brier |
| Pappalardo et al. 2019 | Sport prediction baseline | G_FAIR_08/09 BSS |
| Yurdakul & Naranjo 2020 | "Stability Index for ML" | G_FAIR_10 PSI |
| Goodfellow et al. 2015 | "Explaining Adversarial Examples" | G_ROB_01 ε=0.01 |
| Madry et al. 2018 | "PGD Resistant Models" ICLR | G_ROB_02/03 ε-bounded |
| Tran et al. 2022 | "Distribution Shifts in ML" NeurIPS | G_ROB_04 roster turnover |
| Recht et al. 2019 | "Do ImageNet Classifiers Generalize?" | G_ROB_05 strict shift |
| Vovk 2024 | "Conformal Prediction" book + AOS reviews | G_ROB_06/07 conformal |
| Angelopoulos & Bates 2023 | "Gentle Intro CP" FnT-ML | G_ROB_07 efficiency |
| Sinha et al. 2018 | "Distributional Robustness" ICLR | G_ROB_08 DRO ε=0.05 |
| Duchi & Namkoong 2021 | DRO theory JMLR | G_ROB_09 DRO ε=0.10 |
| Holstein, Wallach, Daumé 2024 | "Industry fairness assessments" FAccT | §3 breadth>depth N<500 |
| Buolamwini & Gebru 2018 | "Gender Shades" FAccT | Reporté Phase 5+ |
| Efron 1993 | Bootstrap | §3 statistical power |
| DiCiccio & Efron 1996 | Bootstrap CI | §3 statistical power |
| EEOC §1607.4D | 4/5 disparate impact rule | G_FAIR_07 |
| Naeini 2015 | "Calibrated Probabilities" AAAI | G_FAIR_05 ECE method |

---

**END OF SPEC**
