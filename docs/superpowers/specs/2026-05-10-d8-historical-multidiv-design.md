# D8 Historical-State + Multi-Divisions — Design Spec

**Document ID** : ALICE-SPEC-D8-HISTORICAL-MULTIDIV
**Version** : 0.1.0 (DRAFT, pre-implementation review)
**Status** : PROPOSED — awaiting user sign-off after cascade inventory review
**Author** : Pierre Alexandre Guillemin + Claude Opus 4.7
**Date** : 2026-05-10
**Phase** : 3.5 STRICT — entry gate Phase 4a

---

## 1. Contexte

### 1.1 Origine

D8 Phase 3.5 STRICT spec §1.3 exige N≥200 matches multi-saisons cross-year pour
audit ML champion. Tentative initiale (saisons 2021-2024 × division "Nationale 3"
seule) a échoué à 0 matches sur saisons 2021/2022/2023 — root cause empiriquement
diagnostiquée :

1. **Naming change FFE 2022** : `"Nationale III"` (chiffres romains, ≤2021) vs
   `"Nationale 3"` (chiffre arabe, ≥2022). Résolu via SAISON_DIVISION_FILTER mapping.
2. **CRITIQUE — état historique non reconstruit** : `ALIDataCache.team_to_club` et
   `joueurs_by_club` sont chargés depuis `joueurs.parquet` qui contient UNIQUEMENT
   l'état CURRENT (snapshot scrape FFE le plus récent). Pour saison antérieure,
   les équipes/clubs/joueurs ont muté : départs club, dissolutions, changement
   de roster. enumerate_candidates filtre les matches dont `team_to_club.get
   (equipe_dom)` est None → 0 matches éligibles sur saisons antérieures.

### 1.2 Demande user (2026-05-10)

> "j'exige un produit livrable SOTA ML pour commercialisation"

Refus du trade-off "multi-divisions saison 2024 seule" (= retreat sur scope
multi-saison). Demande la résolution propre du blocker historique.

### 1.3 Cible

D8 audit produit :
- N ≥ 200 matches éligibles cross-saison + cross-division
- 19 gates G-A SOTA strict évalués (zéro INCONCLUSIVE par défaut)
- by_niveau breakdown ISO 24027 §6 sur 5 divisions adultes (Top 16 / N1 / N2 / N3 / N4)
- by_saison breakdown sur saisons disponibles (2024 + saisons antérieures
  reconstructibles)
- R-ALI-01 (PRIVATE rules unverifiable cross-niveau) quantifié empiriquement
- R-ALI-04 (drift FFE rules + roster turnover cross-year) quantifié empiriquement

---

## 2. Solution proposée

### 2.1 Reconstruction état historique depuis echiquiers.parquet

`echiquiers.parquet` contient pour CHAQUE match toutes les infos nécessaires pour
reconstruire l'état d'un club/équipe à un moment donné (saison) :

| Colonne | Source état historique |
|---------|------------------------|
| `saison` | filtre temporel |
| `equipe_dom`, `equipe_ext` | nom équipe à cette saison |
| `nr_blanc`, `nr_noir` | nr_ffe joueur (clé primaire FFE, immutable) |
| `nom_blanc`, `nom_noir`, `prenom_blanc`, `prenom_noir` | identité joueur |
| `elo_blanc`, `elo_noir` | **Elo au moment du match** (vs Elo CURRENT dans joueurs.parquet) |
| `equipe` | club_id explicite |
| `genre`, `categorie` (peut être absent / nullable) | demographics |

**Méthode** : pour saison cible S, regrouper echiquiers[saison=S] par
(equipe, nr_ffe) pour obtenir le roster effectif de chaque équipe à cette saison.
Le `team_to_club` historique se déduit via la colonne `equipe` (club_id).

### 2.2 Architecture pivot — multi-saisons × multi-divisions

D8 audit en 2 phases :

**Phase A — Saison 2024 multi-divisions (5 kernels)** :
- Top 16 : 88 candidates
- Nationale 1 : 180 candidates
- Nationale 2 : 360 candidates
- Nationale 3 : 720 candidates
- Nationale 4 : 1351 candidates
- Total saison 2024 ≈ 2700 candidates → atteinte N≥200 garantie sur saison 2024 seule

**Phase B — Saisons antérieures (4 kernels supplémentaires si historique reconstruit)** :
- 2021/2022/2023 multi-divisions sur reconstructed state
- Permet by_saison breakdown empirique = R-ALI-04 drift validation

Total : **5–9 kernels** Kaggle (Phase A obligatoire, Phase B optionnel selon
faisabilité reconstruction historique).

### 2.3 Limites reconnues de la reconstruction historique

| Champ | Reconstructible depuis echiquiers ? | Workaround |
|-------|-------------------------------------|------------|
| Elo joueur à la saison | ✅ via `elo_blanc/noir` per match | take median Elo across rondes |
| nr_ffe / nom / prénom | ✅ stable | direct |
| Club d'équipe (via nom équipe) | ✅ via colonne `equipe` | direct |
| Catégorie (Sen, Vet, etc.) | ⚠️ nullable | fallback "Sen" si absent |
| Genre | ⚠️ nullable | fallback "M" si absent |
| `mute` status | ❌ non disponible historique | défaut False |
| `licence_active` | ❌ non disponible historique | défaut True (assume actif puisque a joué) |
| age_min / age_max | ❌ non disponible historique | None |

**Implications fonctionnelles** :
- `RuleEngine.A02 §3.7.f noyau` (mutes) ne peut pas être validé empiriquement sur
  données historiques. **Documenté comme dette : R-ALI-05 NEW** "noyau historique
  unverifiable".
- `J02 jeunes` (D3) déjà out-of-scope D8 (Phase 4+).

---

## 3. Cascade inventory — what changes / what breaks

### 3.1 Files to modify

| File | Type modif | Risque casse |
|------|-----------|---------------|
| `services/ali/cache.py` | ADD classmethod `from_parquets_at_saison(saison)` factory | LOW (additive, no signature change) |
| `services/ali/cache.py` | NO change to `from_parquets()` (current API preserved) | NONE |
| `scripts/backtest/harness.py` | ADD optional `historical_saison: int \| None` to `setup()` | LOW (default None = current behavior) |
| `scripts/d8/run.py` | ADD `ALICE_DIVISION` env var; ADD historical_saison passthrough; default = current | MEDIUM (current 4 wrappers `run_2021..2024.py` need update OR pivot to division-keyed wrappers) |
| `scripts/d8/aggregate.py` | extend to fuse N reports keyed by `(saison, division)` | MEDIUM (output schema change) |
| `scripts/d8/upload_d8_dataset.py` | update wrapper file list + dataset metadata template | LOW |
| `scripts/d8/types.py` | extend `D8Lineage` with `division` field | LOW (additive) |
| `scripts/d8/breakdowns.py` | NO change (already handles by_niveau via `niveau_fn`) | NONE |
| `scripts/d8/perturb_runner.py` | verify perturb works on historical cache (no API change expected) | LOW (test) |
| `scripts/d8/kernel-metadata-*.json` | rename / regenerate per (saison, division) — 5 to 9 kernels | MEDIUM (config) |
| `scripts/d8/run_*.py` wrappers | rename / regenerate per (saison, division) | MEDIUM |

### 3.2 Files used at runtime — verify NO regression

| File | Usage | Risk |
|------|-------|------|
| `services/inference.py::StackingInferenceService` | uses ML champion + features, NOT cache structure | NONE |
| `services/ali/generator.py::ScenarioGenerator` | reads `cache.team_to_club` + `joueurs_by_club` via PoolLoader | MEDIUM (must work on historical cache) |
| `services/ali/pool_loader.py::PlayerPoolLoader` | reads `cache.joueurs_by_club[club_id]` → DataFrame | MEDIUM (verify schema match historical) |
| `services/feature_store.py::FeatureStore` | reads `data/feature_store/*.parquet` indep of cache | NONE |
| `scripts/backtest/runner.py::BacktestRunner.run_single` | accepts cache via harness | MEDIUM (verify) |
| `scripts/backtest/run_match.py::run_backtest_match` | uses scenario_generator + inference | LOW |
| `scripts/backtest/ground_truth.py::extract_observed_lineup` | reads echiquiers + cache | MEDIUM (verify equipe lookup historical) |
| `scripts/backtest/runner_sampling.py::enumerate_candidates` | uses `cache.team_to_club` + `joueurs_by_club` | MEDIUM (must work historical) |
| `app/main.py::lifespan` | production API boot — uses `ALIDataCache.load_from_parquets()` (current API) | NONE (preserved) |
| `app/api/routes.py::compose` | production endpoint — uses inference + scenario_generator | NONE (preserved) |
| `services/composer.py` | DELETED (D5 RESOLUE Plan 3 T23) | NONE |

### 3.3 Tests impacted

| Test file | Update required ? |
|-----------|-------------------|
| `tests/services/test_cache.py` | ADD `test_from_parquets_at_saison_*` (new tests) |
| `tests/services/test_*.py` (other services) | NO change (current API preserved) |
| `tests/backtest/test_runner_*.py` | NO change (current API preserved) |
| `tests/backtest/test_runner_sampling.py` | NO change (parametrized via cache fixture) |
| `tests/d8/test_run_e2e_smoke.py` | extend to test (saison, division) parametrization |
| `tests/d8/test_aggregate.py` | update fuse_per_match to handle (saison, division) tuples |
| `tests/d8/test_perturb_runner.py` | NO change (cache opaque) |
| Total tests existing : 296 passing | ADD ~10 new tests; UPDATE ~3 fixtures |

### 3.4 Cascade compute budget impact

| Scope | N kernels | Wallclock per kernel | Total wallclock parallel 4 simultaneous |
|-------|-----------|---------------------|------------------------------------------|
| Phase A (saison 2024 × 5 divisions) | 5 | ~9h chacun | ~12h |
| Phase B (saisons 2021-2023 × 5 divisions) | 15 supplémentaires | ~9h chacun | ~36h |
| Total Phase A + B | 20 kernels | — | ~48h |
| Aggregator | 1 kernel | ~10 min | inclus |

**Recommendation** : démarrer Phase A seule (atteinte spec §1.3 N≥200 garantie),
Phase B = post-V1 si reconstruction historique se révèle fidèle suffisamment.

### 3.5 ISO impact

| Norme | Impact | Action |
|-------|--------|--------|
| ISO 5055 | nouveaux fichiers ≤300L : `cache.py` ADD ~80L | verify post-edit |
| ISO 5259 | lineage SHA-256 enrichi avec `division` field | OK additif |
| ISO 24027 §6 | by_niveau breakdown enrichi (5 divisions) | ✅ amélioration |
| ISO 24029 §6.5 | stress/DRO sur 5 niveaux Elo distincts | ✅ amélioration |
| ISO 42001 | Model Card : ajouter section "audit cross-niveau" | UPDATE |
| ISO 23894 | R-ALI-04 (drift) testable empiriquement Phase B | UPDATE risk register |
| ISO 23894 | R-ALI-05 NEW "noyau historique unverifiable" | ADD risk register |
| ISO 25059 | quality gates F1-F12 / T1-T12 inchangés | NONE |
| ISO 27034 | input validation `historical_saison` int + range check | ADD |
| ISO 29119 | tests coverage maintained ≥70% | ✅ |

### 3.6 Architecture decisions (ADRs proposed)

| ADR | Décision |
|-----|----------|
| ADR-018 NEW | "ALIDataCache historical state reconstruction depuis echiquiers.parquet" — explique trade-offs (Elo via echiquiers vs joueurs, mute défaut False, etc.) |
| ADR-019 NEW | "D8 audit pivot multi-divisions × multi-saisons" — explique rejection trade-off "saison 2024 N3 only" |

---

## 4. Risk register updates

### R-ALI-05 NEW — Noyau historique unverifiable

**Description** : `RuleEngine A02 §3.7.f` (noyau = N joueurs ne peuvent pas
descendre / être mutes) ne peut pas être validé empiriquement sur saisons
antérieures car le statut `mute` au moment de la saison X n'est pas
reconstructible depuis echiquiers.parquet.

**Mitigation** : assume `mute=False` pour reconstruction historique. Tag les
violations potentielles dans D8_FAILURE_ANALYSIS_LOG.md sous "noyau_historical_unverifiable".

**Impact** : potentiellement ↑ false-positive scenarios générés par
ScenarioGenerator. Affecte recall_ali sur saisons antérieures (sous-estimation
ou sur-estimation à mesurer empiriquement).

### R-ALI-01 (existing) — PRIVATE rules unverifiable

Quantifié empiriquement via by_niveau breakdown D8 (5 divisions). Si gap recall
N1 vs N4 > 0.10 → indication forte que PRIVATE rules N1 stricter biaise
champion. Amélioration vs status quo (T22 saison 2024 N3 seule).

### R-ALI-04 (existing) — Drift FFE rules + roster turnover cross-year

Validable Phase B (saisons antérieures reconstruites). Si by_saison drift
> 0.20 PSI → drift réel détecté. Phase A seule ne le valide pas.

---

## 5. Acceptance criteria (avant push kernel D8 v12)

- [ ] `services/ali/cache.py::from_parquets_at_saison()` implémenté + 5 tests dédiés
- [ ] `BacktestHarness.setup(historical_saison=...)` paramètre additionnel + test
- [ ] Local smoke saison 2024 N3 (current state) inchangé : 240 candidates, 38 valid
- [ ] Local smoke saison 2024 N4 (NEW): N candidates >100, M valid >20
- [ ] Local smoke saison 2021 N3 (historical reconstruction): N candidates >50, M valid >20
- [ ] `tests/d8/` : 296 existing pass + 10-15 NEW pass
- [ ] ISO 5055 : tous nouveaux fichiers ≤300L, CC ≤B, MI A
- [ ] `aggregate.py` fuse correctement N×M reports keyed (saison, division)
- [ ] D8_FINDINGS.md template prévoit by_niveau + by_saison sections
- [ ] User sign-off explicite avant Kaggle push

---

## 6. Cross-references

- `docs/superpowers/specs/2026-04-30-d8-fairness-robustness-design.md` (D8 v1 spec)
- `docs/architecture/DECISIONS.md` (ADRs à ajouter : 018 + 019)
- `docs/iso/AI_RISK_REGISTER.md` (ajouter R-ALI-05)
- `memory/project_debt_current.md` (résorber D-2026-05-10-d8-multi-saison-data-limit)
- `services/ali/cache.py` (cible modif)
- `scripts/backtest/harness.py` (cible modif)
- `scripts/d8/run.py` + wrappers + kernel-metadata (cible modif)

---

**END OF DESIGN SPEC v0.1.0 — awaiting user sign-off before implementation**
