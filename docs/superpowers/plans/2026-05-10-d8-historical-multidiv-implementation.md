# D8 Historical-State + Multi-Divisions — Implementation Plan

**Document ID** : ALICE-PLAN-D8-HISTORICAL-MULTIDIV
**Version** : 0.1.0 (DRAFT, pre-approval)
**Status** : PROPOSED — awaiting user sign-off after spec + cascade review
**Spec** : `docs/superpowers/specs/2026-05-10-d8-historical-multidiv-design.md`

---

## Tasks (10) — TDD per task

### Task 1 — Spec ADR-018 + ADR-019 (~30 min)

**Files create** :
- `docs/architecture/ADR-018-ali-data-cache-historical-state.md`
- `docs/architecture/ADR-019-d8-pivot-multidiv-multisaison.md`

**Outcome** : Architecture decisions documentées. Référencées par cascade.

### Task 2 — Update R-ALI-05 in risk register (~15 min)

**Files modify** : `docs/iso/AI_RISK_REGISTER.md`

**Outcome** : R-ALI-05 NEW (noyau historique unverifiable) tracé.

### Task 3 — Implement `ALIDataCache.from_parquets_at_saison()` (~2h)

**Files modify** :
- `services/ali/cache.py` — ADD classmethod
- `tests/services/test_cache.py` — ADD 5 tests TDD

**Pattern** :
```python
@classmethod
def from_parquets_at_saison(cls, joueurs_path, echiquiers_path, saison: int):
    """Reconstruct cache state at saison X from echiquiers.parquet.

    Limitations (R-ALI-05): mute / licence_active / age fields default to
    static values since echiquiers does not capture per-saison state.
    """
    # 1. Load echiquiers, filter by saison
    # 2. Build joueurs_by_club: groupby(equipe).agg(union(nr_blanc, nr_noir))
    #    → DataFrame with (nr_ffe, nom, prenom, elo, club, mute=False, ...)
    # 3. Build team_to_club: groupby(equipe).agg(equipe_first_value())
    # 4. Return ALIDataCache instance
```

**Tests** :
- test_from_parquets_at_saison_2024_matches_current_state (sanity check)
- test_from_parquets_at_saison_2021_returns_historical_teams
- test_from_parquets_at_saison_invalid_raises
- test_historical_state_no_mute_field_default_false
- test_historical_state_elo_from_echiquiers_median

### Task 4 — Update `BacktestHarness.setup()` (~1h)

**Files modify** :
- `scripts/backtest/harness.py` — ADD `historical_saison: int | None = None` param
- `tests/backtest/test_harness.py` — ADD 2 tests (None default = current state, int = historical)

**Pattern** :
```python
def setup(self, *, historical_saison: int | None = None) -> None:
    if historical_saison is not None:
        self.cache = ALIDataCache.from_parquets_at_saison(
            Path(settings.joueurs_parquet),
            Path(settings.echiquiers_parquet),
            historical_saison,
        )
    else:
        self.cache = ALIDataCache.load_from_parquets(...)
    # rest unchanged
```

### Task 5 — Update D8 `run.py` (~1h)

**Files modify** :
- `scripts/d8/run.py` — read `ALICE_DIVISION` env var, pass `historical_saison` to harness
- `scripts/d8/types.py` — extend `D8Lineage` with `division` field

**Pattern** :
```python
saison = int(os.environ["ALICE_SAISON"])
division = os.environ.get("ALICE_DIVISION", SAISON_DIVISION_FILTER[saison])
harness.setup(historical_saison=saison if saison < 2024 else None)
config = RunnerConfig(saison=saison, division_filter=division, ...)
```

### Task 6 — Regenerate D8 wrappers + kernel-metadata (~30 min)

**Files create** (replace existing per-saison wrappers) :
- `scripts/d8/run_2024_top16.py` … `run_2024_n4.py` (5 wrappers Phase A)
- `scripts/d8/kernel-metadata-d8-2024-{top16,n1,n2,n3,n4}.json` (5 metadata)
- Phase B wrappers (15) : optional, deferred

**Pattern** : same 38-line wrapper as existing run_2024.py + ALICE_DIVISION setdefault.

### Task 7 — Update D8 `aggregate.py` (~1h30)

**Files modify** :
- `scripts/d8/aggregate.py` — fuse keyed by (saison, division)
- `scripts/d8/aggregate_render.py` — D8_FINDINGS.md template by_niveau + by_saison
- `scripts/d8/upload_d8_dataset.py` — kernel-metadata-aggregator.json updates 5+ dataset_sources

**Pattern** : `load_saison_reports()` → `load_audit_reports(saisons, divisions)` returning
`dict[tuple[int, str], dict]`.

### Task 8 — Update D8 tests (~1h)

**Files modify** :
- `tests/d8/test_aggregate.py` — extend fuse + render fixtures with (saison, division) keys
- `tests/d8/test_run_e2e_smoke.py` — add multi-division smoke
- `tests/d8/test_perturb_runner.py` — verify on historical cache

**Outcome** : 296 + ~10 NEW = ~306 D8 tests pass.

### Task 9 — Local pre-push smoke (~30 min)

**Sequence** :
1. Smoke saison 2024 N3 (existing) : SMOKE_MAX_MATCHES=3, must reproduce 1-2 valid matches
2. Smoke saison 2024 N4 (NEW) : SMOKE_MAX_MATCHES=3, expected ≥1 valid match
3. Smoke saison 2021 N3 historical (NEW) : SMOKE_MAX_MATCHES=3, expected ≥1 valid match
4. Verify d8_saison_S_division_D.json output schema valid

**Acceptance** : 3 smokes pass under 10min total local wallclock.

### Task 10 — Push 5 Phase A kernels + monitoring (~30 min push + ~12h Kaggle wallclock)

**Sequence** :
1. Re-stage alice-d8-code (361+ .py)
2. Force version dataset
3. Push 5 saison-2024 kernels (Top 16, N1, N2, N3, N4)
4. Status monitor toutes 30 min
5. Si tous COMPLETE → outputs as datasets → push aggregator
6. Si ERROR → SDK log diag → fix LOCAL → re-push

**Acceptance** : 5 kernels COMPLETE + d8_saison_2024_*.json + aggregator full report.

---

## Total estimate

| Phase | Tasks | Local effort | Kaggle wallclock |
|-------|-------|--------------|------------------|
| Pre-implementation (specs + ADRs + risk) | 1, 2 | ~45 min | 0 |
| Core code | 3, 4, 5 | ~4h | 0 |
| Wrappers + metadata + aggregator | 6, 7 | ~2h | 0 |
| Tests | 8 | ~1h | 0 |
| Local smoke | 9 | ~30 min (3 × 3min compute) | 0 |
| Kaggle push + monitor (Phase A) | 10 | ~30 min push + monitor | ~12h parallel 4 simultaneous |
| **TOTAL Phase A** | 10 | **~8h dev** | **~12h Kaggle** |
| Phase B (optional, post-V1) | +5 | +1h | +36h |

---

## Self-review (sections 1-7)

**Section 1 spec — completeness** :
- ✅ contexte clair (data semantics blocker identifié empiriquement)
- ✅ user demand explicit
- ✅ cible mesurable (N≥200, 19 gates jugés, by_niveau + by_saison)

**Section 2 solution — feasibility** :
- ✅ reconstruction historique fait sense (data complète dans echiquiers)
- ⚠️ limites documentées (mute, licence_active, age) → R-ALI-05 NEW
- ✅ phase A seule suffit pour spec §1.3

**Section 3 cascade — completeness** :
- ✅ tous fichiers cible identifiés
- ✅ tous fichiers runtime à vérifier listés
- ✅ tests impactés identifiés
- ✅ compute budget estimé

**Section 4 risks** :
- ✅ R-ALI-05 NEW tracé
- ✅ R-ALI-01 + R-ALI-04 mitigations renforcées par audit

**Section 5 acceptance** :
- ✅ critères mesurables avant push
- ✅ user sign-off explicit avant Kaggle

**Section 6 ADRs** :
- ✅ ADR-018 + ADR-019 listés

**Section 7 plan tâches** :
- ✅ 10 tâches TDD-friendly
- ✅ effort estimé honnête (8h dev)
- ✅ Phase B explicite optional post-V1

---

**END OF PLAN v0.1.0 — awaiting user sign-off before Task 1**
