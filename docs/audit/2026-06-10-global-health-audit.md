# Global Project Health Audit — Alice Engine

**Date**: 2026-06-10
**Scope**: Full repository — architecture, tests/CI, ML pipeline coherence, technical debt & documentation consistency
**Method**: 4 parallel read-only exploration passes (architecture, tests/CI, ML pipeline, debt/docs) + manual verification of disputed counts
**Trigger**: Owner concern — "plans keep succeeding each other, I can't judge whether development is healthy"

---

## 1. Executive Verdict

**Overall health: GOOD (B+).** The project is unusually rigorous for a solo/hobby
context: the announced ML champion is real, wired end-to-end, and its metrics are
reproducible from committed metadata. Technical debt is *visible* (26 open items,
all tracked with target phases), failures are documented (8 postmortems), and the
succession of plans reflects quality-driven iteration, not chaos.

**The single most important fact for the owner**: the system is **not in
production** and one **structural flaw (D-P3-19)** was found at the end of
Phase 3 — the opponent-lineup simulator (ALI) did not account for clubs fielding
several teams the same weekend. This flaw failed 13 of 19 audit gates in the
2026-05-16 empirical stress test and is the reason Phase 4a was inserted. That
insertion is the correct engineering response, but it is the main driver of the
"plans keep piling up" feeling.

---

## 2. What is verifiably solid

### 2.1 ML champion — real and wired (verified end-to-end)
- Artifact exists: `models/cache/mlp_meta_learner.joblib` + `temperature_T.joblib`
  + `mlp_champion_metadata.json` (SHA-256 lineage, created 2026-04-28).
- Metadata matches doc claims: `log_loss_test_calibrated 0.55303`,
  `ece_draw_test_calibrated 0.00165`, `temperature_T 1.0216`.
- Full inference path verified: `app/api/routes.py` → `compose_scenarios.py` →
  `services/inference.py::StackingInferenceService` → 3 GBMs (LGB 44.8 MB /
  XGB 63.2 MB / CB 16.0 MB, all present) → 18 meta-features → MLP → temperature.
- DVC pipeline (`dvc.yaml`): 3 stages (refit_mlp_champion, backtest_holdout_2024,
  d8_audit) consume the champion artifacts. Timeline coherent (OOF parquets
  2026-04-16 → MLP refit 2026-04-28).

### 2.2 Code architecture
- All served code (`app/`, `services/`) ≤ 300 lines/file (ISO 5055), strict SRP,
  no broken imports, no circular dependencies.
- Zero hardcoded secrets; config via Pydantic Settings + env vars; PII filtering
  in logging; rate limiting; async MongoDB audit logging.
- Broad `except Exception` handlers (5 sites in `app/`) are all logged with
  `logger.exception()` and have graceful fallbacks — acceptable.
- 9 training/batch scripts exceed 300 lines (max `scripts/cloud/optuna_kaggle.py`
  485L) — non-served code, tolerated.

### 2.3 Tests & CI
- 226 test files, 2 127 test functions (manually verified). Local coverage ~73%
  against a 70% enforced threshold (`--cov-fail-under=70`).
- `services/ali/` (the active development area) is well covered: 15/17 modules
  have dedicated test files.
- Pre-commit (Gitleaks, Ruff lint+format, MyPy, Bandit) + pre-push (<90s fast
  tests + xenon + pip-audit) + full CI (quality/tests/security/complexity).

### 2.4 Governance
- 23 ADRs; only 1 superseded (ADR-015) + 1 method pivot (ADR-017 McNemar→Wilcoxon)
  — low decision churn for a 6-month ML project.
- 8 postmortems documenting real failures (AutoGluon ×2, training divergence,
  data bugs) — failures are surfaced, not hidden.
- 10/10 spot-checked CLAUDE.md claims verified TRUE (e.g. `/compose` wired,
  `verify_plan3_dod.sh`, `scripts/robustness/`, `scripts/monitoring/bias_tracker.py`).

---

## 3. Weaknesses found (honest list)

| # | Finding | Severity | Detail |
|---|---------|----------|--------|
| W1 | **D-P3-19 structural flaw** (ALI multi-team conditioning) | **CRITICAL — known, being fixed** | Found at T22 review (2026-04-28), confirmed empirically 2026-05-16 (13/19 D8 Phase A gates FAIL). Phase 4a is the fix; T1–T3 of it are shipped, integration (T4+) pending. |
| W2 | Phase 4a modules are **orphan code** today | Medium — expected | `preference_model.py`, `adverse_ce.py`, `diversification.py` are tested but imported by NOTHING in production. They await T4+ integration. Until wired, the running system still has the D-P3-19 flaw. |
| W3 | 5 core modules with **zero dedicated unit tests** | Medium | `app/api/routes.py`, `compose_helpers.py`, `compose_scenarios.py`, `app/logging_config.py`, `services/data_loader.py`. Covered only indirectly via `test_compose_e2e.py`, which is itself excluded from pre-push (runs in CI only). |
| W4 | **Non-fatal CI gates** | Medium | MyPy `continue-on-error: true` (ci.yml:52), Bandit `\|\| true` (ci.yml:55), pip-audit `\|\| true` with 6 ignored CVEs (ci.yml:168-174). These report but never block. |
| W5 | Stale doc note | Low | `config/MODEL_SPECS.md:927` still names "LGB + Dirichlet" as champion (obsolete since 2026-04-28 MLP refit). |
| W6 | Stale TODOs in `services/inference.py` | Low | Legacy backward-compat stub with "TODO Phase 3" comments; dead path, real path goes through `StackingInferenceService`. |
| W7 | Data freshness | Low (pre-prod) | `data/*.parquet` frozen at 2026-03-18; refresh contract is Phase 5 (tracked D-2026-06-01-historical-store-refresh). |
| W8 | Documentation weight | Structural | 104 docs / ~21 000 lines vs ~120 production files — ratio ~2.6:1 in lines. Disciplined but a real maintenance cost for a solo project. |
| W9 | Pre-V1 audit gate not yet executed | Expected | NON-NEGOTIABLE gate per CLAUDE.md; scheduled Phase 4.5, before prod. Nothing skipped — but prod must not happen without it. |

Open debt: **26 items** in `memory/project_debt_current.md`, ~30 resolved since
Phase 2. All open items carry an explicit target phase. The critical ones:
D-P3-19 (Phase 4a, in progress), D8 fairness breakdown (Phase 3.5 STRICT,
blocking Phase 4), pre-V1 audit gate (Phase 4.5, blocks v1).

---

## 4. Why "plans keep succeeding each other" — root-cause answer

Measured causes, in order of impact:

1. **A real flaw found late** — D-P3-19 forced inserting Phase 4a upstream of the
   already-planned Phase 4b. This is the correct response (fixing before building
   on top), but it added a full phase to the roadmap.
2. **Quality-first doctrine** — the project refuses silent debt: every review
   finding becomes either an inline fix or a tracked debt with a phase. This
   makes progress *look* slower because problems are surfaced instead of buried.
3. **ISO/process overhead** — 14 ISO standards, audit gates, session memos,
   9-step Kaggle push process. High assurance, high wall-clock cost.

**This is iteration, not drift**: only 1 of 23 ADRs was reversed, plans form a
logical chain (data → training → API → ALI → audit → fix), and no plan was
abandoned mid-flight.

## 5. Honest risk statement for the owner

- The product **delivers no user value yet** (no production deployment). The
  distance to a usable v1 is: Phase 4a integration (T4+) → Phase 4b CE user
  (OR-Tools) → Phase 4.5 pre-V1 audit → Phase 5 deploy. That is several phases,
  each historically taking weeks.
- The running `/compose` endpoint works but embeds the known D-P3-19 flaw until
  Phase 4a is wired in. It must not be exposed to real users before that.
- Solo-dev + heavy process = the main schedule risk is process cost, not code
  quality. Quality indicators are green; velocity is the trade-off.

## 6. Recommended next actions (priority order)

1. **Finish Phase 4a integration (T4+)** — wires the 3 orphan modules, resolves
   the only critical structural flaw. Highest-leverage work; everything else is
   secondary.
2. **D8 / Phase 3.5 STRICT debt** (fairness/robustness breakdown) — declared
   blocking for Phase 4; confirm it is actually closed or consciously re-scoped.
3. Quick fixes (<1h total): update `MODEL_SPECS.md:927` champion note; delete or
   clearly mark the dead `services/inference.py` legacy stub.
4. Medium-term hardening (post-4a, pre-prod): make MyPy and Bandit fatal in CI;
   add dedicated tests for `app/api/routes.py`, `compose_helpers.py`,
   `data_loader.py`; document the rationale for each ignored CVE.
5. **Do not deploy v1 without the Pre-V1 audit gate** (Phase 4.5) — already
   mandated, restated here for the record.

---

*Method note: findings from 4 parallel Explore agents; champion metrics, test
counts (226 files / 2 127 functions) and doc count (104) manually re-verified.
Agent-reported file line counts and CI line numbers were not all individually
re-verified; treat ±5% tolerance on counts, but all existence claims (artifacts,
files, wiring) were grep/path-verified.*
