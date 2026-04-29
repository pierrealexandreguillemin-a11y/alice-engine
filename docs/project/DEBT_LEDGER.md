# Debt Ledger — ALICE Engine (versioned mirror)

**Document ID** : ALICE-DEBT-LEDGER
**Version** : 1.0.0
**Last updated** : 2026-04-29 (Plan 3 V2 T25 finalization, JALON #3 peer review Minor #5)
**ISO Compliance** : ISO/IEC 42001:2023 Annex A.6 (Lifecycle traceability),
ISO/IEC 42010 (architecture description)

## Purpose

Versioned mirror of critical open dette technique referenced from versioned
ISO docs (ADR, Risk Register, Quality Gates Reports). The full debt
inventory lives in `memory/project_debt_current.md` (source of truth,
hosted outside repo at `~/.claude/projects/...`). This file mirrors only
the **open dettes referenced from versioned docs** to close the ISO 42001
traceability gap (cross-references from `docs/iso/*` to non-versioned
memory).

## Open Critical Debt (referenced from versioned ISO docs)

### D-P3-19 — ALI multi-équipes joint conditionné (CRITICAL, Phase 4a)

- **Source** : T22 review post-mortem 2026-04-28 (commit `88ba3a1`)
- **Severity** : 🔴 CRITICAL OPEN, score 20 (per `AI_RISK_REGISTER.md`
  R-ALI-06)
- **Why** : ALI Phase 3 sample dans pool club total au lieu de pool
  conditionné sur équipes simultanées du club adverse. Bloquant gates
  absolus P3G07-P3G11. Évidence : 117 clubs N3 saison 2024 alignent
  2-4 équipes simultanées, gap recall by_size = 0.28.
- **Resolution** : Phase 4a Approche A SOTA (CE-adverse miroir OR-Tools)
- **Cross-references** :
  - `docs/architecture/adr/ADR-016-ali-conditioned-multi-team-adverse-ce-mirror.md` (status: Proposed)
  - `docs/iso/AI_RISK_REGISTER.md` §2.7 R-ALI-06
  - `docs/iso/AI_RISK_ASSESSMENT.md` §R-ALI-06
  - `docs/iso/ALI_QUALITY_GATES_REPORT.md` §6.2 §7.5
  - `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md` §Phase 4a + 4b
  - `memory/project_debt_current.md` D-P3-19 (full detail)

### D8 — ALI fairness/robustness breakdown rigoureux (Phase 3.5 STRICT)

- **Source** : Phase 3 scope (R-ALI-01 mitigation upgrade)
- **Severity** : 🟡 Monitor, bloquant Phase 4a acceptance gate
- **Why** : sans breakdown N≥200 multi-saisons par genre / taille club /
  niveau, on ne peut pas valider que les biais R-ALI-01 (PRIVATE rules
  unverifiable) restent <10% recall variance par stratum.
- **Resolution** : Phase 3.5 STRICT (compute ~52 min × 4 saisons = 4h)
- **Cross-references** :
  - `docs/iso/AI_RISK_REGISTER.md` §R-ALI-01
  - `docs/iso/ALI_QUALITY_GATES_REPORT.md` §6.3 (Phase 3.5 STRICT levers)
  - `memory/project_debt_current.md` D8

### D9 — Adaptive Importance Sampling + drift monitoring (Phase 5+)

- **Source** : Phase 3 brainstorm finding (R-ALI-04 mitigation)
- **Severity** : ⚠️ score 12 (highest ALI register)
- **Why** : λ=0.9 + n_topk=10 + n_mc_pairs=5 statique calibré sur
  saisons 2021-2024. En prod, drift FFE rules + roster turnover dégrade
  silencieusement les prédictions.
- **Resolution** : Phase 5+ (après volume data prod, AIS Veach 1995 +
  drift dashboard PSI/KL)
- **Cross-references** :
  - `docs/iso/AI_RISK_REGISTER.md` §R-ALI-04
  - `memory/project_debt_current.md` D9

### D6/D7 — DVC versioning + Kaggle lineage (PARTIAL T24, Phase 5)

- **Source** : V8 historique
- **Severity** : 🟢 acceptable Phase 3, requis Phase 5 deploy
- **Status T24 (2026-04-29, Plan 3 V2)** : `dvc.yaml` 2 stages locaux
  (refit_mlp_champion + backtest_holdout_2024) avec dvc.lock. DAG
  reproducible. Pas de remote DVC. Kaggle training non couvert.
- **Resolution** : Phase 5 (`dvc remote add -d s3://...` + stages
  training Kaggle CatBoost/XGB/LGB)
- **Cross-references** :
  - `docs/devops/ML_MODEL_VERSIONING_STANDARDS.md` §"ALICE Engine —
    Implémentation T24 Phase 3"
  - `dvc.yaml` racine repo
  - `memory/project_debt_current.md` D6/D7

## Resolved Debt (Plan 3 V2 session 2026-04-28..29)

11 dettes résorbées fix-on-sight cycle T22 + T23 + T25 :

| ID | Status | Commit | Note |
|----|--------|--------|------|
| ~~D-P3-13~~ | RESOLUE T22.0 | `bb9c434` | MLP champion refit OOF, ECE_draw 0.001648 |
| ~~D-P3-01~~ | RESOLUE T22 | `e2ce349` | env var `CHESS_APP_RULES_DIR` + graceful CI skip |
| ~~D-P3-02~~ | RESOLUE T22 | `e2ce349` | mutability documenté délibéré (lifespan reload) |
| ~~D-P3-05~~ | RESOLUE T22 | `e2ce349` | imports cleanup |
| ~~D-P3-06~~ | RESOLUE T22 | `e2ce349` | O(N²) → O(N log N) |
| ~~D-P3-07~~ | RESOLUE T22 | `e2ce349` | Pydantic VerifiabilityEntry |
| ~~D-P3-08~~ | OBSOLETE T22 | `e2ce349` | déjà couvert checkers.py |
| ~~D-P3-09~~ | RESOLUE T22 | `e2ce349` | Literal `out_of_scope` reclassement |
| ~~D-P3-11~~ | RESOLUE T22 | `4c848e1` | services/ffe_rules.py supprimé |
| ~~D-P2-02~~ | RESOLUE T22 | `0656fdf` | partition_rules wired |
| ~~D-P2-05~~ | RESOLUE T22 | `e2ce349` | --full opt-in design |
| ~~D-P3-18~~ | RESOLUE 2026-04-28 | `e2ce349` | ADR-017 Wilcoxon SOTA pivot |

Plus T23 résorption précédente (commit `cdf6a7c`) : D5 + D-P2-03 + D-P2-04 + D-P3-12.

## How to Sync This File

This versioned mirror **must be updated** when :

1. A new dette is added to `memory/project_debt_current.md` AND a versioned
   ISO doc references it.
2. A dette is résolue → move to "Resolved Debt" table with commit SHA.
3. A new ADR/risk references an open dette → cross-ref ajouté.

**Trigger** : pre-merge JALON peer review checklist (cf.
`scripts/verify_plan3_dod.sh` Plan 3 V2 T25 pattern).
