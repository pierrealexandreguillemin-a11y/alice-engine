---
name: pre-v1-audit
description: Use BEFORE any production launch (v1, vN+1, SaaS deploy, public release) of ALICE Engine. Generates the 8-section pre-V1 audit report from `docs/process/PRE_V1_AUDIT_TEMPLATE.md`, populated with current champion state (MLP(32,16) 18f stacking + temp scaling), CE OR-Tools status, ISO 14-norm scan, and cost/effort matrix on improvement candidates. NON NÉGOCIABLE per CLAUDE.md. Triggers on /pre-v1-audit, "audit pre-prod", "avant lancement v1", "before production launch", "pre-V1 quality gate", or any pre-production checkpoint intent.
---

# Pre-V1 Production Audit Skill (ALICE Engine)

Generates the formal audit report required BEFORE any ALICE Engine production
launch. Confronts champion ML, CE optimization, features, performance, ISO
compliance, and risk against published SOTA — outputs a gain/effort matrix and
ML-senior recommendation for user decision (accept/override/adopt-subset/reject).

## When to invoke

| Trigger | Condition |
|---|---|
| **T1 user-explicit** | `/pre-v1-audit <scope>` OR "audit pre-prod" / "avant lancement v1" / "pre-V1 quality gate" / "before production launch" |
| **T2 phase milestone** | Phase 4b done → Phase 4.5 entry (mandatory before Phase 5 SaaS deploy) |
| **T3 vN+1 launch** | Avant TOUT incrément production (vN.X→vN+1, public release, multi-tenant rollout) |

**NON NÉGOCIABLE per CLAUDE.md** : pas de prod sans audit signé user.

## ALICE-specific context (read FIRST before audit)

Sources de vérité à charger :

```bash
# Champion ML state
cat config/MODEL_SPECS.md
cat docs/project/V9_HP_SEARCH_RESULTS.md | head -50

# Architecture decisions
cat docs/architecture/DECISIONS.md  # ADR-001..ADR-017+

# ISO compliance status
cat docs/iso/IMPLEMENTATION_STATUS.md
cat docs/iso/AI_RISK_REGISTER.md

# Quality gates
cat docs/requirements/QUALITY_GATES.md  # F1-F12 / T1-T12

# Open dette
cat ~/.claude/projects/C--Dev-Alice-Engine/memory/project_debt_current.md

# Roadmap context
sed -n '/^## Phase 4\.5/,/^## Phase 5/p' docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md
```

## 8-step protocol (ALICE-specific)

### Step 1 — Read context + current state

Charger sources ci-dessus + `docs/process/PRE_V1_AUDIT_TEMPLATE.md`.
Identifier scope = quoi est launched (v1 = champion + CE + API ; vN+1 = quel delta).

### Step 2 — Section 1 ML self-challenge

Pour le champion **MLP(32,16) 18f + temp scaling** :

- Métriques canonical : log_loss 0.5530, ECE_draw 0.001648, T=1.0216 (re-vérifier
  dans `models/cache/mlp_champion_metadata.json`)
- Lineage SHA-256 : `models/cache/mlp_meta_learner.joblib` + `temperature_T.joblib`
- Comparer aux V9 GBM single (LGB 0.5619, XGB 0.5622, CB 0.5708)
- **Alternatives obligatoires à évaluer** :
  - Deep Ensemble 5×MLP (Lakshminarayanan 2017 NeurIPS) — gain attendu +0.005-0.015 log_loss
  - Test-time augmentation (TTA) avec Elo perturbés — gain +0.002-0.005
  - Bayesian MLP (uncertainty-aware) — calibration edge cases
  - Mixture of Experts gated par ronde/saison — drift adaptation
  - Conformal wrapping E2E (D15) — IC sur E[score]
- **Test empirique** : SI gain > 0.005 log_loss attendu, refit + comparer OOF AVANT swap
- **Output** : tableau ROI sections 1 du template (Source / Gain / Effort / Risque / Decision)

### Step 3 — Section 2 Feature/Data self-challenge

Revue dette features deferred :
- D13 zone_enjeu (Phase 4+ couplé CE) — SHAP-attendu ?
- F4 streak features (Phase 3 remontés option B) — déjà intégrés ?
- D11 NLP PDF FFE → JSON (Phase ultérieure) — gap éventuel ?

Drift FFE :
- `chess-app/scripts/sync_ffe_rules.py --check` last run ?
- `data/echiquiers.parquet` saison max ? `data/joueurs.parquet` count ?

SHAP top-20 features champion : reste-t-il signal résiduel non capté ?

### Step 4 — Section 3 CE / Optimization self-challenge

Status Phase 4 (CE OR-Tools multi-équipes) :
- Toutes contraintes FFE A02 §3.7.b/c/d/f encodées ? (vérifier `services/composer.py`
  ou successeur Phase 4)
- Coupes J02 jeunes / S65 vétérans / Loubatière supportés ? (D3, D4 dette)
- Sensitivity analysis poids objective E[score] ?
- Phase 4a ALI joint conditionné (D-P3-19 / R-ALI-06) intégré ?

### Step 5 — Section 4 Performance / UX

Mesurer (créer scripts si absents) :
- `scripts/benchmarks/api_latency.py` : P50/P95/P99 sur `/compose` + `/recompose`
- Cold start : `time .venv/Scripts/python.exe -m app.main` jusqu'à ready
- RSS memory : `psutil` au runtime sous load
- Error handling : revue `app/api/routes.py` exception handlers

Cible : P99 < 2s (CE on-demand spec), cold start < 30s, RSS < 4GB sur Oracle VM
ARM 24GB.

### Step 6 — Section 5 Security/Compliance ISO (14 normes)

Pour chaque norme dans `docs/iso/ISO_STANDARDS_REFERENCE.md` :
- Run scan (radon, gitleaks, bandit, mkdocs --strict, pytest --cov, etc.)
- Identifier gaps vs `docs/iso/IMPLEMENTATION_STATUS.md`
- Document PASS / FAIL / GAP / DEFERRED par norme

### Step 7 — Section 6 Risk ISO 23894

Pour chaque R-ALI-* dans `docs/iso/AI_RISK_REGISTER.md` :
- Status mitigation (validated / pending / failed)
- D8 outputs disponibles (gates_19_status.json) → quantif risques
- Risques résiduels nécessitant user-acceptance explicite

### Step 8 — Sections 7+8 ROI matrix + Senior reco

Synthétiser findings 1-6 dans tableau unique `<gain> × <effort> × <risk> = ROI`.
Senior reco 200-400 mots :
- Ship as-is OR adopter N améliorations
- Raisons techniques (pas marketing)
- Hypothèses + risques résiduels
- Deadline impact

User signe formellement (accept / override / adopt-subset / reject) en bas du
rapport.

## Output deliverable

Écrire `reports/pre_v1_audit/<YYYY-MM-DD>-<scope>.md` (template rempli intégral).

Update side-effects :
- `memory/project_debt_current.md` : nouvelles dettes identifiées
- `docs/architecture/DECISIONS.md` : ADR proposé pour chaque amélioration adoptée
- Commit : `audit(pre-v1): <scope> — N items pre-V1, M items post-V1`

## Cross-refs

- Template : `docs/process/PRE_V1_AUDIT_TEMPLATE.md`
- Standing rule : `CLAUDE.md` §"Pre-V1 Production Audit"
- Roadmap : `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md` §Phase 4.5
- Dette : `D-2026-05-10-pre-v1-audit-gate` dans `memory/project_debt_current.md`
- Skill source : `.claude/skills/pre-v1-audit/SKILL.md` (this file)

## ALICE-specific anti-patterns to avoid

- ❌ "Le champion marche, on ship" — confrontation alternatives obligatoire
- ❌ "ECE 0.0016 est parfait, rien à dire" — quantifier l'asymptote sur features
  + base ensemble actuels, pas absolu
- ❌ "ISO 14 normes c'est documenté" — re-scanner empiriquement, pas relire docs
- ❌ Skipper Section 7 ROI matrix — le tableau gain/effort EST le délivrable
- ❌ Senior reco vague ("ça semble OK") — raisons techniques précises requises
- ❌ Pas de signature user — l'audit est nul sans decision formelle signée
