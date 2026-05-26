---
name: project-session-end
description: Use when CLAUDE.md STANDING PROCEDURE post-task triggers fire — T1 (task >30min wall OR >5 tool-calls OR 1+ commit shipped), T2 (fatigue ALERT ≥1 indicator), T3 (fatigue STOP ≥1 indicator), T4 (Plan-T-N task SHIPPED OR Kaggle kernel push validé OR Phase milestone P1-P7 atomic SHIPPED OR ADR adopted OR debt D-XX résolue), T5 (user-explicit "memo update"/"doc de reprise"/"session-end"/"STOP"/"pause"/"fin de session"). Use before fresh-Claude reprise without conversational context, before commit body planning, before user-facing recap message.
---

# alice-engine — Project Session-End Memo Generator

Implements alice-engine `CLAUDE.md` STANDING PROCEDURE NON-NÉGOCIABLE post-task memory + fatigue self-assessment. **18-section autoportable resume memo** for fresh-Claude reprise sans context conversationnel.

Stack-aware : **Python 3.x + venv `.venv/`** (FastAPI + ML stack CatBoost/XGBoost/LightGBM/MLP champion + Optuna V9), **Kaggle CPU training cloud** (kernels Optuna + Training Final + OOF ; AG ÉLIMINÉ ADR-011), **DVC partial** (refit MLP + backtest, remote = Phase 5), **14 ISO normes** mandatory à chaque étape, **R-PRE-PUSH-01 push <90s** (ruff+mypy+xenon+pip-audit+pytest fast). Bench gates = **15 quality gates F1-F12/T1-T12 + ECE_draw + log_loss + RPS + ALL 3 models converged + skill kernel-push 9 étapes + ATTENDRE validation user**.

## When to invoke (T1-T5 triggers obligatoire)

| Trigger | Condition |
|---|---|
| **T1 post-task-long** | tâche > 30 min wall OU > 5 tool-calls OU 1+ commit shippé |
| **T2 fatigue ALERT** | ≥ 1 indicateur ALERT (per `feedback_post_task_procedure_and_fatigue.md`) |
| **T3 fatigue STOP** | ≥ 1 indicateur STOP |
| **T4 milestone naturelle** | Plan-T-N task SHIPPED OU Kaggle kernel push validé OU Phase milestone (P1-P7) atomic SHIPPED OU ADR adopted OR debt D-XX résolue |
| **T5 user-explicit** | "memo update" / "doc de reprise" / "session-end" / "STOP" / "pause" / "fin de session" |

Self-trigger when planning a commit body — verify session-end memo updated BEFORE final user-facing message.

**Cohérence avec `feedback_self_contained_resume_after_each_task.md` existant** : per-task discipline (CRITIQUE) reste applicable comme version allégée entre les tâches ; ce skill (T1-T5) est la version session-end ceremonial avec les 18 sections + 7 critères + history archive.

## What to do when invoked (11-step protocol)

### Step 1 — Read STANDING PROCEDURE source-of-truth

```bash
# Don't reinvent — read the canonical sources
cat C:/Dev/alice-engine/CLAUDE.md  # §"STANDING PROCEDURE — POST-TASK MEMORY + FATIGUE SELF-ASSESSMENT"
cat C:/Users/pierr/.claude/projects/C--Dev-alice-engine/memory/feedback_session_end_template.md
cat C:/Users/pierr/.claude/projects/C--Dev-alice-engine/memory/feedback_post_task_procedure_and_fatigue.md
```

### Step 2 — Sanity bash collect state (parallel)

```bash
cd C:/Dev/alice-engine
git rev-parse HEAD
git log --oneline -10                                                            # recent commits
git status --short                                                                # uncommitted
git rev-list --count origin/master..HEAD 2>/dev/null || echo "no remote ahead"   # commits ahead
.venv/Scripts/python -m pytest --no-header -q -m "not slow" 2>&1 | tail -3       # test counts (slow markers excluded for speed)
.venv/Scripts/python -m ruff check . 2>&1 | tail -3                              # lint state
.venv/Scripts/python -m mypy . 2>&1 | tail -3                                     # type check
ls -t models/*.joblib 2>/dev/null | head -3                                       # champion artifacts
ls -t docs/postmortem/*.md 2>/dev/null | head -3                                  # latest postmortems
kaggle kernels status pierrax/<latest-slug> 2>&1 | head -5                        # Kaggle kernel state
ls C:/Users/pierr/.claude/projects/C--Dev-alice-engine/memory/MEMORY.md            # check current entry 0
```

### Step 3 — Determine session segment scope

- **Commits this segment** : count commits since session start (cumul user actions).
- **Files modified** : `git diff --stat origin/master..HEAD` (or `HEAD~N..HEAD` if no remote) to enumerate scope.
- **Tests delta** : compare current PASS count vs baseline at session start (pytest --cov 70% target).
- **Kaggle kernels created/pushed** : new kernel slugs cette session + version bumps.
- **Champion artifact deltas** : `models/mlp_meta_learner.joblib` SHA256 changed?
- **Debt resolutions** : `~~Dx~~` strikethroughs ajoutées dans `project_debt_current.md` ou CLAUDE.md.
- **User-corrective-loops** : count instances of "user-flag → claude-fix" pattern (= candidate anti-patterns pour `feedback_diagnostic_first_doctrine.md`).

### Step 4 — Determine task tag

Concise kebab-case identifier for filename. Examples alice-engine :
- `plan3-t22-mlp-champion-refit`
- `plan3-t25-verify-dod`
- `phase4a-d-p3-19-ali-joint-conditional`
- `adr-017-wilcoxon-pivot`
- `audit-iso-pre-push`
- `optuna-v9-kernel-catboost-final`
- `r-pre-push-01-resolution`

### Step 5 — Generate filename

```
C:/Users/pierr/.claude/projects/C--Dev-alice-engine/memory/project_session_end_<YYYY-MM-DD>-<task-tag>.md
```

### Step 6 — Build 18 sections per template (literal)

| § | Section | Content source |
|---|---|---|
| 1 | Resume protocol bash | literal copy-paste sanity checks (git + pytest + ruff + mypy + kaggle + paths runtime) |
| 2 | Session totals + HEAD | git state + tests + commits-ahead + push state + pre-push hook PASS list (R-PRE-PUSH-01) |
| 3 | Deliverables (commits + files + LOC + bench/Kaggle/champion deltas) | `git log` + `git diff --stat` + Kaggle kernel pushes + champion artifact SHA256 deltas + debt resolutions |
| 4 | Hard signals fresh-Claude must respect | compose from CLAUDE.md projet + CLAUDE.md racine + feedbacks slow-changing alice-engine — `WebSearch avant choix techniques ; ALL 3 models converged + ISO 14 normes avant champion ; alpha per-model (LGB=0.1, XGB=0.5, CB=0.3) ; rsm=0.7 CB >50 features ; init_scores avant filtrage features ; skill kernel-push 9 étapes MANDATORY ; quality gates 15 conditions F1-F12/T1-T12 ; pre-push <90s R-PRE-PUSH-01 ; fix-on-sight ; ATTENDRE validation user avant push Kaggle (étape 8 9 étapes) ; AutoGluon ÉLIMINÉ ADR-011 ; NE JAMAIS optuna.integration (v4+ = optuna_integration) ; documenter immédiatement (postmortem)` |
| 5 | Next decision options A/B/C/D | + effort estimé + ship-gate empirique (15 quality gates F1-F12/T1-T12) + rollback path + recommendation. ATTENDRE user avant push Kaggle. |
| 6 | Tests/lint/quality status | pytest --cov 70% + ruff + mypy + bandit + xenon + pip-audit + gitleaks + commitizen + pre-commit/pre-push hooks (R-PRE-PUSH-01 <90s) + check_quality_gates (15 conditions F1-F12/T1-T12) + ISO 14 normes compliance status |
| 7 | Kaggle/champion/runtime state | Kaggle kernels (slugs + COMPLETE/RUNNING/ERROR + version) + champion artifacts (`models/mlp_meta_learner.joblib` SHA256) + DVC stages (refit MLP + backtest) + push origin master state + ML metrics (ECE_draw, log_loss, RPS, E[score] MAE) + Phase status P1-P7 + ISO 14 norms — VALEURS ABSOLUES, pas de "voir tool externe" |
| 8 | Mapping commit hash ↔ Kaggle dataset/kernel/champion identity | anti faux-ami "commit-hash = champion validé/déployé" |
| 9 | Contrat champion deploy / Kaggle push explicit | bench gates empirical : 15 quality gates F1-F12/T1-T12 ALL PASS + ECE_draw < 0.005 + log_loss improvement vs Elo + ALL 3 models converged + skill kernel-push 9 étapes complete + WebSearch API doc paramwise (3 AG fails postmortem) + verify dataset on Kaggle BEFORE push + pre-push <90s + ISO 14 + ATTENDRE user OK |
| 10 | Plans status DRIFT-check | `docs/superpowers/plans/2026-04-07-optuna-v9-implementation.md` + `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md` + `docs/architecture/DECISIONS.md` ADR-001..017 + `memory/project_debt_current.md` debt tracking D1-D16 + D-P3-XX |
| 11 | Anti-pattern counter unifié | pointer `feedback_diagnostic_first_doctrine.md` (au démarrage : stub vide, à enrichir au fil des sessions) |
| 12 | Dead code post-pivot tracking | symboles landed mais call-site caduc (lien avec `project_debt_current.md` debt formelle) |
| 13 | Phase status précis (P1 → P7) | composite quality gate run? + status DONE/IN-PROGRESS/PENDING/NOT-STARTED + verdict PASS/FAIL/PENDING + Kaggle deployed YES/NO. P1 ML training DONE champion MLP(32,16) 0.5530 / P2 API DONE Plan 3 V2 T1-T25 / P3 Plan 3 V2 DONE 6 dettes + ADR-016/017 / P4a ALI joint OPEN / P4b CE OR-Tools NOT-STARTED / P5 SaaS NOT-STARTED / P6 SDK NOT-STARTED / P7 monétisation NOT-STARTED |
| 14 | Read order critical pour fresh-Claude | 15-20 docs ordonnés : CLAUDE.md projet → MEMORY.md → THIS memo → MODEL_SPECS.md → DECISIONS.md ADR → debt → QUALITY_GATES → ISO 14 → plans/specs → docs-index → feedbacks CRITIQUES |
| 15 | Critique restante (TODO future) | unfinished items + estimated effort + push origin pending USER OK + Kaggle kernel push pending (rappel skill kernel-push 9 étapes) |
| 16 | OPEN QUESTION fresh-Claude | Q1 + Q2 + Q3 anti-décision-implicite (question 1-2 phrases + 2-3 hypothèses + recommandation par défaut + condition switch) |
| 17 | Fatigue self-assessment + 1-line user-facing | indicateurs OK/ALERT/STOP table (6 indicateurs per `feedback_post_task_procedure_and_fatigue.md`) |
| 18 | Instructions de reprise IDE-pasteable | bloc literal copy-paste depuis §1+§4+§5+§14+§16+§17. Auto-généré à chaque memo refresh. NE PAS générer ad-hoc dans chat — persister dans le memo |

### Step 7 — Verify 7 fresh-Claude-resume self-contained criteria

Per template "Checklist autoportée" :

1. **Sanity bash literal** : un fresh-Claude peut copier-coller §1 sans interprétation (commandes pytest/ruff/mypy/kaggle + paths absolus).
2. **State complete** : §2-§3 capture HEAD + commits + tests + Kaggle/champion état sans ambiguïté.
3. **Hard signals enumerated** : §4 cite TOUS rules + invariants critiques (WebSearch ; ALL 3 models converged ; skill kernel-push 9 étapes ; quality gates 15 conditions ; alpha per-model ; rsm=0.7 ; AutoGluon ÉLIMINÉ ; pre-push <90s ; fix-on-sight ; ATTENDRE user).
4. **Next options actionnable** : §5 propose A/B/C/D avec effort + ship-gate (15 quality gates) + recommandation explicite.
5. **Read order ordered** : §14 priorise lecture pour fresh-Claude (CLAUDE.md → MODEL_SPECS → ADR → debt → QUALITY_GATES → ISO 14).
6. **OPEN QUESTION explicit** : §16 force user decision, NOT décision implicite Claude.
7. **IDE-pasteable bloc** : §18 contient TOUT le minimum pour fresh-Claude reprise.

Si < 7/7 → re-edit memo BEFORE write.

### Step 8 — Write memo file

`Write` tool to `C:/Users/pierr/.claude/projects/C--Dev-alice-engine/memory/project_session_end_<DATE>-<task-tag>.md`.

### Step 9 — Update MEMORY.md entry 0 (= "CURRENT")

- Promote NEW memo as entry 0 "**CURRENT**".
- Demote previous entry 0 to "==== PRIOR SESSION ... — SUPERSEDED by 0 ====" (in place inside MEMORY.md).
- Keep entry 0 concise (~1 paragraph) — full detail is in memo file linked.
- Verify MEMORY.md size < 24.4 KB warning threshold (compress predecessor entries if needed).

### Step 10 — Update history archive

`C:/Users/pierr/.claude/projects/C--Dev-alice-engine/memory/project_session_history_archive.md` : prepend current-date entry sous `## YYYY-MM-DD` header. One paragraph summary literal (= chronological full archive preserved across MEMORY.md compactions).

If file does not exist yet, create with header `# alice-engine session history archive` then prepend.

### Step 11 — Output user-facing recap

Format obligatoire :

```
Memory protocol complete. <N> fichiers memory updated/created cette session.

## Session-end memory bus complet
<table>

## Recap session <DATE> (1 paragraphe)
<HEAD + commits + key deliverables (Kaggle pushes + champion deltas + debt resolutions) + next session prep>

## Next decision points (per memo §16)
- Q1 : ...
- Q2 : ...
- Q3 : ...

## Fatigue final
<STOP/ALERT/OK> — <indicateur principal>. Break clean obligatoire si STOP.

Reprise next session : copy-paste §18 IDE-pasteable du memo pour fresh-Claude. Protocol respecté.
```

## Anti-patterns (refuse to bless)

- **Threshold strict T1** : seuils > 30 min wall / > 5 tool-calls / 1+ commit shippé sont **fire-on-equality**, pas dilutables. 30:01 min trigger ; 5+1 tools trigger ; 1 commit trigger. Pas de "reste léger" / "pas de seuil atteint" / "encore tôt" / "pas vraiment long-task" — la doctrine ANTICIPE le bug RED 2026-05-10 (orbit-wars + bitnet-fr) où l'agent voyait ">30min juste atteint" comme "pas atteint". Threshold = strict.
- **Chaining bypass T1** : si user enchaîne immédiatement sur next task après T1 fire ("on push T14 Kaggle?" / "et la suite?" / "et maintenant?"), memo refresh OBLIGATOIRE AVANT next task. "Transition naturelle vers X" = pattern de bypass à refuser. Per CLAUDE.md "AVANT de demander la prochaine décision à l'utilisateur" — le memo précède la décision suivante, pas l'inverse.
- **Skip section** : tous les 18 sections obligatoires, "moins important" = NO. Si une section n'a pas de contenu, écrire `N/A — <raison>`, jamais blank.
- **Section vide ou TBD** : remplir avec valeur réelle ou "N/A — <raison>" minimum, jamais blank ni "TBD" ni "expected" (cf. `feedback_calculate_dont_guess.md` CRITIQUE).
- **Fresh-Claude resume incomplet** : 7 self-contained criteria DOIVENT toutes PASS. Si < 7/7 → re-edit avant write.
- **MEMORY.md entry 0 size > 24.4 KB** : compresser predecessor entries simultanément.
- **Manual assembly bypass** : si user invoque "session-end" → INVOKE ce skill, ne pas réimplémenter manually (= drift risk inter-sessions).
- **Skip fatigue self-assessment** : §17 obligatoire avec 6 indicateurs table + 1-line user-facing.
- **Hardcode Kaggle/champion results** : §7 valeurs absolues, pas "voir tool externe" — copier les valeurs DANS le memo.
- **Bypass ATTENDRE validation user** : si memo propose "push Kaggle" en §5, expliciter que c'est étape 8 du process 9 étapes — décision user, pas auto.
- **Continuer le pattern legacy `project_session_resume.md` rolling unique** : depuis 2026-05-09, migration vers `project_session_end_<DATE>-<tag>.md` dated. Le fichier `project_session_resume.md` reste comme legacy frozen reference, NE PAS y écrire.

## Cross-references

- `C:/Dev/alice-engine/CLAUDE.md` `## STANDING PROCEDURE — POST-TASK MEMORY + FATIGUE SELF-ASSESSMENT` (= source-of-truth doctrinale projet).
- `C:/Dev/CLAUDE.md` racine (= rigueur + anti-hallucination + bench-every-step — applicable partout).
- `memory/feedback_session_end_template.md` (= template original 18 sections + checklist autoportée — projection locale du GENERIC).
- `memory/feedback_post_task_procedure_and_fatigue.md` (= STANDING PROCEDURE rationale + 6-indicator fatigue).
- `memory/feedback_per_task_memory_updates.md` (= per-task memo discipline, project-agnostic projection).
- `memory/feedback_self_contained_resume_after_each_task.md` (= alice-engine existing CRITIQUE — coexiste, intent identique, version allégée).
- `memory/feedback_diagnostic_first_doctrine.md` (= anti-pattern counter source-of-truth, stub à enrichir).
- `memory/feedback_audit_before_push.md` (= 9 étapes Kaggle push, CRITIQUE).
- `memory/feedback_converge_all_models_before_champion.md` (= ALL 3 + ISO avant champion, CRITIQUE).
- `memory/feedback_no_silent_debt.md` (= debt visible, CRITIQUE).
- `memory/feedback_complete_or_nothing.md` (= 0 livraison partielle, CRITIQUE).
- `memory/feedback_fix_on_sight.md` (= fix-on-sight solo dev, CRITIQUE).
- `memory/feedback_quality_gates_process.md` (= 15 quality gates F1-F12/T1-T12, CRITIQUE).
- `memory/MEMORY.md` (= index target update entry 0).
- `memory/project_session_history_archive.md` (= chronological archive target).
- `memory/project_debt_current.md` (= debt tracking D1-D16 + D-P3-XX).
- `memory/project_session_resume.md` (= LEGACY, frozen 2026-05-09, ne plus écrire).
- `config/MODEL_SPECS.md` (= source vérité per-model 590 configs).
- `docs/architecture/DECISIONS.md` (= ADR-001..017).
- `docs/requirements/QUALITY_GATES.md` (= F1-F12/T1-T12).
- `docs/iso/ISO_STANDARDS_REFERENCE.md` (= 14 normes).
- `.claude/rules/docs-index.md` (= 65 docs index).
- `C:/Dev/wiki/entities/alice-engine.md` (= wiki entity).
- Source canonique de la doctrine : `C:/Dev/feedback_session_end_template_GENERIC.md`.
- Source canonique du skill template : `C:/Dev/skills-templates/project-session-end/SKILL.md`.

## Outputs deliverables (literal report back)

After invocation completes, report :

1. Memo filename + LOC + sections count (= 18).
2. MEMORY.md entry 0 promotion verified.
3. Archive prepend verified.
4. 7 fresh-Claude self-contained criteria : `<X>/7` PASS.
5. Fatigue verdict + 1-line user-facing.
6. Total time spent (wall) — should be < 2 min.
