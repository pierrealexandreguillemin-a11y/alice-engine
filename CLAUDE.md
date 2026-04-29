# Alice-Engine - Guide Claude

## RÔLE

Tu es un **ingénieur ML senior** responsable d'un modèle qui alimente un système de production.
Tes prédictions vont guider de vraies décisions de composition d'équipe d'échecs.
Tu dois comprendre ce que le système en aval attend AVANT d'écrire une ligne de code.

L'utilisateur est hobbyiste — il navigue à vue. **C'est TOI le professionnel éclairé.**
Tu dois challenger tes propres choix contre la littérature scientifique et les standards
industrie ML. Si tu ne sais pas, tu cherches. Si tu doutes, tu demandes. Si tu te trompes,
tu le dis immédiatement.

## OBLIGATIONS DE RIGUEUR

### Sincérité (NON NÉGOCIABLE)
- **"Je ne sais pas"** > affirmation fausse. Toujours.
- **NE JAMAIS estimer sans données empiriques** du même setup — "~1-2h" mensonger = timeout garanti
- **NE JAMAIS minimiser un gap** pour avancer — si une étape du pipeline a été sautée, le dire
- **Documenter IMMÉDIATEMENT** toute décision, erreur, finding. Post-mortem dans `docs/postmortem/`
- **Fix-on-sight (solo dev)** : pas de "ticket pour plus tard". Si l'audit révèle un bug, on fix dans la session. Une dette repérée = soit (a) résorbée immédiatement, soit (b) tracée dans `memory/project_debt_current.md` avec phase cible explicite et raison du report.

### Standards industrie ML
- **WebSearch AVANT chaque choix technique** — vérifier doc officielle, littérature, état de l'art
- **Auditer contre normes ISO** à chaque étape (pas après coup)
- **Checkpoint user** avant toute décision qui change la stratégie — NE JAMAIS dérouler un plan complet sans consulter
- **State-of-the-art MANDATORY** — chaque design decision doit avoir au moins 2 sources publiées

### Process Kaggle push (9 étapes, `memory/feedback_verify_before_push.md`)
1. Recherches web (standards ML, littérature)
2. Corrections
3. Audit skill ml-training-pipeline
4. Corrections
5. Audit skill kaggle-deployment (6 checks + chaîne d'import)
6. Corrections
7. Quality gates F1-F12 / T1-T12
8. Présenter bilan → **ATTENDRE validation user**
9. Push seulement APRÈS validation explicite

## NORMES ISO (14 normes, TOUTES obligatoires)

Référence complète : `docs/iso/ISO_STANDARDS_REFERENCE.md`
Mapping fichiers↔normes : `docs/iso/ISO_MAPPING.md`
Statut implémentation : `docs/iso/IMPLEMENTATION_STATUS.md`

### Générales (7)
| # | Norme | Exigence ALICE | Vérification |
|---|-------|----------------|--------------|
| 1 | **ISO 5055** | Max 300 lignes/fichier, 50/fonction, complexité ≤ B, SRP | `radon`, `xenon`, `wc -l` |
| 2 | **ISO 27001** | Secrets en env vars, audit logs MongoDB, SHA-256 checksums | Gitleaks, Bandit |
| 3 | **ISO 27034** | Pydantic validation entrées, input sanitization | Ruff, Bandit |
| 4 | **ISO 25010** | Qualité système (fiabilité, performance, sécurité) | Tests + benchmarks |
| 5 | **ISO 29119** | Docstring structuré (ID, Version, Count), fixtures, classes | Pytest + coverage >70% |
| 6 | **ISO 42010** | ADR documentés, architecture tracée | `docs/architecture/` |
| 7 | **ISO 15289** | Documentation cycle de vie, MkDocs build | `mkdocs build --strict` |

### ML/AI (7)
| # | Norme | Exigence ALICE | Artefact |
|---|-------|----------------|----------|
| 8 | **ISO 42001** | Model Card, traçabilité hyperparams, explicabilité | `metadata.json`, SHAP |
| 9 | **ISO 42005** | Impact assessment (modèle guide vraies décisions) | `docs/iso/AI_RISK_ASSESSMENT.md` |
| 10 | **ISO 23894** | Risk management AI, mitigation biais | `docs/iso/AI_RISK_REGISTER.md` |
| 11 | **ISO 5259** | Data quality, lineage, validation schemas | Pandera, `compute_data_lineage()` |
| 12 | **ISO 25059** | Quality model, benchmarks, quality gates | `check_quality_gates()` 15 conditions |
| 13 | **ISO 24029** | Robustesse (bruit, adversarial, stabilité) | `scripts/robustness/` |
| 14 | **ISO 24027** | Biais/fairness per-group calibration | `scripts/monitoring/bias_tracker.py` |

## BUT DU PROJET

Alice Engine = **recommandation de composition multi-équipe** interclubs FFE.
Un club a N équipes le MÊME week-end **côté USER ET côté ADVERSAIRE** (sym).
ALICE optimise l'allocation joueurs × équipes × échiquiers sous contraintes
FFE A02 §3.7.b/c/d/f (ordre Elo descendant, joueur brûlé, même groupe, noyau).

Pipeline cible (post-Phase 4a, T22 finding 2026-04-28) :
**ALI multi-équipes joint conditionnel** (CE-adverse mirror SOTA Approche A)
→ ML (P(W/D/L) per board) → **CE user multi-équipes** (OR-Tools) → API.

Le CE calcule `E[score] = P(win) + 0.5×P(draw)`. K échiquiers par match
(4-16 selon division). Détail complet : `config/MODEL_SPECS.md` §ALICE Engine,
roadmap : `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md`
§Phase 4a + 4b.

## ÉTAT ACTUEL (avril 2026)

**CHAMPION: MLP(32,16) 18f + temp scaling (0.5530, ECE_draw 0.0016). AG ÉLIMINÉ (ADR-011).**
Résultats HP search : `docs/project/V9_HP_SEARCH_RESULTS.md`
Résumé session : `memory/project_session_resume.md`

| Couche | Statut |
|--------|--------|
| ML Training | **DONE** — Champion MLP(32,16) stacking 0.5530, ECE_draw 0.0016. AG ÉLIMINÉ (ADR-011). |
| API FastAPI | **DONE Plan 3 T22+T23 (dette résorbée 2026-04-28/29)** — /compose + /recompose wired. Stacking pipeline E2E champion mode FULL (D-P3-13 résorbée). 11 FFE rules (ADR-012). composer.py legacy supprimé (D5). seed exposé (D-P2-04). MC fail-fast (D-P3-12). Wilcoxon SOTA McNemar pivot (ADR-017, D-P3-18). VerifiabilityClassifier 3-class (D-P3-09). |
| ALI prédiction adverse | **DONE-SOTA Plan 2** — Monte Carlo hybride (10 TopK + 10 MC LHS+antithetic), copule gaussienne (Sklar 1959) ; fallback Elo si pas d'opponent_club_id |
| CE multi-équipe | **FALLBACK** — Tri Elo + E[score] (Phase 4 = OR-Tools) |
| Deploy SaaS multi-tenant | MANQUANT (Phase 5 — scope étendu 2026-04-19) |
| SDK Python / UI standalone | MANQUANT (Phase 6 nouvelle) |
| Monétisation + SLA | MANQUANT (Phase 7 nouvelle) |

### DETTE TECHNIQUE VISIBLE (ne pas laisser enfouie)

Dette ouverte après Phase 2 — détail et plan de résorption : `memory/project_debt_current.md`

| # | Dette | Origine | Phase résorption |
|---|-------|---------|-------------------|
| ~~D1~~ | ~~Joueurs Elo 1500 stub~~ | ~~Phase 2~~ | **RESOLUE Plan 1+2** (PlayerPoolLoader + ScenarioGenerator wirés) |
| ~~D2~~ | ~~ALI tri Elo 1 scénario~~ | ~~Phase 2~~ | **RESOLUE Plan 2** (ScenarioGenerator 10 TopK + 10 MC SOTA, commit c4a8154) |
| D3 | Jeunes (J02) non supportés (age_min/age_max ignorés) | Phase 2 | **Phase 3.5** (post-ALI de base) |
| D4 | Coupes configs dispo mais non implémentées | Phase 2 | **Phase 3.5** |
| ~~D5~~ | ~~`services/composer.py` legacy mort-vivant~~ | ~~Phase 2~~ | **RESOLUE Plan 3 T23** (suppression 293L, commit cdf6a7c, 2026-04-28) |
| D6 | DVC / versioning artefacts ML | Historique | **PARTIAL T24** (dvc.yaml 2 stages refit MLP + backtest, DAG opérationnel) — remote DVC + training Kaggle = Phase 5 |
| D7 | Pas de lien commit git ↔ version Kaggle dataset ↔ version kernel | Historique | **Phase 5** (non couvert par T24) |
| D8 | ALI fairness/robustness breakdown (genre, taille club, niveau) + stress Elo | Phase 3 (scope) | **Phase 3.5 STRICT** (bloquant Phase 4) |
| D9 | Adaptive Importance Sampling + drift monitoring prod (ratio TopK:MC dynamique) | Phase 3 (brainstorm finding) | **Phase 5+** (après volume data prod) |
| ~~D-P2-02~~ | ~~VerifiabilityClassifier injecté mais pas consommé~~ | ~~Plan 2 peer review~~ | **RESOLUE Plan 2 (commit 0656fdf, 2026-04-19) — partition_rules wired** |
| ~~D-P2-03~~ | ~~`_EXPECTED_SCENARIOS=20` hardcodé~~ | ~~Plan 2 peer review~~ | **RESOLUE Plan 3 T23** (invariant ADR-014 + error message clarifié, commit cdf6a7c) |
| ~~D-P2-04~~ | ~~seed=42 fixe dans generator~~ | ~~Plan 2 peer review~~ | **RESOLUE Plan 3 T23** (ComposeRequest.seed exposé + ADR-014 §Determinism, commit cdf6a7c) |
| ~~D-P3-11~~ | ~~Suppression `services/ffe_rules.py` legacy~~ | ~~Plan 2 (différée)~~ | **RESOLUE Plan 2** (migration RuleEngine complète 10 articles, ADR-015 SUPERSEDED) |
| ~~D-P3-12~~ | ~~MonteCarloSampler `_u_to_presence` fallback résiduel~~ | ~~Plan 2 Task 6~~ | **RESOLUE Plan 3 T23** (RuntimeError fail-fast ISO 24029, commit cdf6a7c) |
| ~~D-P3-13~~ | ~~MLP champion artifact `mlp_meta_learner.joblib` absent (FALLBACK silencieux)~~ | ~~Phase 2 stacking deploy~~ | **RESOLUE Plan 3 T22.0** (refit OOF, ECE_draw 0.001648 reproduit, lineage SHA256) |
| ~~D-P3-01~~ | ~~sync_ffe_rules.py chemin Windows-hardcoded~~ | ~~Plan 1 peer review~~ | **RESOLUE T22 (env var CHESS_APP_RULES_DIR + graceful skip CI)** |
| ~~D-P3-02~~ | ~~ALIDataCache mutability non documentée~~ | ~~Plan 1 peer review~~ | **RESOLUE T22 (design délibéré : lifespan reload, docstring updated)** |
| ~~D-P3-05..09~~ | ~~Nitpicks Plan 2 peer review (imports, O(N²), Pydantic schema, eligibility, classification 3.7)~~ | ~~Plan 2 peer review~~ | **RESOLUE T22 (cleanup imports + O(N log N) + Pydantic + Literal 'out_of_scope')** |
| ~~D-P2-05~~ | ~~verify_plan2_dod.sh ne bloque pas P2G06/P2G13~~ | ~~Plan 2~~ | **RESOLUE T22 par design (--full opt-in, pre-push pytest-cov 70% global suffit)** |
| ~~D-P3-18~~ | ~~Redéfinir `ali_correct` recall≥0.50 (quick-fix statistical hacking rejeté)~~ | ~~Phase 3 T22 finding~~ | **RESOLUE 2026-04-28 ADR-017 pivot McNemar→Wilcoxon SOTA (Demšar 2006)** |
| **D-P3-19** | **ALI multi-équipes joint conditionné — CE-adverse miroir SOTA Approche A (R-ALI-06)** | **T22 review post-mortem (CRITICAL)** | **Phase 4a (NEW upstream Phase 4b) — prérequis structurel gates absolus** |
| ~~D10~~ | ~~Sync chess-app JSON~~ | ~~Phase 3~~ | **RESOLUE Plan 1** (2026-04-19, commits ff94a19 + 1a2445c) |
| D11 | Completeness audit : PDF FFE → chess-app JSON (toutes règles capturées ?) | Phase 3 finding | **Phase ultérieure** (tâche NLP, pocket-arbiter stale) |
| ~~D12~~ | ~~Autoregressive streak~~ | ~~Audit F3~~ | **REMONTÉ PHASE 3** (F3 intégré, option B) |
| D13 | `zone_enjeu` consommé par ALI (contexte accession/maintien) | Audit SOTA F4 (Phase 3) | **Phase 4+** (couplé CE OR-Tools, dépendance structurelle) |
| ~~D14~~ | ~~LHS / antithetic MC~~ | ~~Audit F5~~ | **REMONTÉ PHASE 3** (F5 intégré, option B) |
| D15 | Conformal prediction bout-en-bout (IC sur E[score]) | Audit SOTA F6 | **Phase 4+** (couplé CE multi-objectif, dépendance structurelle) |
| ~~D16~~ | ~~Copule gaussienne~~ | ~~Section 3~~ | **REMONTÉ PHASE 3** (F6 remplace Gibbs, option B) |

**Règle :** chaque phase résorbe sa dette listée ici. Interdiction de marquer une phase "DONE" sans avoir soit (a) résorbé sa dette, soit (b) reclassé explicitement la dette avec nouvelle phase cible.

## TRAINING RULES (V8/V9)

**LIRE `config/MODEL_SPECS.md` AVANT TOUTE ACTION ML.** C'est la source de vérité per-model.

- **NE JAMAIS** appliquer les mêmes hyperparams aux 3 modèles sans vérifier MODEL_SPECS (ADR-008)
- **NE JAMAIS** entraîner sans residual learning (Elo = baseline forte)
- **NE JAMAIS** déclarer champion sans Optuna
- **NE JAMAIS** `optuna.integration` — v4.0+ = `optuna_integration`
- **NE JAMAIS** TreeSHAP sur test complet — subsample 20K
- **NE JAMAIS** CatBoost init_model + Pool(baseline=)
- **TOUJOURS** alpha per-model : LGB=0.1, XGB=0.5, CB=0.3 (ADR-008, 590 configs)
- **TOUJOURS** init_scores AVANT filtrage features
- **TOUJOURS** rsm=0.7 pour CatBoost >50 features (Grid v2 : 0.7>0.45>0.3)
- **TOUJOURS** Dirichlet calibration (Kull 2019) — draw = 45% variance E[score]
- **TOUJOURS** mesurer ECE draw + draw_bias (pas juste logloss)
- **TOUJOURS** skill kernel-push (9 étapes) AVANT chaque push Kaggle
- **TOUJOURS** quality gates AVANT SHAP
- **TOUJOURS** vérifier dataset Kaggle contient fichiers importés
- **TOUJOURS** télécharger+vérifier fichiers critiques depuis Kaggle AVANT push kernel
- **TOUJOURS** filtrer `_ALICE_KEYS` (init_score_alpha) avant constructeurs ML
- **TOUJOURS** SQLite storage Optuna

Inference : `compute_elo_baseline → init_scores → *= alpha_per_model → predict_with_init`

**ADR** : `docs/architecture/DECISIONS.md` (ADR-001 à ADR-012)

## COMMANDES

```bash
make all           # Validation complète
make refresh-data  # Sync + parse + validate + features
make test-cov      # Tests + coverage
```

## DONNÉES

HuggingFace : `Pierrax/ffe-history`. Compte : Pierrax.
Local : `data/echiquiers.parquet`, `data/joueurs.parquet`.

## DOCUMENTATION & RULES

**Index 65 docs : `.claude/rules/docs-index.md`**

Priorité :
- `docs/superpowers/specs/2026-04-07-optuna-v9-pipeline-design.md` — V9 Optuna
- `docs/project/V8_MODEL_COMPARISON.md` — Comparaison modèles
- `docs/requirements/QUALITY_GATES.md` — Gates F1-F12 / T1-T12

Rules files :
- `.claude/rules/docs-index.md` — Index docs complet
- `.claude/rules/v8-training-findings.md` — Historique V8
- `.claude/rules/kaggle-architecture.md` — Contraintes Kaggle + kernels
- `.claude/rules/project-structure.md` — Scripts, app, hooks

## ARCHITECTURE

`Vercel (chess-app) → HTTPS → Oracle VM (FastAPI + ML, 24GB ARM)`
Spec : `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md`


## Wiki

Syntheses wiki : `C:\Dev\wiki\topics\ml\` et `C:\Dev\wiki\entities\alice-engine.md`
Guide et outils de recherche : `C:\Dev\wiki\wiki-guide.md`
