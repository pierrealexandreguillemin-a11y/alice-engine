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
Un club a N équipes le MÊME week-end. ALICE optimise l'allocation joueurs × équipes × échiquiers.
Pipeline : ALI (prédire adversaire) → ML (P(win/draw/loss) per board) → CE (optimiser E[score] multi-équipe) → API.
Le CE calcule `E[score] = P(win) + 0.5×P(draw)`. K échiquiers par match (4-16 selon division).
Détail complet : `config/MODEL_SPECS.md` §ALICE Engine.

## ÉTAT ACTUEL (avril 2026)

**CHAMPION: MLP(32,16) 18f + temp scaling (0.5530, ECE_draw 0.0016). AG ÉLIMINÉ (ADR-011).**
Résultats HP search : `docs/project/V9_HP_SEARCH_RESULTS.md`
Résumé session : `memory/project_session_resume.md`

| Couche | Statut |
|--------|--------|
| ML Training | **DONE** — Champion MLP(32,16) stacking 0.5530, ECE_draw 0.0016. AG ÉLIMINÉ (ADR-011). |
| API FastAPI | **DONE** — /compose + /recompose wired. Stacking pipeline E2E. 11 FFE rules (ADR-012). |
| ALI prédiction adverse | **FALLBACK** — Elo ranking (Phase 3 = Monte Carlo 20 scénarios) |
| CE multi-équipe | **FALLBACK** — Tri Elo + E[score] (Phase 4 = OR-Tools) |
| Deploy Oracle VM | MANQUANT (Phase 5) |

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
