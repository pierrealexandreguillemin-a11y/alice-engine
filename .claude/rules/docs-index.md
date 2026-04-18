# Documentation Index — Alice Engine

Index complet de tous les docs du projet. Consulter AVANT de poser des questions.

## Architecture & ADR
- `docs/architecture/ARCHITECTURE.md` — Vue d'ensemble architecture
- `docs/architecture/DECISIONS.md` — Registre des decisions (ADR-001 a ADR-012, incl. ADR-011 AG YAGNI, ADR-012 FFE autonome)
- `docs/architecture/ADR-002-inference-feature-construction.md` — Feature store decision
- `docs/architecture/ADR-003-single-model-kernels.md` — Architecture 4 kernels V8

## Projet & Suivi
- `docs/project/V8_MODEL_COMPARISON.md` — **PRIORITE** Comparaison 3 modeles + stacking
- `docs/project/TRAINING_PROGRESS.md` — Suivi V8 complet, resultats v1-v18+resume
- `docs/project/METHODOLOGIE_ML_TRAINING.md` — Pipeline ML 5 etapes, Optuna Phase 2
- `docs/project/ML_EVALUATION_RESULTS.md` — Resultats evaluation
- `docs/project/CHANGELOG.md` — Historique versions
- `docs/project/ANALYSE_INITIALE_ALICE.md` — Analyse initiale
- `docs/project/BILAN_DONNEES.md` — Bilan donnees FFE
- `docs/project/BILAN_PARSING.md` — Bilan parsing HTML
- `docs/project/INSTRUCTIONS_PROJET.md` — Instructions originales
- `docs/project/PLAN_ML_TRAINING.md` — Plan training initial
- `docs/project/RAPPORT_SCRAPING.md` — Rapport scraping FFE
- `docs/bilan-v8-fe-complete.md` — Bilan FE V8 (196-220 cols)

## Requirements & Metier
- `docs/requirements/CDC_ALICE.md` — Cahier des charges
- `docs/requirements/FEATURE_DOMAIN_LOGIC.md` — **PRIORITE** Logique metier features
- `docs/requirements/FEATURE_SPECIFICATION.md` — Spec formelle features (ISO 5259)
- `docs/requirements/QUALITY_GATES.md` — Gates F1-F12 / T1-T12
- `docs/requirements/REGLES_FFE_ALICE.md` — Regles FFE applicables
- `docs/requirements/CONTEXTE_DATASET_FFE.md` — Contexte donnees FFE
- `docs/requirements/CONTEXTE_INTEGRATION.md` — Integration chess-app
- `docs/requirements/DONNEES_ALICE_ENGINE.md` — Structure donnees

## ISO Conformite
- `docs/iso/ISO_STANDARDS_REFERENCE.md` — Normes ISO applicables + mapping
- `docs/iso/IMPLEMENTATION_STATUS.md` — Statut implementation (auto-genere)
- `docs/iso/ISO_COMPLIANCE_TODOS.md` — Scores ISO reels V8
- `docs/iso/ISO_MAPPING.md` — Mapping normes-fichiers
- `docs/iso/ISO_PIPELINE_HOOKS.md` — Hooks pre-commit ISO
- `docs/iso/AI_DEVELOPMENT_DISCLOSURE.md` — LLM co-authorship (ISO 42001)
- `docs/iso/AI_POLICY.md` — Politique IA
- `docs/iso/AI_RISK_ASSESSMENT.md` — Evaluation risques IA
- `docs/iso/AI_RISK_REGISTER.md` — Registre risques
- `docs/iso/STATEMENT_OF_APPLICABILITY.md` — Declaration applicabilite
- `docs/iso/AG_ASSISTANT_ANALYSIS.md` — Analyse assistant IA

## Specs (Superpowers)
- `docs/superpowers/specs/2026-04-07-optuna-v9-pipeline-design.md` — **ACTIF** V9 Optuna (11 etapes)
- `docs/superpowers/specs/2026-04-07-phase2-serving-design.md` — Phase 2 serving (revisee)
- `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md` — Roadmap 5 phases prod
- `docs/superpowers/specs/2026-03-21-multiclass-v8-design.md` — V8 MultiClass
- `docs/superpowers/specs/2026-03-27-differential-features-design.md` — Differentiels
- `docs/superpowers/specs/2026-03-18-kaggle-cloud-training-design.md` — Kaggle training
- `docs/superpowers/specs/2026-03-17-data-refresh-pipeline-design.md` — Data refresh

## Plans (Superpowers)
- `docs/superpowers/plans/2026-04-07-optuna-v9-implementation.md` — **ACTIF** Plan Optuna
- `docs/superpowers/plans/2026-04-07-stacking-evaluation.md` — Stacking (invalide)
- `docs/superpowers/plans/2026-03-27-differential-features.md` — Differentiels
- `docs/superpowers/plans/2026-03-25-shap-feature-validation.md` — SHAP validation
- `docs/superpowers/plans/2026-03-23-residual-learning-phase1.md` — Residual Phase 1
- `docs/superpowers/plans/2026-03-21-multiclass-v8-training.md` — V8 training
- `docs/superpowers/plans/2026-03-21-multiclass-v8-features.md` — V8 features
- `docs/superpowers/plans/2026-03-18-kaggle-cloud-training.md` — Kaggle training
- `docs/superpowers/plans/2026-03-17-data-refresh-pipeline.md` — Data refresh

## Postmortems
- `docs/postmortem/2026-04-07-skipped-optuna-tuning.md` — Optuna skippe
- `docs/postmortem/2026-03-28-split-temporal-nan-features.md` — 61 features mortes
- `docs/postmortem/2026-03-25-resultat-blanc-2.0-bug.md` — 62K victoires jeunes
- `docs/postmortem/2026-03-22-training-v8-divergence.md` — V8 v1-v11 divergence
- `docs/postmortem/2026-03-21-autogluon-kaggle-postmortem.md` — AutoGluon echec (ÉLIMINÉ ADR-011)
- `docs/postmortem/2026-04-16-autogluon-v9-time-allocation-failure.md` — AutoGluon V9 echec (ÉLIMINÉ ADR-011)
- `docs/postmortem/2026-04-16-catboost-oof-snapshot-crash.md` — CB OOF snapshot bug

## Plans Legacy
- `docs/plans/ISO_5055_ARCHITECTURE_PLAN.md` — Plan architecture ISO
- `docs/plans/AUDIT_ISO_PLAN.md` — Plan audit ISO
- `docs/plans/AUTOGLUON_IMPLEMENTATION_PLAN.md` — Plan AutoGluon (ÉLIMINÉ ADR-011)

## DevOps & Operations
- `docs/development/CONTRIBUTING.md` — Guide contribution
- `docs/development/PYTHON-HOOKS-SETUP.md` — Setup hooks Python
- `docs/devops/GITHUB_ACTIONS_DISK_CLEANUP.md` — CI disk cleanup
- `docs/devops/ML_MODEL_VERSIONING_STANDARDS.md` — DVC versioning (TODO)
- `docs/operations/DEPLOIEMENT_RENDER.md` — Deploy Render
