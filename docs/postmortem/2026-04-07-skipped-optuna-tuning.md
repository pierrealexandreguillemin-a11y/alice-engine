# Postmortem: Optuna Hyperparameter Tuning Skipped

**Date:** 2026-04-07
**Severity:** HIGH — Non-conformité pipeline ML documenté
**Status:** OPEN — Decision requise

---

## Résumé

L'étape Optuna hyperparameter tuning (Phase 2 de la méthodologie ML) n'a
jamais été exécutée sur les modèles V8. Les 3 modèles "convergés ALL PASS"
utilisent des hyperparamètres tunés manuellement sur ~18 itérations Kaggle.

L'évaluation de stacking (2026-04-07) a été réalisée sur ces modèles
non-optimisés avec une méthodologie incorrecte (training sur valid set au
lieu de OOF, hyperparamètres MLP arbitraires).

## Chronologie

| Date | Event |
|------|-------|
| 2026-01-10 | Méthodologie ML documentée: Optuna = Phase 2, entre Features et Training |
| 2026-01-11 | Code Optuna écrit: optuna_core.py, optuna_objectives.py, search spaces |
| 2026-03-21 | V8 MultiClass lancé — hyperparams manuels dans hyperparameters.yaml |
| 2026-03-22→04-05 | 18 itérations Kaggle: hyperparams ajustés par essai-erreur |
| 2026-04-06 | V8 Milestone déclaré "Phase 1 complete" — Optuna jamais exécuté |
| 2026-04-07 | Stacking évalué avec méthodologie incorrecte sur modèles non-optimisés |
| 2026-04-07 | Audit révèle le gap — ce postmortem |

## Ce qui existe (prêt mais non utilisé)

| Composant | Fichier | Status |
|-----------|---------|--------|
| Optuna core | `scripts/training/optuna_core.py` | Code prêt, jamais run V8 |
| Objectives CatBoost/XGB/LGB | `scripts/training/optuna_objectives.py` | Code prêt |
| Search spaces | `config/hyperparameters.yaml` section `optuna:` | Configuré mais outdated |
| Stacking OOF | `scripts/ensemble/stacking.py` | Code prêt, binaire (pas multiclass) |
| Meta-learner config | `config/hyperparameters.yaml` section `stacking:` | Defaults sklearn |
| Tests | `tests/training_optuna/` | 5 fichiers test, passent |

## Ce qui manque

1. **Search space V8 outdated** : depth [4,6,8,10] mais V8 utilise depth=4.
   lr max 0.08 mais V8 utilise 0.005-0.03. Le search space ne reflète pas
   les learnings V8 (residual learning, 197 features).

2. **Objectifs multiclass** : `optuna_objectives.py` optimise AUC (binaire).
   V8 est multiclass 3-way (log_loss, RPS, E[score] MAE).

3. **Stacking multiclass** : `stacking.py` utilise `predict_proba(...)[:, 1]`
   (binaire). V8 produit 3 probabilités.

4. **Meta-learner Optuna** : aucun search space pour C, alpha, hidden_layers
   du meta-learner. Les valeurs sont des defaults sklearn.

5. **Exécution Optuna sur Kaggle** : 100 trials × 3 modèles nécessite un
   kernel Kaggle dédié ou HF Jobs. Non planifié.

## Impact

- **Les modèles V8 ne sont PAS optimisés** — ils passent les quality gates
  mais rien ne garantit qu'ils sont proches de l'optimum.
- **Le gain potentiel d'Optuna est INCONNU** — pas "marginal", pas "grand",
  simplement inconnu.
- **L'évaluation stacking est invalide** — méthodologie incorrecte ET sur
  modèles non-optimisés.
- **Non-conformité ISO 42001** — le choix d'hyperparamètres n'est pas
  justifié par une recherche systématique traçable.

## Root Cause

Pression pour déclarer "Phase 1 complete" et avancer sur le wiring (Phase 2).
L'étape Optuna a été implicitement reportée, sans documentation du report
ni évaluation de l'impact.

## Options

### Option A : Optuna maintenant, wiring après
- Mettre à jour search spaces pour V8 multiclass + residual learning
- Adapter optuna_objectives.py pour multiclass (log_loss, RPS)
- Run 100 trials × 3 modèles sur Kaggle CPU (~100-300h)
- Ré-entraîner avec best params → re-evaluate → stacking OOF → champion
- Wiring Phase 2 avec les modèles optimisés

### Option B : Wiring d'abord avec modèles actuels, Optuna en parallèle
- Phase 2 wiring avec XGBoost v5 actuel (fonctionnel, quality gates PASS)
- Lancer Optuna sur Kaggle en parallèle (indépendant du wiring)
- Swap modèles quand Optuna terminé (architecture le permet si bien conçu)

### Option C : Optuna meta-learner seulement, base models inchangés
- Adapter stacking.py au multiclass
- Optuna sur le meta-learner (rapide, 9 features)
- Base models gardent les hyperparams manuels actuels
- Ne résout PAS le gap de conformité ISO sur les base models

## Recommendation

Aucune. C'est une décision de priorité business/conformité qui revient
au responsable projet.

## Références

- `docs/project/METHODOLOGIE_ML_TRAINING.md` Section 2.3 — pipeline Optuna
- `config/hyperparameters.yaml` — search spaces + stacking config
- `scripts/training/optuna_core.py` — code prêt
- `scripts/ensemble/stacking.py` — stacking OOF prêt (binaire)
- scikit-learn: https://scikit-learn.org/stable/modules/grid_search.html
- ISO 42001 AI lifecycle: https://aws.amazon.com/blogs/security/ai-lifecycle-risk-management-iso-iec-420012023-for-ai-governance/
