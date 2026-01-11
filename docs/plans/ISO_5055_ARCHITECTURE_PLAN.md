# Plan 100% Conformite ISO 5055 - Architecture

Document ID: ALICE-PLAN-ISO5055-001
Version: 1.0.0
Date: 2026-01-11
Auteur: ALICE Engine Team

## P0 - CRITIQUE: Conformite Rangement Documents

**AVANT TOUTE IMPLEMENTATION:**
- [ ] Verifier structure `docs/` conforme ISO 15289
- [ ] Plans dans `docs/plans/`
- [ ] Rapports dans `docs/iso/` ou `reports/`
- [ ] Aucun fichier technique a la racine (sauf config)

---

## Resume Executif

| Categorie | Fichiers | Priorite |
|-----------|----------|----------|
| Fichiers >300 lignes | 6 | P1 |
| Fonctions rang C (11-16) | 6 | P2 |
| Modules xenon rang C | 5 | P3 |

**Objectif:** Score 90% -> 100%

---

## Phase 1: Fichiers >300 lignes (Priorite P1)

### 1.1 `scripts/comparison/statistical_comparison.py` (421 lignes)

**Split propose:**
```
scripts/comparison/
    __init__.py           # Re-export (~50 lignes)
    types.py              # ModelComparison dataclass (~40 lignes)
    core.py               # compare_models(), _compute_metrics() (~120 lignes)
    baseline.py           # compare_with_baseline() (~70 lignes)
    report.py             # save_comparison_report() (~100 lignes)
    pipeline.py           # full_comparison_pipeline() (~90 lignes)
```

### 1.2 `scripts/training/optuna_tuning.py` (337 lignes)

**Split propose:**
```
scripts/training/
    optuna_core.py        # optimize_hyperparameters() (~100 lignes)
    optuna_objectives.py  # Dispatch table 3 objectives (~200 lignes)
```

### 1.3 `scripts/autogluon/iso_compliance.py` (331 lignes)

**Split propose:**
```
scripts/autogluon/iso_compliance/
    __init__.py           # Re-export (~40 lignes)
    types.py              # 3 dataclasses (~80 lignes)
    model_card.py         # generate_model_card() (~60 lignes)
    robustness.py         # validate_robustness() (~70 lignes)
    fairness.py           # validate_fairness() (~70 lignes)
```

### 1.4 `scripts/ensemble/stacking.py` (316 lignes) - P1-low

Extraire `oof_compute.py` (~120 lignes).

### 1.5 `scripts/features/ce/transferability.py` (313 lignes) - P1-low

Dispatch table pour scenarios.

### 1.6 `scripts/model_registry/validation.py` (301 lignes) - P1-low

Extraire `retention.py` (~80 lignes).

---

## Phase 2: Fonctions complexite C (Priorite P2)

### 2.1 `_classify_urgency` (16) - features/ce/urgency.py

**Pattern: Dispatch table avec checkers**
```python
URGENCY_CHECKS = [_check_aucune, _check_critique, _check_haute]

def classify_urgency(ctx: UrgencyContext) -> str:
    for check in URGENCY_CHECKS:
        result = check(ctx)
        if result is not None:
            return result
    return "normale"
```
**Reduction:** C(16) -> B(6) + A(3) helpers

### 2.2 `save_production_models` (16) - model_registry/versioning.py

**Pattern: Fonctions step**
```python
def save_production_models(...) -> Path:
    environment = _step1_collect_environment(logger)
    data_lineage = _step2_compute_lineage(...)
    artifacts = _step3_save_artifacts(...)
    ...
```
**Reduction:** C(16) -> B(8) + A(3-4) steps

### 2.3 `extract_feature_importance` (17) - model_registry/artifacts.py

**Pattern: Strategy dispatch**
```python
IMPORTANCE_EXTRACTORS = {
    "CatBoost": _get_catboost_importance,
    "XGBoost": _get_xgboost_importance,
    "LightGBM": _get_lightgbm_importance,
}
```
**Reduction:** C(17) -> A(4) + A(3) extractors

### 2.4 `calculate_selection_patterns` (14) - features/ali/patterns.py

**Pattern: Extraction helpers**
```python
def _classify_role(taux, flexibilite, nb_echiquiers) -> str: ...
def _compute_player_pattern(joueur, group, ...) -> dict: ...
```
**Reduction:** C(14) -> B(7) + A(4) helpers

---

## Phase 3: Modules xenon rang C (Priorite P3)

| Module | Lignes | Action |
|--------|--------|--------|
| `features/advanced/elo_trajectory.py` | 139 | Extraire `_process_player_trajectory()` |
| `features/advanced/pressure.py` | 144 | Extraire `_is_decisive()`, `_classify_pressure_type()` |
| `features/ali/patterns.py` | 131 | Couvert en 2.4 |
| `features/ali/presence.py` | 132 | Extraire `_calculate_player_presence()` |
| `features/ce/urgency.py` | 177 | Couvert en 2.1 |

---

## Ordre d'implementation

```
P0 - Conformite documents (immediat)
[ ] Verifier structure docs/
[ ] Deplacer fichiers mal places

P1 - Fichiers >300 lignes
[1] statistical_comparison.py    421 -> 5 modules
[2] optuna_tuning.py            337 -> 2 modules
[3] iso_compliance.py           331 -> 5 modules
[4] stacking.py                 316 -> 2 modules
[5] transferability.py          313 -> 2 modules
[6] validation.py               301 -> 2 modules

P2 - Complexite C
[7] extract_feature_importance   C(17) -> A(4)
[8] _classify_urgency           C(16) -> B(6)
[9] save_production_models      C(16) -> B(8)
[10] calculate_selection_patterns C(14) -> B(7)

P3 - Xenon C modules
[11-14] Extraction helpers
```

---

## Validation

Apres chaque module:
```bash
# Tests
pytest tests/ --cov --tb=short

# Complexite
python -m xenon scripts/ --max-average B --max-modules B --max-absolute C

# Lignes
wc -l scripts/**/*.py | sort -rn | head -10

# Lint
python -m ruff check scripts/
```

---

## Resultat attendu

| Metrique | Avant | Apres |
|----------|-------|-------|
| Fichiers >300 lignes | 6 | 0 |
| Fonctions rang C | 18 | <10 |
| Fonctions rang D | 0 | 0 |
| Modules xenon C | 5 | 0 |
| **Score conformite** | 90% | **100%** |

---

## References ISO

- ISO/IEC 5055:2021 - Code Quality (SRP, <300 lignes)
- ISO/IEC 15289:2019 - Documentation Structure
- ISO/IEC 42010:2022 - Architecture Description
