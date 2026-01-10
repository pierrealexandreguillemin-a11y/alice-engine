# Plan d'Implémentation AutoGluon - ALICE Engine

Document ID: ALICE-PLAN-AUTOGLUON-001
Version: 1.0.0
Date: 2026-01-10
Status: APPROVED

## 1. Objectif

Intégrer AutoGluon 1.5 avec TabPFN-2.5 pour comparer avec l'ensemble CatBoost/XGBoost/LightGBM existant, avec validation statistique McNemar 5×2cv et conformité ISO 42001/24029/24027.

## 2. Justification Technique

### 2.1 Pourquoi AutoGluon?
- **Multi-layer stacking**: Ensemble automatique de 15+ algorithmes
- **TabPFN-2.5**: Foundation model pré-entraîné (5000× plus rapide)
- **Preset 'extreme'**: Inclut TabPFN, CatBoost, XGBoost, LightGBM, Neural Networks
- **Benchmark**: Médiane rang 3 sur OpenML (TabPFN rang 2)

### 2.2 TabPFN-2.5 Avantages
- Zero-shot learning (pas d'entraînement requis)
- Jusqu'à 50K échantillons supportés
- Incertitude calibrée native
- Idéal pour données tabulaires structurées

## 3. Architecture

```
scripts/
├── autogluon/
│   ├── __init__.py           # Exports package
│   ├── trainer.py            # Train AutoGluon (~300 lignes)
│   ├── predictor_wrapper.py  # Wrapper sklearn (~200 lignes)
│   ├── config.py             # Configuration (~100 lignes)
│   └── iso_compliance.py     # ISO 42001/24029 (~250 lignes)
├── comparison/
│   ├── __init__.py           # Exports package
│   ├── mcnemar_test.py       # McNemar 5×2cv (~250 lignes)
│   └── statistical_comparison.py  # Pipeline comparaison (~300 lignes)
```

## 4. Dépendances

```
# requirements.txt additions
autogluon>=1.5.0
tabpfn>=2.5.0
torch>=2.0.0
mlflow>=2.10.0
scipy>=1.11.0  # McNemar test
```

## 5. Configuration

```yaml
# config/hyperparameters.yaml - section autogluon
autogluon:
  presets: "extreme"
  time_limit: 3600
  num_bag_folds: 5
  num_stack_levels: 2
  verbosity: 2
  eval_metric: "accuracy"

  tabpfn:
    enabled: true
    n_ensemble_configurations: 16

  models:
    include:
      - TabPFN
      - CatBoost
      - XGBoost
      - LightGBM
      - NeuralNetFastAI
      - RandomForest

statistical_comparison:
  method: "mcnemar_5x2cv"
  alpha: 0.05
  n_splits: 5
  bootstrap_samples: 1000
  effect_size_threshold: 0.05
```

## 6. Implémentation Détaillée

### 6.1 Module trainer.py

```python
"""AutoGluon Trainer - ISO 42001 Compliant."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from autogluon.tabular import TabularPredictor

if TYPE_CHECKING:
    from scripts.ml_types.metrics import ModelMetrics

logger = logging.getLogger(__name__)


@dataclass
class AutoGluonTrainingResult:
    """Résultat d'entraînement AutoGluon."""

    predictor: TabularPredictor
    train_time: float
    metrics: ModelMetrics
    leaderboard: pd.DataFrame
    best_model: str
    model_path: Path


def train_autogluon(
    train_data: pd.DataFrame,
    label: str,
    valid_data: pd.DataFrame | None = None,
    config: dict | None = None,
    output_dir: Path | None = None,
) -> AutoGluonTrainingResult:
    """Entraîne un modèle AutoGluon.

    ISO 42001: Traçabilité complète, reproductibilité garantie.
    """
    ...
```

### 6.2 Module mcnemar_test.py

```python
"""McNemar 5×2cv Test - ISO 24029 Statistical Validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


@dataclass
class McNemarResult:
    """Résultat du test McNemar 5×2cv."""

    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    confidence_interval: tuple[float, float]
    model_a_mean_accuracy: float
    model_b_mean_accuracy: float
    winner: str | None


def mcnemar_5x2cv_test(
    model_a_fit: Callable,
    model_b_fit: Callable,
    model_a_predict: Callable,
    model_b_predict: Callable,
    X: NDArray[np.float64] | pd.DataFrame,
    y: NDArray[np.int64] | pd.Series,
    n_splits: int = 5,
    alpha: float = 0.05,
    random_state: int = 42,
) -> McNemarResult:
    """Test McNemar 5×2cv (Dietterich 1998).

    Méthode recommandée pour comparer deux classifieurs.
    Évite les biais du t-test sur données non-indépendantes.

    ISO 24029: Validation statistique robuste.
    """
    ...
```

### 6.3 Module iso_compliance.py

```python
"""ISO Compliance for AutoGluon - 42001/24029/24027."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ISO42001ModelCard:
    """Model Card conforme ISO 42001."""

    model_id: str
    model_name: str
    version: str
    created_at: str
    training_data_hash: str
    hyperparameters: dict
    metrics: dict
    intended_use: str
    limitations: str
    ethical_considerations: str


@dataclass
class ISO24029RobustnessReport:
    """Rapport robustesse conforme ISO 24029."""

    model_id: str
    noise_tolerance: float
    adversarial_robustness: float
    distribution_shift_score: float
    confidence_calibration: float


@dataclass
class ISO24027BiasReport:
    """Rapport biais conforme ISO 24027."""

    model_id: str
    demographic_parity: float
    equalized_odds: float
    calibration_by_group: dict
```

## 7. Tests (80 tests minimum)

### 7.1 test_autogluon_trainer.py (25 tests)
- test_train_basic
- test_train_with_validation
- test_train_preset_extreme
- test_tabpfn_integration
- test_leaderboard_generation
- test_best_model_selection
- test_model_persistence
- test_reproducibility
- test_feature_importance
- test_prediction_probabilities
- ...

### 7.2 test_mcnemar_comparison.py (15 tests)
- test_mcnemar_basic
- test_mcnemar_5x2cv_structure
- test_p_value_calculation
- test_effect_size
- test_confidence_intervals
- test_winner_determination
- test_tie_handling
- test_statistical_power
- ...

### 7.3 test_autogluon_iso_compliance.py (20 tests)
- test_model_card_generation
- test_model_card_fields
- test_robustness_report
- test_bias_report
- test_mlflow_logging
- test_reproducibility_hash
- test_audit_trail
- ...

## 8. Conformité ISO

### 8.1 ISO 42001:2023 - AI Management System
- [x] Model Card obligatoire
- [x] Traçabilité MLflow
- [x] Reproductibilité (seeds, versioning)
- [x] Documentation hyperparamètres

### 8.2 ISO 24029:2021 - Neural Network Robustness
- [x] Tests perturbation (bruit)
- [x] Robustesse adversariale
- [x] Calibration des confiances
- [x] Distribution shift

### 8.3 ISO 24027:2021 - Bias Detection
- [x] Demographic parity
- [x] Equalized odds
- [x] Calibration par groupe
- [x] Rapport de biais

## 9. Intégration MLflow

```python
# Logging automatique
mlflow.autogluon.autolog()

# Experiment tracking
with mlflow.start_run(run_name="autogluon_extreme"):
    result = train_autogluon(...)
    mlflow.log_metrics(result.metrics)
    mlflow.log_artifact(model_card_path)
```

## 10. Validation Finale

### 10.1 Critères d'Acceptation
- [ ] 80+ tests passent
- [ ] Coverage > 80%
- [ ] McNemar test implémenté
- [ ] Model Card générée
- [ ] Rapport ISO 24029
- [ ] Intégration MLflow fonctionnelle

### 10.2 Comparaison Attendue
- AutoGluon (preset=extreme) vs Ensemble actuel
- Métriques: Accuracy, F1, AUC-ROC
- Significativité: p < 0.05 (McNemar)
- Taille d'effet: Cohen's d

## 11. Références

- Dietterich, T.G. (1998). Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms
- AutoGluon Documentation: https://auto.gluon.ai/
- TabPFN Paper: Hollmann et al. (2023)
- ISO/IEC 42001:2023 - AI Management System
- ISO/IEC 24029:2021 - Neural Network Robustness

---

**Approuvé par**: User
**Date d'approbation**: 2026-01-10
