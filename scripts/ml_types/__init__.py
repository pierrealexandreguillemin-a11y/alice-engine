"""Types stricts pour le pipeline ML ALICE.

Ce package definit les types et protocols pour garantir un typage strict
sans utilisation de `Any` dans les scripts ML.

Modules:
- protocols: Interfaces MLClassifier, CatBoost, XGBoost, LightGBM
- configs: TypedDict configurations
- results: Dataclasses resultats

Conformite:
- ISO/IEC 5055 (Code Quality)
- PEP 544 (Protocols)
- Ruff ANN401 (no Any)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Configs
from scripts.ml_types.configs import (
    CatBoostConfig,
    GlobalConfig,
    LightGBMConfig,
    MLConfig,
    StackingConfig,
    XGBoostConfig,
)

# Protocols
from scripts.ml_types.protocols import (
    CatBoostModel,
    LightGBMModel,
    MLClassifier,
    XGBoostModel,
)

# Results
from scripts.ml_types.results import (
    ModelMetrics,
    ModelRegistry,
    StackingResult,
    TrainingResult,
)

# ==============================================================================
# TYPE ALIASES
# ==============================================================================

# Features et labels
Features = pd.DataFrame
Labels = pd.Series
FeaturesArray = NDArray[np.float64]
LabelsArray = NDArray[np.int64]

# Resultats
ModelResults = dict[str, TrainingResult]

# Noms de modeles
ModelName = str
VALID_MODEL_NAMES: tuple[ModelName, ...] = ("CatBoost", "XGBoost", "LightGBM")

__all__ = [
    # Protocols
    "MLClassifier",
    "CatBoostModel",
    "XGBoostModel",
    "LightGBMModel",
    # Configs
    "CatBoostConfig",
    "XGBoostConfig",
    "LightGBMConfig",
    "StackingConfig",
    "GlobalConfig",
    "MLConfig",
    # Results
    "ModelMetrics",
    "TrainingResult",
    "StackingResult",
    "ModelRegistry",
    # Type aliases
    "Features",
    "Labels",
    "FeaturesArray",
    "LabelsArray",
    "ModelResults",
    "ModelName",
    "VALID_MODEL_NAMES",
]
