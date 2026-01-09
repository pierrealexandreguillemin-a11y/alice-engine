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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray

# Configs
# ==============================================================================
# TYPE ALIASES (runtime-safe avec TypeAlias)
# ==============================================================================
from typing import TypeAlias

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
    StackingResult,
    TrainingResult,
)

# Features et labels (strings pour éviter import pandas/numpy à runtime)
Features: TypeAlias = "pd.DataFrame"
Labels: TypeAlias = "pd.Series"
FeaturesArray: TypeAlias = "NDArray[np.float64]"
LabelsArray: TypeAlias = "NDArray[np.int64]"

# Resultats
ModelResults: TypeAlias = dict[str, TrainingResult]

# Noms de modeles
ModelName: TypeAlias = str
VALID_MODEL_NAMES: tuple[str, ...] = ("CatBoost", "XGBoost", "LightGBM")

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
    # Type aliases
    "Features",
    "Labels",
    "FeaturesArray",
    "LabelsArray",
    "ModelResults",
    "ModelName",
    "VALID_MODEL_NAMES",
]
