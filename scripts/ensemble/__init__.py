"""Package Ensemble Stacking - ISO 5055.

Ce package contient les modules pour l'ensemble stacking:
- types.py: Dataclasses (StackingMetrics, StackingResult)
- model_factory.py: Creation de modeles
- oof.py: Calcul OOF predictions
- voting.py: Soft voting
- stacking.py: Logique de stacking
- save.py: Sauvegarde avec conformite ISO
"""

from scripts.ensemble.model_factory import (
    create_catboost_model,
    create_lightgbm_model,
    create_model_by_name,
    create_xgboost_model,
)
from scripts.ensemble.oof import compute_oof_for_model
from scripts.ensemble.save import save_stacking_ensemble
from scripts.ensemble.stacking import create_stacking_ensemble
from scripts.ensemble.types import StackingMetrics, StackingResult
from scripts.ensemble.voting import MODEL_NAMES, compute_soft_voting

__all__ = [
    # Types
    "StackingMetrics",
    "StackingResult",
    # Model factory
    "create_catboost_model",
    "create_xgboost_model",
    "create_lightgbm_model",
    "create_model_by_name",
    # OOF
    "compute_oof_for_model",
    # Voting
    "compute_soft_voting",
    "MODEL_NAMES",
    # Stacking
    "create_stacking_ensemble",
    # Save
    "save_stacking_ensemble",
]
