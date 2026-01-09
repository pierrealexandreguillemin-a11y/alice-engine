"""Dataclasses resultats ML - ISO 5055.

Ce module definit les dataclasses pour les resultats:
- ModelMetrics: Metriques d'un modele
- TrainingResult: Resultat d'entrainement
- StackingResult: Resultat du stacking
- ModelRegistry: Registre des modeles

Conformite:
- ISO/IEC 5055 (Code Quality)
- Ruff ANN401 (no Any)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from scripts.ml_types.protocols import MLClassifier


@dataclass
class ModelMetrics:
    """Metriques d'un modele."""

    auc_roc: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    log_loss: float
    train_time_s: float = 0.0
    test_auc: float = 0.0
    test_accuracy: float = 0.0
    test_f1: float = 0.0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_positives: int = 0

    def to_dict(self) -> dict[str, float | int]:
        """Convertit en dictionnaire."""
        return {
            "auc_roc": self.auc_roc,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "log_loss": self.log_loss,
            "train_time_s": self.train_time_s,
            "test_auc": self.test_auc,
            "test_accuracy": self.test_accuracy,
            "test_f1": self.test_f1,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_positives": self.true_positives,
        }


@dataclass
class TrainingResult:
    """Resultat d'entrainement d'un modele."""

    model: MLClassifier | None
    train_time: float
    metrics: ModelMetrics

    @property
    def is_valid(self) -> bool:
        """Verifie si l'entrainement a reussi."""
        return self.model is not None


@dataclass
class StackingResult:
    """Resultat du stacking ensemble."""

    meta_model: MLClassifier
    oof_predictions: NDArray[np.float64]
    test_predictions: NDArray[np.float64]
    stacking_test_proba: NDArray[np.float64]
    model_weights: dict[str, float]
    model_names: list[str]
    single_model_metrics: dict[str, dict[str, float]]
    stacking_metrics: dict[str, float]
    best_single_name: str
    best_single_auc: float
    use_stacking: bool
    n_folds: int


@dataclass
class ModelRegistry:
    """Registre des modeles sauvegardes."""

    version: str
    version_dir: Path
    models: dict[str, Path] = field(default_factory=dict)
    metadata_path: Path | None = None
    encoders_path: Path | None = None
