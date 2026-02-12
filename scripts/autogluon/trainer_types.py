"""Types pour AutoGluon Trainer - ISO 42001.

Ce module contient les dataclasses de résultat d'entraînement.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (Traçabilité)
- ISO/IEC 5055:2021 - Code Quality (SRP)

Document ID: ALICE-MOD-AUTOGLUON-TRAINER-TYPES-001
Version: 1.0.0
Author: ALICE Engine Team
Last Updated: 2026-02-12
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from autogluon.tabular import TabularPredictor

    from scripts.autogluon.config import AutoGluonConfig


@dataclass
class AutoGluonTrainingResult:
    """Resultat d'entrainement AutoGluon.

    Attributes
    ----------
        predictor: TabularPredictor entraine
        train_time: Temps d'entrainement en secondes
        leaderboard: DataFrame du classement des modeles
        best_model: Nom du meilleur modele
        model_path: Chemin de sauvegarde du modele
        data_hash: Hash des donnees d'entrainement (tracabilite)
        config: Configuration utilisee
    """

    predictor: TabularPredictor
    train_time: float
    leaderboard: pd.DataFrame
    best_model: str
    model_path: Path
    data_hash: str
    config: AutoGluonConfig
    metrics: dict[str, float] = field(default_factory=dict)
