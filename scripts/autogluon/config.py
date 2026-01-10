"""Module: scripts/autogluon/config.py - AutoGluon Configuration.

Document ID: ALICE-MOD-AUTOGLUON-CONFIG-001
Version: 1.0.0

Configuration pour AutoGluon TabularPredictor.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (parametres documentes)
- ISO/IEC 5055:2021 - Code Quality (<100 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-10
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class AutoGluonConfig:
    """Configuration AutoGluon ISO 42001 compliant.

    Attributes
    ----------
        presets: Niveau de qualite ('medium_quality', 'high_quality',
                 'best_quality', 'extreme')
        time_limit: Temps maximum d'entrainement en secondes
        eval_metric: Metrique d'evaluation (roc_auc, accuracy, f1, etc.)
        num_bag_folds: Nombre de folds pour bagging
        num_stack_levels: Nombre de niveaux de stacking
        models_include: Liste des modeles a inclure
        random_seed: Graine pour reproductibilite
        verbosity: Niveau de verbosity (0-4)
    """

    presets: str = "extreme"
    time_limit: int = 3600
    eval_metric: str = "roc_auc"
    num_bag_folds: int = 5
    num_stack_levels: int = 2
    models_include: list[str] = field(
        default_factory=lambda: [
            "TabPFN",
            "CatBoost",
            "XGBoost",
            "LightGBM",
            "NeuralNetFastAI",
            "RandomForest",
        ]
    )
    tabpfn_enabled: bool = True
    tabpfn_n_ensemble: int = 16
    random_seed: int = 42
    verbosity: int = 2
    output_dir: Path = field(default_factory=lambda: Path("models/autogluon"))

    def to_fit_kwargs(self) -> dict[str, Any]:
        """Convertit la config en kwargs pour TabularPredictor.fit().

        Returns
        -------
            Dict des arguments pour .fit()
        """
        return {
            "presets": self.presets,
            "time_limit": self.time_limit,
            "num_bag_folds": self.num_bag_folds,
            "num_stack_levels": self.num_stack_levels,
            "verbosity": self.verbosity,
        }


def load_autogluon_config(
    config_path: str | Path = "config/hyperparameters.yaml",
) -> AutoGluonConfig:
    """Charge la configuration AutoGluon depuis le fichier YAML.

    Args:
    ----
        config_path: Chemin vers le fichier de configuration

    Returns:
    -------
        AutoGluonConfig avec les valeurs du fichier

    ISO 42001: Configuration externalisee et versionnable.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return AutoGluonConfig()

    with open(config_path, encoding="utf-8") as f:
        full_config = yaml.safe_load(f)

    ag_config = full_config.get("autogluon", {})

    return AutoGluonConfig(
        presets=ag_config.get("presets", "extreme"),
        time_limit=ag_config.get("time_limit", 3600),
        eval_metric=ag_config.get("eval_metric", "roc_auc"),
        num_bag_folds=ag_config.get("num_bag_folds", 5),
        num_stack_levels=ag_config.get("num_stack_levels", 2),
        models_include=ag_config.get("models", {}).get("include", []),
        tabpfn_enabled=ag_config.get("tabpfn", {}).get("enabled", True),
        tabpfn_n_ensemble=ag_config.get("tabpfn", {}).get("n_ensemble_configurations", 16),
        random_seed=ag_config.get("random_seed", 42),
        verbosity=ag_config.get("verbosity", 2),
    )
