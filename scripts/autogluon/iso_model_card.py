"""Generation de Model Card - ISO 42001.

Ce module contient la generation de Model Card.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (Model Card)
- ISO/IEC 5055:2021 - Code Quality (<80 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.autogluon.iso_types import ISO42001ModelCard

if TYPE_CHECKING:
    from scripts.autogluon.trainer import AutoGluonTrainingResult

logger = logging.getLogger(__name__)


def generate_model_card(
    result: AutoGluonTrainingResult,
    output_path: Path | None = None,
) -> ISO42001ModelCard:
    """Genere une Model Card ISO 42001 pour le modele AutoGluon.

    Args:
    ----
        result: Resultat d'entrainement AutoGluon
        output_path: Chemin de sauvegarde (optionnel)

    Returns:
    -------
        ISO42001ModelCard complete

    ISO 42001: Documentation obligatoire du modele.
    """
    # Extraire l'importance des features si disponible
    feature_importance = _extract_feature_importance(result)

    model_card = ISO42001ModelCard(
        model_id=str(result.model_path.name),
        model_name=f"AutoGluon_{result.config.presets}",
        version="1.0.0",
        created_at=datetime.now().isoformat(),
        training_data_hash=result.data_hash,
        hyperparameters={
            "presets": result.config.presets,
            "time_limit": result.config.time_limit,
            "eval_metric": result.config.eval_metric,
            "num_bag_folds": result.config.num_bag_folds,
            "num_stack_levels": result.config.num_stack_levels,
        },
        metrics=result.metrics,
        best_model=result.best_model,
        num_models_trained=int(result.metrics.get("num_models", 0)),
        feature_importance=feature_importance,
    )

    if output_path:
        _save_model_card(model_card, output_path)

    return model_card


def _extract_feature_importance(result: AutoGluonTrainingResult) -> dict[str, float]:
    """Extrait l'importance des features."""
    try:
        fi_df = result.predictor.feature_importance()
        return fi_df["importance"].to_dict()
    except Exception:
        logger.debug("Feature importance not available")
        return {}


def _save_model_card(model_card: ISO42001ModelCard, output_path: Path) -> None:
    """Sauvegarde la Model Card."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(model_card), f, indent=2, default=str)
    logger.info(f"Model card saved to {output_path}")
