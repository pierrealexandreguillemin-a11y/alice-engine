"""Sauvegarde de rapport de comparaison - ISO 42001.

Ce module contient les fonctions de sauvegarde de rapport.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System
- ISO/IEC 5055:2021 - Code Quality (<80 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from scripts.comparison.types import ModelComparison

logger = logging.getLogger(__name__)


def _convert_to_native(obj: Any) -> Any:
    """Convertit les types numpy en types Python natifs."""
    if isinstance(obj, np.integer | np.floating):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_convert_to_native(v) for v in obj]
    return obj


def save_comparison_report(
    comparison: ModelComparison,
    output_path: Path,
) -> None:
    """Sauvegarde le rapport de comparaison.

    Args:
    ----
        comparison: Resultat de la comparaison
        output_path: Chemin du fichier JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "model_a_name": comparison.model_a_name,
        "model_b_name": comparison.model_b_name,
        "winner": comparison.winner,
        "practical_significance": bool(comparison.practical_significance),
        "recommendation": comparison.recommendation,
        "timestamp": comparison.timestamp,
        "mcnemar": {
            "statistic": float(comparison.mcnemar_result.statistic),
            "p_value": float(comparison.mcnemar_result.p_value),
            "significant": bool(comparison.mcnemar_result.significant),
            "effect_size": float(comparison.mcnemar_result.effect_size),
            "confidence_interval": [
                float(x) for x in comparison.mcnemar_result.confidence_interval
            ],
        },
        "metrics_a": _convert_to_native(comparison.metrics_a),
        "metrics_b": _convert_to_native(comparison.metrics_b),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Comparison report saved to {output_path}")
