"""Validation ISO complete - ISO 42001/24029/24027.

Ce module contient la validation ISO complete.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System
- ISO/IEC 24029:2021 - Neural Network Robustness
- ISO/IEC TR 24027:2021 - Bias Detection
- ISO/IEC 5055:2021 - Code Quality (<80 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.autogluon.iso_fairness import validate_fairness
from scripts.autogluon.iso_model_card import generate_model_card
from scripts.autogluon.iso_robustness import validate_robustness

if TYPE_CHECKING:
    from scripts.autogluon.trainer import AutoGluonTrainingResult

logger = logging.getLogger(__name__)


def validate_iso_compliance(
    result: AutoGluonTrainingResult,
    test_data: Any,
    protected_attribute: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Validation complete ISO 42001/24029/24027.

    Args:
    ----
        result: Resultat d'entrainement AutoGluon
        test_data: Donnees de test
        protected_attribute: Attribut protege pour test de biais
        output_dir: Repertoire de sauvegarde des rapports

    Returns:
    -------
        Dict avec Model Card et rapports de conformite

    ISO 42001/24029/24027: Validation complete obligatoire.
    """
    output_dir = output_dir or result.model_path
    output_dir = Path(output_dir)

    # Model Card (ISO 42001)
    model_card = generate_model_card(result, output_path=output_dir / "model_card.json")

    # Robustesse (ISO 24029)
    robustness = validate_robustness(result.predictor, test_data)
    _save_report(robustness, output_dir / "robustness_report.json")

    # Biais (ISO 24027) - si attribut protege specifie
    fairness = None
    if protected_attribute:
        fairness = validate_fairness(result.predictor, test_data, protected_attribute)
        _save_report(fairness, output_dir / "fairness_report.json")

    logger.info(f"ISO compliance reports saved to {output_dir}")

    return {
        "model_card": model_card,
        "robustness": robustness,
        "fairness": fairness,
        "compliant": _is_compliant(robustness, fairness),
    }


def _save_report(report: Any, path: Path) -> None:
    """Sauvegarde un rapport JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)


def _is_compliant(robustness: Any, fairness: Any) -> bool:
    """Determine si le modele est conforme."""
    if robustness.status == "FRAGILE":
        return False
    if fairness is not None and fairness.status == "CRITICAL":
        return False
    return True
