"""Refinement helper functions for iterative ISO compliance.

Extracted from iterative_refinement.py for SRP compliance (ISO 5055).

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management (Continuous Improvement)
- ISO/IEC TR 24027:2021 - Bias Treatment Strategies

Author: ALICE Engine Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts.agents.iterative_refinement import IterativeRefinement, RefinementResult

logger = logging.getLogger(__name__)


def should_reweight(group_analyses: list[dict[str, Any]]) -> bool:
    """Determine si le reweighting est approprie.

    Reweighting si desequilibre > 2x entre groupes.
    """
    if not group_analyses:
        return False

    counts = [g.get("sample_count", 0) for g in group_analyses]
    if not counts:
        return False

    max_count = max(counts)
    min_count = min(c for c in counts if c > 0)
    return max_count / min_count > 2


def calculate_reweighting(group_analyses: list[dict[str, Any]]) -> dict[str, float]:
    """Calcule les poids de reweighting par groupe.

    Poids inversement proportionnel a la taille du groupe.
    """
    if not group_analyses:
        return {}

    total = sum(g.get("sample_count", 1) for g in group_analyses)
    n_groups = len(group_analyses)

    weights = {}
    for g in group_analyses:
        name = g.get("group_name", "unknown")
        count = g.get("sample_count", 1)
        weights[name] = (total / n_groups) / count if count > 0 else 1.0

    return weights


def run_refinement_loop(
    refinement: IterativeRefinement,
    validation_fn: Any,
    train_fn: Any,  # noqa: ARG001
    max_iterations: int | None = None,
) -> list[RefinementResult]:
    """Execute une boucle de raffinement iteratif.

    Args:
    ----
        refinement: Instance IterativeRefinement
        validation_fn: Fonction de validation ISO
        train_fn: Fonction d'entrainement
        max_iterations: Nombre max d'iterations

    Returns:
    -------
        Liste des resultats de raffinement
    """
    from scripts.agents.iterative_refinement import RefinementAction

    iterations = max_iterations or refinement.max_iterations
    results: list[RefinementResult] = []

    for i in range(iterations):
        logger.info(f"Refinement iteration {i + 1}/{iterations}")

        report = validation_fn()

        fairness_result = refinement.refine_fairness(report.get("fairness", {}))
        results.append(fairness_result)

        if fairness_result.action == RefinementAction.NONE:
            logger.info("Fairness compliant, checking robustness...")

            robustness_result = refinement.refine_robustness(report.get("robustness", {}))
            results.append(robustness_result)

            if robustness_result.action == RefinementAction.NONE:
                logger.info("All validations passed!")
                break

        if fairness_result.action == RefinementAction.BLOCK_DEPLOYMENT:
            logger.error("Critical failure, stopping refinement")
            break

        if fairness_result.action in (RefinementAction.REWEIGHT, RefinementAction.RETRAIN):
            logger.info(f"Applying {fairness_result.action.value}...")

    return results
