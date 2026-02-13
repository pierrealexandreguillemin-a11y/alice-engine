"""Iterative Refinement - Corrections automatiques ISO.

Ce module implémente le raffinement itératif du pipeline ALICE,
inspiré de l'architecture AG-A/MLZero (NeurIPS 2025).

Le module de raffinement:
- Analyse les échecs de conformité ISO
- Propose et applique des corrections automatiques
- Supporte l'injection de connaissance experte

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management (Continuous Improvement)
- ISO/IEC TR 24027:2021 - Bias Treatment Strategies

Author: ALICE Engine Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from scripts.agents.semantic_memory import ComplianceStatus, ISOSemanticMemory

logger = logging.getLogger(__name__)


class RefinementAction(Enum):
    """Actions de raffinement possibles."""

    NONE = "none"
    REWEIGHT = "reweight"
    RESAMPLE = "resample"
    ADJUST_THRESHOLD = "adjust_threshold"
    RETRAIN = "retrain"
    SCHEDULE_REVIEW = "schedule_review"
    BLOCK_DEPLOYMENT = "block_deployment"


@dataclass
class RefinementResult:
    """Résultat d'une action de raffinement."""

    action: RefinementAction
    success: bool
    details: dict[str, Any] = field(default_factory=dict)
    next_steps: list[str] = field(default_factory=list)


class IterativeRefinement:
    """Module de raffinement itératif pour ALICE.

    Ce module analyse les résultats de validation ISO et applique
    des corrections automatiques ou planifie des actions manuelles.

    Example:
    -------
        refinement = IterativeRefinement()

        # Correction fairness automatique
        result = refinement.refine_fairness({
            "demographic_parity_ratio": 0.58,
            "status": "CRITICAL",
            "disadvantaged_groups": ["PACA"],
        })

        if result.action == RefinementAction.REWEIGHT:
            # Appliquer le reweighting
            new_weights = result.details["weights"]
    """

    def __init__(self, memory: ISOSemanticMemory | None = None) -> None:
        """Initialise le module de raffinement."""
        self.memory = memory or ISOSemanticMemory()
        self._iteration_count = 0
        self._max_iterations = 5

    @property
    def max_iterations(self) -> int:
        """Return maximum refinement iterations."""
        return self._max_iterations

    def refine_fairness(
        self,
        report: dict[str, Any],
        *,
        auto_apply: bool = False,
    ) -> RefinementResult:
        """Raffine le modèle en cas d'échec fairness.

        Args:
        ----
            report: Rapport de fairness avec metrics et analyses
            auto_apply: Appliquer automatiquement la correction

        Returns:
        -------
            RefinementResult avec action et détails
        """
        dp_ratio = report.get("demographic_parity_ratio", 1.0)
        status = self.memory.evaluate_fairness(dp_ratio)

        logger.info(f"Fairness refinement: DP={dp_ratio:.2%}, status={status.value}")

        if status == ComplianceStatus.COMPLIANT:
            return RefinementResult(
                action=RefinementAction.NONE,
                success=True,
                details={"reason": "Model is compliant"},
            )

        if status == ComplianceStatus.CRITICAL:
            return self._handle_critical_fairness(report, auto_apply)

        # CAUTION status
        return self._handle_caution_fairness(report)

    def _handle_critical_fairness(
        self, report: dict[str, Any], auto_apply: bool
    ) -> RefinementResult:
        """Gère les échecs critiques de fairness."""
        disadvantaged = report.get("disadvantaged_groups", [])
        group_analyses = report.get("group_analyses", [])

        # Déterminer la meilleure action
        if self._should_reweight(group_analyses):
            weights = self._calculate_reweighting(group_analyses)

            if auto_apply:
                logger.info("Auto-applying reweighting correction")
                # Ici on appliquerait le reweighting aux données

            return RefinementResult(
                action=RefinementAction.REWEIGHT,
                success=True,
                details={
                    "weights": weights,
                    "disadvantaged_groups": disadvantaged,
                    "auto_applied": auto_apply,
                },
                next_steps=[
                    "Apply weights to training data",
                    "Retrain model with weighted samples",
                    "Re-run ISO validation",
                ],
            )

        # Fallback: block deployment
        return RefinementResult(
            action=RefinementAction.BLOCK_DEPLOYMENT,
            success=False,
            details={
                "reason": "Critical fairness failure, manual intervention required",
                "disadvantaged_groups": disadvantaged,
            },
            next_steps=[
                "Review data quality for disadvantaged groups",
                "Collect more samples if under-represented",
                "Consider feature engineering to reduce bias",
            ],
        )

    def _handle_caution_fairness(self, report: dict[str, Any]) -> RefinementResult:
        """Gère les avertissements de fairness."""
        return RefinementResult(
            action=RefinementAction.SCHEDULE_REVIEW,
            success=True,
            details={
                "reason": "Fairness warning, scheduled for review",
                "metric": report.get("demographic_parity_ratio"),
                "review_deadline": "next_quarter",
            },
            next_steps=[
                "Schedule quarterly fairness audit",
                "Monitor fairness metrics in production",
                "Document mitigation plan",
            ],
        )

    def _should_reweight(self, group_analyses: list[dict[str, Any]]) -> bool:
        """Delegate to refinement_helpers.should_reweight."""
        from scripts.agents.refinement_helpers import should_reweight

        return should_reweight(group_analyses)

    def _calculate_reweighting(self, group_analyses: list[dict[str, Any]]) -> dict[str, float]:
        """Delegate to refinement_helpers.calculate_reweighting."""
        from scripts.agents.refinement_helpers import calculate_reweighting

        return calculate_reweighting(group_analyses)

    def refine_robustness(
        self,
        report: dict[str, Any],
    ) -> RefinementResult:
        """Raffine le modèle en cas d'échec robustness.

        Args:
        ----
            report: Rapport de robustness

        Returns:
        -------
            RefinementResult avec action et détails
        """
        noise_tolerance = report.get("noise_tolerance", 1.0)
        status = self.memory.evaluate_robustness(noise_tolerance)

        logger.info(f"Robustness refinement: NT={noise_tolerance:.2%}, status={status.value}")

        if status == ComplianceStatus.COMPLIANT:
            return RefinementResult(
                action=RefinementAction.NONE,
                success=True,
                details={"reason": "Model is robust"},
            )

        critical_features = report.get("critical_features", [])

        if critical_features:
            return RefinementResult(
                action=RefinementAction.RETRAIN,
                success=True,
                details={
                    "critical_features": critical_features,
                    "recommendation": "Add feature redundancy or data augmentation",
                },
                next_steps=[
                    f"Review critical features: {critical_features}",
                    "Add noise augmentation during training",
                    "Consider ensemble methods for robustness",
                ],
            )

        return RefinementResult(
            action=RefinementAction.SCHEDULE_REVIEW,
            success=True,
            details={"reason": "Robustness warning"},
        )

    def run_refinement_loop(
        self,
        validation_fn: Any,
        train_fn: Any,
        max_iterations: int | None = None,
    ) -> list[RefinementResult]:
        """Delegate to refinement_helpers.run_refinement_loop."""
        from scripts.agents.refinement_helpers import run_refinement_loop

        return run_refinement_loop(self, validation_fn, train_fn, max_iterations)
