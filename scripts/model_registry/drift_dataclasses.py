"""Dataclasses Drift - ISO 5259/42001.

Structures de données pour drift monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Seuils de drift
PSI_THRESHOLD_WARNING = 0.1
PSI_THRESHOLD_CRITICAL = 0.25
ACCURACY_DROP_THRESHOLD = 0.05
ELO_SHIFT_THRESHOLD = 50


@dataclass
class DriftMetrics:
    """Métriques de drift pour monitoring modèle."""

    round_number: int
    timestamp: str
    predictions_count: int
    accuracy: float
    auc_roc: float | None
    elo_mean_shift: float
    elo_std_shift: float
    psi_score: float
    has_warning: bool = False
    has_critical: bool = False
    alerts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convertit en dictionnaire."""
        return {
            "round": self.round_number,
            "timestamp": self.timestamp,
            "predictions": {
                "count": self.predictions_count,
                "accuracy": self.accuracy,
                "auc_roc": self.auc_roc,
            },
            "drift": {
                "elo_mean_shift": self.elo_mean_shift,
                "elo_std_shift": self.elo_std_shift,
                "psi_score": self.psi_score,
            },
            "status": {
                "has_warning": self.has_warning,
                "has_critical": self.has_critical,
                "alerts": self.alerts,
            },
        }


@dataclass
class DriftReport:
    """Rapport de drift sur la saison."""

    season: str
    model_version: str
    baseline_elo_mean: float
    baseline_elo_std: float
    rounds: list[DriftMetrics] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convertit en dictionnaire."""
        return {
            "season": self.season,
            "model_version": self.model_version,
            "baseline": {
                "elo_mean": self.baseline_elo_mean,
                "elo_std": self.baseline_elo_std,
            },
            "rounds": [r.to_dict() for r in self.rounds],
            "summary": self.get_summary(),
        }

    def get_summary(self) -> dict[str, object]:
        """Résumé du drift sur la saison."""
        if not self.rounds:
            return {"status": "no_data"}

        accuracies = [r.accuracy for r in self.rounds]
        psi_scores = [r.psi_score for r in self.rounds]
        warnings = sum(1 for r in self.rounds if r.has_warning)
        criticals = sum(1 for r in self.rounds if r.has_critical)

        return {
            "rounds_monitored": len(self.rounds),
            "accuracy_trend": {
                "first": accuracies[0],
                "last": accuracies[-1],
                "min": min(accuracies),
                "max": max(accuracies),
                "degradation": accuracies[0] - accuracies[-1],
            },
            "psi_trend": {
                "max": max(psi_scores),
                "avg": sum(psi_scores) / len(psi_scores),
            },
            "alerts": {
                "warnings": warnings,
                "criticals": criticals,
            },
            "recommendation": self._get_recommendation(accuracies, psi_scores, criticals),
        }

    def _get_recommendation(
        self, accuracies: list[float], psi_scores: list[float], criticals: int
    ) -> str:
        """Génère une recommandation basée sur le drift."""
        if criticals > 2:
            return "RETRAIN_URGENT"
        if max(psi_scores) > PSI_THRESHOLD_CRITICAL:
            return "RETRAIN_RECOMMENDED"
        if accuracies[0] - accuracies[-1] > ACCURACY_DROP_THRESHOLD:
            return "MONITOR_CLOSELY"
        return "OK"
