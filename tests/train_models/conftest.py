"""Fixtures Training Pipeline - ISO 29119.

Document ID: ALICE-TEST-TRAIN-CONFTEST
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockMetricsTPP:
    """Mock pour ModelMetrics (train_models_parallel tests)."""

    auc_roc: float = 0.85
    accuracy: float = 0.80
    precision: float = 0.78
    recall: float = 0.82
    f1_score: float = 0.80
    log_loss: float = 0.45
    train_time_s: float = 10.0
    test_auc: float | None = None
    test_accuracy: float | None = None
    test_f1: float | None = None

    def to_dict(self) -> dict[str, Any]:
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
        }


@dataclass
class MockModelResultTPP:
    """Mock pour SingleModelResult."""

    model: Any = None
    metrics: MockMetricsTPP = field(default_factory=MockMetricsTPP)
