"""Types partagés pour baseline models - ISO 5055.

Module unique pour les types, évitant la duplication.

Author: ALICE Engine Team
Version: 1.0.0
"""

from dataclasses import dataclass


@dataclass
class BaselineMetrics:
    """Métriques du modèle baseline."""

    model_name: str
    auc_roc: float
    accuracy: float
    f1_score: float
    log_loss: float
    train_time_s: float
    n_train_samples: int
    n_test_samples: int
