"""Model Rollback Module - ISO 23894.

Detection automatique de degradation et rollback vers version N-1.

ISO Compliance:
- ISO/IEC 23894:2023 - AI Risk Management
- Patterns: Uber/Netflix shadow -> canary -> rollout

Author: ALICE Engine Team
Version: 1.0.0
"""

from scripts.model_registry.rollback.detector import detect_degradation
from scripts.model_registry.rollback.executor import execute_rollback, log_rollback_event
from scripts.model_registry.rollback.types import (
    DegradationThresholds,
    RollbackDecision,
    RollbackResult,
)

__all__ = [
    "DegradationThresholds",
    "RollbackDecision",
    "RollbackResult",
    "detect_degradation",
    "execute_rollback",
    "log_rollback_event",
]
