"""AIMMS Module - ISO/IEC 42001:2023 AI Management System.

Post-training orchestration: calibration, uncertainty, alerting.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (lifecycle management)
- ISO/IEC 24029:2021 - Neural Network Robustness (calibration, uncertainty)
- ISO/IEC 23894:2023 - AI Risk Management (alerting)

Author: ALICE Engine Team
Version: 1.0.0
"""

from scripts.aimms.aimms_types import (
    AIMSConfig,
    AIMSResult,
    LifecyclePhase,
)
from scripts.aimms.postprocessor import (
    AIMSPostprocessor,
    run_postprocessing,
)

__all__ = [
    "AIMSConfig",
    "AIMSResult",
    "AIMSPostprocessor",
    "LifecyclePhase",
    "run_postprocessing",
]
