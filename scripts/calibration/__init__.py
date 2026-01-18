"""Calibration Module - ISO 24029.

Module de calibration des probabilités ML.

ISO Compliance:
- ISO/IEC 24029:2021 - Neural Network Robustness

Author: ALICE Engine Team
Version: 1.0.0
"""

from scripts.calibration.calibrator import (
    Calibrator,
    calibrate_model,
    compute_ece,
)
from scripts.calibration.calibrator_types import (
    CalibrationConfig,
    CalibrationMethod,
    CalibrationMetrics,
    CalibrationResult,
)

__all__ = [
    "CalibrationConfig",
    "CalibrationMethod",
    "CalibrationMetrics",
    "CalibrationResult",
    "Calibrator",
    "calibrate_model",
    "compute_ece",
]
