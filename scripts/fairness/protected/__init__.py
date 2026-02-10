"""Protected Attributes Module - ISO 24027.

Validation des attributs proteges dans les features ML.

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems
- EEOC 80% rule - Disparate impact
- GDPR Art.9 - Special categories of data
- EU AI Act Art.10(5) - Bias detection

Author: ALICE Engine Team
Version: 1.0.0
"""

from scripts.fairness.protected.config import (
    DEFAULT_PROTECTED_ATTRIBUTES,
    PROXY_CORRELATION_THRESHOLD,
)
from scripts.fairness.protected.types import (
    ProtectedAttribute,
    ProtectionLevel,
    ProxyCorrelation,
    ValidationResult,
)
from scripts.fairness.protected.validator import (
    detect_proxy_correlations,
    validate_features,
)

__all__ = [
    "DEFAULT_PROTECTED_ATTRIBUTES",
    "PROXY_CORRELATION_THRESHOLD",
    "ProtectedAttribute",
    "ProtectionLevel",
    "ProxyCorrelation",
    "ValidationResult",
    "detect_proxy_correlations",
    "validate_features",
]
