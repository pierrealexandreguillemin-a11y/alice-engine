"""Configuration des attributs proteges FFE - ISO 24027.

Attributs proteges identifies dans le dataset FFE:
- ligue_code / ligue: discrimination geographique
- blanc_titre / noir_titre: proxy genre via titres feminins

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems
- GDPR Art.9 - Special categories of data

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from scripts.fairness.protected.types import ProtectedAttribute, ProtectionLevel

PROXY_CORRELATION_THRESHOLD: float = 0.7

DEFAULT_PROTECTED_ATTRIBUTES: list[ProtectedAttribute] = [
    ProtectedAttribute(
        name="ligue_code",
        level=ProtectionLevel.PROXY_CHECK,
        reason="discrimination geographique regionale",
        proxy_for="region",
    ),
    ProtectedAttribute(
        name="ligue",
        level=ProtectionLevel.PROXY_CHECK,
        reason="discrimination geographique regionale",
        proxy_for="region",
    ),
    # FORBIDDEN: titres feminins (WGM, WIM, WFM) = proxy direct du genre.
    # Rationale ISO 24027 + GDPR Art.9: les titres feminins permettent
    # de reconstruire le genre avec certitude. Usage = discrimination directe.
    ProtectedAttribute(
        name="blanc_titre",
        level=ProtectionLevel.FORBIDDEN,
        reason="proxy direct genre via titres feminins (WGM, WIM, WFM)",
        proxy_for="gender",
    ),
    ProtectedAttribute(
        name="noir_titre",
        level=ProtectionLevel.FORBIDDEN,
        reason="proxy direct genre via titres feminins (WGM, WIM, WFM)",
        proxy_for="gender",
    ),
]
