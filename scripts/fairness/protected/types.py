"""Types pour Protected Attributes - ISO 24027.

Ce module contient les types de base pour la validation
des attributs proteges:
- ProtectionLevel: Enum des niveaux de protection
- ProtectedAttribute: Model Pydantic d'un attribut protege
- ProxyCorrelation: Model Pydantic d'une correlation proxy
- ValidationResult: Model Pydantic du resultat de validation

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems
- ISO/IEC 27034 - Secure Coding (Pydantic validation)
- ISO/IEC 5055:2021 - Code Quality (SRP)

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ProtectionLevel(str, Enum):
    """Niveau de protection d'un attribut (ISO 24027).

    FORBIDDEN: bloque le training (violation directe).
    PROXY_CHECK: warning + log + rapport (ne bloque pas).
    """

    FORBIDDEN = "forbidden"
    PROXY_CHECK = "proxy_check"


class ProtectedAttribute(BaseModel):
    """Attribut protege avec son niveau et sa raison.

    Attributes
    ----------
        name: Nom de la colonne dans le dataset
        level: Niveau de protection (FORBIDDEN ou PROXY_CHECK)
        reason: Raison de la protection (ex: discrimination regionale)
        proxy_for: Attribut sensible dont c'est un proxy (optionnel)
    """

    name: str = Field(min_length=1)
    level: ProtectionLevel
    reason: str = Field(min_length=1)
    proxy_for: str | None = None


class ProxyCorrelation(BaseModel):
    """Correlation entre une feature et un attribut protege.

    Attributes
    ----------
        feature: Nom de la feature du modele
        protected_attr: Nom de l'attribut protege
        correlation: Valeur de correlation (0 a 1)
        method: Methode utilisee (pearson ou cramers_v)
    """

    feature: str = Field(min_length=1)
    protected_attr: str = Field(min_length=1)
    correlation: float = Field(ge=0.0, le=1.0)
    method: Literal["pearson", "cramers_v"]


class ValidationResult(BaseModel):
    """Resultat complet de la validation des attributs proteges.

    Attributes
    ----------
        is_valid: True si aucune violation FORBIDDEN
        violations: Liste des violations FORBIDDEN
        warnings: Liste des warnings PROXY_CHECK
        proxy_correlations: Correlations proxy detectees
        timestamp: Date/heure de la validation
    """

    is_valid: bool
    violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    proxy_correlations: list[ProxyCorrelation] = Field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour serialisation JSON."""
        return self.model_dump()
