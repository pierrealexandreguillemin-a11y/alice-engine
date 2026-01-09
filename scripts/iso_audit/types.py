"""Types pour l'audit ISO - ISO 5055.

Dataclasses et types pour les résultats d'audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ISOViolation:
    """Une violation ISO identifiée."""

    norm: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    file: str
    message: str
    fix_suggestion: str = ""


@dataclass
class AuditResult:
    """Résultat complet de l'audit."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    violations: list[ISOViolation] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    compliant: bool = False

    def add_violation(
        self,
        norm: str,
        severity: str,
        file: str,
        message: str,
        fix: str = "",
    ) -> None:
        """Ajoute une violation."""
        self.violations.append(
            ISOViolation(
                norm=norm,
                severity=severity,
                file=file,
                message=message,
                fix_suggestion=fix,
            )
        )

    def to_dict(self) -> dict:
        """Convertit en dictionnaire."""
        return {
            "timestamp": self.timestamp,
            "compliant": self.compliant,
            "stats": self.stats,
            "violations": [
                {
                    "norm": v.norm,
                    "severity": v.severity,
                    "file": v.file,
                    "message": v.message,
                    "fix_suggestion": v.fix_suggestion,
                }
                for v in self.violations
            ],
            "summary": {
                "total": len(self.violations),
                "critical": sum(1 for v in self.violations if v.severity == "CRITICAL"),
                "high": sum(1 for v in self.violations if v.severity == "HIGH"),
                "medium": sum(1 for v in self.violations if v.severity == "MEDIUM"),
                "low": sum(1 for v in self.violations if v.severity == "LOW"),
            },
        }
