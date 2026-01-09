"""Dataclasses pour Model Registry - ISO 42001/5259/27001.

Ce module contient les structures de données pour:
- Traçabilité des données (DataLineage)
- Informations d'environnement (EnvironmentInfo)
- Artefacts de modèle (ModelArtifact)
- Model Card production (ProductionModelCard)

Conformité ISO/IEC 42001, 5259, 27001, 5055.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataLineage:
    """Traçabilité des données d'entraînement (ISO 5259)."""

    train_path: str
    valid_path: str
    test_path: str
    train_samples: int
    valid_samples: int
    test_samples: int
    train_hash: str
    valid_hash: str
    test_hash: str
    feature_count: int
    target_distribution: dict[str, float]
    created_at: str

    def to_dict(self) -> dict[str, object]:
        """Convertit en dictionnaire."""
        return {
            "train": {
                "path": self.train_path,
                "samples": self.train_samples,
                "hash": self.train_hash,
            },
            "valid": {
                "path": self.valid_path,
                "samples": self.valid_samples,
                "hash": self.valid_hash,
            },
            "test": {
                "path": self.test_path,
                "samples": self.test_samples,
                "hash": self.test_hash,
            },
            "feature_count": self.feature_count,
            "target_distribution": self.target_distribution,
            "created_at": self.created_at,
        }


@dataclass
class EnvironmentInfo:
    """Informations d'environnement pour reproductibilité."""

    python_version: str
    platform_system: str
    platform_release: str
    platform_machine: str
    git_commit: str | None
    git_branch: str | None
    git_dirty: bool
    packages: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        """Convertit en dictionnaire."""
        return {
            "python_version": self.python_version,
            "platform": {
                "system": self.platform_system,
                "release": self.platform_release,
                "machine": self.platform_machine,
            },
            "git": {
                "commit": self.git_commit,
                "branch": self.git_branch,
                "dirty": self.git_dirty,
            },
            "packages": self.packages,
        }


@dataclass
class ModelArtifact:
    """Artefact de modèle avec métadonnées de sécurité."""

    name: str
    path: Path
    format: str
    checksum: str
    size_bytes: int
    onnx_path: Path | None = None
    onnx_checksum: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convertit en dictionnaire."""
        result: dict[str, object] = {
            "name": self.name,
            "path": str(self.path),
            "format": self.format,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
        }
        if self.onnx_path:
            result["onnx"] = {
                "path": str(self.onnx_path),
                "checksum": self.onnx_checksum,
            }
        return result


@dataclass
class ProductionModelCard:
    """Model Card complète conforme ISO 42001."""

    version: str
    created_at: str
    environment: EnvironmentInfo
    data_lineage: DataLineage
    artifacts: list[ModelArtifact]
    metrics: dict[str, dict[str, float]]
    feature_importance: dict[str, dict[str, float]]
    hyperparameters: dict[str, object]
    best_model: dict[str, object]
    limitations: list[str] = field(default_factory=list)
    use_cases: list[str] = field(default_factory=list)
    conformance: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convertit en dictionnaire complet."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "environment": self.environment.to_dict(),
            "data_lineage": self.data_lineage.to_dict(),
            "artifacts": [a.to_dict() for a in self.artifacts],
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "hyperparameters": self.hyperparameters,
            "best_model": self.best_model,
            "limitations": self.limitations,
            "use_cases": self.use_cases,
            "conformance": self.conformance,
        }
