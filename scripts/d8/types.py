"""D8 — typed data structures (ISO 29119, ISO 42001 traceability).

Document ID: ALICE-D8-TYPES
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class D8GroupStats:
    """Per-group metrics (extension fairness.GroupStats + BSS)."""

    group_key: str
    n: int
    recall_mean: float
    jaccard_mean: float
    brier_mean: float
    ece_mean: float
    mae_mean: float
    bss_mean: float


@dataclass(frozen=True)
class D8GroupBreakdown:
    """One dimension breakdown : dim_name -> {group_key: GroupStats}."""

    dim_name: str
    groups: dict[str, D8GroupStats]


@dataclass(frozen=True)
class D8Lineage:
    """SHA-256 lineage per ISO 5259 §lineage."""

    joueurs_sha256: str
    echiquiers_sha256: str
    mlp_artefact_sha256: str
    temp_scaler_sha256: str
    code_sha256: str
    ali_seed: int
    ali_n_topk: int
    ali_n_mc_pairs: int
    ali_decay_lambda: float
    kernel_id: str
    kernel_version_kaggle: str
    run_at_utc: str
    division: str = ""  # ADR-019 multi-division audit (Top 16/N1/N2/N3/N4)
    saison: int = 0  # ADR-019 explicit saison (vs default kernel_id parsing)


class D8GateStatus(Enum):
    """Gate evaluation outcome (3 states)."""

    PASS = "pass"  # noqa: S105 - enum value, not a credential
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class D8GateEvaluation:
    """One gate result + threshold + measured value."""

    gate_id: str
    threshold: float
    measured_value: float
    status: D8GateStatus
    source: str  # SOTA reference


@dataclass
class D8SaisonReport:
    """Per-saison aggregated report (kernel output schema d8.v1)."""

    schema_version: str
    saison: int
    n_matches: int
    lineage: D8Lineage
    per_match: list[Any] = field(default_factory=list)  # MatchStats serialized
    breakdowns: dict[str, D8GroupBreakdown] = field(default_factory=dict)
    multicalibration: dict[str, dict[str, float]] = field(default_factory=dict)
    stress_elo: dict[float, float] = field(default_factory=dict)
    stress_roster: dict[float, float] = field(default_factory=dict)
    conformal: dict[str, Any] = field(default_factory=dict)
    dro_wasserstein: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class D8FullReport:
    """Aggregator output schema d8-aggregator.v1."""

    schema_version: str
    n_matches: int
    saisons: list[int]
    lineage_per_saison: dict[int, D8Lineage]
    breakdowns_global: dict[str, D8GroupBreakdown]
    multicalibration_global: dict[str, dict[str, float]]
    stress_elo_global: dict[float, float]
    stress_roster_global: dict[float, float]
    conformal_global: dict[str, Any]
    dro_global: dict[str, dict[str, Any]]
    gates_19: list[D8GateEvaluation]
