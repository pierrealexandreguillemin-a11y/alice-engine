"""D8 gates — 19 G-A SOTA strict + case-by-case failure logger.

Sources :
- spec §5.1 (10 fairness gates) + §5.2 (9 robustness gates) + §5.3 (case-by-case)
- ISO/IEC TR 24027:2021 §6 (fairness)
- ISO/IEC TR 24029-2:2024 §6.5 (robustness)
- Mehrabi 2021, Hardt 2016, Pleiss 2017, Hébert-Johnson 2018, Feldman 2015,
  Brier 1950, Pappalardo 2019, Yurdakul 2020, Goodfellow 2015, Madry 2018,
  Tran 2022, Recht 2019, Vovk 2024, Angelopoulos 2023, Sinha 2018, Duchi 2021

Document ID: ALICE-D8-GATES
Version: 1.0.0
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from scripts.d8.types import D8GateEvaluation, D8GateStatus

# ---- Gate thresholds (G-A SOTA strict, spec §5) ----

THRESHOLDS: dict[str, float] = {
    # Fairness (10 gates)
    "G_FAIR_01_max_gap_recall": 0.10,
    "G_FAIR_02_recall_per_group_min": 0.85,
    "G_FAIR_03_demographic_parity_diff": 0.10,
    "G_FAIR_04_equalized_odds_diff": 0.10,
    "G_FAIR_05_calibration_ECE_per_group": 0.05,
    "G_FAIR_06_multicalibration_alpha": 0.05,
    "G_FAIR_07_TPR_ratio_min": 0.80,
    "G_FAIR_08_brier_per_group": 0.30,
    "G_FAIR_09_BSS_per_group": 0.30,
    "G_FAIR_10_PSI_per_dim": 0.20,
    # Robustness (9 gates)
    "G_ROB_01_recall_drop_1pct": 0.02,
    "G_ROB_02_recall_drop_5pct": 0.05,
    "G_ROB_03_recall_drop_10pct": 0.10,
    "G_ROB_04_roster_5pct": 0.05,
    "G_ROB_05_roster_20pct": 0.15,
    "G_ROB_06_conformal_coverage_90": 0.90,
    "G_ROB_07_conformal_set_size_max": 3.0,
    "G_ROB_08_DRO_eps_005_min": 0.70,
    "G_ROB_09_DRO_eps_010_min": 0.55,
}

SOURCES: dict[str, str] = {
    "G_FAIR_01_max_gap_recall": "Mehrabi 2021 §4.1",
    "G_FAIR_02_recall_per_group_min": "P3G07 - 5pts",
    "G_FAIR_03_demographic_parity_diff": "Hardt 2016",
    "G_FAIR_04_equalized_odds_diff": "Hardt 2016 §3.2",
    "G_FAIR_05_calibration_ECE_per_group": "Pleiss 2017 §4",
    "G_FAIR_06_multicalibration_alpha": "Hébert-Johnson 2018",
    "G_FAIR_07_TPR_ratio_min": "EEOC §1607.4D + Feldman 2015",
    "G_FAIR_08_brier_per_group": "Brier 1950 + Pappalardo 2019",
    "G_FAIR_09_BSS_per_group": "Pappalardo 2019 §3.4",
    "G_FAIR_10_PSI_per_dim": "Yurdakul 2020",
    "G_ROB_01_recall_drop_1pct": "Goodfellow 2015 ε=0.01",
    "G_ROB_02_recall_drop_5pct": "Madry 2018",
    "G_ROB_03_recall_drop_10pct": "Madry 2018 strict",
    "G_ROB_04_roster_5pct": "Tran 2022 §3.4",
    "G_ROB_05_roster_20pct": "Recht 2019 §5",
    "G_ROB_06_conformal_coverage_90": "Vovk 2024 §2.3",
    "G_ROB_07_conformal_set_size_max": "Angelopoulos 2023 §4.2",
    "G_ROB_08_DRO_eps_005_min": "Sinha 2018 §4",
    "G_ROB_09_DRO_eps_010_min": "Duchi 2021 §6",
}

# Gates where PASS = measured <= threshold (drops, gaps, ECE, set size).
MAX_THRESHOLD_GATES: frozenset[str] = frozenset(
    {
        "G_FAIR_01_max_gap_recall",
        "G_FAIR_03_demographic_parity_diff",
        "G_FAIR_04_equalized_odds_diff",
        "G_FAIR_05_calibration_ECE_per_group",
        "G_FAIR_06_multicalibration_alpha",
        "G_FAIR_08_brier_per_group",
        "G_FAIR_10_PSI_per_dim",
        "G_ROB_01_recall_drop_1pct",
        "G_ROB_02_recall_drop_5pct",
        "G_ROB_03_recall_drop_10pct",
        "G_ROB_04_roster_5pct",
        "G_ROB_05_roster_20pct",
        "G_ROB_07_conformal_set_size_max",
    }
)

# Gates where PASS = measured >= threshold (recall, coverage, BSS, DRO).
MIN_THRESHOLD_GATES: frozenset[str] = frozenset(
    {
        "G_FAIR_02_recall_per_group_min",
        "G_FAIR_07_TPR_ratio_min",
        "G_FAIR_09_BSS_per_group",
        "G_ROB_06_conformal_coverage_90",
        "G_ROB_08_DRO_eps_005_min",
        "G_ROB_09_DRO_eps_010_min",
    }
)


def _validate_known_gate(gate_id: str) -> None:
    """ISO 27034 input validation : reject unknown gate ids."""
    if gate_id not in THRESHOLDS:
        msg = f"Unknown gate id: {gate_id!r}"
        raise ValueError(msg)


def _validate_measured(gate_id: str, measured_value: float) -> None:
    """ISO 27034 input validation : reject NaN/Inf measurements."""
    if not math.isfinite(measured_value):
        msg = f"measured_value for {gate_id} must be finite, got {measured_value!r}"
        raise ValueError(msg)


def evaluate_max_threshold(
    gate_id: str,
    measured_value: float,
    threshold: float | None = None,
) -> D8GateEvaluation:
    """Evaluate a max-threshold gate (PASS if measured <= threshold)."""
    _validate_known_gate(gate_id)
    _validate_measured(gate_id, measured_value)
    threshold = THRESHOLDS[gate_id] if threshold is None else threshold
    status = D8GateStatus.PASS if measured_value <= threshold else D8GateStatus.FAIL
    return D8GateEvaluation(
        gate_id=gate_id,
        threshold=threshold,
        measured_value=measured_value,
        status=status,
        source=SOURCES[gate_id],
    )


def evaluate_min_threshold(
    gate_id: str,
    measured_value: float,
    threshold: float | None = None,
) -> D8GateEvaluation:
    """Evaluate a min-threshold gate (PASS if measured >= threshold)."""
    _validate_known_gate(gate_id)
    _validate_measured(gate_id, measured_value)
    threshold = THRESHOLDS[gate_id] if threshold is None else threshold
    status = D8GateStatus.PASS if measured_value >= threshold else D8GateStatus.FAIL
    return D8GateEvaluation(
        gate_id=gate_id,
        threshold=threshold,
        measured_value=measured_value,
        status=status,
        source=SOURCES[gate_id],
    )


def evaluate_inconclusive(gate_id: str, threshold: float | None = None) -> D8GateEvaluation:
    """Mark gate as INCONCLUSIVE (e.g. DRO non-convergence)."""
    _validate_known_gate(gate_id)
    threshold = THRESHOLDS[gate_id] if threshold is None else threshold
    return D8GateEvaluation(
        gate_id=gate_id,
        threshold=threshold,
        measured_value=float("nan"),
        status=D8GateStatus.INCONCLUSIVE,
        source=SOURCES[gate_id],
    )


_METRIC_TO_GATE: list[tuple[str, str]] = [
    ("max_gap_recall_max_dim", "G_FAIR_01_max_gap_recall"),
    ("recall_per_group_min", "G_FAIR_02_recall_per_group_min"),
    ("demographic_parity_diff", "G_FAIR_03_demographic_parity_diff"),
    ("equalized_odds_diff", "G_FAIR_04_equalized_odds_diff"),
    ("ece_per_group_max", "G_FAIR_05_calibration_ECE_per_group"),
    ("multicalibration_alpha", "G_FAIR_06_multicalibration_alpha"),
    ("tpr_ratio_min", "G_FAIR_07_TPR_ratio_min"),
    ("brier_per_group_max", "G_FAIR_08_brier_per_group"),
    ("bss_per_group_min", "G_FAIR_09_BSS_per_group"),
    ("psi_per_dim_max", "G_FAIR_10_PSI_per_dim"),
    ("recall_drop_1pct", "G_ROB_01_recall_drop_1pct"),
    ("recall_drop_5pct", "G_ROB_02_recall_drop_5pct"),
    ("recall_drop_10pct", "G_ROB_03_recall_drop_10pct"),
    ("roster_5pct_recall_drop", "G_ROB_04_roster_5pct"),
    ("roster_20pct_recall_drop", "G_ROB_05_roster_20pct"),
    ("coverage_global", "G_ROB_06_conformal_coverage_90"),
    ("conformal_set_size_mean", "G_ROB_07_conformal_set_size_max"),
    ("dro_eps_005_recall_worst", "G_ROB_08_DRO_eps_005_min"),
    ("dro_eps_010_recall_worst", "G_ROB_09_DRO_eps_010_min"),
]


def evaluate_19_gates(metrics: dict[str, Any]) -> list[D8GateEvaluation]:
    """Evaluate all 19 G-A gates from a metrics dict."""
    out: list[D8GateEvaluation] = []
    for metric_key, gate_id in _METRIC_TO_GATE:
        if metric_key not in metrics:
            msg = f"metrics dict missing required key {metric_key!r} for {gate_id}"
            raise KeyError(msg)
        value = float(metrics[metric_key])
        if gate_id in MAX_THRESHOLD_GATES:
            out.append(evaluate_max_threshold(gate_id, value))
        elif gate_id in MIN_THRESHOLD_GATES:
            out.append(evaluate_min_threshold(gate_id, value))
        else:  # pragma: no cover - defensive guard
            msg = f"gate {gate_id} missing from MAX/MIN classification sets"
            raise RuntimeError(msg)
    return out


def render_failure_analysis_md(
    failures: list[D8GateEvaluation],
    today: str | None = None,
) -> str:
    """Render D8_FAILURE_ANALYSIS_LOG.md case-by-case template (spec §5.3)."""
    if today is None:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
    if not failures:
        return "# D8 Failure Analysis Log\n\nAll 19 gates PASS — no failures to analyze.\n"
    lines: list[str] = ["# D8 Failure Analysis Log\n"]
    for f in failures:
        delta = f.measured_value - f.threshold
        lines.append(
            f"\n## Gate {f.gate_id} FAIL — analysis {today}\n\n"
            f"**Measured** : {f.measured_value:.4f}\n"
            f"**Threshold** : {f.threshold:.4f}\n"
            f"**Δ from threshold** : {delta:+.4f}\n"
            f"**Source** : {f.source}\n\n"
            "### Gate validity\n\n"
            f"Le seuil {f.threshold:.4f} est-il approprié pour ALICE Engine\n"
            "(domaine échecs + Interclubs FFE) ?\n\n"
            "- Argument validité : <à remplir>\n"
            "- Argument inapplicabilité : <à remplir>\n\n"
            "### Utilité métier\n\n"
            "La gate mesure-t-elle un risque concret pour les utilisateurs ALICE ?\n"
            f"- Impact si métrique reste à {f.measured_value:.4f} : <à remplir>\n"
            "- Mitigation produit possible : <à remplir>\n\n"
            "### Seuil recalibré (proposé)\n\n"
            f"- Si gate validity ✓ : threshold reste {f.threshold:.4f}, fix code\n"
            "- Si validité ✗ : threshold proposé <new_threshold> avec justification\n\n"
            "### Mitigations options (3 max)\n\n"
            "1. <option>\n2. <option>\n3. <option>\n\n"
            "### Décision user (à remplir)\n\n"
            "[ ] Accepter mitigation N°1\n"
            "[ ] Recalibrer seuil à <new_threshold>\n"
            "[ ] Bloquer Phase 4a entry gate jusqu'à fix\n"
        )
    return "".join(lines)


def filter_failures(evaluations: list[D8GateEvaluation]) -> list[D8GateEvaluation]:
    """Return only gates with status FAIL (INCONCLUSIVE excluded)."""
    return [e for e in evaluations if e.status is D8GateStatus.FAIL]


def gates_summary(evaluations: list[D8GateEvaluation]) -> dict[str, int]:
    """Tally PASS / FAIL / INCONCLUSIVE counts (for summary reporting)."""
    summary = {"pass": 0, "fail": 0, "inconclusive": 0}
    for e in evaluations:
        if e.status is D8GateStatus.PASS:
            summary["pass"] += 1
        elif e.status is D8GateStatus.FAIL:
            summary["fail"] += 1
        else:
            summary["inconclusive"] += 1
    return summary
