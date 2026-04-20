"""T14 Smoke robustness (Plan 3 V2 Phase 3, ISO 24029).

Stress-test minimal : perturbe les Elos du pool adversaire avec un bruit
gaussien borne et mesure la degradation du recall. Gate P3G13 : recall
ne doit pas chuter > 10% en absolu sur perturbation 5% Elo.

Reutilise scripts/robustness/ framework (perturbation primitives).

Sources SOTA
------------
- ISO/IEC TR 24029:2021 Robustness of neural networks
- Goodfellow et al. 2015 "Explaining and Harnessing Adversarial Examples"
- Madry et al. 2018 "Towards Deep Learning Models Resistant to
  Adversarial Attacks" (PGD baseline).

Document ID: ALICE-BACKTEST-ROBUSTNESS
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROBUSTNESS_RECALL_DROP_GATE = 0.10  # P3G13 : recall drop <= 10% abs


@dataclass(frozen=True)
class RobustnessResult:
    """Outcome d'une passe de stress-test Elo."""

    baseline_recall: float
    perturbed_recall: float
    recall_drop: float
    noise_pct: float
    seed: int

    def passes_gate(self, gate: float = ROBUSTNESS_RECALL_DROP_GATE) -> bool:
        """Return True if recall_drop <= gate (absolute, not relative)."""
        return self.recall_drop <= gate


def perturb_elos(elos: list[int], noise_pct: float, seed: int = 42) -> list[int]:
    """Apply gaussian noise centered 0 with sigma = noise_pct * mean_elo.

    @param elos: list of Elo ratings.
    @param noise_pct: stddev as fraction (0.05 = 5%).
    @param seed: rng seed (reproductibilite ISO 29119).

    @raises ValueError: noise_pct < 0 or >= 1.
    @returns perturbed elos clipped to [800, 2900] (FFE valid range).
    """
    if not 0 <= noise_pct < 1:
        msg = f"noise_pct must be in [0, 1), got {noise_pct}"
        raise ValueError(msg)
    if not elos:
        return []
    rng = np.random.default_rng(seed)
    arr = np.array(elos, dtype=float)
    sigma = noise_pct * float(arr.mean())
    noise = rng.normal(0, sigma, size=len(arr))
    perturbed = np.clip(arr + noise, 800, 2900).astype(int)
    return perturbed.tolist()


def compute_recall_drop(
    baseline_recall: float,
    perturbed_recall: float,
) -> float:
    """Absolute recall drop ∈ [0, 1] (clamped to 0 if improvement)."""
    return max(0.0, baseline_recall - perturbed_recall)


def robustness_smoke(
    baseline_recall: float,
    perturbed_recall: float,
    noise_pct: float,
    seed: int = 42,
) -> RobustnessResult:
    """Assemble RobustnessResult depuis 2 recall mesures.

    Usage (runner integration) :
    1. Run backtest baseline -> recall_mean_0
    2. Perturb opponent Elos by noise_pct
    3. Re-run backtest -> recall_mean_1
    4. robustness_smoke(recall_mean_0, recall_mean_1, noise_pct)
    """
    return RobustnessResult(
        baseline_recall=baseline_recall,
        perturbed_recall=perturbed_recall,
        recall_drop=compute_recall_drop(baseline_recall, perturbed_recall),
        noise_pct=noise_pct,
        seed=seed,
    )
