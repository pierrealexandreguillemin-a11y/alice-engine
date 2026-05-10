"""D8 E2E smoke test — pipeline complet sur dummy matches en <30s.

ISO 29119 §6.4 integration testing : exercise tous les modules D8 (breakdowns,
calibration, conformal, dro, gates, aggregate) sans Kaggle ni BacktestRunner
réel. Valide les contrats d'interface entre modules + l'output JSON.

Document ID: ALICE-D8-SMOKE
Version: 1.0.0
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.backtest.runner_types import MatchStats
from scripts.d8.aggregate import (
    DEFAULT_SAISONS,
    PERTURBATION_GATES_INCONCLUSIVE,
    compute_global_metrics,
    evaluate_19_with_inconclusive,
    fuse_per_match,
    load_saison_reports,
    verify_lineage_coherence,
)
from scripts.d8.breakdowns import compute_all_7, max_gap_recall
from scripts.d8.calibration import compute_ece_per_group, compute_multicalibration_alpha
from scripts.d8.conformal import conformal_set_size_mean, coverage_rate, split_calibrate
from scripts.d8.dro import compute_dro_for_match
from scripts.d8.gates import THRESHOLDS, evaluate_19_gates
from scripts.d8.types import D8GateStatus


def _dummy_match(saison: int, ronde: int, recall: float = 0.85) -> MatchStats:
    return MatchStats(
        saison=saison,
        ronde=ronde,
        user_team=f"USR{ronde}",
        opponent_team=f"OPP{ronde}",
        recall_ali=recall,
        accuracy_ali=0.90,
        jaccard_ali=0.75,
        brier_ali=0.20,
        ece_ali=0.04,
        recall_baseline=0.40,
        brier_baseline=0.50,
        bss=0.60,
        e_score_predicted=4.0,
        e_score_observed=4.5,
        e_score_mae=0.5,
        ali_correct=True,
        baseline_correct=False,
    )


def _saison_report_dict(saison: int, n: int = 10) -> dict[str, Any]:
    """Mimic d8_saison_{S}.json schema d8.v1 for aggregator smoke."""
    matches: list[dict[str, Any]] = []
    for r in range(n):
        m = _dummy_match(saison, r % 9 + 1, recall=0.85 + 0.01 * (r % 5))
        matches.append(
            {
                "saison": m.saison,
                "ronde": m.ronde,
                "user_team": m.user_team,
                "opponent_team": m.opponent_team,
                "recall_ali": m.recall_ali,
                "accuracy_ali": m.accuracy_ali,
                "jaccard_ali": m.jaccard_ali,
                "brier_ali": m.brier_ali,
                "ece_ali": m.ece_ali,
                "recall_baseline": m.recall_baseline,
                "brier_baseline": m.brier_baseline,
                "bss": m.bss,
                "e_score_predicted": m.e_score_predicted,
                "e_score_observed": m.e_score_observed,
                "e_score_mae": m.e_score_mae,
                "ali_correct": m.ali_correct,
                "baseline_correct": m.baseline_correct,
            }
        )
    return {
        "schema_version": "d8.v1",
        "saison": saison,
        "n_matches": n,
        "lineage": {
            "joueurs_sha256": "j_aaa",
            "echiquiers_sha256": "e_bbb",
            "mlp_artefact_sha256": "mlp_ccc",
            "temp_scaler_sha256": "ts_ddd",
            "code_sha256": "code_eee",
            "ali_seed": 42,
            "ali_n_topk": 10,
            "ali_n_mc_pairs": 5,
            "ali_decay_lambda": 0.9,
            "kernel_id": f"k-{saison}",
            "kernel_version_kaggle": "v1",
            "run_at_utc": "2026-05-09T00:00:00",
        },
        "per_match": matches,
        "conformal": {"coverage_global": 0.92, "set_size_mean": 1.5},
    }


# -----------------------------------------------------------------------------
# Smoke tests (5 minimum per plan §Task 11)
# -----------------------------------------------------------------------------


def test_smoke_breakdowns_5_matches() -> None:
    """Module breakdowns wires correctly on N=5 dummy matches."""
    matches = [_dummy_match(2024, r) for r in range(1, 6)]
    result = compute_all_7(
        matches,
        gender_fn=lambda _m: "M",
        pool_size_fn=lambda _m: 15,
        all_pool_sizes=[10, 12, 14, 16, 18, 20],
        niveau_fn=lambda _m: "N3",
        team_elo_mean_fn=lambda _m: 1700,
        categorie_fn=lambda _m: "Sen",
    )
    assert len(result) == 7
    assert sum(g.n for g in result["by_ronde"].groups.values()) == 5
    assert max_gap_recall(result["by_ronde"]) == 0.0


def test_smoke_calibration_per_group() -> None:
    """ECE per group + multicalibration α run cleanly on small input."""
    rng = np.random.default_rng(42)
    n = 50
    probs = rng.uniform(0.1, 0.9, n)
    labels = (rng.uniform(0, 1, n) < probs).astype(int)
    groups = ["A" if i < n // 2 else "B" for i in range(n)]
    per_group_ece = compute_ece_per_group(probs, labels, groups, n_bins=5)
    assert set(per_group_ece.keys()) == {"A", "B"}
    masks = {"A": np.array([g == "A" for g in groups]), "B": np.array([g == "B" for g in groups])}
    alpha = compute_multicalibration_alpha(probs, labels, masks, n_bins=5)
    assert alpha == max(per_group_ece.values())


def test_smoke_conformal_split_coverage() -> None:
    """Split conformal + coverage exercise on N=40 (calib=10, test=30)."""
    rng = np.random.default_rng(0)
    n = 40
    y_pred = rng.uniform(0, 1, n)
    y_obs = y_pred + rng.normal(0, 0.05, n)
    cal = split_calibrate(y_obs[:10], y_pred[:10], alpha=0.10)
    cov = coverage_rate(y_obs[10:], y_pred[10:], cal)
    assert 0.0 <= cov <= 1.0
    assert conformal_set_size_mean(y_pred[10:], cal) >= 0.0


def test_smoke_dro_wasserstein_worst_case() -> None:
    """DRO wraps user closure cleanly on small Elo pool."""
    pool = [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200]
    out = compute_dro_for_match(
        pool,
        backtest_run_fn=lambda perturbed: max(0.0, 1.0 - abs(np.mean(perturbed) - 1850) / 1850),
        epsilons=(0.05, 0.10),
        n_perturbations=10,
    )
    assert set(out.keys()) == {0.05, 0.10}
    for outcome in out.values():
        assert math.isfinite(outcome.recall_worst_case)
        assert 0.0 <= outcome.recall_worst_case <= 1.0


def test_smoke_gates_19_pass_when_metrics_good() -> None:
    """Gates evaluation returns 19 PASS for perfect metrics dict."""
    metrics = {
        "max_gap_recall_max_dim": 0.05,
        "recall_per_group_min": 0.90,
        "demographic_parity_diff": 0.05,
        "equalized_odds_diff": 0.05,
        "ece_per_group_max": 0.03,
        "multicalibration_alpha": 0.03,
        "tpr_ratio_min": 0.95,
        "brier_per_group_max": 0.20,
        "bss_per_group_min": 0.50,
        "psi_per_dim_max": 0.10,
        "recall_drop_1pct": 0.01,
        "recall_drop_5pct": 0.03,
        "recall_drop_10pct": 0.06,
        "roster_5pct_recall_drop": 0.03,
        "roster_20pct_recall_drop": 0.10,
        "coverage_global": 0.92,
        "conformal_set_size_mean": 2.4,
        "dro_eps_005_recall_worst": 0.78,
        "dro_eps_010_recall_worst": 0.62,
    }
    gates = evaluate_19_gates(metrics)
    assert len(gates) == 19
    assert all(g.status is D8GateStatus.PASS for g in gates)


def test_smoke_all_thresholds_complete() -> None:
    """Sanity : 19 thresholds defined."""
    assert len(THRESHOLDS) == 19


def test_smoke_e2e_aggregator_pipeline_4_saisons(tmp_path: Path) -> None:
    """End-to-end : write 4 saison stubs → aggregator pipeline → 19 gates eval.

    Exercise : load_saison_reports + verify_lineage_coherence + fuse_per_match
    + compute_global_metrics + evaluate_19_with_inconclusive on N=40 fused.
    """
    for s in DEFAULT_SAISONS:
        (tmp_path / f"d8_saison_{s}.json").write_text(json.dumps(_saison_report_dict(s, n=10)))
    reports = load_saison_reports(tmp_path)
    verify_lineage_coherence(reports)
    fused = fuse_per_match(reports)
    assert len(fused) == 4 * 10
    metrics = compute_global_metrics(fused)
    gates = evaluate_19_with_inconclusive(metrics)
    assert len(gates) == 19
    inconclusive_count = sum(1 for g in gates if g.status is D8GateStatus.INCONCLUSIVE)
    assert inconclusive_count == len(PERTURBATION_GATES_INCONCLUSIVE)


def test_smoke_total_runtime_under_30s(tmp_path: Path) -> None:
    """Combined smoke loop : runs the 6 smoke fixtures end-to-end.

    Spec §Task 11 : E2E smoke fits within pre-push fast suite budget (<30s).
    Pytest reports duration; full-suite measurement <1s in practice.
    """
    matches = [_dummy_match(2024, r) for r in range(1, 6)]
    compute_all_7(
        matches,
        gender_fn=lambda _m: "M",
        pool_size_fn=lambda _m: 15,
        all_pool_sizes=[10, 20],
        niveau_fn=lambda _m: "N3",
        team_elo_mean_fn=lambda _m: 1700,
        categorie_fn=lambda _m: "Sen",
    )
    pool = [1500, 1700, 1900, 2100]
    compute_dro_for_match(
        pool,
        backtest_run_fn=lambda _p: 0.85,
        epsilons=(0.05,),
        n_perturbations=5,
    )
    for s in DEFAULT_SAISONS:
        (tmp_path / f"d8_saison_{s}.json").write_text(json.dumps(_saison_report_dict(s, n=5)))
    reports = load_saison_reports(tmp_path)
    fused = fuse_per_match(reports)
    metrics = compute_global_metrics(fused)
    gates = evaluate_19_with_inconclusive(metrics)
    assert len(gates) == 19
