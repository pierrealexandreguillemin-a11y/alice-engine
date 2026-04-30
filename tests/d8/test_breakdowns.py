"""Tests scripts/d8/breakdowns — 7 dims (ISO 24027)."""

from __future__ import annotations

import pytest

from scripts.backtest.runner_types import MatchStats
from scripts.d8.breakdowns import (
    breakdown_by_categorie_age,
    breakdown_by_elo_strata,
    breakdown_by_gender,
    breakdown_by_niveau,
    breakdown_by_pool_size,
    breakdown_by_ronde,
    breakdown_by_saison,
    bucket_categorie_age,
    bucket_elo_strata,
    bucket_pool_size_quartile,
    compute_all_7,
    max_gap_recall,
)


def _ms(saison: int = 2024, ronde: int = 1, recall: float = 0.5) -> MatchStats:
    return MatchStats(
        saison=saison,
        ronde=ronde,
        user_team="USR",
        opponent_team="OPP",
        recall_ali=recall,
        accuracy_ali=0.85,
        jaccard_ali=0.60,
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


# ---- Bucket helpers ----


def test_bucket_categorie_age_u12_petits_poussins() -> None:
    assert bucket_categorie_age("PpoM") == "U12"
    assert bucket_categorie_age("PupF") == "U12"


def test_bucket_categorie_age_u18_minimes() -> None:
    assert bucket_categorie_age("MinM") == "U18"
    assert bucket_categorie_age("CadF") == "U18"


def test_bucket_categorie_age_u20() -> None:
    assert bucket_categorie_age("JunM") == "U20"


def test_bucket_categorie_age_sen() -> None:
    assert bucket_categorie_age("SenM") == "Sen"
    assert bucket_categorie_age("SenF") == "Sen"


def test_bucket_categorie_age_s50_plus() -> None:
    assert bucket_categorie_age("SepM") == "S50+"
    assert bucket_categorie_age("VetF") == "S50+"


def test_bucket_categorie_age_unknown_returns_unknown() -> None:
    assert bucket_categorie_age("XYZ") == "unknown"


def test_bucket_pool_size_quartile_small() -> None:
    sizes = [10, 12, 14, 16, 18, 20, 22, 24]
    assert bucket_pool_size_quartile(11, sizes) == "small"


def test_bucket_pool_size_quartile_xlarge() -> None:
    sizes = [10, 12, 14, 16, 18, 20, 22, 24]
    assert bucket_pool_size_quartile(23, sizes) == "xlarge"


def test_bucket_pool_size_empty_sizes_raises() -> None:
    with pytest.raises(ValueError):
        bucket_pool_size_quartile(15, [])


def test_bucket_elo_strata_q1_low() -> None:
    assert bucket_elo_strata(1450) == "Q1_lt_1500"


def test_bucket_elo_strata_q2() -> None:
    assert bucket_elo_strata(1600) == "Q2_1500_1700"


def test_bucket_elo_strata_q3() -> None:
    assert bucket_elo_strata(1800) == "Q3_1700_1900"


def test_bucket_elo_strata_q4_high() -> None:
    assert bucket_elo_strata(2000) == "Q4_gte_1900"


def test_bucket_elo_strata_boundary_1500() -> None:
    """1500 falls into Q2 (>=1500)."""
    assert bucket_elo_strata(1500) == "Q2_1500_1700"


def test_bucket_elo_strata_boundary_1900() -> None:
    """1900 falls into Q4 (>=1900)."""
    assert bucket_elo_strata(1900) == "Q4_gte_1900"


# ---- Breakdown by ronde ----


def test_breakdown_by_ronde_groups_by_ronde() -> None:
    matches = [_ms(ronde=1), _ms(ronde=1), _ms(ronde=3)]
    b = breakdown_by_ronde(matches)
    assert "ronde_1" in b.groups
    assert "ronde_3" in b.groups
    assert b.groups["ronde_1"].n == 2


def test_breakdown_by_ronde_recall_mean() -> None:
    matches = [_ms(ronde=1, recall=0.4), _ms(ronde=1, recall=0.6)]
    b = breakdown_by_ronde(matches)
    assert b.groups["ronde_1"].recall_mean == pytest.approx(0.5)


def test_breakdown_by_ronde_empty() -> None:
    b = breakdown_by_ronde([])
    assert b.groups == {}


# ---- Breakdown by saison ----


def test_breakdown_by_saison_4_saisons() -> None:
    matches = [_ms(saison=s) for s in (2021, 2022, 2023, 2024)]
    b = breakdown_by_saison(matches)
    assert len(b.groups) == 4


def test_breakdown_by_saison_dim_name() -> None:
    b = breakdown_by_saison([_ms()])
    assert b.dim_name == "saison"


# ---- Breakdown by gender ----


def test_breakdown_by_gender_requires_external_lookup() -> None:
    """Gender is not in MatchStats, breakdown takes a key_fn."""
    matches = [_ms()]
    b = breakdown_by_gender(matches, gender_fn=lambda m: "M")
    assert "M" in b.groups


def test_breakdown_by_gender_two_groups() -> None:
    matches = [_ms(), _ms()]
    b = breakdown_by_gender(matches, gender_fn=lambda m: "M" if m.ronde == 1 else "F")
    assert "M" in b.groups


# ---- Breakdown by pool_size ----


def test_breakdown_by_pool_size_4_buckets() -> None:
    matches = [_ms(ronde=r) for r in range(1, 9)]
    sizes = list(range(10, 26, 2))
    b = breakdown_by_pool_size(matches, pool_size_fn=lambda m: sizes[m.ronde - 1])
    assert all(b in {"small", "medium", "large", "xlarge"} for b in b.groups)


# ---- Breakdown by niveau ----


def test_breakdown_by_niveau_groups_by_niveau() -> None:
    matches = [_ms()]
    b = breakdown_by_niveau(matches, niveau_fn=lambda m: "N3")
    assert "N3" in b.groups


# ---- Breakdown by elo_strata ----


def test_breakdown_by_elo_strata_q4() -> None:
    matches = [_ms()]
    b = breakdown_by_elo_strata(matches, team_elo_mean_fn=lambda m: 2000)
    assert "Q4_gte_1900" in b.groups


# ---- Breakdown by categorie_age ----


def test_breakdown_by_categorie_age_sen_dominant() -> None:
    matches = [_ms()]
    b = breakdown_by_categorie_age(matches, categorie_fn=lambda m: "SenM")
    assert "Sen" in b.groups


# ---- compute_all_7 ----


def test_compute_all_7_returns_7_dims() -> None:
    matches = [_ms()]
    result = compute_all_7(
        matches,
        gender_fn=lambda m: "M",
        pool_size_fn=lambda m: 15,
        all_pool_sizes=[10, 12, 14, 16, 18, 20],
        niveau_fn=lambda m: "N3",
        team_elo_mean_fn=lambda m: 1700,
        categorie_fn=lambda m: "SenM",
    )
    assert set(result.keys()) == {
        "by_gender",
        "by_pool_size",
        "by_ronde",
        "by_saison",
        "by_niveau",
        "by_elo_strata",
        "by_categorie_age",
    }


# ---- max_gap_recall ----


def test_max_gap_recall_normal() -> None:
    matches = [_ms(ronde=1, recall=0.4), _ms(ronde=3, recall=0.8)]
    b = breakdown_by_ronde(matches)
    assert max_gap_recall(b) == pytest.approx(0.4)


def test_max_gap_recall_single_group() -> None:
    matches = [_ms(ronde=1, recall=0.5)]
    b = breakdown_by_ronde(matches)
    assert max_gap_recall(b) == 0.0


def test_max_gap_recall_empty_breakdown() -> None:
    from scripts.d8.types import D8GroupBreakdown

    b = D8GroupBreakdown(dim_name="empty", groups={})
    assert max_gap_recall(b) == 0.0


# ---- Edge cases ----


def test_breakdown_by_ronde_nan_recall() -> None:
    matches = [_ms(ronde=1, recall=float("nan"))]
    b = breakdown_by_ronde(matches)
    import math

    assert math.isnan(b.groups["ronde_1"].recall_mean)


def test_compute_all_7_consistent_n_per_dim() -> None:
    matches = [_ms() for _ in range(10)]
    result = compute_all_7(
        matches,
        gender_fn=lambda m: "M",
        pool_size_fn=lambda m: 15,
        all_pool_sizes=[10, 12, 14, 16, 18, 20],
        niveau_fn=lambda m: "N3",
        team_elo_mean_fn=lambda m: 1700,
        categorie_fn=lambda m: "SenM",
    )
    for dim, breakdown in result.items():
        total_n = sum(g.n for g in breakdown.groups.values())
        assert total_n == 10


def test_breakdown_recall_mean_ignores_nan_or_propagates() -> None:
    """Document behavior on NaN — propagate (caller must skip)."""
    matches = [_ms(ronde=1, recall=0.5), _ms(ronde=1, recall=float("nan"))]
    b = breakdown_by_ronde(matches)
    # NaN propagates by mean default
    import math

    assert math.isnan(b.groups["ronde_1"].recall_mean)


def test_breakdown_jaccard_brier_ece_mae_present() -> None:
    matches = [_ms()]
    b = breakdown_by_ronde(matches)
    g = b.groups["ronde_1"]
    assert hasattr(g, "jaccard_mean")
    assert hasattr(g, "brier_mean")
    assert hasattr(g, "ece_mean")
    assert hasattr(g, "mae_mean")
    assert hasattr(g, "bss_mean")


def test_categorie_age_5_buckets_complete() -> None:
    """All 12 FFE categories map to one of 5 buckets."""
    cats = [
        "PpoM",
        "PpoF",
        "PouM",
        "PouF",
        "PupM",
        "PupF",
        "BenM",
        "BenF",
        "MinM",
        "MinF",
        "CadM",
        "CadF",
        "JunM",
        "JunF",
        "SenM",
        "SenF",
        "SepM",
        "SepF",
        "VetM",
        "VetF",
    ]
    buckets = {bucket_categorie_age(c) for c in cats}
    assert buckets == {"U12", "U18", "U20", "Sen", "S50+"}


def test_pool_size_quartile_with_4_sizes() -> None:
    """Smaller dataset still produces 4 buckets."""
    sizes = [10, 12, 14, 16]
    assert bucket_pool_size_quartile(10, sizes) == "small"
    assert bucket_pool_size_quartile(16, sizes) == "xlarge"


def test_breakdown_by_pool_size_dim_name() -> None:
    matches = [_ms()]
    b = breakdown_by_pool_size(matches, pool_size_fn=lambda m: 15, all_pool_sizes=[10, 20])
    assert b.dim_name == "pool_size"


def test_breakdown_by_elo_strata_dim_name() -> None:
    matches = [_ms()]
    b = breakdown_by_elo_strata(matches, team_elo_mean_fn=lambda m: 1700)
    assert b.dim_name == "elo_strata_team"


def test_breakdown_by_categorie_age_dim_name() -> None:
    matches = [_ms()]
    b = breakdown_by_categorie_age(matches, categorie_fn=lambda m: "SenM")
    assert b.dim_name == "categorie_age"


def test_breakdown_by_gender_dim_name() -> None:
    matches = [_ms()]
    b = breakdown_by_gender(matches, gender_fn=lambda m: "M")
    assert b.dim_name == "gender"


def test_breakdown_by_niveau_dim_name() -> None:
    matches = [_ms()]
    b = breakdown_by_niveau(matches, niveau_fn=lambda m: "N3")
    assert b.dim_name == "niveau"


def test_breakdown_by_ronde_dim_name() -> None:
    matches = [_ms()]
    b = breakdown_by_ronde(matches)
    assert b.dim_name == "ronde"


def test_breakdown_by_saison_dim_name_2() -> None:
    matches = [_ms()]
    b = breakdown_by_saison(matches)
    assert b.dim_name == "saison"


def test_breakdown_groups_immutable() -> None:
    matches = [_ms()]
    b = breakdown_by_ronde(matches)
    g = b.groups["ronde_1"]
    with pytest.raises(Exception):  # noqa: B017, PT011 - frozen dataclass raises FrozenInstanceError
        g.n = 999  # type: ignore[misc]


def test_compute_all_7_with_empty_matches() -> None:
    result = compute_all_7(
        matches=[],
        gender_fn=lambda m: "M",
        pool_size_fn=lambda m: 15,
        all_pool_sizes=[10, 20],
        niveau_fn=lambda m: "N3",
        team_elo_mean_fn=lambda m: 1700,
        categorie_fn=lambda m: "SenM",
    )
    assert all(b.groups == {} for b in result.values())


def test_compute_all_7_dimension_names_match_spec() -> None:
    result = compute_all_7(
        matches=[_ms()],
        gender_fn=lambda m: "M",
        pool_size_fn=lambda m: 15,
        all_pool_sizes=[10, 20],
        niveau_fn=lambda m: "N3",
        team_elo_mean_fn=lambda m: 1700,
        categorie_fn=lambda m: "SenM",
    )
    expected_dim_names = {
        "by_gender": "gender",
        "by_pool_size": "pool_size",
        "by_ronde": "ronde",
        "by_saison": "saison",
        "by_niveau": "niveau",
        "by_elo_strata": "elo_strata_team",
        "by_categorie_age": "categorie_age",
    }
    for key, dim_name in expected_dim_names.items():
        assert result[key].dim_name == dim_name
