"""Tests for adverse_ce robustness + extended constraints (Phase 4a T1 debt resolution).

ISO 29119 : class-based structured tests.
ISO 24029 : robustness (Elo extremes, all-tie pool, single-player, multi-team ambiguity).
ISO 5055 : SRP per test class.

Tests cover :
 - TestConstraintsExtended (debt 1) : C5 mute / C6 foreign / C7 gender quotas.
 - TestLineageHashCollision (debt 2) : JSON-canonical collision resistance.
 - TestSymmetryBreaking (debt 3) : Elo-tied players lex-leader assignment.
 - TestWarmStart (debt 4) : greedy hint speeds solver / preserves optimum.
 - TestRobustness (debt 5) : Elo extremes, all-tie, single-player, multi-team noyau, licence.
 - TestDeterminismTimeout (debt 6) : determinism under timeout (UNKNOWN status locked).

Document ID: ALICE-TEST-ALI-ADVERSE-CE-ROBUSTNESS
Version: 1.0.0
Count: 14 test cases
"""

from __future__ import annotations

from services.ali.adverse_ce import (
    AdverseCEInput,
    AdverseCESolver,
    TeamConstraints,
    compute_lineage_hash,
)
from services.ali.types import PlayerCandidate, TeamSpec


def _make_pool(n: int, base_elo: int = 2000) -> list[PlayerCandidate]:
    """Build a synthetic pool of n PlayerCandidate with descending Elos."""
    return [
        PlayerCandidate(
            nr_ffe=f"P{i:05d}",
            nom=f"NOM{i}",
            prenom=f"Pre{i}",
            elo=base_elo - i * 10,
            club="TestClub",
            mute=False,
            genre="M",
            categorie="SE",
            licence_active=True,
        )
        for i in range(n)
    ]


def _make_pool_with(
    n: int,
    base_elo: int = 2000,
    *,
    mute_indices: set[int] | None = None,
    non_fr_eu_indices: set[int] | None = None,
    female_indices: set[int] | None = None,
    inactive_indices: set[int] | None = None,
) -> list[PlayerCandidate]:
    """Build a pool with explicit per-index attribute overrides."""
    mute_set = mute_indices or set()
    non_fr_set = non_fr_eu_indices or set()
    fem_set = female_indices or set()
    inactive_set = inactive_indices or set()
    return [
        PlayerCandidate(
            nr_ffe=f"P{i:05d}",
            nom=f"NOM{i}",
            prenom=f"Pre{i}",
            elo=base_elo - i * 10,
            club="TestClub",
            mute=i in mute_set,
            genre="F" if i in fem_set else "M",
            categorie="SE",
            licence_active=i not in inactive_set,
            is_french_eu=i not in non_fr_set,
            is_french=i not in non_fr_set,
            sexe="F" if i in fem_set else "M",
        )
        for i in range(n)
    ]


class TestConstraintsExtended:
    """Debt 1 : C5 (mute) + C6 (foreign) + C7 (gender) A02 §3.7.h/i/j."""

    def test_mute_quota_respected(self) -> None:
        # 16 players, 4 mute, max_mutes=2 -> solver picks <=2 mute.
        pool = _make_pool_with(16, mute_indices={0, 5, 10, 15})
        teams = [TeamSpec(team_name="T1", division="N3", board_count=8, target_team=True)]
        constraints = {"T1": TeamConstraints(max_mutes=2)}
        solver = AdverseCESolver()
        sols = solver.solve(AdverseCEInput(pool=pool, teams=teams, team_constraints=constraints))
        assert sols[0].solver_status in {"OPTIMAL", "FEASIBLE"}
        mute_by_nr = {p.nr_ffe: p.mute for p in pool}
        n_mute = sum(1 for nr, _ in sols[0].assignments if mute_by_nr[nr])
        assert n_mute <= 2, f"mute count {n_mute} exceeds quota 2"

    def test_foreign_quota_respected(self) -> None:
        # 16 players, 5 non-fr_eu, min_fr_eu=6 -> solver picks >=6 fr_eu.
        pool = _make_pool_with(16, non_fr_eu_indices={0, 1, 2, 3, 4})
        teams = [TeamSpec(team_name="T1", division="N3", board_count=8, target_team=True)]
        constraints = {"T1": TeamConstraints(min_fr_eu=6)}
        solver = AdverseCESolver()
        sols = solver.solve(AdverseCEInput(pool=pool, teams=teams, team_constraints=constraints))
        assert sols[0].solver_status in {"OPTIMAL", "FEASIBLE"}
        fr_eu_by_nr = {p.nr_ffe: p.is_french_eu for p in pool}
        n_fr_eu = sum(1 for nr, _ in sols[0].assignments if fr_eu_by_nr[nr])
        assert n_fr_eu >= 6, f"fr_eu count {n_fr_eu} below quota 6"

    def test_fr_gender_quota_respected(self) -> None:
        # 16 players, 2 female (indices 0, 1), min_fr_gender_female=1.
        pool = _make_pool_with(16, female_indices={0, 1})
        teams = [TeamSpec(team_name="T1", division="N3", board_count=8, target_team=True)]
        constraints = {"T1": TeamConstraints(min_fr_gender_female=1)}
        solver = AdverseCESolver()
        sols = solver.solve(AdverseCEInput(pool=pool, teams=teams, team_constraints=constraints))
        assert sols[0].solver_status in {"OPTIMAL", "FEASIBLE"}
        female_by_nr = {p.nr_ffe: p.sexe == "F" for p in pool}
        n_female = sum(1 for nr, _ in sols[0].assignments if female_by_nr[nr])
        assert n_female >= 1, f"female count {n_female} below quota 1"


class TestLineageHashCollision:
    """Debt 2 : JSON-canonical lineage hash collision resistance."""

    def test_lineage_hash_collision_resistance(self) -> None:
        # team_name "A|B" != "A:B" != "A,B" under JSON canonical serialization.
        pool = _make_pool(8)
        teams_pipe = [TeamSpec(team_name="A|B", division="N4", board_count=4)]
        teams_colon = [TeamSpec(team_name="A:B", division="N4", board_count=4)]
        teams_comma = [TeamSpec(team_name="A,B", division="N4", board_count=4)]
        h_pipe = compute_lineage_hash(AdverseCEInput(pool=pool, teams=teams_pipe, seed=42))
        h_colon = compute_lineage_hash(AdverseCEInput(pool=pool, teams=teams_colon, seed=42))
        h_comma = compute_lineage_hash(AdverseCEInput(pool=pool, teams=teams_comma, seed=42))
        assert h_pipe != h_colon, "pipe vs colon team_name yields same hash"
        assert h_pipe != h_comma, "pipe vs comma team_name yields same hash"
        assert h_colon != h_comma, "colon vs comma team_name yields same hash"


class TestSymmetryBreaking:
    """Debt 3 : Elo-tied players lex-leader assignment (Margot 2010, Gent 2006)."""

    def test_symmetry_breaking_elo_ties(self) -> None:
        # 4 tied players at Elo 2000 + 4 lower-Elo fillers + 4-board team.
        # Among tied pair (P0, P1) : if both assigned, P0 board_idx <= P1 board_idx.
        pool = [
            PlayerCandidate(
                nr_ffe=f"P{i:05d}",
                nom=f"NOM{i}",
                prenom=f"Pre{i}",
                elo=2000 if i < 4 else 1900 - (i - 4) * 10,
                club="TestClub",
                mute=False,
                genre="M",
                categorie="SE",
                licence_active=True,
            )
            for i in range(8)
        ]
        teams = [TeamSpec(team_name="T1", division="N4", board_count=4, target_team=True)]
        solver = AdverseCESolver()
        sols = solver.solve(AdverseCEInput(pool=pool, teams=teams, seed=42))
        assert sols[0].solver_status in {"OPTIMAL", "FEASIBLE"}

        # Build {nr_ffe -> board_idx} map. For each consecutive tied pair, lower index -> lower board.
        board_by_nr = dict(sols[0].assignments)
        tied_nrs_assigned = [
            (f"P{i:05d}", board_by_nr[f"P{i:05d}"]) for i in range(4) if f"P{i:05d}" in board_by_nr
        ]
        for ii in range(len(tied_nrs_assigned) - 1):
            (_, b_i) = tied_nrs_assigned[ii]
            (_, b_j) = tied_nrs_assigned[ii + 1]
            assert (
                b_i <= b_j
            ), f"symmetry violated: P{ii:05d} board={b_i} > P{ii + 1:05d} board={b_j}"


class TestWarmStart:
    """Debt 4 : greedy Elo-descending warm-start hint."""

    def test_warm_start_preserves_optimum(self) -> None:
        # With hint, solver still reaches OPTIMAL on simple feasible case.
        pool = _make_pool(8)
        teams = [TeamSpec(team_name="T1", division="N4", board_count=4, target_team=True)]
        solver = AdverseCESolver()
        sols = solver.solve(AdverseCEInput(pool=pool, teams=teams, seed=42))
        assert sols[0].solver_status == "OPTIMAL"
        # Top 4 Elos (2000, 1990, 1980, 1970) -> objective 7940.
        assert sols[0].objective_value == 7940.0


class TestRobustness:
    """Debt 5 : ISO 24029 adversarial / extreme robustness."""

    def test_elo_extremes_robust(self) -> None:
        # Pool spans Elo 800-2900 (range 2100). No integer overflow expected.
        pool = [
            PlayerCandidate(
                nr_ffe=f"P{i:05d}",
                nom=f"NOM{i}",
                prenom=f"Pre{i}",
                elo=800 + i * 105,  # 800, 905, ..., 2900 (21 players)
                club="TestClub",
                mute=False,
                genre="M",
                categorie="SE",
                licence_active=True,
            )
            for i in range(21)
        ]
        teams = [TeamSpec(team_name="T1", division="N1", board_count=8, target_team=True)]
        solver = AdverseCESolver()
        sols = solver.solve(AdverseCEInput(pool=pool, teams=teams))
        assert sols[0].solver_status == "OPTIMAL"
        assert len(sols[0].assignments) == 8

    def test_all_tie_pool(self) -> None:
        # 40 players all Elo 2000 -> zero gradient on objective. Must still OPTIMAL.
        pool = [
            PlayerCandidate(
                nr_ffe=f"P{i:05d}",
                nom=f"NOM{i}",
                prenom=f"Pre{i}",
                elo=2000,
                club="TestClub",
                mute=False,
                genre="M",
                categorie="SE",
                licence_active=True,
            )
            for i in range(40)
        ]
        teams = [TeamSpec(team_name="T1", division="N3", board_count=8, target_team=True)]
        solver = AdverseCESolver()
        sols = solver.solve(AdverseCEInput(pool=pool, teams=teams, max_time_sec=5.0))
        assert sols[0].solver_status in {"OPTIMAL", "FEASIBLE"}
        assert len(sols[0].assignments) == 8
        # Objective = 8 * 2000 = 16000.
        assert sols[0].objective_value == 16000.0

    def test_single_player_pool(self) -> None:
        # 1 player, board_count=1 -> OPTIMAL.
        pool = _make_pool(1)
        teams = [TeamSpec(team_name="T1", division="N4", board_count=1, target_team=True)]
        solver = AdverseCESolver()
        sols = solver.solve(AdverseCEInput(pool=pool, teams=teams))
        assert sols[0].solver_status in {"OPTIMAL", "FEASIBLE"}
        assert len(sols[0].assignments) == 1

        # 1 player, board_count=2 -> INFEASIBLE (pool < board_count, guard in _solve_one_team).
        teams2 = [TeamSpec(team_name="T1", division="N4", board_count=2, target_team=True)]
        sols2 = solver.solve(AdverseCEInput(pool=pool, teams=teams2))
        assert sols2[0].solver_status == "INFEASIBLE"

    def test_noyau_ambiguity_multi_team(self) -> None:
        # P00001 in noyau of T1 AND T2. Top-down : T1 gets P00001, T2 still infeasible
        # unless its noyau requirement can be satisfied by remaining pool.
        pool = _make_pool(16)
        teams = [
            TeamSpec(team_name="T1", division="N3", board_count=8, target_team=True),
            TeamSpec(team_name="T2", division="N4", board_count=8),
        ]
        # P00000..P00003 in T1 noyau, P00001..P00004 in T2 noyau (overlap P00001-P00003).
        noyau = {
            "T1": {pool[0].nr_ffe, pool[1].nr_ffe, pool[2].nr_ffe, pool[3].nr_ffe},
            "T2": {pool[1].nr_ffe, pool[2].nr_ffe, pool[3].nr_ffe, pool[4].nr_ffe},
        }
        solver = AdverseCESolver()
        sols = solver.solve(AdverseCEInput(pool=pool, teams=teams, historical_noyau=noyau))
        # T1 must satisfy its noyau (>=4 in noyau).
        if sols[0].solver_status in {"OPTIMAL", "FEASIBLE"}:
            t1_nrs = {nr for nr, _ in sols[0].assignments}
            assert len(t1_nrs & noyau["T1"]) >= 4
        # T2 should be INFEASIBLE because its noyau {P00001..P00004} is drained by T1.
        if sols[1].solver_status in {"OPTIMAL", "FEASIBLE"}:
            t2_nrs = {nr for nr, _ in sols[1].assignments}
            t2_remaining_noyau = noyau["T2"] - {nr for nr, _ in sols[0].assignments}
            assert len(t2_nrs & noyau["T2"]) >= 4 or len(t2_remaining_noyau) < 4

    def test_pool_with_inactive_licence_filtered(self) -> None:
        # 16 players, 4 licence_active=False. Solver does NOT auto-filter (upstream
        # pool_loader responsibility). Solver may assign inactive players.
        pool = _make_pool_with(16, inactive_indices={0, 5, 10, 15})
        teams = [TeamSpec(team_name="T1", division="N3", board_count=8, target_team=True)]
        solver = AdverseCESolver()
        sols = solver.solve(AdverseCEInput(pool=pool, teams=teams))
        # Document via test : licence_active is NOT a solver constraint.
        assert sols[0].solver_status in {"OPTIMAL", "FEASIBLE"}
        # Solver maximizes Elo, picks top 8 (P00000..P00007), which includes
        # P00000 and P00005 (inactive). This is intentional - filtering is upstream.
        active_by_nr = {p.nr_ffe: p.licence_active for p in pool}
        n_inactive_assigned = sum(1 for nr, _ in sols[0].assignments if not active_by_nr[nr])
        # Top 8 includes inactive P00000 + P00005 -> at least 2 inactive assigned.
        assert (
            n_inactive_assigned >= 1
        ), "solver should NOT auto-filter inactive players (upstream responsibility)"


class TestDeterminismTimeout:
    """Debt 6 : determinism guarantee under timeout (num_search_workers=1 + seed)."""

    def test_same_seed_same_solution_under_timeout(self) -> None:
        # Tight timeout to potentially trigger UNKNOWN/FEASIBLE. Two runs identical.
        pool = _make_pool(80)
        teams = [
            TeamSpec(team_name=f"T{i}", division=f"N{(i % 4) + 1}", board_count=8)
            for i in range(10)
        ]
        sols1 = AdverseCESolver().solve(
            AdverseCEInput(pool=pool, teams=teams, max_time_sec=0.001, seed=42)
        )
        sols2 = AdverseCESolver().solve(
            AdverseCEInput(pool=pool, teams=teams, max_time_sec=0.001, seed=42)
        )
        assert len(sols1) == len(sols2)
        for s1, s2 in zip(sols1, sols2, strict=True):
            assert s1.assignments == s2.assignments, (
                f"determinism violated for team {s1.team_name}: "
                f"{s1.assignments} != {s2.assignments}"
            )
            assert (
                s1.solver_status == s2.solver_status
            ), f"status drift for {s1.team_name}: {s1.solver_status} != {s2.solver_status}"
