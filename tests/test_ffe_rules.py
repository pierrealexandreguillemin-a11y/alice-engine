"""Tests for FFE blocking rules — ALICE autonomous (ISO 29119).

11 blocking rules from REGLES_FFE_ALICE.md (ADR-012).
"""

from services.ffe_rules import (
    check_elo_max,
    check_elo_order,
    check_foreign_quota,
    check_fr_gender,
    check_mutes_limit,
    check_noyau,
    check_team_size,
    check_unique_assignment,
    filter_brule,
    filter_match_count,
    filter_same_group,
    sort_by_elo,
)


class TestBrule:
    """A02 3.7.c: player with 3+ matchs in stronger team is blocked."""

    def test_brule_player_excluded(self):
        players = [
            {"ffe_id": "A00001", "elo": 2000, "matchs_equipe_sup": {"team1": 3}},
            {"ffe_id": "A00002", "elo": 1800, "matchs_equipe_sup": {"team1": 1}},
        ]
        result = filter_brule(players, target_team="team2", team_rank=2)
        assert len(result) == 1
        assert result[0]["ffe_id"] == "A00002"

    def test_brule_same_team_ok(self):
        players = [
            {"ffe_id": "A00001", "elo": 2000, "matchs_equipe_sup": {"team1": 3}},
        ]
        result = filter_brule(players, target_team="team1", team_rank=1)
        assert len(result) == 1

    def test_custom_threshold(self):
        players = [
            {"ffe_id": "A00001", "elo": 2000, "matchs_equipe_sup": {"team1": 2}},
        ]
        # F01 (women): threshold = 1
        result = filter_brule(players, target_team="team2", team_rank=2, seuil=1)
        assert len(result) == 0


class TestMatchCount:
    """A02 3.7.e: matchs played must be < round number."""

    def test_exceeded_excluded(self):
        players = [
            {"ffe_id": "A00001", "matchs_joues": 5},
            {"ffe_id": "A00002", "matchs_joues": 3},
        ]
        result = filter_match_count(players, ronde=4)
        assert len(result) == 1
        assert result[0]["ffe_id"] == "A00002"

    def test_equal_excluded(self):
        players = [{"ffe_id": "A00001", "matchs_joues": 4}]
        result = filter_match_count(players, ronde=4)
        assert len(result) == 0


class TestNoyau:
    """A02 3.7.f: 50% core players after round 1."""

    def test_noyau_respected(self):
        selected = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
        noyau = {"A1", "A2", "A3", "A4", "A5"}
        assert check_noyau(selected, noyau, ronde=3) is True

    def test_noyau_violated(self):
        selected = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
        noyau = {"A1", "A2", "A3"}
        assert check_noyau(selected, noyau, ronde=3) is False

    def test_noyau_skip_ronde_1(self):
        assert check_noyau(["A1", "A2", "A3", "A4"], set(), ronde=1) is True

    def test_noyau_absolute(self):
        selected = ["A1", "A2", "A3", "A4", "A5"]
        noyau = {"A1", "A2"}
        assert check_noyau(selected, noyau, ronde=3, noyau_min=2, noyau_type="absolu") is True


class TestMutes:
    """A02 3.7.g: max transferred players per match."""

    def test_within_limit(self):
        selected = [
            {"ffe_id": "A1", "is_muted": True},
            {"ffe_id": "A2", "is_muted": True},
            {"ffe_id": "A3", "is_muted": False},
        ]
        assert check_mutes_limit(selected, max_mutes=3) is True

    def test_exceeded(self):
        selected = [{"ffe_id": f"A{i}", "is_muted": True} for i in range(4)]
        assert check_mutes_limit(selected, max_mutes=3) is False


class TestUniqueAssignment:
    """1 player = 1 team: unique assignment constraint."""

    def test_no_duplicates(self):
        assert check_unique_assignment([["A1", "A2"], ["A3", "A4"]]) is True

    def test_duplicate(self):
        assert check_unique_assignment([["A1", "A2"], ["A2", "A3"]]) is False


class TestEloSort:
    """Elo descending sort for board assignment."""

    def test_descending(self):
        players = [{"ffe_id": "A1", "elo": 1500}, {"ffe_id": "A2", "elo": 2000}]
        result = sort_by_elo(players)
        assert result[0]["elo"] == 2000


class TestEloOrder:
    """A02 3.6.e: Elo descending with 100pt tolerance."""

    def test_valid_order(self):
        assert check_elo_order([2100, 2000, 1900, 1800]) is True

    def test_invalid_order(self):
        assert check_elo_order([1800, 2100, 1900, 1800]) is False

    def test_within_tolerance(self):
        assert check_elo_order([2050, 2000, 1990, 1900]) is True


class TestForeignQuota:
    """A02 3.7.h: min French/EU players."""

    def test_quota_met(self):
        players = [{"is_french_eu": True}] * 6 + [{"is_french_eu": False}] * 2
        assert check_foreign_quota(players, min_fr_eu=5) is True

    def test_quota_violated(self):
        players = [{"is_french_eu": True}] * 3 + [{"is_french_eu": False}] * 5
        assert check_foreign_quota(players, min_fr_eu=5) is False


class TestTeamSize:
    """A02 3.7.a: team must meet minimum size requirement."""

    def test_valid(self):
        assert check_team_size(8, required=8) is True

    def test_invalid(self):
        assert check_team_size(6, required=8) is False


class TestSameGroup:
    """A02 3.7.d: one player per group."""

    def test_blocked_in_group(self):
        players = [{"ffe_id": "A00001"}, {"ffe_id": "A00002"}]
        history = {"A00001": "groupA"}  # already played in groupA
        result = filter_same_group(players, target_group="groupB", group_history=history)
        assert len(result) == 1
        assert result[0]["ffe_id"] == "A00002"

    def test_same_group_ok(self):
        players = [{"ffe_id": "A00001"}]
        history = {"A00001": "groupA"}
        result = filter_same_group(players, target_group="groupA", group_history=history)
        assert len(result) == 1


class TestFrGender:
    """A02 3.7.i: 1 French male + 1 French female for N1/N2."""

    def test_valid(self):
        players = [
            {"is_french": True, "sexe": "M"},
            {"is_french": True, "sexe": "F"},
            {"is_french": False, "sexe": "M"},
        ]
        assert check_fr_gender(players) is True

    def test_missing_female(self):
        players = [
            {"is_french": True, "sexe": "M"},
            {"is_french": False, "sexe": "F"},
        ]
        assert check_fr_gender(players) is False


class TestEloMax:
    """A02 3.7.j: Elo max per division."""

    def test_within_limit(self):
        players = [{"elo": 2300}, {"elo": 2100}]
        assert check_elo_max(players, elo_max=2400) is True

    def test_exceeded(self):
        players = [{"elo": 2500}, {"elo": 2100}]
        assert check_elo_max(players, elo_max=2400) is False
