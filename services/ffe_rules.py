"""FFE blocking rules — ALICE autonomous eligibility filter (ISO 42001).

Document ID: ALICE-FFE-RULES
Version: 1.0.0

11 blocking rules from REGLES_FFE_ALICE.md (ADR-012).
Parametrable per competition via ReglesCompetition config.
ALICE does NOT depend on chess-app Flat-Six.

Source: A02_2025_26_Championnat_de_France_des_Clubs.pdf
"""

from __future__ import annotations


def filter_brule(
    players: list[dict], target_team: str, team_rank: int, seuil: int = 3
) -> list[dict]:
    """A02 3.7.c: exclude players burned for this team.

    Player with seuil+ matchs in a STRONGER team (lower rank) cannot play for weaker team.
    team_rank: 1 = strongest, higher = weaker.
    """
    result = []
    for p in players:
        matchs_sup = p.get("matchs_equipe_sup", {})
        blocked = False
        for team, info in matchs_sup.items():
            # info can be int (count) or dict with rank
            count = info if isinstance(info, int) else info.get("count", 0)
            team_r = 0 if isinstance(info, int) else info.get("rank", 0)
            # Blocked only if played in stronger team (lower rank) >= seuil times
            if count >= seuil and team != target_team and team_r < team_rank:
                blocked = True
                break
        if not blocked:
            result.append(p)
    return result


def filter_match_count(players: list[dict], ronde: int) -> list[dict]:
    """A02 3.7.e: matchs played must be < round number."""
    return [p for p in players if p.get("matchs_joues", 0) < ronde]


def check_noyau(
    selected_ids: list[str],
    noyau: set[str],
    ronde: int,
    noyau_min: int = 50,
    noyau_type: str = "pourcentage",
) -> bool:
    """A02 3.7.f: core players requirement after round 1."""
    if ronde <= 1:
        return True
    core_count = sum(1 for pid in selected_ids if pid in noyau)
    if noyau_type == "absolu":
        return core_count >= noyau_min
    return core_count >= len(selected_ids) / 2


def check_mutes_limit(selected: list[dict], max_mutes: int = 3) -> bool:
    """A02 3.7.g: max transferred players per match."""
    mute_count = sum(1 for p in selected if p.get("is_muted", False))
    return mute_count <= max_mutes


def check_unique_assignment(teams: list[list[str]]) -> bool:
    """1 player = 1 team only."""
    all_ids = [pid for team in teams for pid in team]
    return len(all_ids) == len(set(all_ids))


def sort_by_elo(players: list[dict], key: str = "elo") -> list[dict]:
    """Sort players by Elo descending for board assignment."""
    return sorted(players, key=lambda p: p.get(key, 1500), reverse=True)


def check_elo_order(elos: list[int], tolerance: int = 100) -> bool:
    """A02 3.6.e: Elo descending with tolerance."""
    for i in range(len(elos) - 1):
        if elos[i + 1] - elos[i] > tolerance:
            return False
    return True


def check_foreign_quota(players: list[dict], min_fr_eu: int = 5) -> bool:
    """A02 3.7.h: minimum French/EU players."""
    fr_eu = sum(1 for p in players if p.get("is_french_eu", True))
    return fr_eu >= min_fr_eu


def check_team_size(actual: int, required: int = 8) -> bool:
    """A02 3.7.a: team must have exactly required players."""
    return actual >= required


def filter_same_group(
    players: list[dict], target_group: str, group_history: dict[str, str]
) -> list[dict]:
    """A02 3.7.d: if club has multiple teams in same group, player can only play for one."""
    return [
        p for p in players if group_history.get(p.get("ffe_id", ""), target_group) == target_group
    ]


def check_fr_gender(selected: list[dict]) -> bool:
    """A02 3.7.i: N1/N2 must have at least 1 French male + 1 French female."""
    has_fr_male = any(p.get("is_french", False) and p.get("sexe", "M") == "M" for p in selected)
    has_fr_female = any(p.get("is_french", False) and p.get("sexe", "M") == "F" for p in selected)
    return has_fr_male and has_fr_female


def check_elo_max(selected: list[dict], elo_max: int) -> bool:
    """A02 3.7.j: no player above elo_max (e.g., 2400 for N4)."""
    return all(p.get("elo", 0) <= elo_max for p in selected)
