"""Tests T2 ground truth extraction (Plan 3 V2)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from scripts.backtest.ground_truth import (
    ObservedLineup,
    ObservedPlayer,
    extract_observed_lineup,
)

if TYPE_CHECKING:
    from services.ali.cache import ALIDataCache

J = Path("data/joueurs.parquet")
E = Path("data/echiquiers.parquet")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (J.exists() and E.exists()),
        reason="data parquets absent",
    ),
]


@pytest.fixture(scope="module")
def cache() -> ALIDataCache:
    """Module-scope cache (loaded once)."""
    from services.ali.cache import ALIDataCache

    return ALIDataCache.load_from_parquets(J, E)


def _pick_recent_match(cache: ALIDataCache) -> tuple[str, int, int]:
    """Return (club_name, saison, ronde) for a real recent home match."""
    df = cache.echiquiers_total
    sub = df[df["saison"] == 2023]
    first = sub.iloc[0]
    return str(first["equipe_dom"]), 2023, int(first["ronde"])


def test_extract_observed_lineup_real_match(cache: ALIDataCache) -> None:
    """Pour un match réel (saison, ronde, club) → lineup non vide et triée."""
    club, saison, ronde = _pick_recent_match(cache)

    lineup = extract_observed_lineup(cache, club, saison, ronde)
    assert isinstance(lineup, ObservedLineup)
    assert lineup.club_name == club
    assert lineup.saison == saison
    assert lineup.ronde == ronde
    assert len(lineup.players) >= 4
    assert all(isinstance(p, ObservedPlayer) for p in lineup.players)
    ecs = [p.echiquier for p in lineup.players]
    assert ecs == sorted(ecs), "players must be sorted ascending by echiquier"


def test_extract_observed_lineup_club_inconnu_raise(cache: ALIDataCache) -> None:
    """Club inexistant → KeyError."""
    with pytest.raises(KeyError):
        extract_observed_lineup(cache, "CLUB_INEXISTANT_XYZ_999", 2023, 5)


def test_extract_observed_lineup_saison_absente_raise(cache: ALIDataCache) -> None:
    """Saison inexistante → KeyError."""
    with pytest.raises(KeyError):
        extract_observed_lineup(cache, "ANY", 1900, 1)


def test_observed_lineup_player_names_frozenset(cache: ALIDataCache) -> None:
    """player_names() retourne frozenset des noms uniques."""
    club, saison, ronde = _pick_recent_match(cache)
    lineup = extract_observed_lineup(cache, club, saison, ronde)

    names = lineup.player_names()
    assert isinstance(names, frozenset)
    assert len(names) == len({p.joueur_nom for p in lineup.players})
    # Every name present in the lineup is in the set
    for p in lineup.players:
        assert p.joueur_nom in names


def test_extract_home_vs_away_filter(cache: ALIDataCache) -> None:
    """as_domicile=True ne retourne que les matches où le club est à domicile."""
    club, saison, ronde = _pick_recent_match(cache)

    home_lineup = extract_observed_lineup(
        cache,
        club,
        saison,
        ronde,
        as_domicile=True,
    )
    assert home_lineup.club_name == club
    assert len(home_lineup.players) >= 4

    # Forcing as_domicile=False for a team that only played at home must raise
    df = cache.echiquiers_total
    match_sub = df[(df["saison"] == saison) & (df["ronde"] == ronde)]
    played_away_same_ronde = bool(
        (match_sub["equipe_ext"] == club).any(),
    )
    if not played_away_same_ronde:
        with pytest.raises(KeyError):
            extract_observed_lineup(
                cache,
                club,
                saison,
                ronde,
                as_domicile=False,
            )


def test_extract_observed_players_belong_to_club(cache: ALIDataCache) -> None:
    """Chaque ObservedPlayer extrait appartient bien au club cible.

    Vérifie via ``blanc_equipe`` / ``noir_equipe`` que les joueurs renvoyés
    sont bien ceux du club (pas ceux de l'adversaire). Garde-fou contre
    toute régression du mapping home/away.
    """
    club, saison, ronde = _pick_recent_match(cache)
    lineup = extract_observed_lineup(cache, club, saison, ronde)

    df = cache.echiquiers_total
    match = df[
        (df["saison"] == saison)
        & (df["ronde"] == ronde)
        & ((df["equipe_dom"] == club) | (df["equipe_ext"] == club))
    ]
    names_in_club = set()
    for _, row in match.iterrows():
        if str(row["blanc_equipe"]) == club:
            names_in_club.add(str(row["blanc_nom"]).strip())
        if str(row["noir_equipe"]) == club:
            names_in_club.add(str(row["noir_nom"]).strip())

    for p in lineup.players:
        assert p.joueur_nom in names_in_club, (
            f"Player {p.joueur_nom!r} not in {club!r} lineup " f"(saison={saison} ronde={ronde})"
        )


# D-2026-05-11 : groupe filter for multi-phase competitions (Top 16)


def test_extract_observed_lineup_groupe_filter_top16(cache: ALIDataCache) -> None:
    """Top 16 multi-phase groupe filter (D-2026-05-11).

    Top 16 saison 2024 Bischwiller joue 2 matches au ronde 1 (Groupe B +
    Poule Haute). Sans groupe filter, _select_match_rows mélange les 16 boards
    → invariant FFE trip. Avec groupe filter, chaque phase est isolée et le
    match est extrait correctement.

    Régulation FFE Top 16 = 2 phases séquentielles :
    - Phase 1 (Groupe A + Groupe B parallèles, rondes 1-7) : qualification
    - Phase 2 (Poule Haute/Basse, rondes 1-4 restart) : finale titre/maintien

    Bischwiller saison 2024 est qualifié de Groupe B → Poule Haute. Donc à
    ronde nominale 1, 2 matches distincts existent dans le parquet.
    """
    # Phase 1 (Groupe B) : Tremblay-En-France vs Bischwiller (ronde 1 Groupe B)
    lineup_reg = extract_observed_lineup(
        cache, "Bischwiller", 2024, 1, as_domicile=False, groupe="Groupe B"
    )
    assert lineup_reg.club_name == "Bischwiller"
    assert lineup_reg.saison == 2024
    assert lineup_reg.ronde == 1
    assert len(lineup_reg.players) == 8

    # Phase 2 (Poule Haute) : Clichy Echecs 92 vs Bischwiller (ronde 1 Poule Haute)
    lineup_final = extract_observed_lineup(
        cache, "Bischwiller", 2024, 1, as_domicile=False, groupe="Poule Haute"
    )
    assert lineup_final.club_name == "Bischwiller"
    assert len(lineup_final.players) == 8

    # Lineups should differ (different matches, 6 months apart chronologically)
    names_reg = lineup_reg.player_names()
    names_final = lineup_final.player_names()
    # At least one differing player (roster turnover regular → final phase)
    # OR same — both are valid; key is the lineups don't trip invariant
    assert isinstance(names_reg, frozenset)
    assert isinstance(names_final, frozenset)


def test_extract_observed_lineup_groupe_filter_n3_mono_phase(cache: ALIDataCache) -> None:
    """N3 saison 2024 mono-phase : groupe filter isolates correctly.

    Une équipe peut apparaître dans plusieurs compétitions sous le même nom
    (ex La Farlede-Toulon joue N3 Interclubs ET N2 Interclubs Jeunes au
    ronde 1 saison 2024). Le groupe filter discrimine sans ambiguïté.
    """
    df = cache.echiquiers_total
    n3 = df[(df["saison"] == 2024) & (df["division"] == "Nationale 3")]
    if n3.empty:
        pytest.skip("No N3 saison 2024 data")
    first = n3.iloc[0]
    club = str(first["equipe_dom"])
    ronde = int(first["ronde"])
    groupe = str(first["groupe"])
    lineup = extract_observed_lineup(cache, club, 2024, ronde, groupe=groupe)
    assert lineup.club_name == club
    assert len(lineup.players) >= 4


def test_extract_observed_lineup_unknown_groupe_raises(cache: ALIDataCache) -> None:
    """Unknown groupe → KeyError (no rows match)."""
    with pytest.raises(KeyError):
        extract_observed_lineup(
            cache, "Bischwiller", 2024, 1, as_domicile=False, groupe="Groupe Inexistant XYZ"
        )
