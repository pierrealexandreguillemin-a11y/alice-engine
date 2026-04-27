"""T17 Edge cases tests (Plan 3 V2 §T17).

5 tests obligatoires (DoD) :
    1. ronde 1 (no history) — pas d'historique pour recency F2/F3
    2. club <12 joueurs — pool trop petit, runner skip defensivement
    3. ronde = derniere — fin de saison, pas de leak forward-looking
    4. saison incomplete (ronde inexistante) — KeyError fail-fast
    5. match avec forfaits — ground_truth filtre les noms vides

ISO 24029 (robustness) : edge cases couverts par tests deterministes.
ISO 29119 (testing) : 1 test par edge case, fixtures reelles.

Document ID: ALICE-BACKTEST-EDGE-CASES
Version: 1.0.0
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from scripts.backtest.ground_truth import extract_observed_lineup
from scripts.backtest.harness import BacktestHarness
from scripts.backtest.runner import BacktestRunner
from scripts.backtest.runner_types import RunnerConfig

if TYPE_CHECKING:
    from services.ali.cache import ALIDataCache

J = Path("data/joueurs.parquet")
E = Path("data/echiquiers.parquet")

pytestmark = pytest.mark.skipif(
    not (J.exists() and E.exists()),
    reason="data parquets absent",
)


@pytest.fixture(scope="module")
def cache() -> ALIDataCache:
    """Module-scope cache (loaded once for all 5 edge case tests)."""
    from services.ali.cache import ALIDataCache

    return ALIDataCache.load_from_parquets(J, E)


@pytest.fixture(scope="module")
def harness() -> BacktestHarness:
    """Module-scope harness for runner edge cases."""
    h = BacktestHarness()
    h.setup()
    return h


def _pick_team_with_match_at(cache: ALIDataCache, saison: int, ronde: int) -> str:
    """Pick a team_name (equipe_dom) that has a real match at (saison, ronde)."""
    df = cache.echiquiers_total
    sub = df[(df["saison"] == saison) & (df["ronde"] == ronde)]
    if sub.empty:
        pytest.skip(f"No match found at saison={saison} ronde={ronde}")
    return str(sub.iloc[0]["equipe_dom"])


# ---------------------------------------------------------------------------
# T17.1 : ronde 1 (no history)
# ---------------------------------------------------------------------------


def test_edge_ronde_1_no_history_extractable(cache: ALIDataCache) -> None:
    """Ronde 1 : ground_truth ne depend pas de l'historique, doit extraire.

    F2 (recency) et F3 (streak) consomment l'historique mais ground_truth
    ne fait que filtrer le match courant. Aucune dependance forward-looking,
    aucun crash sur ronde 1.
    """
    team = _pick_team_with_match_at(cache, saison=2024, ronde=1)
    lineup = extract_observed_lineup(cache, team, 2024, 1)
    assert lineup.ronde == 1
    assert lineup.saison == 2024
    assert len(lineup.players) >= 1, "ronde 1 doit produire au moins 1 joueur observe"
    # Sanity : echiquiers tries croissants (invariant ObservedLineup)
    ecs = [p.echiquier for p in lineup.players]
    assert ecs == sorted(ecs)


# ---------------------------------------------------------------------------
# T17.2 : club <12 joueurs (pool trop petit)
# ---------------------------------------------------------------------------


def test_edge_club_too_small_skipped_by_runner(harness: BacktestHarness) -> None:
    """Pool insuffisant : runner.sample_matches skip silencieusement.

    Le runner verifie ``len(pool) >= team_size`` avant d'inclure un match
    (runner.py L108). team_size = 1000 (au-dela du plus grand club FFE,
    max observe 972 joueurs) -> aucun match viable, sample_matches
    retourne liste vide sans crash.

    Egalement validation indirecte du cas concret D3 (jeunes <12 joueurs) :
    le code defensif ne distingue pas la cause, seul l'invariant compte.
    """
    assert harness.cache is not None
    max_pool = max(len(df) for df in harness.cache.joueurs_by_club.values())
    too_big = max_pool + 1
    runner = BacktestRunner(
        harness=harness,
        config=RunnerConfig(max_matches=10, team_size=too_big, rondes=(5,)),
    )
    matches = runner.sample_matches()
    assert matches == [], (
        f"team_size={too_big} (> max pool {max_pool}) doit eliminer tous "
        f"les clubs (defensive guard runner.py L108) — got {len(matches)}"
    )


# ---------------------------------------------------------------------------
# T17.3 : ronde = derniere (fin de saison)
# ---------------------------------------------------------------------------


def test_edge_last_round_extractable(cache: ALIDataCache) -> None:
    """Derniere ronde de la saison : extractable comme une ronde normale.

    Pas de logique forward-looking dans ground_truth, donc la derniere
    ronde n'est pas un cas particulier. Validation : pas de off-by-one
    sur la borne sup.
    """
    df = cache.echiquiers_total
    sub_2024 = df[df["saison"] == 2024]
    last_ronde = int(sub_2024["ronde"].max())
    team = _pick_team_with_match_at(cache, saison=2024, ronde=last_ronde)

    lineup = extract_observed_lineup(cache, team, 2024, last_ronde)
    assert lineup.ronde == last_ronde
    assert len(lineup.players) >= 1


# ---------------------------------------------------------------------------
# T17.4 : saison incomplete (ronde inexistante)
# ---------------------------------------------------------------------------


def test_edge_inexistent_round_raises_keyerror(cache: ALIDataCache) -> None:
    """Saison incomplete : ronde au-dela du max -> KeyError fail-fast.

    Pas de fallback silencieux qui inventerait une ground truth. ISO 24029
    fail-fast : un appelant qui demande la ronde 99 d'une saison qui n'en
    a que 18 doit recevoir une erreur explicite.
    """
    with pytest.raises(KeyError, match="No match found"):
        extract_observed_lineup(cache, "any_team", 2024, 99)


# ---------------------------------------------------------------------------
# T17.5 : match avec forfaits (noms vides filtres)
# ---------------------------------------------------------------------------


def test_edge_forfait_excludes_empty_player(cache: ALIDataCache) -> None:
    """Match avec forfait : ground_truth filtre les noms vides.

    Cas reel : Aix-Les Bains (home, equipe_dom unique en ronde 1)
    vs Meximieux saison 2024 ronde 1 a 2 forfaits cote Aix-Les Bains :
    - echiquier 3 : forfait_blanc (Aix-Les Bains blanc, blanc_nom='')
    - echiquier 4 : forfait_noir (Aix-Les Bains noir, noir_nom='')

    Le filtre ``_extract_players`` (ground_truth.py L207-208 : ``if not name``)
    exclut ces 2 echiquiers -> Aix-Les Bains lineup = 2 joueurs sur 4 boards.
    ``as_domicile=True`` evite l'ambiguite si le team_name reapparait ailleurs.
    """
    df = cache.echiquiers_total
    match_rows = df[
        (df["saison"] == 2024)
        & (df["ronde"] == 1)
        & (df["equipe_dom"] == "Aix-Les Bains")
        & (df["equipe_ext"] == "Meximieux")
    ]
    if match_rows.empty:
        pytest.skip("fixture forfait Aix-Les Bains/Meximieux ronde 1 absent")
    n_boards = len(match_rows)
    # Sanity : le match doit contenir au moins un forfait pour que le test ait du sens
    has_forfait = match_rows["type_resultat"].str.startswith("forfait", na=False).any()
    assert has_forfait, "fixture invalide : aucun forfait sur ce match"

    lineup_home = extract_observed_lineup(cache, "Aix-Les Bains", 2024, 1, as_domicile=True)
    # 2 forfaits cote home -> strictement moins de joueurs que d'echiquiers
    assert len(lineup_home.players) < n_boards, (
        f"forfaits non filtres : attendu < {n_boards} joueurs, " f"got {len(lineup_home.players)}"
    )
    # Aucun joueur extrait n'a un nom vide (invariant strip() != '')
    for p in lineup_home.players:
        assert p.joueur_nom.strip() != "", f"joueur avec nom vide non filtre : {p!r}"
