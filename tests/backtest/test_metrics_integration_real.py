"""Integration test metrics on REAL FFE data (concern #2 audit fix).

Plan 3 V2. ISO 29119 : integration layer above unit tests.

Prevents silent failures :
- Canonical name matching `f"{nom} {prenom}".strip()` must resolve correctly
  against echiquiers.parquet "NOM Prenom" format (accents, dashes, spaces).
- Ground truth ObservedLineup.player_names() must match canonical names
  built from joueurs.parquet → metrics non-zero on realistic data.

Design note (fast integration, no harness) :
Le harness ML est couvert par tests/test_phase3_plan3_smoke.py (pilote 10
matches). Ici, on isole SPECIFIQUEMENT la jointure des noms canoniques :
- extraire observed lineup réel (echiquiers.blanc_nom / noir_nom)
- construire un ScenarioSet SYNTHETIQUE avec exactement ces joueurs
- vérifier que les metrics retournent > 0 (sinon le format est divergent)

Ce test failerait si `_canonical_name(PlayerCandidate)` produisait
"Nom Prenom" alors qu'echiquiers stocke "NOM Prenom" (casse).

Document ID: ALICE-BACKTEST-METRICS-INTEGRATION
Version: 1.1.0
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.backtest.ground_truth import extract_observed_lineup
from scripts.backtest.metrics import (
    accuracy_at_k,
    brier_presence,
    jaccard_max,
    top_k_recall,
)
from services.ali.scenario import BoardAssignment, Lineup, Scenario, ScenarioSet
from services.ali.types import PlayerCandidate

J = Path("data/joueurs.parquet")
E = Path("data/echiquiers.parquet")

pytestmark = pytest.mark.skipif(
    not (J.exists() and E.exists()),
    reason="data parquets absent",
)


@pytest.fixture(scope="module")
def cache():
    from services.ali.cache import ALIDataCache

    return ALIDataCache.load_from_parquets(J, E)


def _find_viable_match(cache) -> tuple[str, int, int] | None:
    """Find any (club, saison, ronde) with >= 4 observed players."""
    df = cache.echiquiers_total
    recent = df[df["saison"] >= 2023]
    for _, row in (
        recent.drop_duplicates(subset=["equipe_ext", "saison", "ronde"]).head(100).iterrows()
    ):
        club = str(row["equipe_ext"])
        try:
            obs = extract_observed_lineup(
                cache, club, int(row["saison"]), int(row["ronde"]), as_domicile=False
            )
        except Exception:  # noqa: BLE001, S112
            continue
        if len(obs.players) >= 4:
            return club, int(row["saison"]), int(row["ronde"])
    return None


def _mk_candidate_from_observed_name(observed_name: str) -> PlayerCandidate:
    """Parse 'NOM Prenom' echiquiers format back to (nom, prenom) candidate.

    echiquiers stores canonical "NOM Prenom" (FFE convention). Our
    `_canonical_name(p)` produces `f"{p.nom} {p.prenom}".strip()`. Pour
    round-trip fidèle, on split sur le dernier espace : NOM = tout sauf
    prenom.
    """
    parts = observed_name.rsplit(" ", 1)
    if len(parts) != 2:
        nom, prenom = observed_name, ""
    else:
        nom, prenom = parts
    return PlayerCandidate(
        nr_ffe=f"SYN_{abs(hash(observed_name)) % 10**9}",
        nom=nom,
        prenom=prenom,
        elo=1800,
        club="synthetic",
        mute=False,
        genre="M",
        categorie="SE",
        licence_active=True,
    )


def test_canonical_matching_on_real_ffe_names(cache):
    """CRITICAL concern #2 : canonical name round-trip sur data réelle.

    Si `_canonical_name(PlayerCandidate(nom, prenom))` ne reproduit pas le
    format exact de `echiquiers.blanc_nom` / `noir_nom`, les metrics
    retournent silencieusement 0 sur tout le backtest → désastre masqué.

    Protocole :
    1. Extraire un observed lineup réel (≥ 4 joueurs)
    2. Recréer des PlayerCandidate à partir des noms observés
    3. Construire ScenarioSet synthétique avec ces candidates
    4. Vérifier recall == 1.0 (round-trip parfait attendu)
    """
    match = _find_viable_match(cache)
    if match is None:
        pytest.skip("No viable match (>= 4 players) in saisons >= 2023")

    club, saison, ronde = match
    observed = extract_observed_lineup(cache, club, saison, ronde, as_domicile=False)
    assert len(observed.players) >= 4

    # Build ScenarioSet from observed names (round-trip through PlayerCandidate)
    candidates = [_mk_candidate_from_observed_name(n) for n in observed.player_names()]
    assignments = tuple(
        BoardAssignment(board=i + 1, player=c, p_assignment=1.0) for i, c in enumerate(candidates)
    )
    lineup = Lineup(team_size=len(candidates), assignments=assignments)
    scenario = Scenario(lineup=lineup, joint_prob=1.0, weight=1.0, source="monte_carlo")
    scenario_set = ScenarioSet(
        scenarios=(scenario,),
        opponent_club_id=club,
        round_date=f"{saison}-01-01",
        generated_at=f"{saison}-01-01T00:00:00Z",
        lineage_hash="integration" + "0" * 54,
    )

    recall = top_k_recall(observed, scenario_set)
    accuracy = accuracy_at_k(observed, scenario_set)
    jaccard = jaccard_max(observed, scenario_set)
    brier = brier_presence(observed, scenario_set)

    assert recall == pytest.approx(1.0), (
        f"Canonical round-trip BROKEN : recall={recall} (expected 1.0). "
        f"observed sample : {list(observed.player_names())[:3]}, "
        f"candidates sample : {[(c.nom, c.prenom) for c in candidates[:3]]}"
    )
    assert accuracy == pytest.approx(1.0)
    assert jaccard == pytest.approx(1.0)
    assert brier == pytest.approx(0.0)
