"""Match sampling + stratification for BacktestRunner (ISO 5055 split).

Plan 3 V2 T22 fix-on-sight : extrait depuis runner.py pour respecter
ISO 5055 (max 300 lignes/fichier, SRP). Implémente le pipeline 2-phases
décrit dans `BacktestRunner.sample_matches()` :

1. Enumeration filtre strict (type_competition + division + dedup).
2. Stratification per-ronde (T15 module wired) — Bergmeir 2012/2018,
   Pappalardo 2019, ISO 24027 §6.

Document ID: ALICE-BACKTEST-SAMPLING
Version: 1.0.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.backtest.runner_types import MatchCandidate, RunnerConfig
from scripts.backtest.stratified_sampler import (
    StratifiedSamplerConfig,
    stratified_sample,
)

if TYPE_CHECKING:
    from services.ali.cache import ALIDataCache


def enumerate_candidates(cache: ALIDataCache, config: RunnerConfig) -> list[MatchCandidate]:
    """Phase 1 : strict filter type_competition + division + dedup.

    Strict filter design (Plan 3 V2 périmètre, fix-on-sight 2026-04-28) :
    - ``type_competition='national'`` exclut scolaire, J02 (D3),
      coupes (D4), national_jeunes, regional, national_feminin.
    - ``division_filter='Nationale 3'`` exact-match (sécurise vs N4/N2 ; les
      variantes ``Nationale III Jeunes`` rejetées par type_competition).

    @param cache: ALIDataCache déjà chargé (echiquiers + team_to_club).
    @param config: RunnerConfig avec saison, type_competition,
                   division_filter, rondes, team_size.
    @returns list de MatchCandidate (pre-stratification, filtre strict).
    """
    df = cache.echiquiers_total
    team_to_club = cache.team_to_club
    joueurs_by_club = cache.joueurs_by_club

    sub = df[
        (df["saison"] == config.saison)
        & (df["type_competition"] == config.type_competition)
        & (df["division"] == config.division_filter)
        & (df["ronde"].isin(list(config.rondes)))
    ]
    # D-2026-05-11 : include `groupe` in dedup key — Top 16 saison 2024 a 4 groupes
    # (Groupe A/B rondes 1-7 régulière + Poule Haute/Basse rondes 1-4 finale).
    # Une équipe qualifiée (ex Bischwiller) apparaît en Groupe B ET Poule Haute
    # à la même ronde nominale → 2 matches distincts mais (ronde, dom, ext)
    # collision sans `groupe`. Resilient à colonne absente (tests + données historiques).
    has_groupe = "groupe" in sub.columns
    dedup_cols = ["saison", "ronde", "equipe_dom", "equipe_ext"]
    if has_groupe:
        dedup_cols.append("groupe")
    sub = sub.drop_duplicates(subset=dedup_cols)

    out: list[MatchCandidate] = []
    seen: set[tuple[int, str, str, str]] = set()
    for _, row in sub.iterrows():
        ronde = int(row["ronde"])
        user_team = str(row["equipe_dom"])
        opp_team = str(row["equipe_ext"])
        if has_groupe:
            groupe_raw = row.get("groupe")
            groupe = (
                str(groupe_raw) if groupe_raw is not None and str(groupe_raw) != "nan" else ""
            )
        else:
            groupe = ""
        # FFE bye teams ("Exempt") are not real matches — skip (D-2026-05-10-bye)
        if user_team == "Exempt" or opp_team == "Exempt":
            continue
        key = (ronde, user_team, opp_team, groupe)
        if key in seen:
            continue
        opp_club = team_to_club.get(opp_team)
        user_club = team_to_club.get(user_team)
        if opp_club is None or user_club is None:
            continue
        user_pool = joueurs_by_club.get(user_club)
        opp_pool = joueurs_by_club.get(opp_club)
        if user_pool is None or opp_pool is None:
            continue
        if len(user_pool) < config.team_size or len(opp_pool) < config.team_size:
            continue
        out.append(
            MatchCandidate(
                saison=config.saison,
                ronde=ronde,
                user_team=user_team,
                opp_team=opp_team,
                opp_club=opp_club,
                groupe=groupe,
            )
        )
        seen.add(key)
    return out


def stratify_per_ronde(
    candidates: list[MatchCandidate], config: RunnerConfig
) -> list[MatchCandidate]:
    """Phase 2 : balanced sampling per-ronde (T15 wired, SOTA ML).

    Sources :
    - Bergmeir & Benitez 2012 : walk-forward (préserver ordre temporel intra-ronde)
    - Pappalardo 2019 PlayeRank : per-round balanced (sport SOTA)
    - ISO/IEC TR 24027:2021 §6 : group-level fairness audit minimum N
    - Barocas/Hardt/Narayanan 2019 §3 : equal stratum representation

    ``max_per_stratum = ceil(max_matches / N_rondes_present)`` garantit
    que chaque ronde contribue ≈ équitablement, sous le total.
    """
    if not candidates:
        return []
    n_rondes = max(1, len({c.ronde for c in candidates}))
    max_per = max(1, -(-config.max_matches // n_rondes))
    sampler_cfg = StratifiedSamplerConfig(
        min_per_stratum=config.stratify_min_per_ronde,
        max_per_stratum=max_per,
        seed=config.seed,
    )
    sampled = stratified_sample(
        candidates,
        strata_fn=lambda c: f"ronde_{c.ronde}",  # type: ignore[attr-defined,arg-type]
        config=sampler_cfg,
    )
    return sampled[: config.max_matches]
