r"""Build offline clubs-teams fixture from ALICE parquet (Phase 4a T4, ADR-023).

Derives ``(saison, club, ronde) -> simultaneous teams`` from
``data/echiquiers.parquet`` ONLY: offline, deterministic, zero network (in
PROD ``simultaneous_teams`` arrives in the ``/compose`` payload; this
fixture serves backtest/Kaggle/tests). Data-driven derivations:
- **board_count**: empirical modal distinct ``echiquier`` per match per
  division-saison; zero counts (all-NaN echiquier) are dropped and a
  division left without data fails fast — NEVER a silent board_count=0.
- **Team->club grouping**: trailing team-number token ("Clichy 2", roman
  "Vandoeuvre IV"), stripped ONLY if corpus-corroborated (bare base OR
  >= 2 siblings; protects "Pau Henri IV"); residuals stay own club key.
- ``date`` in each entry = filtering field, NOT a TeamSpec member: rondes
  do NOT align calendar-wise across divisions (date_coherence_rate ~ 0.40)
  so consumers MUST filter simultaneous teams by ``date``, not ronde alone.
- **Size contract** (no silent cap; ISO 27001 pre-commit caps 1000 KB): only
  >= 2-team (club, ronde) groups written (lookup miss => single-team ronde);
  entries = compact arrays ordered per top-level ``entry_columns`` (first 3
  map onto TeamSpec); quality metrics = FULL corpus before this write-filter.

ISO 5259: parquet SHA-256 embedded, output SHA-256 logged. CLI: ``--saison 2024``.

Document ID: ALICE-SCRIPT-BUILD-CLUBS-TEAMS
Version: 1.2.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_PARQUET = REPO_ROOT / "data" / "echiquiers.parquet"
SCHEMA_VERSION = "1.2.0"
ENTRY_COLUMNS = ["team_name", "division", "board_count", "date"]
# 'national' + 'regional' (lower teams; bucket also tags youth divisions — debt D3).
DEFAULT_TYPES = ("national", "regional")
GROUPING_RATE_THRESHOLD = 0.95
_ROMAN = {r: i for i, r in enumerate("I II III IV V VI VII VIII IX X".split(), 1)}
_SUFFIX_RE = re.compile(r"^(?P<base>.*\S)\s+(?P<num>\d{1,2}|II|III|IV|V|VI|VII|VIII|IX|X|I)$")
_MATCH_KEY = ["division", "ronde", "equipe_dom", "equipe_ext"]
_COLUMNS = ["saison", "type_competition", "echiquier", "date", *_MATCH_KEY]


def compute_file_sha256(path: Path) -> str:
    """Return hex SHA-256 of a file's content (ISO 5259 lineage)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_team_suffix(name: str) -> tuple[str, int] | None:
    """Split 'Clichy 2' -> ('Clichy', 2); None if no trailing team-number token."""
    match = _SUFFIX_RE.match(name)
    if match is None:
        return None
    return match["base"], _ROMAN.get(match["num"]) or int(match["num"])


def build_club_index(team_names: set[str]) -> tuple[dict[str, str], Counter[str]]:
    """Map raw name -> club key via corroboration; stats counts bare/grouped/uncorroborated."""
    candidates = {name: split_team_suffix(name) for name in team_names}
    base_counts = Counter(cand[0] for cand in candidates.values() if cand is not None)
    index: dict[str, str] = {}
    stats: Counter[str] = Counter()
    for name, cand in candidates.items():
        if cand is None:
            index[name] = name
            stats["bare"] += 1
        elif cand[0] in team_names or base_counts[cand[0]] >= 2:
            index[name] = cand[0]
            stats["grouped"] += 1
        else:
            index[name] = name
            stats["uncorroborated"] += 1
    return index, stats


def load_team_rounds(
    parquet: Path, saison: int, types: tuple[str, ...]
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """Return (teams_df, board_count_by_division, skip_counts); SystemExit on bad data."""
    df = pd.read_parquet(parquet, columns=_COLUMNS)
    df = df[(df["saison"] == saison) & (df["type_competition"].isin(types))]
    if df.empty:
        raise SystemExit(f"ERROR: no rows saison={saison} types={list(types)} in {parquet}")
    boards = df.groupby(_MATCH_KEY)["echiquier"].nunique()  # all-NaN echiquier => count 0
    board_count = {d: int(s.mode().iloc[0]) for d, s in boards[boards > 0].groupby("division")}
    keep = ["division", "ronde", "date"]
    frames = [df[[*keep, c]].rename(columns={c: "team"}) for c in ("equipe_dom", "equipe_ext")]
    teams = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["division", "ronde", "team"]
    )
    named = teams[teams["team"].astype(str).str.strip() != ""]
    kept = named[named["ronde"].notna()]
    missing = sorted(set(kept["division"]) - set(board_count))
    if missing:
        raise SystemExit(f"ERROR: no usable echiquier data (board_count) for divisions {missing}")
    skips = {"n_dropped_empty_names": len(teams) - len(named)}
    skips["n_rows_ronde_nan"] = len(named) - len(kept)
    return kept, board_count, skips


def build_clubs(
    teams: pd.DataFrame, index: dict[str, str], board_count: dict[str, int]
) -> dict[str, Any]:
    """Group entries under club keys; precondition: `index` covers all team names in `teams`."""
    clubs: dict[str, Any] = {}
    for row in teams.itertuples(index=False):
        entry = {
            "team_name": row.team,
            "division": row.division,
            "board_count": board_count[row.division],
            "date": row.date.strftime("%Y-%m-%d") if pd.notna(row.date) else None,
        }
        rondes = clubs.setdefault(index[row.team], {"rondes": {}})["rondes"]
        rondes.setdefault(str(int(row.ronde)), []).append(entry)
    for club in clubs.values():
        for entries in club["rondes"].values():
            entries.sort(key=lambda e: (e["team_name"], e["division"]))
    return clubs


def filter_simultaneous(clubs: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    """Keep >=2-team (club, ronde) groups as ENTRY_COLUMNS arrays + count date coherence."""
    out: dict[str, Any] = {}
    multi = coherent = 0
    for club, data in clubs.items():
        rondes = {}
        for ronde, ent in data["rondes"].items():
            if len(ent) >= 2:
                multi += 1
                coherent += len({e["date"] for e in ent if e["date"] is not None}) <= 1
                rondes[ronde] = [[e[c] for c in ENTRY_COLUMNS] for e in ent]
        if rondes:
            out[club] = {"rondes": rondes}
    return out, coherent, multi


def build_payload(parquet: Path, saison: int, types: tuple[str, ...]) -> dict[str, Any]:
    """Assemble the canonical fixture payload (deterministic, no timestamps)."""
    teams, board_count, skips = load_team_rounds(parquet, saison, types)
    index, stats = build_club_index(set(teams["team"]))
    clubs_full = build_clubs(teams, index, board_count)
    clubs_out, coherent, multi = filter_simultaneous(clubs_full)
    grouping_rate = (stats["bare"] + stats["grouped"]) / max(stats.total(), 1)
    return {
        "schema_version": SCHEMA_VERSION,
        "saison": saison,
        "source": "data/echiquiers.parquet",
        "source_parquet_sha256": compute_file_sha256(parquet),
        "generator": "scripts/build_clubs_teams.py",
        "type_competition": sorted(types),
        "entry_columns": ENTRY_COLUMNS,
        "simultaneity_filter": ">=2 teams per (club, ronde); lookup miss => single-team ronde",
        "metrics": {
            "grouping_rate": round(grouping_rate, 4),
            "n_uncorroborated_suffixes": stats["uncorroborated"],
            **skips,
            "date_coherence_rate": round(coherent / multi, 4) if multi else None,
            "n_multi_team_club_rondes": multi,
            "n_clubs_total": len(clubs_full),
            "n_clubs_written": len(clubs_out),
            "n_team_entries_total": int(len(teams)),
            "n_team_entries_written": sum(
                len(entries) for c in clubs_out.values() for entries in c["rondes"].values()
            ),
        },
        "clubs": clubs_out,
    }


def main() -> int:
    """CLI entry point: build fixture JSON + log SHA-256 and quality metrics."""
    parser = argparse.ArgumentParser(description="Build offline clubs-teams fixture from parquet")
    parser.add_argument("--saison", type=int, required=True)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--type-competition", nargs="+", default=list(DEFAULT_TYPES))
    args = parser.parse_args()
    output = args.output or REPO_ROOT / "config" / f"clubs_teams_{args.saison}.json"
    payload = build_payload(args.parquet, args.saison, tuple(args.type_competition))
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    output.write_bytes((text + "\n").encode("utf-8"))  # LF-only: OS-independent SHA-256
    print(f"WROTE: {output} sha256={compute_file_sha256(output)}")
    print(f"METRICS: {json.dumps(payload['metrics'], sort_keys=True)}")
    if payload["metrics"]["grouping_rate"] < GROUPING_RATE_THRESHOLD:
        print(f"WARNING: grouping_rate < {GROUPING_RATE_THRESHOLD} - trace debt", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
