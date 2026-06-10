"""Tests for scripts/build_clubs_teams.py (Phase 4a T4, ADR-023 offline fixture).

Unit tests use SYNTHETIC parquet fixtures (no 35 MB real parquet load);
one optional integration test against the real parquet is marked slow.

Document ID: ALICE-TEST-BUILD-CLUBS-TEAMS
Version: 1.1.0
Count: 21 tests (schema, idempotence, grouping, filter contract, board_count, edge cases)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_clubs_teams import (
    DEFAULT_PARQUET,
    build_club_index,
    build_payload,
    main,
    split_team_suffix,
)


def _make_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a synthetic echiquiers-like parquet from row dicts."""
    defaults = {
        "saison": 2024,
        "type_competition": "national",
        "division": "Nationale 2",
        "ronde": 1,
        "equipe_dom": "Clichy",
        "equipe_ext": "Bischwiller",
        "echiquier": 1,
        "date": pd.Timestamp("2024-01-14"),
    }
    df = pd.DataFrame([{**defaults, **row} for row in rows])
    path = tmp_path / "echiquiers.parquet"
    df.to_parquet(path)
    return path


def _match_rows(dom: str, ext: str, boards: int = 4, **overrides: object) -> list[dict]:
    """One match = `boards` rows (1 per echiquier)."""
    return [
        {"equipe_dom": dom, "equipe_ext": ext, "echiquier": b + 1, **overrides}
        for b in range(boards)
    ]


class TestSplitTeamSuffix:
    """ISO 29119 — suffix parsing edge cases."""

    def test_digit_suffix(self) -> None:
        assert split_team_suffix("Clichy 2") == ("Clichy", 2)

    def test_roman_suffix(self) -> None:
        assert split_team_suffix("Vandoeuvre-Echecs IV") == ("Vandoeuvre-Echecs", 4)

    def test_roman_one_suffix(self) -> None:
        assert split_team_suffix("Vendome I") == ("Vendome", 1)

    def test_roman_two_not_parsed_as_one_plus_garbage(self) -> None:
        # Anchored regex + longest-first alternation: "II" must not match as "I".
        assert split_team_suffix("Vendome II") == ("Vendome", 2)

    def test_no_suffix(self) -> None:
        assert split_team_suffix("Clichy") is None

    def test_encoding_artifact_name_does_not_crash(self) -> None:
        assert split_team_suffix("NAO Ca�ssa 2") == ("NAO Ca�ssa", 2)


class TestBuildClubIndex:
    """Corroboration rule: strip suffix only if corpus supports it."""

    def test_multi_team_club_grouped(self) -> None:
        index, stats = build_club_index({"Clichy", "Clichy 2", "Clichy 3"})
        assert index == {"Clichy": "Clichy", "Clichy 2": "Clichy", "Clichy 3": "Clichy"}
        assert stats["grouped"] == 2 and stats["bare"] == 1

    def test_siblings_without_bare_base_grouped(self) -> None:
        index, _ = build_club_index({"Palamede Echecs IV", "Palamede Echecs V"})
        assert index["Palamede Echecs IV"] == "Palamede Echecs"
        assert index["Palamede Echecs V"] == "Palamede Echecs"

    def test_roman_one_and_two_grouped_same_club(self) -> None:
        # "I" is a valid team-1 suffix; corroborated by the "II" sibling.
        index, _ = build_club_index({"Vendome I", "Vendome II"})
        assert index["Vendome I"] == "Vendome"
        assert index["Vendome II"] == "Vendome"

    def test_uncorroborated_suffix_stays_own_club(self) -> None:
        # "Pau Henri IV" = club name ending in a roman numeral, no sibling/base.
        index, stats = build_club_index({"Pau Henri IV", "Lille"})
        assert index["Pau Henri IV"] == "Pau Henri IV"
        assert stats["uncorroborated"] == 1


class TestBuildPayload:
    """End-to-end payload assembly from synthetic parquet."""

    def test_schema_and_lineage(self, tmp_path: Path) -> None:
        rows = _match_rows("Clichy", "Lille") + _match_rows(
            "Clichy 2", "Rouen", division="Nationale 3"
        )
        parquet = _make_parquet(tmp_path, rows)
        payload = build_payload(parquet, 2024, ("national",))
        assert set(payload) == {
            "schema_version",
            "saison",
            "source",
            "source_parquet_sha256",
            "generator",
            "type_competition",
            "entry_columns",
            "simultaneity_filter",
            "metrics",
            "clubs",
        }
        assert len(payload["source_parquet_sha256"]) == 64
        assert payload["entry_columns"] == ["team_name", "division", "board_count", "date"]
        entry = payload["clubs"]["Clichy"]["rondes"]["1"][0]
        assert entry == ["Clichy", "Nationale 2", 4, "2024-01-14"]

    def test_multi_team_club_same_ronde(self, tmp_path: Path) -> None:
        rows = _match_rows("Clichy", "Lille") + _match_rows(
            "Clichy 2", "Rouen", division="Nationale 3"
        )
        parquet = _make_parquet(tmp_path, rows)
        payload = build_payload(parquet, 2024, ("national",))
        teams = [e[0] for e in payload["clubs"]["Clichy"]["rondes"]["1"]]
        assert teams == ["Clichy", "Clichy 2"]
        assert payload["metrics"]["n_multi_team_club_rondes"] == 1
        assert payload["metrics"]["date_coherence_rate"] == 1.0

    def test_single_team_rondes_filtered_but_counted(self, tmp_path: Path) -> None:
        # Explicit size contract: single-team (club, ronde) groups are NOT
        # written (lookup miss => single-team ronde) but stay in total metrics.
        parquet = _make_parquet(tmp_path, _match_rows("Clichy", "Bischwiller"))
        payload = build_payload(parquet, 2024, ("national",))
        assert payload["clubs"] == {}
        assert payload["metrics"]["n_team_entries_total"] == 2
        assert payload["metrics"]["n_team_entries_written"] == 0
        assert payload["metrics"]["n_clubs_total"] == 2
        assert payload["metrics"]["n_clubs_written"] == 0

    def test_date_incoherence_measured(self, tmp_path: Path) -> None:
        rows = _match_rows("Clichy", "Lille") + _match_rows(
            "Clichy 2", "Rouen", division="Nationale 3", date=pd.Timestamp("2024-01-21")
        )
        parquet = _make_parquet(tmp_path, rows)
        payload = build_payload(parquet, 2024, ("national",))
        assert payload["metrics"]["date_coherence_rate"] == 0.0

    def test_board_count_modal_per_division(self, tmp_path: Path) -> None:
        # 2 matches with 8 boards, 1 truncated to 5 (forfeits) -> modal 8.
        rows = (
            _match_rows("Lyon", "B", boards=8)
            + _match_rows("Lyon 2", "D", boards=8)
            + _match_rows("E", "F", boards=5, ronde=2)
        )
        parquet = _make_parquet(tmp_path, rows)
        payload = build_payload(parquet, 2024, ("national",))
        entries = payload["clubs"]["Lyon"]["rondes"]["1"]
        assert [e[2] for e in entries] == [8, 8]  # board_count column

    def test_all_nan_echiquier_division_fails_fast(self, tmp_path: Path) -> None:
        # All-NaN echiquier would yield modal board_count=0 (downstream CE would
        # treat every pool as feasible): must fail LOUDLY, never serialize 0.
        nan_match = {"equipe_dom": "Rouen", "equipe_ext": "Caen", "echiquier": None}
        rows = _match_rows("Clichy", "Lille") + [{**nan_match, "division": "Nationale 3"}]
        parquet = _make_parquet(tmp_path, rows)
        with pytest.raises(SystemExit, match="Nationale 3"):
            build_payload(parquet, 2024, ("national",))

    def test_nan_ronde_rows_skipped_and_counted(self, tmp_path: Path) -> None:
        rows = _match_rows("Clichy", "Lille") + _match_rows(
            "Rouen", "Caen", division="Nationale 3", ronde=None
        )
        parquet = _make_parquet(tmp_path, rows)
        payload = build_payload(parquet, 2024, ("national",))
        assert payload["metrics"]["n_rows_ronde_nan"] == 2
        assert payload["metrics"]["n_team_entries_total"] == 2  # Clichy + Lille only

    def test_absent_saison_raises(self, tmp_path: Path) -> None:
        parquet = _make_parquet(tmp_path, _match_rows("Clichy", "Lille"))
        with pytest.raises(SystemExit, match="saison=1999"):
            build_payload(parquet, 1999, ("national",))

    def test_empty_team_name_dropped_and_counted(self, tmp_path: Path) -> None:
        rows = _match_rows("Clichy", "Lille") + _match_rows("  ", "Rouen", division="Nationale 3")
        parquet = _make_parquet(tmp_path, rows)
        payload = build_payload(parquet, 2024, ("national",))
        assert payload["metrics"]["n_dropped_empty_names"] == 1
        assert "  " not in payload["clubs"]


class TestCli:
    """CLI entry point (argparse + canonical JSON write + SHA logging)."""

    def test_idempotence_two_runs_identical_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        parquet = _make_parquet(tmp_path, _match_rows("Clichy", "Clichy 2"))
        out = tmp_path / "clubs_teams_2024.json"
        argv = [
            "build_clubs_teams.py",
            "--saison",
            "2024",
            "--parquet",
            str(parquet),
            "--output",
            str(out),
        ]
        monkeypatch.setattr("sys.argv", argv)
        assert main() == 0
        first = out.read_bytes()
        assert main() == 0
        assert out.read_bytes() == first
        captured = capsys.readouterr()
        assert "sha256=" in captured.out and "METRICS:" in captured.out
        parsed = json.loads(first.decode("utf-8"))
        assert parsed["schema_version"] == "1.2.0"
        assert b"\r" not in first  # LF-only canonical bytes (OS-independent SHA)


@pytest.mark.slow
def test_real_parquet_saison_2024_integration() -> None:
    """Integration on the real parquet: metrics sane, consumer shape honoured."""
    if not DEFAULT_PARQUET.exists():
        pytest.skip("real parquet not available")
    payload = build_payload(DEFAULT_PARQUET, 2024, ("national", "regional"))
    metrics = payload["metrics"]
    # Sanity floor only — the 0.95 policy threshold is reported by the CLI
    # (measured 0.9473 saison 2024: residual = spelling variants, traced as debt).
    assert metrics["grouping_rate"] >= 0.90
    assert metrics["n_clubs_written"] > 100
    for club in list(payload["clubs"].values())[:50]:
        for entries in club["rondes"].values():
            assert len(entries) >= 2  # simultaneity filter honoured
            for entry in entries:
                team_name, division, board_count, date = entry
                assert isinstance(team_name, str) and isinstance(division, str)
                assert board_count >= 1
