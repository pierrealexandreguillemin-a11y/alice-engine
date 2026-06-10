# config/ — Vendored configuration & offline fixtures

| File | What | Source / lineage |
|------|------|------------------|
| `MODEL_SPECS.md` | Per-model training specs (READ FIRST before any ML action) | hand-maintained |
| `hyperparameters.yaml` | ML hyperparams + Optuna search spaces | hand-maintained |
| `ffe_rules/` | FFE A02 rules JSON vendored from chess-app (ADR-013) | `scripts/sync_ffe_rules.py` (SHA-256 drift check) |
| `clubs_teams_<saison>.json` | Offline clubs-teams fixture (Phase 4a T4) | `scripts/build_clubs_teams.py` from `data/echiquiers.parquet` |

## clubs_teams_<saison>.json — build procedure

```bash
make build-clubs-teams SAISON=2024
# equivalent: python scripts/build_clubs_teams.py --saison 2024 && git diff --stat
```

Derives `(saison, club, ronde) -> simultaneous teams` (team→club grouping via
trailing team-number normalization with corpus corroboration, plus empirical
modal `board_count` per division). Entries are compact arrays ordered per the
top-level `entry_columns` key — `[team_name, division, board_count, date]`,
mapping onto `services/ali/types.py::TeamSpec`.

Properties:

- **Simultaneity filter (explicit, no silent cap)** — only (club, ronde)
  groups with **≥ 2 simultaneous teams** are written (ISO 27001 pre-commit
  gate caps committed files at 1000 KB). Contract: a lookup miss means the
  club fields a single team that ronde. Quality metrics are computed on the
  FULL corpus before this write-filter (`n_*_total` vs `n_*_written`).
- **Deterministic / idempotent** — canonical sorted compact JSON, LF-only,
  no timestamps; two runs produce byte-identical output. Commit the
  regenerated file when `data/echiquiers.parquet` changes.
- **ISO 5259 lineage** — `source_parquet_sha256` embedded in the JSON; the
  output JSON SHA-256 is logged to stdout at build time.
- **Quality metrics embedded** (`metrics` key): `grouping_rate` (team→club
  parsing quality, warn < 0.95) and `date_coherence_rate` (% of multi-team
  (club, ronde) groups whose match dates agree — ronde numbers do NOT align
  calendar-wise across divisions; consumers must filter by `date`).

## ADR-023 note (offline-only — no chess-app)

This fixture serves **only the OFFLINE path** (backtest / Kaggle / tests).
In PROD, `simultaneous_teams` arrives in the `/compose` request payload
assembled by chess-app; ALICE is stateless and **never reads the chess-app
database** (ADR-023 V2, supersedes CDC §4.3 shared-DB-read). The build script
is zero-network and has no chess-app dependency.
