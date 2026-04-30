r"""Sync chess-app flat-six JSON rules into ALICE config/ffe_rules/.

ISO 5259 : lineage tracé via SHA-256 des JSONs sync.
ISO 42001 : source_ref = chess-app commit + date.

Path resolution (D-P3-01 résorption 2026-04-28) :
- Lit env var ``CHESS_APP_RULES_DIR`` en priorité (CI Linux + collègue
  multi-machine portable).
- Fallback ``C:/Dev/chess-app/backend/flat-six/rules`` (dev local Windows
  pierrax). Si introuvable, hook pre-commit `ffe-rules-drift` skip
  gracieusement avec exit 0 + warning sur stderr (pas blocker CI sans
  chess-app présent).

Usage:
    python scripts/sync_ffe_rules.py          # sync all
    python scripts/sync_ffe_rules.py --check  # drift check only (CI)
    CHESS_APP_RULES_DIR=/path/to/chess-app/backend/flat-six/rules \
        python scripts/sync_ffe_rules.py --check  # custom path
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

CHESS_APP_RULES_DIR_DEFAULT = Path("C:/Dev/chess-app/backend/flat-six/rules")
ALICE_RULES_DIR = Path(__file__).parent.parent / "config" / "ffe_rules"


def resolve_chess_app_dir() -> Path | None:
    """Resolve chess-app rules dir from env var or default fallback.

    Returns None if neither env var nor default exists (CI sans chess-app
    cloné, ex GitHub Actions Linux). Hook pre-commit skip gracieusement.
    """
    env_path = os.environ.get("CHESS_APP_RULES_DIR")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        print(
            f"WARNING: CHESS_APP_RULES_DIR={env_path} does not exist, falling back to default",
            file=sys.stderr,
        )
    if CHESS_APP_RULES_DIR_DEFAULT.exists():
        return CHESS_APP_RULES_DIR_DEFAULT
    return None


# Phase 3 scope : A02 uniquement. Extension J02/Coupes en Phase 3.5.
RULES_TO_SYNC = [
    ("national/a02.json", "a02.json"),
]


def compute_file_sha256(path: Path) -> str:
    """Return hex SHA-256 of a file's content."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def detect_drift(source: Path, target: Path) -> bool:
    """Return True if source and target differ (or target missing)."""
    if not target.exists():
        return True
    return compute_file_sha256(source) != compute_file_sha256(target)


def sync_rules(source: Path, target: Path) -> None:
    """Copy source JSON to target, creating dirs as needed."""
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> int:
    """Sync FFE rules from chess-app or check drift (CLI entry point)."""
    parser = argparse.ArgumentParser(description="Sync FFE rules from chess-app")
    parser.add_argument("--check", action="store_true", help="drift check only")
    args = parser.parse_args()

    chess_app_dir = resolve_chess_app_dir()
    if chess_app_dir is None:
        print(
            "WARNING: chess-app rules dir not found (set CHESS_APP_RULES_DIR "
            "or clone chess-app at C:/Dev/chess-app/). Skipping drift check.",
            file=sys.stderr,
        )
        return 0  # graceful skip (CI sans chess-app)

    drift_found = False
    for src_rel, tgt_rel in RULES_TO_SYNC:
        src = chess_app_dir / src_rel
        tgt = ALICE_RULES_DIR / tgt_rel
        if not src.exists():
            print(f"ERROR: source missing: {src}", file=sys.stderr)
            return 1
        if detect_drift(src, tgt):
            drift_found = True
            if args.check:
                print(f"DRIFT: {tgt_rel}", file=sys.stderr)
            else:
                print(f"SYNC: {src_rel} -> {tgt_rel}")
                sync_rules(src, tgt)

    if args.check and drift_found:
        print(
            "FAIL: FFE rules drift detected. Run scripts/sync_ffe_rules.py",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
