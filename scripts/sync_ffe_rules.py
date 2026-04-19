"""Sync chess-app flat-six JSON rules into ALICE config/ffe_rules/.

ISO 5259 : lineage tracé via SHA-256 des JSONs sync.
ISO 42001 : source_ref = chess-app commit + date.

Usage:
    python scripts/sync_ffe_rules.py          # sync all
    python scripts/sync_ffe_rules.py --check  # drift check only (CI)
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

CHESS_APP_RULES_DIR = Path("C:/Dev/chess-app/backend/flat-six/rules")
ALICE_RULES_DIR = Path(__file__).parent.parent / "config" / "ffe_rules"

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

    drift_found = False
    for src_rel, tgt_rel in RULES_TO_SYNC:
        src = CHESS_APP_RULES_DIR / src_rel
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
