"""Train PreferenceModel on echiquiers.parquet for a given saison (Phase 4a T2).

Usage:
    python scripts/train_preference_model.py --saison 2024 \
        --output models/preference_model_2024.joblib

ISO 5259 lineage : input SHA-256 -> artifact SHA-256 logged to stdout.
ISO 42001 reproducibility : `--seed` + `--alpha` fully control fit.

Document ID: ALICE-SCRIPT-TRAIN-PREFERENCE
Version: 1.0.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from services.ali.preference_model import PreferenceModel, save_artifact


def main() -> None:
    """Parse CLI args, fit PreferenceModel, persist artifact, log lineage SHAs."""
    parser = argparse.ArgumentParser(description="Train PreferenceModel (Phase 4a T2)")
    parser.add_argument("--saison", type=int, required=True, help="FFE saison (e.g. 2024)")
    parser.add_argument("--output", type=Path, required=True, help="Artifact output path")
    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace prior alpha")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/echiquiers.parquet"),
        help="Input parquet path (default: data/echiquiers.parquet)",
    )
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    print(
        f"Training PreferenceModel saison={args.saison} "
        f"alpha={args.alpha} seed={args.seed} input_rows={len(df)}"
    )
    model = PreferenceModel(laplace_alpha=args.alpha, seed=args.seed)
    artifact = model.fit(df, args.saison)
    save_artifact(artifact, args.output)
    print(f"Saved artifact to {args.output}")
    print(f"  Input SHA-256     : {artifact.input_sha256}")
    print(f"  Artifact SHA-256  : {artifact.artifact_sha256}")
    print(f"  Train size        : {artifact.train_size}")
    print(f"  N teams max       : {artifact.n_teams_max}")
    print(f"  Bias gate skipped : {artifact.bias_gate_skipped}")


if __name__ == "__main__":
    main()
