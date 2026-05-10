"""Upload alice-d8-input + alice-d8-code datasets to Kaggle.

Stages :
- `alice-d8-input` : data parquets + ML artefacts (8 files) + FFE rules (2) +
  feature store (training_mean.parquet) + dataset-metadata.json
- `alice-d8-code` : Python source tree (39 modules) + dataset-metadata.json

The 4 saison kernels + d8-aggregator depend on both datasets.

Also injects the current git rev-parse HEAD into ALICE_CODE_SHA in each
`kernel-metadata-saison-*.json` (ISO 5259 lineage).

Usage :
    python -m scripts.d8.upload_d8_dataset           # stage + push to Kaggle
    python -m scripts.d8.upload_d8_dataset --check   # validate sources only
    python -m scripts.d8.upload_d8_dataset --stage   # stage locally, no push

Document ID: ALICE-D8-UPLOAD
Version: 2.0.0
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_INPUT_SLUG = "alice-d8-input"
DATASET_CODE_SLUG = "alice-d8-code"
KAGGLE_USERNAME = "pguillemin"
REPO = Path(__file__).resolve().parent.parent.parent

# (source, staged_subpath) for alice-d8-input — fail-fast if missing.
INPUT_ARTEFACT_SOURCES: tuple[tuple[Path, str], ...] = (
    # Parquets
    (REPO / "data" / "joueurs.parquet", "data/joueurs.parquet"),
    (REPO / "data" / "echiquiers.parquet", "data/echiquiers.parquet"),
    # Feature store (FeatureStore.load() needs this)
    (
        REPO / "data" / "feature_store" / "training_mean.parquet",
        "data/feature_store/training_mean.parquet",
    ),
    (
        REPO / "data" / "feature_store" / "joueur_features.parquet",
        "data/feature_store/joueur_features.parquet",
    ),
    # ML artefacts (load_models requires LGB; XGB/CB optional fallback to Dirichlet)
    (REPO / "models" / "cache" / "LightGBM.txt", "artefacts/LightGBM.txt"),
    (REPO / "models" / "cache" / "XGBoost.ubj", "artefacts/XGBoost.ubj"),
    (REPO / "models" / "cache" / "CatBoost.cbm", "artefacts/CatBoost.cbm"),
    (
        REPO / "models" / "cache" / "draw_rate_lookup.parquet",
        "artefacts/draw_rate_lookup.parquet",
    ),
    (REPO / "models" / "cache" / "encoders.joblib", "artefacts/encoders.joblib"),
    (
        REPO / "models" / "cache" / "lightgbm_dirichlet.joblib",
        "artefacts/lightgbm_dirichlet.joblib",
    ),
    (
        REPO / "models" / "cache" / "mlp_meta_learner.joblib",
        "artefacts/mlp_meta_learner.joblib",
    ),
    (REPO / "models" / "cache" / "temperature_T.joblib", "artefacts/temperature_T.joblib"),
    # FFE rules (RuleEngine + VerifiabilityClassifier)
    (REPO / "config" / "ffe_rules" / "a02.json", "config/ffe_rules/a02.json"),
    (
        REPO / "config" / "ffe_rules" / "alice_verifiability.json",
        "config/ffe_rules/alice_verifiability.json",
    ),
)

# Python source tree to stage in alice-d8-code (Kaggle prepends /kaggle/input/<slug>
# to sys.path at kernel runtime; run.py / aggregate.py rely on package layout).
CODE_PACKAGE_DIRS: tuple[str, ...] = (
    "app",
    "scripts/d8",
    "scripts/backtest",
    "scripts/features",
    "scripts/serving",
    "services/ali",
    "services/ffe",
)
CODE_TOPLEVEL_FILES: tuple[str, ...] = (
    "scripts/__init__.py",
    "scripts/baselines.py",
    "scripts/kaggle_metrics.py",
    "scripts/kaggle_quality_gates.py",
    "scripts/d8/kaggle-requirements.txt",  # pip install at kernel boot
    "services/__init__.py",
    "services/inference.py",
    "services/feature_store.py",
    "services/data_loader.py",  # imported by services/__init__.py
)
# kaggle-deployment skill : `kaggle kernels push` uploads ONLY code_file.
# Per-saison thin wrappers (run_2021.py … run_2024.py) are referenced by
# kernel-metadata-saison-*.json::code_file. They live in scripts/d8/ and
# are also staged in alice-d8-code so cross-imports resolve identically.
CODE_PER_SAISON_WRAPPERS: tuple[str, ...] = (
    "scripts/d8/run_2021.py",
    "scripts/d8/run_2022.py",
    "scripts/d8/run_2023.py",
    "scripts/d8/run_2024.py",
)


def _validate_input_sources() -> None:
    """ISO 5259 fail-fast if any input file missing."""
    missing = [str(src) for src, _ in INPUT_ARTEFACT_SOURCES if not src.exists()]
    if missing:
        msg = f"Missing input files for upload: {missing}"
        raise FileNotFoundError(msg)


def _validate_code_sources() -> None:
    """Fail-fast if any required Python module missing."""
    missing: list[str] = []
    for pkg in CODE_PACKAGE_DIRS:
        if not (REPO / pkg).is_dir():
            missing.append(pkg)
    for f in (*CODE_TOPLEVEL_FILES, *CODE_PER_SAISON_WRAPPERS):
        if not (REPO / f).is_file():
            missing.append(f)
    if missing:
        msg = f"Missing Python sources: {missing}"
        raise FileNotFoundError(msg)


def _stage_input(staging: Path) -> None:
    """Stage data + artefacts + config under alice-d8-input/."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    for src, dst_subpath in INPUT_ARTEFACT_SOURCES:
        dst = staging / dst_subpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)


def _copy_python_tree(src_dir: Path, dst_dir: Path) -> None:
    """Recursive copy of *.py only (skip __pycache__, .pyc, .ipynb_checkpoints)."""
    for src in src_dir.rglob("*.py"):
        rel = src.relative_to(REPO)
        if "__pycache__" in rel.parts:
            continue
        dst = dst_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)


def _stage_code(staging: Path) -> None:
    """Stage Python source tree under alice-d8-code/."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    for pkg in CODE_PACKAGE_DIRS:
        _copy_python_tree(REPO / pkg, staging)
    for f in CODE_TOPLEVEL_FILES:
        dst = staging / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / f, dst)


def _write_dataset_metadata(staging: Path, slug: str, title: str) -> None:
    """Write dataset-metadata.json per Kaggle CLI requirements."""
    metadata = {
        "title": title,
        "id": f"{KAGGLE_USERNAME}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (staging / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _git_head_sha() -> str:
    """Resolve current git HEAD SHA for ALICE_CODE_SHA lineage injection."""
    result = subprocess.run(  # noqa: S603 - args are static literals
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    )
    return result.stdout.strip()


def _write_code_sha(code_staging: Path, code_sha: str) -> None:
    """Stage CODE_SHA.txt in alice-d8-code at upload time.

    kaggle-deployment skill 2026-03-29 : env vars not honored at runtime, so
    run.py reads CODE_SHA.txt at boot via _resolve_code_sha().
    """
    (code_staging / "CODE_SHA.txt").write_text(code_sha + "\n", encoding="utf-8")


def _kaggle_create_or_version(staging: Path, msg: str) -> None:
    """Try `kaggle datasets create`; fall back to `version` if exists."""
    create = subprocess.run(  # noqa: S603 - args are static literals
        ["kaggle", "datasets", "create", "-p", str(staging), "-r", "zip"],
        capture_output=False,
        text=True,
        check=False,
    )
    if create.returncode == 0:
        return
    subprocess.run(  # noqa: S603
        ["kaggle", "datasets", "version", "-p", str(staging), "-m", msg, "-r", "zip"],
        check=True,
    )


def main() -> int:
    """Stage + (optionally) push both datasets to Kaggle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate sources only")
    parser.add_argument("--stage", action="store_true", help="Stage locally, no Kaggle push")
    args = parser.parse_args()

    _validate_input_sources()
    _validate_code_sources()
    if args.check:
        sys.stdout.write(
            f"OK: {len(INPUT_ARTEFACT_SOURCES)} input files + "
            f"{len(CODE_PACKAGE_DIRS)} package dirs + "
            f"{len(CODE_TOPLEVEL_FILES)} top-level python files present.\n"
        )
        return 0

    code_sha = _git_head_sha()
    sys.stdout.write(f"Git HEAD SHA = {code_sha}\n")

    input_staging = REPO / "build" / "kaggle" / DATASET_INPUT_SLUG
    code_staging = REPO / "build" / "kaggle" / DATASET_CODE_SLUG
    _stage_input(input_staging)
    _write_dataset_metadata(input_staging, DATASET_INPUT_SLUG, "alice d8 input")
    _stage_code(code_staging)
    _write_dataset_metadata(code_staging, DATASET_CODE_SLUG, "alice d8 code")

    _write_code_sha(code_staging, code_sha)
    sys.stdout.write(f"Staged CODE_SHA.txt in alice-d8-code (sha={code_sha[:7]})\n")

    if args.stage:
        sys.stdout.write(f"Staged at {input_staging} + {code_staging} (no push).\n")
        return 0

    _kaggle_create_or_version(input_staging, f"D8 input refresh code_sha={code_sha[:7]}")
    _kaggle_create_or_version(code_staging, f"D8 code refresh code_sha={code_sha[:7]}")
    sys.stdout.write(f"Datasets {DATASET_INPUT_SLUG} + {DATASET_CODE_SLUG} uploaded.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
