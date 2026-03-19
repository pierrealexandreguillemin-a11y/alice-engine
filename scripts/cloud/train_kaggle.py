"""Kaggle Cloud Training — ALICE Engine orchestration (ISO 42001/5259/5055).

Orchestrates feature engineering + model training on Kaggle.
ML training logic is in scripts/kaggle_trainers.py (SRP).
Usage: python train_kaggle.py  (Kaggle kernel)
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Fix cudf device detection on Kaggle T4 (cudaErrorInvalidDevice)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# GPU-accelerate pandas via cudf.pandas — test in subprocess to catch C++ aborts
_CUDF_AVAILABLE = False
try:
    _r = subprocess.run(  # noqa: S603, S607
        [
            sys.executable,
            "-c",
            "import os; os.environ['CUDA_VISIBLE_DEVICES']='0'; "
            "import cudf.pandas; cudf.pandas.install(); import pandas as pd; "
            "pd.DataFrame({'a':['x','y','x']}).groupby('a').size(); print('OK')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if _r.returncode == 0 and "OK" in _r.stdout:
        import cudf.pandas  # noqa: F401, PLC0415

        cudf.pandas.install()
        _CUDF_AVAILABLE = True
except Exception:  # noqa: BLE001, S110
    pass  # CPU pandas fallback — logged in main()

import pandas as pd  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_REPO_ID = "Pierrax/alice-engine"
OUTPUT_DIR = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """SHA256 of hash_pandas_object, 16 hex chars."""
    hash_values = pd.util.hash_pandas_object(df, index=True)
    return hashlib.sha256(hash_values.values.tobytes()).hexdigest()[:16]


def build_lineage(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    data_dir: Path,
    label_column: str = "resultat_blanc",
) -> dict:
    """ISO 5259 data lineage."""
    return {
        "train": {
            "path": str(data_dir / "train.parquet"),
            "samples": len(train),
            "hash": compute_dataframe_hash(train),
        },
        "valid": {
            "path": str(data_dir / "valid.parquet"),
            "samples": len(valid),
            "hash": compute_dataframe_hash(valid),
        },
        "test": {
            "path": str(data_dir / "test.parquet"),
            "samples": len(test),
            "hash": compute_dataframe_hash(test),
        },
        "feature_count": len(train.columns) - 1,
        "target_distribution": {
            "positive_ratio": float((train[label_column] == 1.0).mean()),
            "total_samples": len(train),
        },
        "created_at": datetime.now(tz=UTC).isoformat(),
    }


def collect_environment() -> dict:
    """Python version, platform, package versions, Kaggle kernel ID."""
    pkgs = {}
    for pkg in ("catboost", "xgboost", "lightgbm", "pandas", "scikit-learn", "huggingface_hub"):
        try:
            import importlib.metadata as im  # noqa: PLC0415

            pkgs[pkg] = im.version(pkg)
        except Exception:
            pkgs[pkg] = "unknown"
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "kaggle_kernel_id": os.environ.get("KAGGLE_KERNEL_RUN_SLUG", "local"),
        "packages": pkgs,
    }


def _artifact_entry(name: str, path: Path) -> dict:
    """Single artifact dict with sha256 + size_bytes."""
    if not path.exists():
        return {"name": name, "path": str(path), "sha256": "n/a", "size_bytes": 0}
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"name": name, "path": str(path), "sha256": sha, "size_bytes": path.stat().st_size}


def build_model_card(
    results: dict,
    lineage: dict,
    gate: dict,
    config: dict,
    model_extensions: dict,
    *,
    out_dir: Path | None = None,
) -> dict:
    """ISO 42001 Model Card with status=CANDIDATE."""
    env = collect_environment()
    version = datetime.now(tz=UTC).strftime("v%Y%m%d_%H%M%S")
    metrics = {n: r["metrics"] for n, r in results.items() if r["model"] is not None}
    importance = {n: r["importance"] for n, r in results.items() if r["model"] is not None}
    artifact_dir = out_dir if out_dir else OUTPUT_DIR
    artifacts = [
        _artifact_entry(n, artifact_dir / f"{n}{model_extensions.get(n, '.pkl')}") for n in metrics
    ]
    # fmt: off
    return {
        "version": version, "created_at": datetime.now(tz=UTC).isoformat(),
        "status": "CANDIDATE", "environment": env, "data_lineage": lineage,
        "artifacts": artifacts, "metrics": metrics, "feature_importance": importance,
        "hyperparameters": config, "best_model": {"name": gate.get("best_model"), "auc": gate.get("best_auc")},
        "quality_gate_result": gate,
        "limitations": ["Trained on FFE interclub data only", "Not suitable for tournament games"],
        "use_cases": ["Team composition outcome prediction"],
        "conformance": {"ISO_42001": "CANDIDATE", "ISO_5259": "COMPLIANT", "ISO_5055": "COMPLIANT"},
    }
    # fmt: on


def fetch_champion_auc() -> float | None:
    """Fetch best_model.auc from latest metadata.json on HF Hub."""
    try:
        import json  # noqa: PLC0415

        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        path = hf_hub_download(HF_REPO_ID, "metadata.json", repo_type="model")
        with open(path) as fh:
            data = json.load(fh)
        best_model = data.get("best_model", {})
        auc = float(best_model.get("auc", 0.0)) if best_model else 0.0
        return auc if auc > 0 else None
    except Exception:
        logger.warning("Could not fetch champion AUC — first run assumed.")
        return None


def _get_hf_token() -> str | None:
    """Get HF token from Kaggle Secrets, env var, or HF cache."""
    for attempt in range(3):
        try:
            from kaggle_secrets import UserSecretsClient  # noqa: PLC0415

            token = UserSecretsClient().get_secret("HF_TOKEN")
            if token and token.strip():
                logger.info("HF token found via Kaggle Secrets (len=%d)", len(token.strip()))
                return token.strip()
            logger.warning("Kaggle Secret HF_TOKEN is empty")
            break
        except Exception as exc:  # noqa: BLE001
            logger.info("Kaggle Secrets attempt %d/3: %s", attempt + 1, exc)
            if attempt < 2:
                import time  # noqa: PLC0415

                time.sleep(2)
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        logger.info("HF token found via env var (len=%d)", len(token))
        return token
    cache = Path.home() / ".cache" / "huggingface" / "token"
    if cache.exists():
        token = cache.read_text().strip()
        if token:
            logger.info("HF token found via HF cache (len=%d)", len(token))
            return token
    logger.warning("No HF token found (tried: Kaggle Secrets, env var, HF cache)")
    return None


def save_metadata_and_push(metadata: dict, out_dir: Path) -> None:
    """Save metadata.json and optionally push to HF Hub."""
    import json  # noqa: PLC0415

    token = _get_hf_token()
    with open(out_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    if token:
        from huggingface_hub import HfApi  # noqa: PLC0415

        version = metadata["version"]
        HfApi().upload_folder(
            folder_path=str(out_dir),
            repo_id=HF_REPO_ID,
            repo_type="model",
            path_in_repo=version,
            token=token,
        )
        logger.info("Pushed %s to HF Hub: %s", version, HF_REPO_ID)
    else:
        logger.warning("HF_TOKEN not set — skipping HF Hub push.")


def _setup_hf_auth() -> None:
    """Authenticate with HuggingFace Hub early."""
    token = _get_hf_token()
    if token:
        try:
            from huggingface_hub import login  # noqa: PLC0415

            login(token=token, add_to_git_credential=False)
            logger.info("HuggingFace Hub authenticated")
        except Exception as exc:  # noqa: BLE001
            logger.warning("HF login failed: %s", exc)


def _setup_kaggle_imports() -> None:
    """Find alice-code dataset and add to sys.path."""
    import zipfile  # noqa: PLC0415

    kaggle_input = None
    for candidate in [
        Path("/kaggle/input/alice-code"),
        Path("/kaggle/input/datasets/pguillemin/alice-code"),
    ]:
        if candidate.exists():
            kaggle_input = candidate
            break

    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        all_files = list(kaggle_root.rglob("*"))
        logger.info(
            "/kaggle/input/ tree (%d items): %s", len(all_files), [str(f) for f in all_files[:30]]
        )
    logger.info("kaggle_input=%s", kaggle_input)

    if kaggle_input:
        zips = list(kaggle_input.rglob("*.zip"))
        if zips:
            work_dir = Path("/kaggle/working/code")
            work_dir.mkdir(parents=True, exist_ok=True)
            for zf in zips:
                logger.info("Unzipping %s", zf.name)
                with zipfile.ZipFile(zf) as z:
                    z.extractall(work_dir)
            sys.path.insert(0, str(work_dir))
            logger.info("sys.path += %s (unzipped)", work_dir)
        else:
            sys.path.insert(0, str(kaggle_input))
            logger.info("sys.path += %s (direct)", kaggle_input)


def _load_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    """Compute features from raw parquets. Returns (train, valid, test, features_dir)."""
    logger.info("Computing features from raw data")
    out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    data_dir = out / "code" / "data"

    kaggle_data = Path("/kaggle/input/datasets/pguillemin/alice-code/data")
    if (kaggle_data / "echiquiers.parquet").exists():
        data_dir = kaggle_data
    elif not (data_dir / "echiquiers.parquet").exists():
        data_dir = Path("data")
    logger.info(
        "Raw data dir: %s, contents: %s",
        data_dir,
        list(data_dir.iterdir()) if data_dir.exists() else [],
    )

    from scripts.feature_engineering import run_feature_engineering_v2  # noqa: PLC0415

    features_dir = out / "data" / "features"
    run_feature_engineering_v2(data_dir=data_dir, output_dir=features_dir, include_advanced=True)

    return (
        pd.read_parquet(features_dir / "train.parquet"),
        pd.read_parquet(features_dir / "valid.parquet"),
        pd.read_parquet(features_dir / "test.parquet"),
        features_dir,
    )


def main() -> None:
    """Full Kaggle training pipeline orchestration (ISO 42001)."""
    logger.info("ALICE Engine — Kaggle Cloud Training (cudf=%s)", _CUDF_AVAILABLE)
    _setup_kaggle_imports()
    _setup_hf_auth()

    # Late import: scripts.kaggle_trainers is in the Kaggle dataset (via sys.path)
    from scripts.kaggle_trainers import (  # noqa: PLC0415
        LABEL_COLUMN,
        MODEL_EXTENSIONS,
        check_quality_gates,
        default_hyperparameters,
        evaluate_on_test,
        prepare_features,
        save_models,
        train_all_sequential,
    )

    train, valid, test, features_dir = _load_features()
    lineage = build_lineage(train, valid, test, features_dir, label_column=LABEL_COLUMN)
    logger.info("Lineage: train=%d valid=%d test=%d", len(train), len(valid), len(test))

    X_train, y_train, X_valid, y_valid, X_test, y_test, encoders = prepare_features(
        train,
        valid,
        test,
    )
    config = default_hyperparameters()
    results = train_all_sequential(X_train, y_train, X_valid, y_valid, config)
    evaluate_on_test(results, X_test, y_test)

    champion_auc = fetch_champion_auc()
    gate = check_quality_gates(results, champion_auc=champion_auc)
    logger.info("Quality gate: %s", gate)

    version = datetime.now(tz=UTC).strftime("v%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / version
    save_models(results, encoders, out_dir)
    metadata = build_model_card(results, lineage, gate, config, MODEL_EXTENSIONS, out_dir=out_dir)
    metadata["version"] = version
    save_metadata_and_push(metadata, out_dir)
    logger.info(
        "Done. Status=%s Best=%s AUC=%.4f",
        metadata["status"],
        gate.get("best_model"),
        gate.get("best_auc", 0),
    )


if __name__ == "__main__":
    main()
