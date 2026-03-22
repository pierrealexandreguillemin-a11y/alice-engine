"""Kaggle training artifacts — lineage, model card, metadata (ISO 42001/5259)."""

from __future__ import annotations

import hashlib  # noqa: E401
import json
import logging
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd  # noqa: TCH002

logger = logging.getLogger(__name__)
HF_REPO_ID = "Pierrax/alice-engine"


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """SHA256 of hash_pandas_object, 16 hex chars."""
    h = pd.util.hash_pandas_object(df, index=True)
    return hashlib.sha256(h.values.tobytes()).hexdigest()[:16]


def build_lineage(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    data_dir: Path,
    label_column: str = "resultat_blanc",
) -> dict:
    """ISO 5259 data lineage."""

    def _entry(name: str, df: pd.DataFrame) -> dict:
        return {
            "path": str(data_dir / f"{name}.parquet"),
            "samples": len(df),
            "hash": compute_dataframe_hash(df),
        }

    return {
        "train": _entry("train", train),
        "valid": _entry("valid", valid),
        "test": _entry("test", test),
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
    artifact_dir = out_dir if out_dir else Path("/kaggle/working")
    artifacts = [
        _artifact_entry(n, artifact_dir / f"{n}{model_extensions.get(n, '.pkl')}") for n in metrics
    ]
    # fmt: off
    return {
        "version": version, "created_at": datetime.now(tz=UTC).isoformat(),
        "status": "CANDIDATE", "environment": env, "data_lineage": lineage,
        "artifacts": artifacts, "metrics": metrics, "feature_importance": importance,
        "hyperparameters": config,
        "best_model": {"name": gate.get("best_model"), "log_loss": gate.get("best_log_loss")},
        "quality_gate_result": gate,
        "limitations": ["Trained on FFE interclub data only", "Not suitable for tournament games",
                        "LightGBM: CPU only (pip package lacks GPU/OpenCL support on Kaggle)"],
        "use_cases": ["Team composition outcome prediction"],
        "conformance": {"ISO_42001": "CANDIDATE", "ISO_5259": "COMPLIANT", "ISO_5055": "COMPLIANT"},
    }
    # fmt: on


def _artifact_entry(name: str, path: Path) -> dict:
    if not path.exists():
        return {"name": name, "path": str(path), "sha256": "n/a", "size_bytes": 0}
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"name": name, "path": str(path), "sha256": sha, "size_bytes": path.stat().st_size}


def fetch_champion_auc() -> float | None:
    """Fetch best_model.auc from latest metadata.json on HF Hub (legacy)."""
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        path = hf_hub_download(HF_REPO_ID, "metadata.json", repo_type="model")
        with open(path) as fh:
            data = json.load(fh)
        best = data.get("best_model", {})
        auc = float(best.get("auc", 0.0)) if best else 0.0
        return auc if auc > 0 else None
    except Exception:
        logger.warning("Could not fetch champion AUC — first run assumed.")
        return None


def fetch_champion_ll() -> float | None:
    """Fetch best_log_loss from latest metadata.json on HF Hub (multiclass gate)."""
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        path = hf_hub_download(HF_REPO_ID, "metadata.json", repo_type="model")
        with open(path) as fh:
            data = json.load(fh)
        gate = data.get("quality_gate_result", {})
        ll = float(gate.get("best_log_loss", 0.0)) if gate else 0.0
        return ll if ll > 0 else None
    except Exception:
        logger.warning("Could not fetch champion log_loss — first run assumed.")
        return None


def save_metadata_and_push(metadata: dict, out_dir: Path) -> None:
    """Save metadata.json and optionally push to HF Hub."""
    token = _get_hf_token()
    with open(out_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    if token:
        from huggingface_hub import HfApi  # noqa: PLC0415

        HfApi().upload_folder(
            folder_path=str(out_dir),
            repo_id=HF_REPO_ID,
            repo_type="model",
            path_in_repo=metadata["version"],
            token=token,
        )
        logger.info("Pushed %s to HF Hub: %s", metadata["version"], HF_REPO_ID)
    else:
        logger.warning("HF_TOKEN not set — skipping HF Hub push.")


def _get_hf_token() -> str | None:
    """Get HF token from Kaggle Secrets, env var, or HF cache."""
    import time as _time  # noqa: PLC0415

    for attempt in range(3):
        try:
            from kaggle_secrets import UserSecretsClient  # noqa: PLC0415

            token = UserSecretsClient().get_secret("HF_TOKEN")
            if token and token.strip():
                return token.strip()
            break
        except Exception as exc:  # noqa: BLE001
            logger.info("Kaggle Secrets attempt %d/3: %s", attempt + 1, exc)
            if attempt < 2:
                _time.sleep(2)
    for source, val in [("env", os.environ.get("HF_TOKEN", "")), ("cache", _read_hf_cache())]:
        if val and val.strip():
            logger.info("HF token found via %s", source)
            return val.strip()
    logger.warning("No HF token found")
    return None


def _read_hf_cache() -> str:
    cache = Path.home() / ".cache" / "huggingface" / "token"
    return cache.read_text().strip() if cache.exists() else ""


def setup_hf_auth() -> None:
    """Authenticate with HuggingFace Hub early."""
    token = _get_hf_token()
    if token:
        try:
            from huggingface_hub import login  # noqa: PLC0415

            login(token=token, add_to_git_credential=False)
            logger.info("HuggingFace Hub authenticated")
        except Exception as exc:  # noqa: BLE001
            logger.warning("HF login failed: %s", exc)


def save_models(
    results: dict, encoders: dict, out_dir: Path, model_extensions: dict | None = None
) -> None:
    """Save model artifacts in native formats + encoders via joblib (I3)."""
    import joblib  # noqa: PLC0415

    _ext = model_extensions or {"CatBoost": ".cbm", "XGBoost": ".ubj", "LightGBM": ".txt"}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, r in results.items():
        model = r["model"]
        if model is None:
            continue
        ext = _ext.get(name, ".pkl")
        path = out_dir / f"{name}{ext}"
        if name == "CatBoost":
            model.save_model(str(path))
        elif name == "XGBoost":
            model.save_model(str(path))
        elif name == "LightGBM":
            model.booster_.save_model(str(path))
        else:
            joblib.dump(model, path)
    joblib.dump(encoders, out_dir / "encoders.joblib")
