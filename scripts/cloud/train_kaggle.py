"""Kaggle Cloud Training Script — ALICE Engine (ISO 42001/5259/5055).

Self-contained: CatBoost / XGBoost / LightGBM → CANDIDATE on HF Hub.
Usage: python train_kaggle.py  (Kaggle kernel)
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# fmt: off
CATEGORICAL_FEATURES = ["type_competition", "division", "ligue_code", "jour_semaine"]
CATBOOST_CAT_FEATURES = [
    "type_competition", "division", "ligue_code",
    "blanc_titre", "noir_titre", "jour_semaine", "zone_enjeu_dom",
]
# fmt: on
LABEL_COLUMN = "resultat_blanc"
AUC_FLOOR = 0.70
HF_REPO_ID = "Pierrax/alice-engine"
DATA_DIR = Path(os.environ.get("KAGGLE_DATA_DIR", "/kaggle/input/alice-features"))
OUTPUT_DIR = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """SHA256 of hash_pandas_object, 16 hex chars. Matches model_registry/utils.py."""
    hash_values = pd.util.hash_pandas_object(df, index=True)
    return hashlib.sha256(hash_values.values.tobytes()).hexdigest()[:16]


def build_lineage(
    train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, data_dir: Path
) -> dict:
    """ISO 5259 data lineage. Nested format matching production metadata.json."""
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
            "positive_ratio": float((train[LABEL_COLUMN] == 1.0).mean()),
            "total_samples": len(train),
        },
        "created_at": datetime.now(tz=UTC).isoformat(),
    }


def _encode_categoricals(
    splits: list[pd.DataFrame],
) -> dict:
    """Label-encode all categorical columns across splits. Fit on first split."""
    from sklearn.preprocessing import LabelEncoder

    all_cat_cols = sorted(set(CATEGORICAL_FEATURES) | set(CATBOOST_CAT_FEATURES))
    encoders: dict = {}
    for split in splits:
        for col in all_cat_cols:
            if col not in split.columns:
                continue
            if col not in encoders:
                enc = LabelEncoder()
                split[col] = enc.fit_transform(split[col].astype(str))
                encoders[col] = enc
            else:
                split[col] = encoders[col].transform(split[col].astype(str))
    return encoders


def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X and y, dropping target + string columns."""
    y = (df[LABEL_COLUMN] == 1.0).astype(int)
    drop_cols = [LABEL_COLUMN, "resultat_noir", "resultat_text"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X = X.drop(columns=obj_cols)
    return X, y


def prepare_features(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Label-encode categoricals, split X/y, drop remaining strings."""
    encoders = _encode_categoricals([train, valid, test])
    X_train, y_train = _split_xy(train)
    X_valid, y_valid = _split_xy(valid)
    X_test, y_test = _split_xy(test)
    return X_train, y_train, X_valid, y_valid, X_test, y_test, encoders


def default_hyperparameters() -> dict:
    """Matching config/hyperparameters.yaml. thread_count/n_jobs=4 for Kaggle."""
    # fmt: off
    return {
        "global": {"random_seed": 42, "early_stopping_rounds": 50, "eval_metric": "auc"},
        "catboost": {
            "iterations": 1000, "learning_rate": 0.03, "depth": 6,
            "l2_leaf_reg": 3, "min_data_in_leaf": 20, "thread_count": 4,
            "task_type": "CPU", "random_seed": 42, "verbose": 100, "early_stopping_rounds": 50,
        },
        "xgboost": {
            "n_estimators": 1000, "learning_rate": 0.03, "max_depth": 6,
            "reg_lambda": 1.0, "reg_alpha": 0.0, "min_child_weight": 1,
            "tree_method": "hist", "n_jobs": 4, "random_state": 42,
            "early_stopping_rounds": 50, "verbosity": 1,
        },
        "lightgbm": {
            "n_estimators": 1000, "learning_rate": 0.03, "num_leaves": 63,
            "max_depth": -1, "reg_lambda": 1.0, "reg_alpha": 0.0,
            "min_child_samples": 20, "n_jobs": 4, "random_state": 42,
            "early_stopping_rounds": 50, "verbose": -1,
        },
    }
    # fmt: on


def compute_validation_metrics(y_true: Any, y_pred: Any, y_proba: Any) -> dict:
    """10-field metrics on validation set. Matches scripts/training/metrics.py."""
    from sklearn.metrics import (  # noqa: PLC0415
        accuracy_score,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    cm = confusion_matrix(y_true, y_pred)
    # fmt: off
    return {
        "auc_roc": float(roc_auc_score(y_true, y_proba)), "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "true_negatives": int(cm[0, 0]), "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]), "true_positives": int(cm[1, 1]),
    }
    # fmt: on


def _eval_model(model: Any, X_valid: Any, y_valid: Any, train_time: float) -> dict:
    """Evaluate model on validation, return {model, metrics, importance}."""
    import numpy as np

    y_proba = model.predict_proba(X_valid)[:, 1]
    y_pred = (y_proba >= 0.5).astype(np.int64)
    metrics = compute_validation_metrics(y_valid.values, y_pred, y_proba)
    metrics["train_time_s"] = train_time
    importance = (
        dict(zip(X_valid.columns, model.feature_importances_, strict=False))
        if hasattr(model, "feature_importances_")
        else {}
    )
    return {"model": model, "metrics": metrics, "importance": importance}


def _fail_result() -> dict:
    """Sentinel result for a model that failed to train (I1)."""
    return {"model": None, "metrics": {}, "importance": {}}


def _train_catboost(X_train: Any, y_train: Any, X_valid: Any, y_valid: Any, params: dict) -> dict:
    """Train CatBoost with partial-failure handling (I1)."""
    try:
        from catboost import CatBoostClassifier  # noqa: PLC0415

        cat_idx = [
            X_train.columns.get_loc(c) for c in CATBOOST_CAT_FEATURES if c in X_train.columns
        ]
        cb = CatBoostClassifier(**params, cat_features=cat_idx, eval_metric="AUC")
        t0 = time.time()
        cb.fit(X_train, y_train, eval_set=(X_valid, y_valid))
        result = _eval_model(cb, X_valid, y_valid, time.time() - t0)
        del cb
        gc.collect()
        return result
    except Exception:
        logger.exception("CatBoost training failed")
        return _fail_result()


def _train_xgboost(X_train: Any, y_train: Any, X_valid: Any, y_valid: Any, params: dict) -> dict:
    """Train XGBoost with partial-failure handling (I1)."""
    try:
        from xgboost import XGBClassifier  # noqa: PLC0415

        xgb = XGBClassifier(**params, eval_metric="auc")
        t0 = time.time()
        xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)
        result = _eval_model(xgb, X_valid, y_valid, time.time() - t0)
        del xgb
        gc.collect()
        return result
    except Exception:
        logger.exception("XGBoost training failed")
        return _fail_result()


def _train_lightgbm(X_train: Any, y_train: Any, X_valid: Any, y_valid: Any, params: dict) -> dict:
    """Train LightGBM with partial-failure handling (I1)."""
    try:
        import lightgbm as lgb_lib  # noqa: PLC0415
        from lightgbm import LGBMClassifier  # noqa: PLC0415

        lgb_p = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        lgbm = LGBMClassifier(**lgb_p)
        t0 = time.time()
        lgbm.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[
                lgb_lib.early_stopping(params.get("early_stopping_rounds", 50)),
                lgb_lib.log_evaluation(100),
            ],
        )
        result = _eval_model(lgbm, X_valid, y_valid, time.time() - t0)
        del lgbm
        gc.collect()
        return result
    except Exception:
        logger.exception("LightGBM training failed")
        return _fail_result()


def train_all_sequential(  # noqa: PLR0914
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
    config: dict,
) -> dict:
    """CatBoost -> gc -> XGBoost -> gc -> LightGBM. Partial-failure safe (I1)."""
    # fmt: off
    return {
        "CatBoost": _train_catboost(X_train, y_train, X_valid, y_valid, config["catboost"]),
        "XGBoost": _train_xgboost(X_train, y_train, X_valid, y_valid, config["xgboost"]),
        "LightGBM": _train_lightgbm(X_train, y_train, X_valid, y_valid, config["lightgbm"]),
    }
    # fmt: on


def evaluate_on_test(results: dict, X_test: Any, y_test: Any) -> None:
    """Compute test_auc, test_accuracy, test_f1 for each model. Mutates results."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: PLC0415

    for _name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = r["model"].predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(np.int64)
        r["metrics"]["test_auc"] = float(roc_auc_score(y_test, y_proba))
        r["metrics"]["test_accuracy"] = float(accuracy_score(y_test, y_pred))
        r["metrics"]["test_f1"] = float(f1_score(y_test, y_pred, zero_division=0))


def check_quality_gates(results: dict, champion_auc: float | None = None) -> dict:
    """ISO 42001: AUC floor + relative degradation (2%) gate."""
    best_name, best_r = max(
        ((n, r) for n, r in results.items() if r["model"] is not None),
        key=lambda x: x[1]["metrics"]["test_auc"],
    )
    best_auc = best_r["metrics"]["test_auc"]
    if best_auc < AUC_FLOOR:
        return {"passed": False, "reason": f"AUC {best_auc:.4f} < {AUC_FLOOR}"}
    if champion_auc and champion_auc > 0:
        drop_pct = (champion_auc - best_auc) / champion_auc * 100
        if drop_pct > 2.0:
            return {"passed": False, "reason": f"Degradation {drop_pct:.1f}% > 2.0%"}
    return {"passed": True, "best_model": best_name, "best_auc": best_auc}


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


def collect_environment() -> dict:
    """Python version, platform, package versions, Kaggle kernel ID."""
    pkgs = {}
    for pkg in ("catboost", "xgboost", "lightgbm", "pandas", "scikit-learn", "huggingface_hub"):
        try:
            import importlib.metadata as im  # noqa: PLC0415

            pkgs[pkg] = im.version(pkg)
        except Exception:
            pkgs[pkg] = "unknown"
    # fmt: off
    return {"python_version": sys.version, "platform": platform.platform(),
            "kaggle_kernel_id": os.environ.get("KAGGLE_KERNEL_RUN_SLUG", "local"), "packages": pkgs}
    # fmt: on


def _artifact_entry(name: str, path: Path) -> dict:
    """Single artifact dict with sha256 + size_bytes."""
    if not path.exists():
        return {"name": name, "path": str(path), "sha256": "n/a", "size_bytes": 0}
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"name": name, "path": str(path), "sha256": sha, "size_bytes": path.stat().st_size}


def build_model_card(
    results: dict, lineage: dict, gate: dict, config: dict, *, out_dir: Path | None = None
) -> dict:
    """ISO 42001 Model Card with status=CANDIDATE."""
    env = collect_environment()
    version = datetime.now(tz=UTC).strftime("v%Y%m%d_%H%M%S")
    metrics = {n: r["metrics"] for n, r in results.items() if r["model"] is not None}
    importance = {n: r["importance"] for n, r in results.items() if r["model"] is not None}
    artifact_dir = out_dir if out_dir else OUTPUT_DIR
    artifacts = [
        _artifact_entry(n, artifact_dir / f"{n}{MODEL_EXTENSIONS.get(n, '.pkl')}") for n in metrics
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


MODEL_EXTENSIONS = {"CatBoost": ".cbm", "XGBoost": ".ubj", "LightGBM": ".txt"}


def save_models(results: dict, encoders: dict, out_dir: Path) -> None:
    """Save model artifacts in native formats + encoders via joblib (I3)."""
    import joblib  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, r in results.items():
        model = r["model"]
        if model is None:
            continue
        ext = MODEL_EXTENSIONS.get(name, ".pkl")
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


def _get_hf_token() -> str | None:
    """Get HF token from Kaggle Secrets, env var, or HF cache."""
    # 1. Kaggle Secrets (preferred on Kaggle)
    try:
        from kaggle_secrets import UserSecretsClient  # noqa: PLC0415

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:  # noqa: BLE001, S110
        logger.debug("Kaggle Secrets not available")
    # 2. Environment variable
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    # 3. HuggingFace cache (local dev)
    cache = Path.home() / ".cache" / "huggingface" / "token"
    if cache.exists():
        return cache.read_text().strip()
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
        # fmt: off
        HfApi().upload_folder(folder_path=str(out_dir), repo_id=HF_REPO_ID,
                              repo_type="model", path_in_repo=version, token=token)
        # fmt: on
        logger.info("Pushed %s to HF Hub: %s", version, HF_REPO_ID)
    else:
        logger.warning("HF_TOKEN not set — skipping HF Hub push.")


def _find_parquet(data_dir: Path, name: str) -> Path:
    """Find parquet file in data_dir or its subdirectories."""
    direct = data_dir / name
    if direct.exists():
        return direct
    # Kaggle zip datasets may nest files in a subdirectory
    found = list(data_dir.rglob(name))
    if found:
        return found[0]
    msg = f"{name} not found in {data_dir}. Contents: {list(data_dir.rglob('*'))[:10]}"
    raise FileNotFoundError(msg)


def _load_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load features from Kaggle input, local path, or HuggingFace Hub."""
    # 1. Kaggle mounted dataset
    if DATA_DIR.exists() and list(DATA_DIR.rglob("*.parquet")):
        logger.info("Loading from Kaggle input: %s", DATA_DIR)
        return (
            pd.read_parquet(_find_parquet(DATA_DIR, "train.parquet")),
            pd.read_parquet(_find_parquet(DATA_DIR, "valid.parquet")),
            pd.read_parquet(_find_parquet(DATA_DIR, "test.parquet")),
        )
    # 2. Local path fallback
    local = Path("data/features")
    if local.exists() and (local / "train.parquet").exists():
        logger.info("Loading from local: %s", local)
        return (
            pd.read_parquet(local / "train.parquet"),
            pd.read_parquet(local / "valid.parquet"),
            pd.read_parquet(local / "test.parquet"),
        )
    # 3. HuggingFace Hub (always works with internet)
    logger.info("Loading from HuggingFace Hub: Pierrax/ffe-history")
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    files = {}
    for name in ("train", "valid", "test"):
        path = hf_hub_download(
            repo_id="Pierrax/ffe-history",
            filename=f"features/{name}.parquet",
            repo_type="dataset",
            local_dir=str(out / "hf_cache"),
        )
        files[name] = pd.read_parquet(path)
        logger.info("Downloaded %s.parquet (%d rows)", name, len(files[name]))
    return files["train"], files["valid"], files["test"]


def main() -> None:
    """Full Kaggle training pipeline orchestration (ISO 42001)."""
    logger.info("ALICE Engine — Kaggle Cloud Training")
    train, valid, test = _load_features()
    lineage = build_lineage(train, valid, test, DATA_DIR)
    logger.info("Lineage: train=%d valid=%d test=%d", len(train), len(valid), len(test))
    X_train, y_train, X_valid, y_valid, X_test, y_test, encoders = prepare_features(
        train, valid, test
    )
    config = default_hyperparameters()
    results = train_all_sequential(X_train, y_train, X_valid, y_valid, config)
    evaluate_on_test(results, X_test, y_test)
    champion_auc = fetch_champion_auc()
    gate = check_quality_gates(results, champion_auc=champion_auc)
    logger.info("Quality gate: %s", gate)
    # C4: save models FIRST so checksums are real in model card
    version = datetime.now(tz=UTC).strftime("v%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / version
    save_models(results, encoders, out_dir)
    metadata = build_model_card(results, lineage, gate, config, out_dir=out_dir)
    metadata["version"] = version  # ensure consistent version
    save_metadata_and_push(metadata, out_dir)
    logger.info(
        "Done. Status=%s Best=%s AUC=%.4f",
        metadata["status"],
        gate.get("best_model"),
        gate.get("best_auc", 0),
    )


if __name__ == "__main__":
    main()
