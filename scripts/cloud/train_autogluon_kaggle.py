"""AutoGluon Kaggle Training — ALICE Engine (ISO 42001/5259/5055).

Standalone script for Kaggle kernel. Loads pre-computed features from
pguillemin/alice-features, trains AutoGluon ensemble, saves all artifacts.

Usage: python train_autogluon_kaggle.py  (Kaggle kernel)
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Config ---
LABEL = "target"
LABEL_SOURCE = "resultat_blanc"
DEDUP_KEYS = ["saison", "ronde", "echiquier", "blanc_nom", "noir_nom", "equipe_dom"]
METADATA_COLS = [
    "saison",
    "competition",
    "division",
    "groupe",
    "ligue",
    "equipe_dom",
    "equipe_ext",
    "score_dom",
    "score_ext",
    "date",
    "date_str",
    "heure",
    "lieu",
    "blanc_nom",
    "blanc_equipe",
    "noir_nom",
    "noir_equipe",
    "resultat_blanc",
    "resultat_noir",
    "resultat_text",
    "type_resultat",
]
AG_PRESETS = "best_quality"
AG_TIME_LIMIT = 21600  # 6 hours
AG_BAG_FOLDS = 5
AG_STACK_LEVELS = 2
AG_SEED = 42
AUC_FLOOR = 0.70
HF_REPO_ID = "Pierrax/alice-engine"

DATA_PATH = Path("/kaggle/input/alice-features")
OUTPUT_DIR = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))


# --- Data ---


def _sha256(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:16]


def load_and_clean(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load parquets, dedup, binarize target, drop metadata."""
    train = pd.read_parquet(path / "train.parquet")
    valid = pd.read_parquet(path / "valid.parquet")
    test = pd.read_parquet(path / "test.parquet")
    logger.info("Raw: train=%d valid=%d test=%d", len(train), len(valid), len(test))

    frames = []
    for name, df in [("train", train), ("valid", valid), ("test", test)]:
        before = len(df)
        df = df.drop_duplicates(subset=DEDUP_KEYS).copy()
        df[LABEL] = (df[LABEL_SOURCE] == 1.0).astype(int)
        drop = [c for c in METADATA_COLS if c in df.columns]
        df = df.drop(columns=drop)
        logger.info("%s: %d -> %d rows, %d features", name, before, len(df), len(df.columns) - 1)
        frames.append(df)
    return frames[0], frames[1], frames[2]


def build_lineage(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> dict:
    """ISO 5259 data lineage."""
    return {
        "train": {"samples": len(train), "hash": _sha256(train)},
        "valid": {"samples": len(valid), "hash": _sha256(valid)},
        "test": {"samples": len(test), "hash": _sha256(test)},
        "feature_count": len(train.columns) - 1,
        "target_positive_rate": float(train[LABEL].mean()),
        "dedup_keys": DEDUP_KEYS,
        "created_at": datetime.now(tz=UTC).isoformat(),
    }


# --- Training ---


def train_autogluon(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    out_dir: Path,
) -> tuple:
    """Fit AutoGluon on train+valid, evaluate on test."""
    from autogluon.tabular import TabularPredictor  # noqa: PLC0415

    model_dir = out_dir / "autogluon_model"
    predictor = TabularPredictor(
        label=LABEL,
        eval_metric="roc_auc",
        path=str(model_dir),
        verbosity=2,
    )
    predictor.fit(
        train_data=train_data,
        presets=AG_PRESETS,
        time_limit=AG_TIME_LIMIT,
        num_bag_folds=AG_BAG_FOLDS,
        num_stack_levels=AG_STACK_LEVELS,
        ag_args_fit={"ag.max_memory_usage_ratio": 3.0},
    )
    leaderboard = predictor.leaderboard(test_data, silent=True)
    logger.info("Trained %d models. Best: %s", len(leaderboard), leaderboard.iloc[0]["model"])
    return predictor, leaderboard


def evaluate(predictor: Any, test_data: pd.DataFrame) -> dict:
    """Compute test metrics."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: PLC0415

    y_true = test_data[LABEL]
    y_proba = predictor.predict_proba(test_data.drop(columns=LABEL))[1]
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "test_auc": float(roc_auc_score(y_true, y_proba)),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "test_samples": len(y_true),
    }


# --- Artifacts ---


def build_model_card(
    metrics: dict,
    lineage: dict,
    robustness: dict,
    fairness: dict,
    leaderboard_len: int,
    best_model: str,
    version: str,
) -> dict:
    """ISO 42001 Model Card."""
    pkgs = {}
    for pkg in ("autogluon", "pandas", "scikit-learn", "numpy"):
        try:
            import importlib.metadata as im  # noqa: PLC0415

            pkgs[pkg] = im.version(pkg)
        except Exception:
            pkgs[pkg] = "unknown"
    return {
        "version": version,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "status": "CANDIDATE",
        "model_type": "AutoGluon_ensemble",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "kaggle_kernel": os.environ.get("KAGGLE_KERNEL_RUN_SLUG", "local"),
            "packages": pkgs,
        },
        "config": {
            "presets": AG_PRESETS,
            "time_limit": AG_TIME_LIMIT,
            "num_bag_folds": AG_BAG_FOLDS,
            "num_stack_levels": AG_STACK_LEVELS,
            "seed": AG_SEED,
        },
        "data_lineage": lineage,
        "metrics": metrics,
        "best_model": best_model,
        "num_models_trained": leaderboard_len,
        "iso_24029_robustness": robustness,
        "iso_24027_fairness": fairness,
        "quality_gate": {"passed": metrics["test_auc"] >= AUC_FLOOR, "auc_floor": AUC_FLOOR},
        "limitations": ["FFE interclub data only", "Not for tournament games"],
        "conformance": {
            "ISO_42001": "CANDIDATE",
            "ISO_5259": "COMPLIANT",
            "ISO_24029": robustness["status"],
            "ISO_24027": fairness["status"],
        },
    }


def push_to_hf(out_dir: Path, version: str) -> None:
    """Push artifacts to HuggingFace Hub."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        cache = Path.home() / ".cache" / "huggingface" / "token"
        token = cache.read_text().strip() if cache.exists() else ""
    if not token:
        logger.warning("No HF token — skipping push.")
        return
    from huggingface_hub import HfApi  # noqa: PLC0415

    HfApi().upload_folder(
        folder_path=str(out_dir),
        repo_id=HF_REPO_ID,
        repo_type="model",
        path_in_repo=f"autogluon/{version}",
        token=token,
    )
    logger.info("Pushed to HF Hub: %s/autogluon/%s", HF_REPO_ID, version)


# --- Main ---


def main() -> None:
    """Full AutoGluon training pipeline."""
    logger.info("ALICE Engine — AutoGluon Kaggle Training")

    # Setup Kaggle imports (alice-code dataset for diagnostics module)
    for candidate in [Path("/kaggle/input/alice-code"), Path(".")]:
        if (candidate / "scripts").exists():
            sys.path.insert(0, str(candidate))
            break

    from scripts.cloud.autogluon_diagnostics import (  # noqa: PLC0415
        save_diagnostics,
        save_predictions,
        test_fairness,
        test_robustness,
    )

    version = datetime.now(tz=UTC).strftime("v%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load + clean
    data_path = DATA_PATH if DATA_PATH.exists() else Path("data/features")
    train, valid, test = load_and_clean(data_path)
    lineage = build_lineage(train, valid, test)

    # 2. Save clean features as output
    for name, df in [("train", train), ("valid", valid), ("test", test)]:
        df.to_parquet(out_dir / f"clean_{name}.parquet", index=False)
    logger.info("Clean features saved to %s", out_dir)

    # 3. Train on train+valid
    combined = pd.concat([train, valid], ignore_index=True)
    logger.info("Training on %d rows (%d features)", len(combined), len(combined.columns) - 1)
    del train, valid
    gc.collect()

    predictor, leaderboard = train_autogluon(combined, test, out_dir)
    del combined
    gc.collect()

    # 4. Evaluate
    metrics = evaluate(predictor, test)
    logger.info(
        "AUC=%.4f Acc=%.4f F1=%.4f",
        metrics["test_auc"],
        metrics["test_accuracy"],
        metrics["test_f1"],
    )

    # 5. ISO diagnostics
    save_predictions(predictor, test, LABEL, out_dir)
    save_diagnostics(predictor, test, LABEL, leaderboard, out_dir)
    robustness = test_robustness(predictor, test, LABEL, seed=AG_SEED)
    fairness = test_fairness(predictor, test, LABEL)
    logger.info("Robustness: %s | Fairness: %s", robustness["status"], fairness["status"])

    # 6. Model card + reports
    model_card = build_model_card(
        metrics,
        lineage,
        robustness,
        fairness,
        len(leaderboard),
        str(leaderboard.iloc[0]["model"]),
        version,
    )
    for fname, data in [
        ("metadata.json", model_card),
        ("robustness_report.json", robustness),
        ("fairness_report.json", fairness),
    ]:
        with open(out_dir / fname, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # 7. Push if quality gate passes
    if model_card["quality_gate"]["passed"]:
        push_to_hf(out_dir, version)
    else:
        logger.error("Quality gate FAILED: AUC %.4f < %.2f", metrics["test_auc"], AUC_FLOOR)

    logger.info(
        "Done. Version=%s AUC=%.4f Best=%s", version, metrics["test_auc"], model_card["best_model"]
    )


if __name__ == "__main__":
    main()
