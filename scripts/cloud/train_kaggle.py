"""Kaggle Cloud Training — ALICE Engine orchestration (ISO 42001/5259/5055)."""

from __future__ import annotations

import logging  # noqa: E401
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Fix cudf device detection on Kaggle T4
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# GPU-accelerate pandas via cudf.pandas — subprocess test to catch C++ aborts
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

OUTPUT_DIR = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))


def _setup_kaggle_imports() -> None:
    """Find alice-code dataset and add to sys.path."""
    import zipfile  # noqa: PLC0415

    candidates = [
        Path("/kaggle/input/alice-code"),
        Path("/kaggle/input/datasets/pguillemin/alice-code"),
    ]
    kaggle_input = next((c for c in candidates if c.exists()), None)
    root = Path("/kaggle/input")
    if root.exists():
        items = list(root.rglob("*"))
        logger.info("/kaggle/input/ tree (%d items): %s", len(items), [str(f) for f in items[:30]])
    logger.info("kaggle_input=%s", kaggle_input)
    if not kaggle_input:
        return
    zips = list(kaggle_input.rglob("*.zip"))
    if zips:
        wd = Path("/kaggle/working/code")
        wd.mkdir(parents=True, exist_ok=True)
        for zf in zips:
            with zipfile.ZipFile(zf) as z:
                z.extractall(wd)
        sys.path.insert(0, str(wd))
    else:
        sys.path.insert(0, str(kaggle_input))
    logger.info("sys.path += %s", sys.path[0])


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

    from scripts.kaggle_artifacts import (  # noqa: PLC0415
        build_lineage,
        build_model_card,
        fetch_champion_auc,
        save_metadata_and_push,
        save_models,
        setup_hf_auth,
    )
    from scripts.kaggle_diagnostics import save_diagnostics  # noqa: PLC0415
    from scripts.kaggle_trainers import (  # noqa: PLC0415
        LABEL_COLUMN,
        MODEL_EXTENSIONS,
        check_quality_gates,
        default_hyperparameters,
        evaluate_on_test,
        prepare_features,
        train_all_sequential,
    )

    setup_hf_auth()
    train, valid, test, features_dir = _load_features()
    lineage = build_lineage(train, valid, test, features_dir, label_column=LABEL_COLUMN)
    logger.info("Lineage: train=%d valid=%d test=%d", len(train), len(valid), len(test))

    X_train, y_train, X_valid, y_valid, X_test, y_test, encoders = prepare_features(
        train, valid, test
    )
    version = datetime.now(tz=UTC).strftime("v%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    config = default_hyperparameters()
    # CatBoost: write training logs inside versioned out_dir (not cwd)
    config["catboost"]["train_dir"] = str(out_dir / "catboost_info")
    results = train_all_sequential(X_train, y_train, X_valid, y_valid, config)
    evaluate_on_test(results, X_test, y_test)

    champion_auc = fetch_champion_auc()
    gate = check_quality_gates(results, champion_auc=champion_auc)
    logger.info("Quality gate: %s", gate)

    save_models(results, encoders, out_dir, model_extensions=MODEL_EXTENSIONS)
    save_diagnostics(results, X_test, y_test, X_valid, y_valid, X_train, out_dir)
    metadata = build_model_card(results, lineage, gate, config, MODEL_EXTENSIONS, out_dir=out_dir)
    metadata["version"] = version
    if gate.get("passed"):
        save_metadata_and_push(metadata, out_dir)
    else:
        logger.error("Quality gate FAILED: %s — saving locally only, NO push.", gate.get("reason"))
        import json  # noqa: PLC0415

        with open(out_dir / "metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2, default=str)
    logger.info(
        "Done. Status=%s Best=%s AUC=%.4f",
        metadata["status"],
        gate.get("best_model"),
        gate.get("best_auc", 0),
    )


if __name__ == "__main__":
    main()
