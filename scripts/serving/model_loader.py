"""Load ML models from HF Hub or local cache (ISO 42001/27001).

Document ID: ALICE-MODEL-LOADER
Version: 1.0.0

Downloads 3 GBMs + MLP + calibrators from Pierrax/alice-engine/v9/.
Falls back to LGB + Dirichlet if any GBM fails to load.

Secrets: HF_TOKEN in env var (ISO 27001 — never hardcoded).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

HF_REPO_ID_DEFAULT = "Pierrax/alice-engine"
HF_SUBFOLDER = "v9"

MODEL_FILES = {
    "lgb": "LightGBM.txt",
    "xgb": "XGBoost.ubj",
    "cb": "CatBoost.cbm",
    "draw_lookup": "draw_rate_lookup.parquet",
    "encoders": "encoders.joblib",
    "lgb_dirichlet": "lightgbm_dirichlet.joblib",
}


@dataclass
class ModelBundle:
    """All models and artifacts needed for inference."""

    lgb_model: Any
    xgb_model: Any
    cb_model: Any
    mlp_model: Any
    temperature: float
    draw_rate_lookup: pd.DataFrame | None
    encoders: Any
    fallback_mode: bool
    version: str


def _sha256(path: Path) -> str:
    """SHA-256 hash of file (ISO 5259 lineage)."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _download_from_hf(cache_dir: Path, hf_repo_id: str) -> None:
    """Download model files from HF Hub to local cache."""
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    for _key, filename in MODEL_FILES.items():
        target = cache_dir / filename
        if target.exists():
            logger.info("Cache hit: %s (%s)", filename, _sha256(target))
            continue
        logger.info("Downloading %s from %s/%s...", filename, hf_repo_id, HF_SUBFOLDER)
        downloaded = hf_hub_download(
            repo_id=hf_repo_id,
            filename=f"{HF_SUBFOLDER}/{filename}",
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
        dl_path = Path(downloaded)
        if dl_path != target:
            dl_path.rename(target)
        logger.info("Downloaded: %s (%s)", filename, _sha256(target))


def load_models(
    cache_dir: Path,
    hf_repo_id: str = HF_REPO_ID_DEFAULT,
    download: bool = True,
) -> ModelBundle:
    """Load all models from local cache. Download from HF first if needed.

    Falls back to LGB + Dirichlet if XGB or CB fail to load.

    Args:
    ----
        cache_dir: Local directory containing model files.
        hf_repo_id: HuggingFace repo ID (default: Pierrax/alice-engine).
        download: If True, download missing files from HF Hub first.

    Returns:
    -------
        ModelBundle with all loaded artifacts and fallback_mode flag.

    Raises:
    ------
        FileNotFoundError: If cache_dir does not exist (download=False) or
            LGB model file is missing.
    """
    if download:
        _download_from_hf(cache_dir, hf_repo_id)

    if not cache_dir.exists():
        raise FileNotFoundError(f"Model cache not found: {cache_dir}")

    fallback = False
    version_parts = []

    # LGB (required)
    lgb_path = cache_dir / MODEL_FILES["lgb"]
    if not lgb_path.exists():
        raise FileNotFoundError(f"LGB model required: {lgb_path}")
    import lightgbm as lgb

    lgb_model = lgb.Booster(model_file=str(lgb_path))
    version_parts.append(f"lgb:{_sha256(lgb_path)}")
    logger.info("Loaded LGB: %s", lgb_path.name)

    # XGB (fallback if missing)
    xgb_model = None
    xgb_path = cache_dir / MODEL_FILES["xgb"]
    if xgb_path.exists():
        import xgboost as xgb

        xgb_model = xgb.Booster(model_file=str(xgb_path))
        version_parts.append(f"xgb:{_sha256(xgb_path)}")
        logger.info("Loaded XGB: %s", xgb_path.name)
    else:
        logger.warning("XGB not found — fallback mode")
        fallback = True

    # CB (fallback if missing)
    cb_model = None
    cb_path = cache_dir / MODEL_FILES["cb"]
    if cb_path.exists():
        from catboost import CatBoostClassifier

        cb_model = CatBoostClassifier()
        cb_model.load_model(str(cb_path))
        version_parts.append(f"cb:{_sha256(cb_path)}")
        logger.info("Loaded CB: %s", cb_path.name)
    else:
        logger.warning("CB not found — fallback mode")
        fallback = True

    # MLP meta-learner + temperature (only if not fallback)
    mlp_model = None
    temperature = 1.0
    if not fallback:
        mlp_path = cache_dir / "mlp_meta_learner.joblib"
        temp_path = cache_dir / "temperature_T.joblib"
        if mlp_path.exists():
            mlp_model = joblib.load(mlp_path)
            logger.info("Loaded MLP meta-learner")
        if temp_path.exists():
            temperature = float(joblib.load(temp_path))
            logger.info("Loaded temperature: T=%.4f", temperature)

    # Dirichlet fallback calibrator
    if fallback:
        dir_path = cache_dir / MODEL_FILES["lgb_dirichlet"]
        if dir_path.exists():
            mlp_model = joblib.load(dir_path)
            logger.info("Loaded Dirichlet fallback calibrator")

    # Draw rate lookup
    draw_lookup = None
    dl_path = cache_dir / MODEL_FILES["draw_lookup"]
    if dl_path.exists():
        draw_lookup = pd.read_parquet(dl_path)
        logger.info("Loaded draw_rate_lookup: %d cells", len(draw_lookup))
    else:
        logger.warning("draw_rate_lookup.parquet MISSING — inference will fail without it")

    # Encoders
    encoders = None
    enc_path = cache_dir / MODEL_FILES["encoders"]
    if enc_path.exists():
        encoders = joblib.load(enc_path)
        logger.info("Loaded encoders")

    mode = "FALLBACK (LGB+Dirichlet)" if fallback else "FULL (3 GBMs + MLP + temp)"
    logger.info("Model bundle ready: %s", mode)

    return ModelBundle(
        lgb_model=lgb_model,
        xgb_model=xgb_model,
        cb_model=cb_model,
        mlp_model=mlp_model,
        temperature=temperature,
        draw_rate_lookup=draw_lookup,
        encoders=encoders,
        fallback_mode=fallback,
        version="|".join(version_parts),
    )
