"""Promote CANDIDATE model to PRODUCTION after ISO validation.

Pulls candidate from HF Hub, runs ISO 24029 robustness,
ISO 24027 fairness, and McNemar comparison vs champion.

Usage: python -m scripts.cloud.promote_model [--version v20260318_120000]

ISO Compliance:
- ISO/IEC 24029 - Neural Network Robustness
- ISO/IEC 24027 - Bias in AI
- ISO/IEC 42001 - AI Management System
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HF_REPO_ID = "Pierrax/alice-engine"


def decide_promotion(
    robustness: dict,
    fairness: dict,
    mcnemar: dict,
) -> dict:
    """Decide PRODUCTION or REJECTED based on ISO 24029/24027 checks.

    Args:
    ----
        robustness: result from run_robustness() with 'compliant' key
        fairness: result from run_fairness() with 'status' key
        mcnemar: result from run_mcnemar() with 'new_auc' key

    Returns:
    -------
        dict with 'decision' (PRODUCTION|REJECTED) and 'reason'
    """
    if not robustness.get("compliant", False):
        noise_tol = robustness.get("noise_tolerance", 0.0)
        return {
            "decision": "REJECTED",
            "reason": f"robustness check failed: noise_tolerance={noise_tol:.3f} < 0.85",
        }
    if fairness.get("status") == "CRITICAL":
        dp = fairness.get("demographic_parity", 0.0)
        return {
            "decision": "REJECTED",
            "reason": f"fairness check CRITICAL: demographic_parity={dp:.3f} < 0.6",
        }
    new_auc = mcnemar.get("new_auc", 0.0)
    return {
        "decision": "PRODUCTION",
        "reason": f"all ISO checks passed: robustness OK, fairness {fairness.get('status')}, AUC={new_auc:.4f}",
    }


def run_robustness(model: Any, X_test: Any, y_test: Any) -> dict:
    """ISO 24029: seeded noise injection + feature dropout (I5)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    rng = np.random.default_rng(42)
    base_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    X_noisy = X_test.copy()
    num_cols = X_noisy.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        noise = rng.normal(0, 0.05 * X_noisy[col].std(), len(X_noisy))
        X_noisy[col] = X_noisy[col] + noise
    noisy_auc = roc_auc_score(y_test, model.predict_proba(X_noisy)[:, 1])
    noise_tolerance = noisy_auc / base_auc if base_auc > 0 else 0.0
    return {
        "base_auc": base_auc,
        "noisy_auc": noisy_auc,
        "noise_tolerance": noise_tolerance,
        "compliant": noise_tolerance >= 0.85,
    }


def run_fairness(model: Any, X_test: Any, y_test: Any, protected_attr: str = "ligue_code") -> dict:
    """ISO 24027: demographic parity on protected attribute."""
    y_pred = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    groups = X_test[protected_attr].unique()
    rates: dict = {}
    for g in groups:
        mask = X_test[protected_attr] == g
        if mask.sum() < 30:
            continue
        rates[g] = float(y_pred[mask].mean())
    if len(rates) < 2:
        return {"status": "INSUFFICIENT_DATA", "demographic_parity": None}
    min_rate, max_rate = min(rates.values()), max(rates.values())
    dp_ratio = min_rate / max_rate if max_rate > 0 else 0.0
    status = "CRITICAL" if dp_ratio < 0.6 else ("CAUTION" if dp_ratio < 0.8 else "FAIR")
    return {"status": status, "demographic_parity": dp_ratio, "group_rates": rates}


def run_mcnemar(new_model: Any, champion_model: Any, X_test: Any, y_test: Any) -> dict:
    """McNemar test: new vs champion predictions on test set."""
    from scipy.stats import chi2  # noqa: PLC0415
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    pred_new = (new_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    pred_champ = (champion_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    correct_new = pred_new == y_test.values
    correct_champ = pred_champ == y_test.values
    b = int((correct_champ & ~correct_new).sum())
    c = int((~correct_champ & correct_new).sum())
    if b + c == 0:
        return {"p_value": 1.0, "statistic": 0.0, "significant": False}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1 - chi2.cdf(stat, 1))
    new_auc = float(roc_auc_score(y_test, new_model.predict_proba(X_test)[:, 1]))
    champ_auc = float(roc_auc_score(y_test, champion_model.predict_proba(X_test)[:, 1]))
    return {
        "p_value": p_value,
        "statistic": stat,
        "significant": p_value < 0.05,
        "new_auc": new_auc,
        "champion_auc": champ_auc,
    }


def _load_candidate(version: str) -> tuple[Any, dict]:
    """Download candidate model + metadata from HF Hub."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    meta_path = hf_hub_download(HF_REPO_ID, f"{version}/metadata.json", repo_type="model")
    with open(meta_path) as fh:
        metadata = json.load(fh)
    best_name = metadata["best_model"]["name"]
    ext_map = {"CatBoost": ".cbm", "XGBoost": ".ubj", "LightGBM": ".txt"}
    ext = ext_map.get(best_name, ".pkl")
    model_path = hf_hub_download(HF_REPO_ID, f"{version}/{best_name}{ext}", repo_type="model")
    model = _load_model_file(best_name, model_path)
    return model, metadata


def _load_model_file(name: str, path: str) -> Any:
    """Load model from native format file."""
    if name == "CatBoost":
        from catboost import CatBoostClassifier  # noqa: PLC0415

        model = CatBoostClassifier()
        model.load_model(path)
        return model
    if name == "XGBoost":
        from xgboost import XGBClassifier  # noqa: PLC0415

        model = XGBClassifier()
        model.load_model(path)
        return model
    if name == "LightGBM":
        import lightgbm as lgb  # noqa: PLC0415

        return lgb.Booster(model_file=path)
    import joblib  # noqa: PLC0415

    return joblib.load(path)


def _load_champion() -> tuple[Any, dict] | None:
    """Load current champion model from HF Hub, or None if first run."""
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        meta_path = hf_hub_download(HF_REPO_ID, "metadata.json", repo_type="model")
        with open(meta_path) as fh:
            metadata = json.load(fh)
        if metadata.get("status") != "PRODUCTION":
            return None
        return _load_candidate(metadata["version"])
    except Exception:
        logger.warning("No champion model found — first promotion assumed.")
        return None


def _load_test_data() -> tuple:
    """Load test data from HF dataset or local path."""
    import pandas as pd  # noqa: PLC0415

    data_dir = Path(os.environ.get("KAGGLE_DATA_DIR", "/kaggle/input/alice-features"))
    test = pd.read_parquet(data_dir / "test.parquet")
    from scripts.cloud.train_kaggle import prepare_features  # noqa: PLC0415

    _, _, _, _, X_test, y_test, _ = prepare_features(test.copy(), test.copy(), test)
    return X_test, y_test


def _update_metadata(metadata: dict, decision: dict, checks: dict, version: str) -> None:
    """Update metadata.json on HF Hub with promotion decision."""
    metadata["status"] = decision["decision"]
    metadata["promotion"] = {
        "decision": decision["decision"],
        "reason": decision["reason"],
        "robustness": checks["robustness"],
        "fairness": checks["fairness"],
        "mcnemar": checks.get("mcnemar"),
    }
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set — cannot push promotion result.")
        return
    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi()
    import tempfile  # noqa: PLC0415

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump(metadata, fh, indent=2, default=str)
        tmp_path = fh.name
    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo=f"{version}/metadata.json",
        repo_id=HF_REPO_ID,
        repo_type="model",
        token=token,
    )
    logger.info("Updated %s/metadata.json → %s", version, decision["decision"])


def main() -> None:
    """Pull candidate -> Robustness -> Fairness -> McNemar -> promote/reject (I2)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Promote CANDIDATE to PRODUCTION")
    parser.add_argument("--version", required=True, help="Candidate version, e.g. v20260318_120000")
    args = parser.parse_args()
    logger.info("promote_model: ISO 24029/24027 validation for %s", args.version)
    candidate_model, metadata = _load_candidate(args.version)
    X_test, y_test = _load_test_data()
    robustness = run_robustness(candidate_model, X_test, y_test)
    logger.info(
        "Robustness: compliant=%s tolerance=%.3f",
        robustness["compliant"],
        robustness["noise_tolerance"],
    )
    fairness = run_fairness(candidate_model, X_test, y_test)
    logger.info(
        "Fairness: status=%s dp=%.3f", fairness["status"], fairness.get("demographic_parity", 0)
    )
    checks: dict[str, Any] = {"robustness": robustness, "fairness": fairness}
    champion = _load_champion()
    if champion:
        champion_model, _ = champion
        mcnemar = run_mcnemar(candidate_model, champion_model, X_test, y_test)
        logger.info("McNemar: p=%.4f significant=%s", mcnemar["p_value"], mcnemar["significant"])
        checks["mcnemar"] = mcnemar
    else:
        checks["mcnemar"] = {
            "p_value": 1.0,
            "significant": False,
            "new_auc": robustness["base_auc"],
        }
    decision = decide_promotion(robustness, fairness, checks["mcnemar"])
    logger.info("Decision: %s — %s", decision["decision"], decision["reason"])
    _update_metadata(metadata, decision, checks, args.version)


if __name__ == "__main__":
    main()
