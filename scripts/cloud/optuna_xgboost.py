"""Entry point: XGBoost Optuna V9 (CPU, 12h). Canary — push first."""

import os
import subprocess
import sys
from pathlib import Path

os.environ["ALICE_MODEL"] = "xgboost"

for p in [
    Path("/kaggle/input/alice-code"),
    Path("/kaggle/input/datasets/pguillemin/alice-code"),
]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "optuna", "optuna-integration"]
)

from scripts.cloud.optuna_kaggle import main  # noqa: E402

main()
