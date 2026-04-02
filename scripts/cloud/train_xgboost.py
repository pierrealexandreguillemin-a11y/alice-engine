"""Entry point: XGBoost only (CPU, ~1h). Canary kernel for fast feedback."""

import os
import sys
from pathlib import Path

os.environ["ALICE_MODEL"] = "xgboost"

# Setup sys.path for Kaggle (alice-code dataset)
for p in [
    Path("/kaggle/input/alice-code"),
    Path("/kaggle/input/datasets/pguillemin/alice-code"),
]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

from scripts.cloud.train_kaggle import main  # noqa: E402

main()
