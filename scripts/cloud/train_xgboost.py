"""Entry point: XGBoost only (CPU, ~1h). Canary kernel for fast feedback."""

import os

os.environ["ALICE_MODEL"] = "xgboost"
from scripts.cloud.train_kaggle import main  # noqa: E402

main()
