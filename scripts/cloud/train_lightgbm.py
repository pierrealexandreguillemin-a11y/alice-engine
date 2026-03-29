"""Entry point: LightGBM only (CPU, ~10h). Best model kernel."""

import os

os.environ["ALICE_MODEL"] = "lightgbm"
from scripts.cloud.train_kaggle import main  # noqa: E402

main()
