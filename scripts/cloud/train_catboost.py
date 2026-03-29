"""Entry point: CatBoost only (CPU, ~9h). SHAP analysis kernel."""

import os

os.environ["ALICE_MODEL"] = "catboost"
from scripts.cloud.train_kaggle import main  # noqa: E402

main()
