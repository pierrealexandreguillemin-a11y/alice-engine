"""Entry point: LightGBM v4 resume from 15K checkpoint (lr=0.005, CPU).

Resume from v3 checkpoint (15K iters, lr=0.003) with lr=0.005 for convergence.
ALICE_INIT_MODEL env var tells _train_lightgbm to load the checkpoint as init_model.
n_estimators=50K = ADDITIONAL iters on top of 15K = max 65K total trees.
early_stopping=200 handles convergence. Budget worst case: ~10h + 1.5h post = 11.5h / 12h.
"""

import os
import sys
from pathlib import Path

os.environ["ALICE_MODEL"] = "lightgbm"

# Setup sys.path for Kaggle (alice-code dataset)
for p in [
    Path("/kaggle/input/alice-code"),
    Path("/kaggle/input/datasets/pguillemin/alice-code"),
]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

# Locate v3 checkpoint (15K iters) from dataset upload
ckpt_candidates = [
    Path("/kaggle/input/alice-lgbm-v3-checkpoint/lgbm_ckpt_15000.txt"),
    Path("/kaggle/input/datasets/pguillemin/alice-lgbm-v3-checkpoint/lgbm_ckpt_15000.txt"),
]
for ckpt in ckpt_candidates:
    if ckpt.exists():
        os.environ["ALICE_INIT_MODEL"] = str(ckpt)
        print(f"Resume checkpoint: {ckpt} ({ckpt.stat().st_size / 1e6:.1f} MB)")
        break
else:
    raise FileNotFoundError(f"No checkpoint found in: {[str(c) for c in ckpt_candidates]}")

from scripts.cloud.train_kaggle import main  # noqa: E402

main()
