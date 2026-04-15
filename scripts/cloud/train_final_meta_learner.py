"""V9 Meta-Learner — MLP on OOF XGB+LGB predictions."""

import sys
from pathlib import Path

for p in [
    Path("/kaggle/input/alice-code"),
    Path("/kaggle/input/datasets/pguillemin/alice-code"),
]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

from scripts.cloud.train_meta_learner import main  # noqa: E402

main()
