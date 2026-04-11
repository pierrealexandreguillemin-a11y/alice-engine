"""Entry point: gap-filling round 2 (LGB alpha extension + XGB depth=6)."""

import sys
from pathlib import Path

for p in [
    Path("/kaggle/input/alice-code"),
    Path("/kaggle/input/datasets/pguillemin/alice-code"),
]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

from scripts.cloud.grid_gaps2 import main  # noqa: E402

main()
