"""Entry point: Grid search saison=2022 (CPU, 12h). Set ALICE_MODEL env var."""

import sys
from pathlib import Path

for p in [
    Path("/kaggle/input/alice-code"),
    Path("/kaggle/input/datasets/pguillemin/alice-code"),
]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

from scripts.cloud.grid_search import main  # noqa: E402

main()
