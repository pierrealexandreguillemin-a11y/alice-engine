"""Utilitaires pour Model Registry - ISO 5055.

Ce module contient les fonctions utilitaires:
- Checksums SHA-256
- Hash DataFrame
- Informations Git
- Versions packages
- Calcul traçabilité données

Conformité ISO/IEC 5055, 5259, 27001.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess  # nosec B404 - subprocess used for internal dev tools only
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from scripts.model_registry.dataclasses import DataLineage, EnvironmentInfo


def compute_file_checksum(file_path: Path) -> str:
    """Calcule le checksum SHA-256 d'un fichier."""
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """Calcule un hash déterministe d'un DataFrame."""
    import pandas as pd  # Lazy import - évite chargement au niveau module

    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[
        :16
    ]


def get_git_info() -> tuple[str | None, str | None, bool]:
    """Récupère les informations git du repository."""
    try:
        # Git commit hash
        commit = subprocess.run(  # nosec B603, B607 - trusted git command for versioning
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        commit_hash = commit.stdout.strip() if commit.returncode == 0 else None

        # Git branch
        branch = subprocess.run(  # nosec B603, B607 - trusted git command for versioning
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        branch_name = branch.stdout.strip() if branch.returncode == 0 else None

        # Git dirty (uncommitted changes)
        status = subprocess.run(  # nosec B603, B607 - trusted git command for versioning
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        is_dirty = bool(status.stdout.strip()) if status.returncode == 0 else False

        return commit_hash, branch_name, is_dirty
    except FileNotFoundError:
        return None, None, False


def get_package_versions() -> dict[str, str]:
    """Récupère les versions des packages ML."""
    versions: dict[str, str] = {}

    try:
        import catboost

        versions["catboost"] = catboost.__version__
    except ImportError:
        pass

    try:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    except ImportError:
        pass

    try:
        import lightgbm

        versions["lightgbm"] = lightgbm.__version__
    except ImportError:
        pass

    try:
        import sklearn

        versions["sklearn"] = sklearn.__version__
    except ImportError:
        pass

    try:
        import numpy

        versions["numpy"] = numpy.__version__
    except ImportError:
        pass

    try:
        import pandas as pd_ver

        versions["pandas"] = pd_ver.__version__
    except ImportError:
        pass

    return versions


def get_environment_info() -> EnvironmentInfo:
    """Collecte les informations d'environnement."""
    commit, branch, dirty = get_git_info()
    return EnvironmentInfo(
        python_version=sys.version,
        platform_system=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
        git_commit=commit,
        git_branch=branch,
        git_dirty=dirty,
        packages=get_package_versions(),
    )


def compute_data_lineage(
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "resultat_blanc",
) -> DataLineage:
    """Calcule la traçabilité des données."""
    import pandas as pd  # Lazy import - évite chargement au niveau module

    # Distribution de la cible
    train_target = train_df[target_col] if target_col in train_df.columns else pd.Series()
    target_dist = {
        "positive_ratio": float(train_target.mean()) if len(train_target) > 0 else 0.0,
        "total_samples": len(train_df) + len(valid_df) + len(test_df),
    }

    return DataLineage(
        train_path=str(train_path),
        valid_path=str(valid_path),
        test_path=str(test_path),
        train_samples=len(train_df),
        valid_samples=len(valid_df),
        test_samples=len(test_df),
        train_hash=compute_dataframe_hash(train_df),
        valid_hash=compute_dataframe_hash(valid_df),
        test_hash=compute_dataframe_hash(test_df),
        feature_count=len(train_df.columns) - 1,  # Excluding target
        target_distribution=target_dist,
        created_at=datetime.now().isoformat(),
    )
