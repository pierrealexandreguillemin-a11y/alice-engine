"""PreferenceModel - Bradley-Terry-Luce MAP fit for ALI Phase 4a.

P(player -> team_rank | Elo, recency, streak, brule_count, historical_rank).
MAP estimation with Laplace prior alpha (sparse data clubs faible volume).

Sources :
- Bradley & Terry 1952 "Rank analysis of incomplete block designs"
- Hunter 2004 "MM algorithms for generalized Bradley-Terry models"
- Marden 1995 "Analyzing and modeling rank data"

ISO 5055 : SRP (fit + predict, no I/O orchestration in core class).
ISO 5259 : Lineage SHA-256 propagation input parquet -> artifact.
ISO 24027 : Bias gate per gender at fit time (fail-fast gap > 0.10).
ISO 29119 : Deterministic via seed, frozen artifact dataclass.
ISO 42001 : Model Card linked (docs/iso/MODEL_CARD_PREFERENCE_<saison>.md).
ISO 42005 : Impact assessment line in Model Card section 8.

Document ID: ALICE-ALI-PREFERENCE-MODEL
Version: 1.0.0
Count: 1 per saison fit
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from numpy.typing import NDArray

    from services.ali.types import PreferenceFeatures

__all__ = [
    "BIAS_RECALL_GAP_THRESHOLD",
    "BiasViolationError",
    "PreferenceModel",
    "PreferenceModelArtifact",
    "load_artifact",
    "save_artifact",
]


BIAS_RECALL_GAP_THRESHOLD = 0.10
"""ISO 24027 threshold : recall delta between any two `sexe` groups."""

_MIN_GROUPS_FOR_BIAS = 2


class BiasViolationError(RuntimeError):
    """Raised when per-gender recall gap exceeds BIAS_RECALL_GAP_THRESHOLD."""


@dataclass(frozen=True)
class PreferenceModelArtifact:
    """Serializable artifact wrapping sklearn estimator + lineage metadata.

    Document ID: ALICE-ALI-PREF-ARTIFACT
    Version: 1.0.0
    Count: 1 per saison fit
    """

    estimator: Any  # sklearn.linear_model.LogisticRegression (Any : sklearn untyped)
    feature_names: tuple[str, ...]
    classes_: tuple[int, ...]
    n_teams_max: int
    saison: int
    input_sha256: str
    artifact_sha256: str
    train_size: int
    laplace_alpha: float
    seed: int
    bias_gate_skipped: bool


class PreferenceModel:
    """Bradley-Terry-Luce MAP fit P(player -> team_rank | features).

    Wraps sklearn `LogisticRegression(solver='lbfgs')` which is multinomial
    by default in sklearn >= 1.5 (`multi_class` kwarg removed in 1.7,
    verified 2026-05-26 sklearn 1.8.0). L2 penalty `C = 1 / alpha` mirrors
    the Laplace prior used in Bayesian BTL MAP (Hunter 2004 eq. 12).
    """

    FEATURE_NAMES: tuple[str, ...] = (
        "elo",
        "recency_decay",
        "streak_count",
        "brule_count",
        "historical_team_rank",
    )

    _REQUIRED_COLUMNS: tuple[str, ...] = (
        "saison",
        "ronde",
        "echiquier",
        "blanc_equipe",
        "blanc_elo",
    )

    def __init__(self, laplace_alpha: float = 1.0, seed: int = 42) -> None:
        """Init with Laplace prior strength `alpha > 0` and deterministic `seed`."""
        if laplace_alpha <= 0:
            raise ValueError("laplace_alpha must be > 0 (Laplace prior strength)")
        self._alpha = laplace_alpha
        self._seed = seed
        self._artifact: PreferenceModelArtifact | None = None

    def fit(self, df: pd.DataFrame, saison: int) -> PreferenceModelArtifact:
        """Fit MAP model on echiquiers.parquet subset for given saison.

        Raises ValueError if no rows match `saison` after filtering. Raises
        KeyError if required columns absent. Raises BiasViolationError if
        ISO 24027 per-gender recall gap > BIAS_RECALL_GAP_THRESHOLD.
        """
        df_train = self._build_features(df, saison)
        if df_train.empty:
            raise ValueError(f"PreferenceModel.fit : no rows for saison={saison}")

        x_train = df_train[list(self.FEATURE_NAMES)].to_numpy()
        y_train = df_train["team_rank"].to_numpy()
        n_teams_max = int(y_train.max()) + 1

        estimator = LogisticRegression(
            C=1.0 / self._alpha,
            solver="lbfgs",
            random_state=self._seed,
            max_iter=2000,
        )
        estimator.fit(x_train, y_train)

        bias_skipped = self._check_bias_gate(df_train, estimator)

        input_sha = self._sha256_dataframe(df_train)
        artifact_sha = self._sha256_estimator(estimator)

        artifact = PreferenceModelArtifact(
            estimator=estimator,
            feature_names=self.FEATURE_NAMES,
            classes_=tuple(int(c) for c in estimator.classes_),
            n_teams_max=n_teams_max,
            saison=saison,
            input_sha256=input_sha,
            artifact_sha256=artifact_sha,
            train_size=int(len(df_train)),
            laplace_alpha=self._alpha,
            seed=self._seed,
            bias_gate_skipped=bias_skipped,
        )
        self._artifact = artifact
        return artifact

    def fit_with_self_return(self, df: pd.DataFrame, saison: int) -> PreferenceModel:
        """Fit then return self (chainable convenience for one-liner tests)."""
        self.fit(df, saison)
        return self

    def predict_proba(self, features: list[PreferenceFeatures]) -> NDArray[np.float64]:
        """Predict P(team_rank | features) for batch of players.

        Returns array of shape `(len(features), n_classes_trained)`.
        """
        if self._artifact is None:
            raise RuntimeError("PreferenceModel not fitted - call fit() first")
        x_pred = np.array(
            [
                [
                    f.elo,
                    f.recency_decay,
                    f.streak_count,
                    f.brule_count,
                    f.historical_team_rank,
                ]
                for f in features
            ],
            dtype=np.float64,
        )
        proba: NDArray[np.float64] = self._artifact.estimator.predict_proba(x_pred)
        return proba

    @property
    def artifact(self) -> PreferenceModelArtifact:
        """Return the fitted artifact (raise RuntimeError if not yet fitted)."""
        if self._artifact is None:
            raise RuntimeError("PreferenceModel not fitted - call fit() first")
        return self._artifact

    def _build_features(self, df: pd.DataFrame, saison: int) -> pd.DataFrame:
        """Extract per-(player, team, rank) features from echiquiers.parquet.

        Filters: saison match, blanc_elo > 0 (unrated scolaires excluded),
        blanc_equipe non-null. MVP placeholders for streak/brule pending
        Phase 4a+T enrichment (see Model Card section 6).
        """
        missing = [c for c in self._REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        df = df[df["saison"] == saison].copy()
        df = df.dropna(subset=["blanc_equipe", "blanc_elo"])
        df = df[df["blanc_elo"] > 0]

        if df.empty:
            return df

        df["team_rank"] = df["echiquier"].astype(int) - 1
        df["elo"] = df["blanc_elo"].astype(int)
        max_ronde = int(df["ronde"].max())
        df["recency_decay"] = 0.9 ** (max_ronde - df["ronde"].astype(int))
        df["streak_count"] = 1  # MVP placeholder (F3, see Model Card section 6)
        df["brule_count"] = 0  # MVP placeholder (A02 §3.7.c, see Model Card section 6)
        # historical_team_rank : MVP placeholder = 0 (uninformative). Using
        # `team_rank` itself would cause label leakage (target == feature).
        # Phase 4a+T enrichment will populate from prior-saison aggregation.
        df["historical_team_rank"] = 0

        keep_cols = ["team_rank", *self.FEATURE_NAMES]
        if "sexe" in df.columns:
            keep_cols.append("sexe")
        return df[keep_cols].reset_index(drop=True)

    def _check_bias_gate(self, df_train: pd.DataFrame, estimator: Any) -> bool:
        """Per-gender recall gap check. Returns True if gate was skipped.

        If `sexe` column missing, gate is skipped (backward-compat) and
        flagged via `bias_gate_skipped=True` for downstream visibility.
        """
        if "sexe" not in df_train.columns:
            return True

        groups = df_train["sexe"].dropna().unique()
        if len(groups) < _MIN_GROUPS_FOR_BIAS:
            return True

        x_full = df_train[list(self.FEATURE_NAMES)].to_numpy()
        y_full = df_train["team_rank"].to_numpy()
        y_pred = estimator.predict(x_full)

        recalls: dict[str, float] = {}
        for grp in groups:
            mask = (df_train["sexe"] == grp).to_numpy()
            if not mask.any():
                continue
            recalls[str(grp)] = float((y_pred[mask] == y_full[mask]).mean())

        if len(recalls) < _MIN_GROUPS_FOR_BIAS:
            return True

        gap = max(recalls.values()) - min(recalls.values())
        if gap > BIAS_RECALL_GAP_THRESHOLD:
            top = max(recalls, key=lambda k: recalls[k])
            bot = min(recalls, key=lambda k: recalls[k])
            raise BiasViolationError(
                f"ISO 24027 bias gate: recall gap {gap:.3f} > "
                f"{BIAS_RECALL_GAP_THRESHOLD} between groups "
                f"{top!r}={recalls[top]:.3f} vs {bot!r}={recalls[bot]:.3f}"
            )
        return False

    @staticmethod
    def _sha256_dataframe(df: pd.DataFrame) -> str:
        """SHA-256 of dataframe content for lineage tracing (ISO 5259)."""
        hashed = pd.util.hash_pandas_object(df, index=True).values
        return hashlib.sha256(hashed.tobytes()).hexdigest()

    @staticmethod
    def _sha256_estimator(estimator: Any) -> str:
        """SHA-256 of joblib-pickled estimator bytes (ISO 5259 + 42001)."""
        buf = io.BytesIO()
        joblib.dump(estimator, buf)
        return hashlib.sha256(buf.getvalue()).hexdigest()


# ----------------------------------------------------------------------
# Persistence helpers (out of class : ISO 5055 SRP, no I/O in model core)
# ----------------------------------------------------------------------
def save_artifact(artifact: PreferenceModelArtifact, path: Path) -> None:
    """Persist artifact via joblib (compressed bytes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: Path) -> PreferenceModelArtifact:
    """Reload artifact via joblib. Caller-owned path validation."""
    artifact: PreferenceModelArtifact = joblib.load(path)
    return artifact
