"""CopulaJointSampler — joint sampling via copule gaussienne (F6 SOTA).

ISO 42001 : SOTA documented (Sklar 1959, Genest & Favre 2007, Nelsen 2006).
ISO 5259 : decay_lambda + saison + current_round traces dans parametres.
ISO 5055 : SRP strict (sampling joint uniquement).

Algorithme :
1. Fit : marginales empiriques (taux_presence) + matrice correlation Spearman
   des co-presences historiques entre joueurs.
2. Cholesky de Sigma pour generer N(0, Sigma).
3. Sample : z ~ N(0, I) -> y = L @ z -> u = Phi(y) -> x_i = 1[u_i < p_i].

References :
- Sklar, A. (1959). Fonctions de repartition a n dimensions et leurs marges.
- Genest, C. & Favre, A.-C. (2007). Everything you always wanted to know about
  copula modeling but were afraid to ask. J. Hydrologic Eng. 12(4), 347-368.
- Nelsen, R. B. (2006). An Introduction to Copulas. Springer.

Document ID: ALICE-ALI-COPULA-SAMPLER
Version: 1.0.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import norm

if TYPE_CHECKING:
    from numpy.typing import NDArray

_NEAREST_PSD_FLOOR = 1e-8
_MIN_ROWS_FOR_SPEARMAN = 2


class CopulaJointSampler:
    """Joint sampling de presence joueurs via copule gaussienne (Sklar 1959).

    Usage:
        sampler = CopulaJointSampler(decay_lambda=0.9)
        sampler.fit(history_df, player_names=[...], saison=2024,
                    nb_rondes_total=11, current_round=10)
        rng = np.random.default_rng(seed)
        presence_vector = sampler.sample(rng)  # shape (N,) binary
    """

    def __init__(self, decay_lambda: float = 0.9) -> None:
        """Initialize sampler with exponential recency-decay factor."""
        self._decay_lambda = decay_lambda
        self._marginales: NDArray[np.float64] | None = None
        self._cholesky: NDArray[np.float64] | None = None
        self._correlation_matrix: NDArray[np.float64] | None = None
        self._n_players: int = 0

    @property
    def correlation_matrix(self) -> NDArray[np.float64]:
        """Return a copy of the fitted Sigma matrix."""
        if self._correlation_matrix is None:
            msg = "CopulaJointSampler not fit yet"
            raise RuntimeError(msg)
        return self._correlation_matrix.copy()

    @property
    def is_fit(self) -> bool:
        """Return True iff fit() has been called with valid data."""
        return self._marginales is not None and self._cholesky is not None

    @property
    def n_players(self) -> int:
        """Return the number of players (marginales dimension)."""
        return self._n_players

    def fit(  # noqa: PLR0913
        self,
        history: pd.DataFrame,
        player_names: list[str],
        saison: int,
        nb_rondes_total: int,
        current_round: int,
    ) -> None:
        """Fit copule depuis history (joueur_nom, saison, ronde, echiquier).

        Construit :
        - marginales p_i = taux_presence_effectif (recency-weighted)
        - matrice correlation Spearman des co-presences
        - Cholesky de la matrice (nearest PSD si besoin)
        """
        self._n_players = len(player_names)

        # 1. Marginales : taux presence par joueur (recency decay)
        self._marginales = np.array(
            [
                self._taux_presence(
                    history,
                    name,
                    saison,
                    nb_rondes_total,
                    current_round,
                )
                for name in player_names
            ],
        )

        # 2. Build presence matrix (rondes x joueurs)
        presence = self._build_presence_matrix(
            history,
            player_names,
            saison,
            nb_rondes_total,
        )

        # 3. Spearman correlation matrix (rank-based, robust to outliers)
        sigma = self._spearman_corr(presence)

        # 4. Nearest PSD if needed
        sigma = self._nearest_psd(sigma)
        self._correlation_matrix = sigma

        # 5. Cholesky decomposition
        self._cholesky = np.linalg.cholesky(sigma)

    def sample(self, rng: np.random.Generator) -> NDArray[np.int8]:
        """Draw 1 binary presence vector via Gaussian copula inverse-transform.

        Returns shape (N,) array with values in {0, 1}.
        """
        if self._cholesky is None or self._marginales is None:
            msg = "CopulaJointSampler not fit yet"
            raise RuntimeError(msg)

        # z ~ N(0, I)
        z = rng.standard_normal(self._n_players)
        # y = L @ z correle
        y = self._cholesky @ z
        # u = Phi(y) marginales uniformes [0,1]
        u = norm.cdf(y)
        # Inverse-CDF binaire : x_i = 1 si u_i < p_i
        result: NDArray[np.int8] = (u < self._marginales).astype(np.int8)
        return result

    def transform_uniform_to_presence(
        self,
        u: NDArray[np.float64],
    ) -> NDArray[np.int8]:
        """Inverse-transform u uniform [0,1]^N -> presence binaire via copule.

        Pour integration avec MonteCarloSampler (LHS samples) :
            u -> z = Phi^-1(u)  (gaussian inverse CDF)
            y = L @ z           (apply Cholesky correlation)
            u_corr = Phi(y)     (back to uniform)
            x_i = 1 if u_corr_i < marginales_i
        """
        if self._cholesky is None or self._marginales is None:
            msg = "CopulaJointSampler not fit yet"
            raise RuntimeError(msg)
        # Clip to avoid Phi^-1(0) = -inf
        u_clipped = np.clip(u, 1e-9, 1 - 1e-9)
        z = norm.ppf(u_clipped)
        y = self._cholesky @ z
        u_corr = norm.cdf(y)
        result: NDArray[np.int8] = (u_corr < self._marginales).astype(np.int8)
        return result

    def _taux_presence(  # noqa: PLR0913
        self,
        history: pd.DataFrame,
        name: str,
        saison: int,
        nb_rondes_total: int,
        current_round: int,
    ) -> float:
        """Taux presence pondere recency decay (reutilise logique F2)."""
        if nb_rondes_total <= 0:
            return 0.0
        if history.empty:
            return 0.0
        sub = history[(history["joueur_nom"] == name) & (history["saison"] == saison)]
        played = set(sub["ronde"].dropna().astype(int).tolist())
        num = 0.0
        den = 0.0
        for r in range(1, nb_rondes_total + 1):
            age = max(current_round - r, 0)
            w = self._decay_lambda**age
            den += w
            if r in played:
                num += w
        return num / den if den > 0 else 0.0

    @staticmethod
    def _build_presence_matrix(
        history: pd.DataFrame,
        player_names: list[str],
        saison: int,
        nb_rondes_total: int,
    ) -> NDArray[np.int8]:
        """Build shape (nb_rondes, N) binary presence matrix from history."""
        n = len(player_names)
        m = nb_rondes_total
        mat: NDArray[np.int8] = np.zeros((m, n), dtype=np.int8)
        if history.empty:
            return mat
        sub = history[history["saison"] == saison]
        name_to_idx = {name: i for i, name in enumerate(player_names)}
        for _, row in sub.iterrows():
            if pd.isna(row["ronde"]):
                continue
            r = int(row["ronde"])
            name = row["joueur_nom"]
            if 1 <= r <= m and name in name_to_idx:
                mat[r - 1, name_to_idx[name]] = 1
        return mat

    @staticmethod
    def _spearman_corr(presence: NDArray[Any]) -> NDArray[np.float64]:
        """Compute Spearman rank correlation matrix.

        Pour des donnees binaires, Spearman ~ Pearson sur rangs.
        Si tous les joueurs ont la meme valeur (toujours 0 ou toujours 1),
        retourne identite pour eviter division par zero.
        """
        n = presence.shape[1]
        if presence.shape[0] < _MIN_ROWS_FOR_SPEARMAN or n == 0:
            identity: NDArray[np.float64] = np.eye(n) if n > 0 else np.zeros((0, 0))
            return identity

        # Use pandas rank for Spearman (Pearson on ranks = Spearman)
        df = pd.DataFrame(presence)
        ranked = df.rank()
        corr: NDArray[np.float64] = ranked.corr().to_numpy()
        # Replace NaN (constant columns) with identity
        nan_mask = np.isnan(corr)
        if nan_mask.any():
            corr = np.where(nan_mask, np.eye(n), corr)
        return corr

    @staticmethod
    def _nearest_psd(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project onto positive semi-definite cone via eigendecomp."""
        if matrix.size == 0:
            return matrix
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, _NEAREST_PSD_FLOOR)
        result: NDArray[np.float64] = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return result
