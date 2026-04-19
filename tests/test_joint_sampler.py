"""Tests for CopulaJointSampler (F6 SOTA, P2-Task 4).

ISO 29119 : structured test suite with fixtures, deterministic seeds.
Document ID: ALICE-ALI-COPULA-SAMPLER-TESTS
Version: 1.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from services.ali.joint_sampler import CopulaJointSampler


def _build_history() -> pd.DataFrame:
    """Mini history : 3 joueurs, 5 rondes, A et B tres correles, C independant."""
    rows = []
    # A et B presents ensemble rondes 1,2,3 ; absents rondes 4,5
    for r in [1, 2, 3]:
        rows.append(("A", 2024, r, 1))
        rows.append(("B", 2024, r, 2))
    # C present rondes 1, 3, 5 (independant de A/B)
    for r in [1, 3, 5]:
        rows.append(("C", 2024, r, 3))
    return pd.DataFrame(rows, columns=["joueur_nom", "saison", "ronde", "echiquier"])


def test_sampler_fit_returns_psd_matrix() -> None:
    hist = _build_history()
    sampler = CopulaJointSampler(decay_lambda=0.9)
    sampler.fit(
        history=hist,
        player_names=["A", "B", "C"],
        saison=2024,
        nb_rondes_total=5,
        current_round=6,
    )
    sigma = sampler.correlation_matrix
    eigvals = np.linalg.eigvalsh(sigma)
    assert (eigvals >= -1e-9).all(), f"sigma not PSD : eigvals = {eigvals}"


def test_sampler_marginales_match_taux_presence() -> None:
    """Sur 1000 samples, marginales empiriques ~ taux_presence input."""
    hist = _build_history()
    sampler = CopulaJointSampler(decay_lambda=1.0)  # plat = simple
    sampler.fit(hist, ["A", "B", "C"], saison=2024, nb_rondes_total=5, current_round=6)
    rng = np.random.default_rng(42)
    samples = np.array([sampler.sample(rng) for _ in range(1000)])
    marginales = samples.mean(axis=0)
    # A, B joues 3/5 = 0.6 ; C joue 3/5 = 0.6
    expected = np.array([0.6, 0.6, 0.6])
    np.testing.assert_allclose(marginales, expected, atol=0.05)


def test_sampler_returns_binary_vectors() -> None:
    hist = _build_history()
    sampler = CopulaJointSampler(decay_lambda=1.0)
    sampler.fit(hist, ["A", "B", "C"], saison=2024, nb_rondes_total=5, current_round=6)
    rng = np.random.default_rng(0)
    s = sampler.sample(rng)
    assert s.shape == (3,)
    assert ((s == 0) | (s == 1)).all()


def test_sampler_reproducibility() -> None:
    hist = _build_history()
    sampler = CopulaJointSampler(decay_lambda=1.0)
    sampler.fit(hist, ["A", "B", "C"], saison=2024, nb_rondes_total=5, current_round=6)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    s1 = np.array([sampler.sample(rng1) for _ in range(50)])
    s2 = np.array([sampler.sample(rng2) for _ in range(50)])
    np.testing.assert_array_equal(s1, s2)


def test_sampler_handles_single_player() -> None:
    """Edge case : 1 seul joueur dans pool."""
    hist = pd.DataFrame(
        [("A", 2024, r, 1) for r in [1, 2, 3]],
        columns=["joueur_nom", "saison", "ronde", "echiquier"],
    )
    sampler = CopulaJointSampler(decay_lambda=1.0)
    sampler.fit(hist, ["A"], saison=2024, nb_rondes_total=5, current_round=6)
    rng = np.random.default_rng(0)
    s = sampler.sample(rng)
    assert s.shape == (1,)


def test_sampler_handles_no_history() -> None:
    """Edge case : joueur sans historique -> marginale 0, jamais sampled."""
    hist = pd.DataFrame(columns=["joueur_nom", "saison", "ronde", "echiquier"])
    sampler = CopulaJointSampler(decay_lambda=1.0)
    sampler.fit(hist, ["UNKNOWN"], saison=2024, nb_rondes_total=5, current_round=6)
    rng = np.random.default_rng(0)
    samples = np.array([sampler.sample(rng) for _ in range(100)])
    assert samples.sum() == 0  # marginale 0 -> toujours 0


def test_sampler_correlated_players_co_occur() -> None:
    """A et B tres correles -> samples montrent co-occurrence > random."""
    hist = _build_history()
    sampler = CopulaJointSampler(decay_lambda=1.0)
    sampler.fit(hist, ["A", "B", "C"], saison=2024, nb_rondes_total=5, current_round=6)
    rng = np.random.default_rng(123)
    samples = np.array([sampler.sample(rng) for _ in range(2000)])
    # P(A=1 AND B=1) doit etre > P(A=1) * P(B=1) si correlation positive
    p_a_and_b = ((samples[:, 0] == 1) & (samples[:, 1] == 1)).mean()
    p_a = (samples[:, 0] == 1).mean()
    p_b = (samples[:, 1] == 1).mean()
    assert p_a_and_b > p_a * p_b * 1.1  # 10% above independence baseline
