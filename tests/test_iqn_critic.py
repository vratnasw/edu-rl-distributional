"""Unit tests for the IQN critic.

Covers:
  - IQN output shape matches (B, n_quantiles)
  - Risk-measure aggregations return scalars (B,) per input
  - aggregate() handles all four supported risk measures
  - quantile_huber_loss returns a non-negative scalar tensor
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.iqn_critic import (  # noqa: E402
    TwinIQNCritic, aggregate, quantile_huber_loss,
)


@pytest.fixture(scope="module")
def critic():
    torch.manual_seed(0)
    return TwinIQNCritic(state_dim=16, action_dim=3, n_quantiles=8)


def test_iqn_output_shape(critic):
    B = 5
    s = torch.randn(B, 16)
    a = torch.randn(B, 3)
    tau = critic.sample_tau(B)
    q1, q2 = critic(s, a, tau)
    assert q1.shape == (B, 8)
    assert q2.shape == (B, 8)


@pytest.mark.parametrize("rm", ["expected", "cvar", "wang", "optimistic"])
def test_aggregate_returns_scalar_per_sample(rm):
    B, N = 4, 16
    q = torch.linspace(-1, 1, B * N).reshape(B, N)
    out = aggregate(q, risk_measure=rm, alpha=0.1)
    assert out.shape == (B,)
    assert torch.isfinite(out).all()


def test_cvar_smaller_than_expected():
    # CVaR(α=0.1) on the bottom decile must be ≤ the mean
    q = torch.randn(20, 32)
    cvar = aggregate(q, risk_measure="cvar", alpha=0.1)
    expected = aggregate(q, risk_measure="expected")
    assert torch.all(cvar <= expected + 1e-6)


def test_optimistic_larger_than_expected():
    q = torch.randn(20, 32)
    opt = aggregate(q, risk_measure="optimistic", alpha=0.1)
    expected = aggregate(q, risk_measure="expected")
    assert torch.all(opt >= expected - 1e-6)


def test_aggregate_rejects_bad_shape():
    with pytest.raises(ValueError):
        aggregate(torch.zeros(3), risk_measure="expected")


def test_aggregate_rejects_unknown_measure():
    with pytest.raises(ValueError):
        aggregate(torch.zeros(2, 8), risk_measure="not_a_measure")


def test_quantile_huber_loss_nonneg(critic):
    B, N = 4, critic.n_quantiles
    pred = torch.randn(B, N)
    tgt = torch.randn(B, N)
    tau = critic.sample_tau(B)
    loss = quantile_huber_loss(pred, tgt, tau, kappa=1.0)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_quantile_huber_zero_when_perfect(critic):
    B, N = 4, critic.n_quantiles
    pred = torch.zeros(B, N)
    tgt = torch.zeros(B, N)
    tau = critic.sample_tau(B)
    loss = quantile_huber_loss(pred, tgt, tau, kappa=1.0)
    assert loss.item() < 1e-6
