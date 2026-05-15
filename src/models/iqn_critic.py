"""Implicit Quantile Network critic.

Twin IQN Q-network for distributional RL. For each (s, a) it predicts a
quantile-conditional Q value Q(s, a, tau) for tau ~ U[0,1] sampled at call
time. The cosine-embedding of tau is mixed with the state-action embedding
via a Hadamard product, following Dabney et al. (2018, ICML).

  - Cosine embedding of tau:
        phi_j(tau) = ReLU(sum_{i=0..n_cos-1} cos(i * pi * tau) * W_{i j} + b_j)
    with n_cos=64 cosine features.
  - Hadamard product: phi(tau) ⊙ psi(s, a) → head → Q(s, a, tau).
  - Loss: quantile Huber regression (kappa=1.0) against bootstrap target
    quantile samples.
  - n_quantiles=32 by default; configurable.

Twin version returns (Q1, Q2) at the same tau samples — Bellman target uses
the elementwise min.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

_PI = 3.141592653589793


def _cosine_embed(tau: torch.Tensor, n_cos: int) -> torch.Tensor:
    """tau: (B, N) → cosine basis (B, N, n_cos): cos(i*pi*tau), i=0..n_cos-1."""
    B, N = tau.shape
    i = torch.arange(n_cos, device=tau.device, dtype=tau.dtype)  # (n_cos,)
    # broadcast: (B, N, 1) * (n_cos,) → (B, N, n_cos)
    return torch.cos(tau.unsqueeze(-1) * _PI * i)


class IQNHead(nn.Module):
    """One IQN Q-network. Forward returns (B, N) quantile values."""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 256, n_cos: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_cos = n_cos
        # state-action encoder
        self.sa_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # tau encoder: cosine features → hidden_dim
        self.tau_encoder = nn.Sequential(
            nn.Linear(n_cos, hidden_dim),
            nn.ReLU(),
        )
        # final head after Hadamard mix
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor,
                tau: torch.Tensor) -> torch.Tensor:
        """state (B, D), action (B, A), tau (B, N) → quantile values (B, N)."""
        B, N = tau.shape
        # encode (s, a) once
        sa = torch.cat([state, action], dim=-1)              # (B, D+A)
        psi = self.sa_encoder(sa)                             # (B, H)
        # cosine embedding for taus
        cos_basis = _cosine_embed(tau, self.n_cos)            # (B, N, n_cos)
        phi = self.tau_encoder(cos_basis)                     # (B, N, H)
        # Hadamard mix: broadcast psi over the N axis
        mix = phi * psi.unsqueeze(1)                          # (B, N, H)
        q = self.head(mix).squeeze(-1)                        # (B, N)
        return q


class TwinIQNCritic(nn.Module):
    """Twin IQN critic. Mirrors the twin-Q structure used in SAC."""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 256, n_cos: int = 64,
                 n_quantiles: int = 32):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles
        self.q1 = IQNHead(state_dim, action_dim, hidden_dim, n_cos)
        self.q2 = IQNHead(state_dim, action_dim, hidden_dim, n_cos)

    def sample_tau(self, batch_size: int, n: Optional[int] = None,
                   device: Optional[torch.device] = None) -> torch.Tensor:
        """Uniform tau samples in (0, 1) of shape (B, n)."""
        n = n if n is not None else self.n_quantiles
        device = device or next(self.parameters()).device
        return torch.rand(batch_size, n, device=device).clamp(1e-5, 1 - 1e-5)

    def forward(self, state: torch.Tensor, action: torch.Tensor,
                tau: torch.Tensor):
        return self.q1(state, action, tau), self.q2(state, action, tau)


# --------------------------------------------------------------------------- #
# Risk-measure aggregations across the quantile axis
# --------------------------------------------------------------------------- #

def aggregate(quantiles: torch.Tensor, *, risk_measure: str = "expected",
              alpha: float = 0.1, wang_eta: float = 0.75) -> torch.Tensor:
    """Reduce a (B, N) quantile tensor to (B,) scalar Q under one of the
    supported risk measures. tau samples are assumed iid U(0,1).

    - expected:    mean over N (standard SAC).
    - cvar:        mean over the bottom alpha-fraction of sorted quantiles.
    - wang:        distortion-weighted mean using a logistic distortion of tau.
                   eta > 0.5 is risk-averse (down-weights large taus); we expose
                   the parameter as `wang_eta`.
    - optimistic:  upper alpha-tail mean (mirror of cvar). Default alpha=0.1
                   uses the top decile (≈ 90th-percentile-ish).
    """
    if quantiles.dim() != 2:
        raise ValueError(f"expected (B, N), got {tuple(quantiles.shape)}")
    rm = risk_measure.lower()
    if rm == "expected":
        return quantiles.mean(dim=-1)
    if rm == "cvar":
        # sort ascending; take the bottom-alpha slice
        sorted_q, _ = quantiles.sort(dim=-1)
        n = sorted_q.shape[-1]
        k = max(1, int(round(alpha * n)))
        return sorted_q[:, :k].mean(dim=-1)
    if rm == "optimistic":
        sorted_q, _ = quantiles.sort(dim=-1)
        n = sorted_q.shape[-1]
        k = max(1, int(round(alpha * n)))
        return sorted_q[:, -k:].mean(dim=-1)
    if rm == "wang":
        # Logistic distortion: phi(p) = sigmoid((logit(p) - wang_eta_logit))
        # Simpler practical form: weight = softmax(- wang_eta * tau_rank).
        # Approximate by sorting and weighting lower quantiles more under risk-aversion.
        sorted_q, _ = quantiles.sort(dim=-1)
        n = sorted_q.shape[-1]
        ranks = torch.arange(1, n + 1, device=quantiles.device,
                             dtype=quantiles.dtype) / (n + 1)
        # under risk-aversion (wang_eta > 0.5) we want to up-weight low quantiles
        weight = torch.exp(-wang_eta * ranks)
        weight = weight / weight.sum()
        return (sorted_q * weight.unsqueeze(0)).sum(dim=-1)
    raise ValueError(f"unknown risk_measure: {risk_measure}")


# --------------------------------------------------------------------------- #
# Quantile Huber loss
# --------------------------------------------------------------------------- #

def quantile_huber_loss(predicted: torch.Tensor, target: torch.Tensor,
                        tau_pred: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Quantile-Huber regression loss (Dabney et al. 2018).

    predicted:  (B, N) — Q(s, a, tau_pred_i) predictions.
    target:     (B, M) — target quantile samples (no grad).
    tau_pred:   (B, N) — the taus the predictions were drawn at.

    Returns scalar mean loss.
    """
    # u = target_j - predicted_i for all i, j → (B, N, M)
    u = target.unsqueeze(1) - predicted.unsqueeze(2)
    # Huber on |u|
    abs_u = u.abs()
    huber = torch.where(abs_u <= kappa,
                        0.5 * u.pow(2),
                        kappa * (abs_u - 0.5 * kappa))
    # Asymmetric quantile weighting
    # weight = |tau - 1(u < 0)| / kappa
    indic = (u.detach() < 0).float()                          # (B, N, M)
    tau = tau_pred.unsqueeze(-1).expand_as(indic)             # (B, N, M)
    weight = (tau - indic).abs() / kappa
    loss = weight * huber                                      # (B, N, M)
    # sum over target axis M, mean over prediction axis N and batch B
    return loss.sum(dim=-1).mean()


# --------------------------------------------------------------------------- #
# Public stub kept for backward compatibility with the Phase A smoke test.
# --------------------------------------------------------------------------- #

def run(fast: bool = False) -> dict:
    """Construct, sanity-check, and return a fresh TwinIQNCritic.

    This is a self-test entry point — actual training happens in
    distributional_sac.py. Returning a metadata dict makes the module
    smoke-checkable from scripts/run_distributional_rl.py.
    """
    state_dim, action_dim = 128, 5
    n_q = 8 if fast else 32
    critic = TwinIQNCritic(state_dim=state_dim, action_dim=action_dim,
                           n_quantiles=n_q)
    B = 4
    s = torch.randn(B, state_dim)
    a = torch.randn(B, action_dim)
    tau = critic.sample_tau(B)
    q1, q2 = critic(s, a, tau)
    return {
        "ok": True,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_quantiles": n_q,
        "q1_shape": tuple(q1.shape),
        "q2_shape": tuple(q2.shape),
        "param_count": sum(p.numel() for p in critic.parameters()),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run(fast=True), indent=2))
