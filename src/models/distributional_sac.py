"""Distributional SAC with IQN critic + configurable risk measure.

Actor mirrors ``edu-rl-agent.src.agent.sac_lagrangian.Actor`` (Gaussian-tanh
3-layer MLP). Critic is a TwinIQNCritic (see :mod:`src.models.iqn_critic`).

Supported risk measures (selectable via config):
  - expected     : mean over quantiles (standard SAC objective)
  - cvar         : mean over the bottom-alpha quantiles (Rawlsian; default 0.1)
  - wang         : distortion-weighted mean (eta=0.75 for risk-aversion)
  - optimistic   : top-alpha quantiles (for exploration; e.g. 90th percentile)

Lagrangian equity constraint: the multiplier is updated by dual ascent on the
mean batch shortfall of the bottom-decile return below an equity floor. This
mirrors the Rawlsian floor constraint from the reference SAC, but applied to
the distribution itself rather than to a separate cost critic.

The "environment" is the Layer 5 world model: rollouts use
``WorldModelEnsemble.predict`` to advance state under the actor's sampled
action; rewards are a deterministic function of the predicted state delta
(see ``_default_reward_fn``). This keeps the implementation self-contained
without requiring the original training environment.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .iqn_critic import TwinIQNCritic, aggregate, quantile_huber_loss

log = logging.getLogger(__name__)

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


# --------------------------------------------------------------------------- #
# Actor
# --------------------------------------------------------------------------- #

class GaussianTanhActor(nn.Module):
    """Stochastic Gaussian-tanh policy. 3-layer MLP, 256 units each."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 action_low: float = -3.0, action_high: float = 3.0):
        super().__init__()
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head_mean = nn.Linear(hidden_dim, action_dim)
        self.head_log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor):
        h = self.body(state)
        mean = self.head_mean(h)
        log_std = self.head_log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        a_tanh = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - a_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        scale = (self.action_high - self.action_low) / 2.0
        offset = (self.action_high + self.action_low) / 2.0
        action = a_tanh * scale + offset
        return action, log_prob

    @torch.no_grad()
    def sample_action(self, state: torch.Tensor,
                      deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.forward(state)
        if deterministic:
            a_tanh = torch.tanh(mean)
        else:
            std = log_std.exp()
            x = mean + std * torch.randn_like(mean)
            a_tanh = torch.tanh(x)
        scale = (self.action_high - self.action_low) / 2.0
        offset = (self.action_high + self.action_low) / 2.0
        return a_tanh * scale + offset


# --------------------------------------------------------------------------- #
# Reward model (default — deterministic function of state delta)
# --------------------------------------------------------------------------- #

def _default_reward_fn(state: torch.Tensor, action: torch.Tensor,
                       next_state: torch.Tensor) -> torch.Tensor:
    """Reward = mean improvement over the embedding plus an action-cost term.

    Using the first 4 dims of the 128-dim Layer-4 embedding as the
    "outcome head" is an opinionated choice consistent with the Layer 5
    world-model interface (state is the embedding itself; outcomes are
    encoded in the leading factors of variance, which is how GNN-CSE
    embeddings are structured).
    """
    delta = next_state[..., :4] - state[..., :4]
    improvement = delta.sum(dim=-1)
    action_cost = 0.01 * action.pow(2).sum(dim=-1)
    return improvement - action_cost


# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #

class ReplayBuffer:
    """Lightweight numpy-backed circular replay buffer."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def push(self, s, a, r, sp, d):
        i = self.idx
        self.state[i] = s; self.action[i] = a; self.reward[i] = r
        self.next_state[i] = sp; self.done[i] = d
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(self, s, a, r, sp, d):
        B = s.shape[0]
        for i in range(B):
            self.push(s[i], a[i], r[i], sp[i], d[i])

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return dict(
            state=self.state[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            next_state=self.next_state[idx],
            done=self.done[idx],
        )


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

@dataclass
class SACConfig:
    state_dim: int = 128
    action_dim: int = 5
    hidden_dim: int = 256
    n_quantiles: int = 32
    kappa: float = 1.0
    gamma: float = 0.99
    tau_target: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    risk_measure: str = "cvar"
    alpha_cvar: float = 0.1
    wang_eta: float = 0.75
    action_low: float = -3.0
    action_high: float = 3.0
    # Equity constraint: penalise expected shortfall of bottom-decile below floor.
    equity_floor: float = -0.5
    lr_lambda: float = 1e-3
    max_lambda: float = 10.0
    initial_lambda: float = 0.5
    # Auxiliary risk-penalty: variance of quantile distribution
    w_variance_penalty: float = 0.0
    # Weight on cvar vs expected in the actor objective (when both are blended)
    w_expected: float = 1.0
    w_cvar: float = 0.0


class DistributionalSAC:
    def __init__(self, cfg: SACConfig, device: torch.device | None = None):
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        D, A = cfg.state_dim, cfg.action_dim

        self.actor = GaussianTanhActor(
            D, A, hidden_dim=cfg.hidden_dim,
            action_low=cfg.action_low, action_high=cfg.action_high,
        ).to(self.device)

        self.critic = TwinIQNCritic(
            D, A, hidden_dim=cfg.hidden_dim, n_quantiles=cfg.n_quantiles,
        ).to(self.device)
        self.target_critic = TwinIQNCritic(
            D, A, hidden_dim=cfg.hidden_dim, n_quantiles=cfg.n_quantiles,
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

        self.target_entropy = -float(A)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)

        self.lam = float(cfg.initial_lambda)
        self.last_metrics: dict = {}

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------ #
    def _risk_value(self, quantiles: torch.Tensor) -> torch.Tensor:
        """Reduce (B, N) → (B,) under the configured risk measure."""
        return aggregate(
            quantiles,
            risk_measure=self.cfg.risk_measure,
            alpha=self.cfg.alpha_cvar,
            wang_eta=self.cfg.wang_eta,
        )

    def _bottom_decile(self, quantiles: torch.Tensor) -> torch.Tensor:
        sorted_q, _ = quantiles.sort(dim=-1)
        n = sorted_q.shape[-1]
        k = max(1, int(round(0.1 * n)))
        return sorted_q[:, :k].mean(dim=-1)

    # ------------------------------------------------------------------ #
    def update(self, batch: dict) -> dict:
        cfg = self.cfg
        s = torch.as_tensor(batch["state"], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device)
        r = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        sp = torch.as_tensor(batch["next_state"], dtype=torch.float32, device=self.device)
        d = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        B = s.shape[0]

        # ---- Critic update ---- #
        with torch.no_grad():
            ap, lp = self.actor.sample(sp)
            tau_targ = self.target_critic.sample_tau(B)
            tq1, tq2 = self.target_critic(sp, ap, tau_targ)
            # min over the twin critics (elementwise across quantiles)
            tq_min = torch.minimum(tq1, tq2)
            # Subtract entropy term broadcast across N
            tq_min = tq_min - self.alpha.detach() * lp  # (B, N)
            # Bellman target: r + gamma * (1 - d) * tq_min, broadcast (B,1)→(B,N)
            target_quantiles = r + (1.0 - d) * cfg.gamma * tq_min

        tau_pred = self.critic.sample_tau(B)
        q1, q2 = self.critic(s, a, tau_pred)
        loss_q1 = quantile_huber_loss(q1, target_quantiles, tau_pred, kappa=cfg.kappa)
        loss_q2 = quantile_huber_loss(q2, target_quantiles, tau_pred, kappa=cfg.kappa)
        loss_q = loss_q1 + loss_q2

        self.opt_critic.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.opt_critic.step()

        # ---- Actor update ---- #
        a_pi, lp_pi = self.actor.sample(s)
        tau_a = self.critic.sample_tau(B)
        q1_pi, q2_pi = self.critic(s, a_pi, tau_a)
        q_quant = torch.minimum(q1_pi, q2_pi)                   # (B, N)
        # Risk-measure value
        q_risk = self._risk_value(q_quant)                      # (B,)
        # Expected value (for blending under Pareto sweeps)
        q_expected = q_quant.mean(dim=-1)                       # (B,)
        # Variance penalty
        var_q = q_quant.var(dim=-1, unbiased=False)             # (B,)
        # Equity floor shortfall (positive when below floor)
        bottom = self._bottom_decile(q_quant)                   # (B,)
        equity_violation = (cfg.equity_floor - bottom).clamp(min=0.0)

        # blended objective
        blended_q = cfg.w_expected * q_expected + cfg.w_cvar * q_risk
        if cfg.w_expected == 0.0 and cfg.w_cvar == 0.0:
            # default: use the configured risk measure directly
            blended_q = q_risk
        loss_actor = (self.alpha.detach() * lp_pi.squeeze(-1)
                       - blended_q
                       + cfg.w_variance_penalty * var_q
                       + self.lam * equity_violation).mean()

        self.opt_actor.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.opt_actor.step()

        # ---- Temperature ---- #
        loss_alpha = -(self.log_alpha * (lp_pi.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        loss_alpha.backward()
        self.opt_alpha.step()

        # ---- Soft target update ---- #
        with torch.no_grad():
            for tp, p in zip(self.target_critic.parameters(),
                              self.critic.parameters()):
                tp.data.mul_(1.0 - cfg.tau_target)
                tp.data.add_(cfg.tau_target * p.data)

        # ---- Lagrangian ---- #
        mean_viol = float(equity_violation.detach().mean().item())
        self.lam = float(np.clip(
            self.lam + cfg.lr_lambda * mean_viol, 0.0, cfg.max_lambda))

        m = {
            "loss_q": float(loss_q.item()),
            "loss_actor": float(loss_actor.item()),
            "alpha": float(self.alpha.detach().item()),
            "q_expected": float(q_expected.mean().item()),
            "q_risk": float(q_risk.mean().item()),
            "q_variance": float(var_q.mean().item()),
            "bottom_decile": float(bottom.mean().item()),
            "equity_violation": mean_viol,
            "lambda": self.lam,
        }
        self.last_metrics = m
        return m

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_quantiles(self, state: torch.Tensor, action: torch.Tensor,
                          n: Optional[int] = None) -> torch.Tensor:
        """Return (B, N) min-of-twin quantile predictions for analysis."""
        n = n or self.cfg.n_quantiles
        tau = self.critic.sample_tau(state.shape[0], n=n)
        q1, q2 = self.critic(state, action, tau)
        return torch.minimum(q1, q2)

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "lambda": self.lam,
            "cfg": self.cfg.__dict__,
        }, path)


# --------------------------------------------------------------------------- #
# Training driver: world-model-rollout SAC
# --------------------------------------------------------------------------- #

def train_dist_sac(world_model, initial_states: torch.Tensor,
                   sac_cfg: SACConfig, *,
                   total_steps: int = 1000,
                   warmup: int = 256,
                   batch_size: int = 256,
                   rollout_horizon: int = 1,
                   replay_capacity: int = 50_000,
                   reward_fn: Callable | None = None,
                   device: torch.device | None = None,
                   log_every: int = 200,
                   seed: int = 0) -> tuple["DistributionalSAC", dict]:
    """Train Distributional-SAC by rolling out the world model from
    randomly-sampled district states."""
    torch.manual_seed(seed); np.random.seed(seed)
    reward_fn = reward_fn or _default_reward_fn
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    world_model = world_model.to(device).eval()

    agent = DistributionalSAC(sac_cfg, device=device)
    buf = ReplayBuffer(replay_capacity, sac_cfg.state_dim, sac_cfg.action_dim)

    initial_states = initial_states.to(device)
    n_states = initial_states.shape[0]
    metrics_log: list[dict] = []

    # Warm-up: random actions on world-model rollouts
    def _rollout_step(state_b: torch.Tensor, action_b: torch.Tensor):
        with torch.no_grad():
            mean_sp, _, _ = world_model.predict(state_b, action_b)
        rwd = reward_fn(state_b, action_b, mean_sp)
        done = torch.zeros(state_b.shape[0], device=device)
        return mean_sp, rwd, done

    t0 = time.time()
    cur_state = initial_states[
        torch.randint(0, n_states, (batch_size,), device=device)]
    for step in range(total_steps):
        # action selection
        if buf.size < warmup:
            action = (sac_cfg.action_low + (sac_cfg.action_high - sac_cfg.action_low)
                       * torch.rand(batch_size, sac_cfg.action_dim, device=device))
        else:
            action, _ = agent.actor.sample(cur_state)

        next_state, reward, done = _rollout_step(cur_state, action)
        buf.push_batch(
            cur_state.detach().cpu().numpy(),
            action.detach().cpu().numpy(),
            reward.detach().cpu().numpy(),
            next_state.detach().cpu().numpy(),
            done.detach().cpu().numpy(),
        )
        cur_state = next_state.detach()
        # Re-seed every rollout_horizon steps to avoid drift
        if (step + 1) % max(1, rollout_horizon) == 0:
            cur_state = initial_states[
                torch.randint(0, n_states, (batch_size,), device=device)]

        if buf.size >= max(warmup, batch_size):
            batch = buf.sample(batch_size)
            m = agent.update(batch)
            if (step + 1) % log_every == 0:
                m_log = {"step": step + 1, **m,
                          "elapsed_s": round(time.time() - t0, 2)}
                metrics_log.append(m_log)
                log.info("step=%d  q_expected=%.4f  q_risk=%.4f  loss_q=%.4f",
                         step + 1, m["q_expected"], m["q_risk"], m["loss_q"])

    return agent, {"metrics_log": metrics_log,
                   "total_steps": total_steps,
                   "elapsed_s": round(time.time() - t0, 2)}


# --------------------------------------------------------------------------- #
# Phase-A-compatible smoke entry point
# --------------------------------------------------------------------------- #

def run(fast: bool = False) -> dict:
    """Tiny self-test that constructs the agent and runs one update."""
    cfg = SACConfig(n_quantiles=8 if fast else 32)
    agent = DistributionalSAC(cfg)
    B = 8
    batch = dict(
        state=np.random.randn(B, cfg.state_dim).astype(np.float32),
        action=np.random.randn(B, cfg.action_dim).astype(np.float32),
        reward=np.random.randn(B).astype(np.float32),
        next_state=np.random.randn(B, cfg.state_dim).astype(np.float32),
        done=np.zeros(B, dtype=np.float32),
    )
    m = agent.update(batch)
    return {"ok": True, "metrics": m}


if __name__ == "__main__":
    import json
    print(json.dumps(run(fast=True), indent=2, default=str))
