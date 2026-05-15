"""3D Pareto surface: efficiency x equity x risk.

Sweep a grid over (w_expected, w_cvar, w_variance_penalty). For each weight
combo, train a short distributional-SAC policy with that actor objective and
record (expected_return, cvar_return, variance_return) at convergence.

A "jointly improvable volume" is reported: the fraction of the grid where all
three metrics strictly dominate a status-quo baseline (the convex-default
status quo = the trio observed under the (w_expected=1, w_cvar=0,
w_variance_penalty=0) standard SAC objective).

Outputs:
  - results/analysis/pareto_surface.json   full surface
  - figures/pareto_surface_3d.pdf          3D scatter, coloured by joint flag
"""
from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


@dataclass
class GridPoint:
    w_expected: float
    w_cvar: float
    w_variance_penalty: float
    expected_return: float
    cvar_return: float
    variance: float
    bottom_decile: float
    elapsed_s: float


def _grid(fast: bool) -> list[tuple[float, float, float]]:
    """3D weight grid. Fast mode uses a 4×4×4 envelope but skips a fraction of
    interior points to keep total runtime under budget on CPU.

    The (1.0, 0.0, 0.0) status-quo point is always included so the baseline
    can be located by exact match.
    """
    if fast:
        # 4 × 4 × 4 = 64; we down-sample to ~30 points but always keep the
        # axis-aligned points (where two weights are zero) and the baseline.
        weights = (0.0, 0.5, 1.0, 1.5)
    else:
        weights = (0.0, 0.2, 0.5, 0.75, 1.0, 1.5)  # 6×6×6 = 216
    pts = list(itertools.product(weights, weights, weights))
    if fast:
        # keep axis-aligned + diagonal + baseline; drop the rest
        kept = []
        for (we, wc, wv) in pts:
            n_zero = sum(int(x == 0.0) for x in (we, wc, wv))
            is_baseline = (we == 1.0 and wc == 0.0 and wv == 0.0)
            is_axis = n_zero >= 2
            is_diag = (we == wc == wv) or (we == wc and wv == 0.0) or (we == wv and wc == 0.0) or (wc == wv and we == 0.0)
            if is_baseline or is_axis or is_diag:
                kept.append((we, wc, wv))
        # ~28 points
        pts = kept
    return pts


def _eval_policy(agent, states: torch.Tensor, n_samples: int = 64) -> dict:
    """Sample one action per state and evaluate the quantile distribution."""
    agent.actor.eval()
    with torch.no_grad():
        action, _ = agent.actor.sample(states)
        q = agent.predict_quantiles(states, action, n=n_samples)   # (B, N)
    agent.actor.train()
    expected = float(q.mean().item())
    sorted_q, _ = q.sort(dim=-1)
    k = max(1, int(round(0.1 * sorted_q.shape[-1])))
    cvar = float(sorted_q[:, :k].mean(dim=-1).mean().item())
    var = float(q.var(dim=-1, unbiased=False).mean().item())
    return {"expected": expected, "cvar": cvar, "variance": var,
            "bottom_decile": cvar}


def run(fast: bool = False, *, world_model=None,
        initial_states: Optional[torch.Tensor] = None,
        steps_per_point: int = 200,
        output_dir: Optional[Path] = None) -> dict:
    """Sweep the 3D weight grid and save results."""
    from src.models.distributional_sac import (
        SACConfig, train_dist_sac,
    )
    if world_model is None or initial_states is None:
        raise ValueError(
            "pareto_surface.run requires `world_model` and `initial_states` "
            "(call via scripts/run_distributional_rl.py)."
        )
    repo = Path(__file__).resolve().parents[2]
    output_dir = output_dir or (repo / "results" / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = repo / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)

    grid = _grid(fast)
    if fast:
        steps_per_point = min(steps_per_point, 120)
    points: list[GridPoint] = []
    import time
    t0 = time.time()
    for i, (we, wc, wv) in enumerate(grid):
        ti = time.time()
        cfg = SACConfig(
            w_expected=float(we), w_cvar=float(wc),
            w_variance_penalty=float(wv),
            risk_measure="cvar",   # used inside the CVaR head regardless
            n_quantiles=(8 if fast else 32),
        )
        try:
            agent, _ = train_dist_sac(
                world_model, initial_states, cfg,
                total_steps=steps_per_point,
                warmup=32 if fast else 64,
                batch_size=64 if fast else 128,
                rollout_horizon=4, log_every=10_000,
            )
            metrics = _eval_policy(agent, initial_states)
        except Exception as e:  # noqa: BLE001
            log.warning("grid point %d failed (%s) — skipping", i, e)
            continue
        gp = GridPoint(
            w_expected=we, w_cvar=wc, w_variance_penalty=wv,
            expected_return=metrics["expected"],
            cvar_return=metrics["cvar"],
            variance=metrics["variance"],
            bottom_decile=metrics["bottom_decile"],
            elapsed_s=round(time.time() - ti, 2),
        )
        points.append(gp)
        log.info("grid %d/%d  (we=%.2f wc=%.2f wv=%.2f)  "
                 "expected=%.4f cvar=%.4f var=%.4f  %.1fs",
                 i + 1, len(grid), we, wc, wv,
                 gp.expected_return, gp.cvar_return, gp.variance,
                 gp.elapsed_s)

    # Status-quo baseline: standard SAC (we=1, wc=0, wv=0)
    baseline = next(
        (p for p in points
         if p.w_expected == 1.0 and p.w_cvar == 0.0 and p.w_variance_penalty == 0.0),
        None,
    )
    if baseline is None and points:
        # If the exact (1,0,0) point was skipped, fall back to nearest neighbour.
        baseline = min(
            points,
            key=lambda p: (p.w_expected - 1.0) ** 2 + p.w_cvar ** 2 + p.w_variance_penalty ** 2,
        )

    if baseline is not None:
        # Higher = better for expected & cvar; LOWER = better for variance
        # (lower variance ↔ less risk for the same return level).
        def dominates(p: GridPoint, b: GridPoint) -> bool:
            return (p.expected_return > b.expected_return
                    and p.cvar_return > b.cvar_return
                    and p.variance < b.variance)
        joint_flags = [int(dominates(p, baseline)) for p in points]
    else:
        joint_flags = [0 for _ in points]
    jointly_improvable_volume = (sum(joint_flags) / len(points)) if points else 0.0

    out = {
        "fast": bool(fast),
        "steps_per_point": int(steps_per_point),
        "n_grid_points": len(points),
        "jointly_improvable_volume": float(jointly_improvable_volume),
        "baseline": (baseline.__dict__ if baseline is not None else None),
        "points": [
            {**p.__dict__, "jointly_improving": bool(jf)}
            for p, jf in zip(points, joint_flags)
        ],
        "elapsed_s": round(time.time() - t0, 2),
    }
    json_path = output_dir / "pareto_surface.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("wrote %s  (volume=%.3f, %d points)",
             json_path, jointly_improvable_volume, len(points))

    # Figure
    try:
        _plot_3d(points, joint_flags, fig_dir / "pareto_surface_3d.pdf")
    except Exception as e:  # noqa: BLE001
        log.warning("pareto figure failed: %s", e)

    return out


def _plot_3d(points: list[GridPoint], joint_flags: list[int],
             out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    xs = [p.expected_return for p in points]
    ys = [p.cvar_return for p in points]
    zs = [p.variance for p in points]
    colors = ["#126342" if jf else "#888888" for jf in joint_flags]
    ax.scatter(xs, ys, zs, c=colors, s=24, alpha=0.85)
    ax.set_xlabel("expected return")
    ax.set_ylabel("CVaR (10%)")
    ax.set_zlabel("variance")
    ax.set_title("Pareto surface: efficiency × equity × risk\n"
                 "green = jointly improves over standard SAC")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    print("Run via scripts/run_distributional_rl.py — needs world model.")
