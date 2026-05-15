"""Distributional RL orchestrator (Phase B).

Pipeline:
  1. Load Layer-5 world model + Layer-4 district embeddings.
  2. Train one CVaR-optimal distributional-SAC agent.
  3. Train an expected-value baseline agent for risk-sensitivity comparison.
  4. Run analyses:
       - Pareto surface (3D)
       - Equity distribution (per archetype)
       - Risk sensitivity (per district)
  5. Emit results/distributional_paper_summary.json.

Flags:
  --fast   smaller grid + 1000 training steps  (target: <15 min on CPU)
  --full   per config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Repo on path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "config"))

# Limit threads for AECF coexistence
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

# Logger
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s :: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("run_dist_rl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="reduced grid + 1000 training steps (default)")
    ap.add_argument("--full", action="store_true",
                    help="full grid + 1M training steps per config")
    args = ap.parse_args()
    fast = not args.full
    if args.full and args.fast:
        log.warning("both --fast and --full given; preferring --full")
        fast = False

    log.info("=== Distributional RL Phase B (fast=%s) ===", fast)
    t0 = time.time()

    # ------------------------------------------------------------------ #
    # 0. Threads (AECF coexistence)
    # ------------------------------------------------------------------ #
    import torch
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.set_num_interop_threads(min(4, torch.get_num_threads()))
    log.info("torch threads=%d interop=%d",
             torch.get_num_threads(), torch.get_num_interop_threads())

    # ------------------------------------------------------------------ #
    # 1. Load world model + embeddings
    # ------------------------------------------------------------------ #
    from utils.world_model_loader import (
        load_world_model, load_district_embeddings, WorldModelLoadError,
    )
    try:
        world_model, arch = load_world_model()
    except WorldModelLoadError as e:
        log.error("world model load failed: %s — wrote results/blocker.json", e)
        return 2
    log.info("loaded world model: state_dim=%d action_dim=%d ensemble=%d",
             arch["state_dim"], arch["action_dim"],
             arch.get("ensemble_size", 5))

    district_states = load_district_embeddings()
    log.info("loaded district embeddings: %s", tuple(district_states.shape))

    state_dim = int(arch["state_dim"])
    action_dim = int(arch["action_dim"])

    # ------------------------------------------------------------------ #
    # 2. Train CVaR-optimal agent
    # ------------------------------------------------------------------ #
    from src.models.distributional_sac import SACConfig, train_dist_sac
    cvar_steps = 1000 if fast else 50_000
    cvar_cfg = SACConfig(
        state_dim=state_dim, action_dim=action_dim,
        risk_measure="cvar", alpha_cvar=0.1,
        w_expected=0.0, w_cvar=1.0, w_variance_penalty=0.0,
        n_quantiles=8 if fast else 32,
    )
    log.info("training CVaR agent (%d steps)…", cvar_steps)
    cvar_agent, cvar_metrics = train_dist_sac(
        world_model, district_states, cvar_cfg,
        total_steps=cvar_steps,
        warmup=128 if fast else 1000,
        batch_size=128 if fast else 256,
        rollout_horizon=4,
        log_every=200 if fast else 1000,
        seed=42,
    )
    cvar_ckpt = REPO / "results" / "distributional_sac" / "cvar_agent.pt"
    cvar_agent.save(cvar_ckpt)
    log.info("CVaR agent saved → %s", cvar_ckpt)

    # ------------------------------------------------------------------ #
    # 3. Train expected-value baseline (for risk-sensitivity comparison)
    # ------------------------------------------------------------------ #
    exp_steps = 1000 if fast else 50_000
    exp_cfg = SACConfig(
        state_dim=state_dim, action_dim=action_dim,
        risk_measure="expected",
        w_expected=1.0, w_cvar=0.0, w_variance_penalty=0.0,
        n_quantiles=8 if fast else 32,
    )
    log.info("training expected-value baseline (%d steps)…", exp_steps)
    exp_agent, exp_metrics = train_dist_sac(
        world_model, district_states, exp_cfg,
        total_steps=exp_steps,
        warmup=128 if fast else 1000,
        batch_size=128 if fast else 256,
        rollout_horizon=4,
        log_every=200 if fast else 1000,
        seed=43,
    )
    exp_ckpt = REPO / "results" / "distributional_sac" / "expected_agent.pt"
    exp_agent.save(exp_ckpt)
    log.info("expected agent saved → %s", exp_ckpt)

    # Save per-agent training metrics
    res_path = REPO / "results" / "distributional_sac" / "results.json"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps({
        "cvar": {"cfg": cvar_cfg.__dict__, **cvar_metrics},
        "expected": {"cfg": exp_cfg.__dict__, **exp_metrics},
        "fast": fast,
    }, indent=2, default=str), encoding="utf-8")
    log.info("training metrics → %s", res_path)

    # ------------------------------------------------------------------ #
    # 4a. Pareto surface
    # ------------------------------------------------------------------ #
    from src.analysis.pareto_surface import run as pareto_run
    pareto = pareto_run(
        fast=fast,
        world_model=world_model,
        initial_states=district_states,
        steps_per_point=200 if fast else 1000,
    )
    log.info("Pareto: jointly_improvable_volume=%.3f over %d grid points",
             pareto["jointly_improvable_volume"], pareto["n_grid_points"])

    # ------------------------------------------------------------------ #
    # 4b. Equity distribution
    # ------------------------------------------------------------------ #
    from src.analysis.equity_distribution import run as equity_run
    equity = equity_run(
        fast=fast, agent=cvar_agent,
        district_states=district_states,
        k=5, n_samples=1000,
    )

    # ------------------------------------------------------------------ #
    # 4c. Risk sensitivity
    # ------------------------------------------------------------------ #
    from src.analysis.risk_sensitivity import run as risk_run
    risk = risk_run(
        fast=fast, expected_agent=exp_agent, cvar_agent=cvar_agent,
        district_states=district_states,
        threshold=0.1,
    )

    # ------------------------------------------------------------------ #
    # 5. Paper summary
    # ------------------------------------------------------------------ #
    per_arch_p10 = {k: v.get("p10", 0.0)
                    for k, v in equity["per_archetype"].items()}
    per_arch_p50 = {k: v.get("p50", 0.0)
                    for k, v in equity["per_archetype"].items()}
    summary = {
        "jointly_improvable_volume": float(pareto["jointly_improvable_volume"]),
        "fraction_districts_risk_sensitive": float(risk["fraction_districts_risk_sensitive"]),
        "cvar_vs_expected_action_distance_mean": float(risk["cvar_vs_expected_action_distance_mean"]),
        "per_archetype_return_p10": per_arch_p10,
        "per_archetype_return_p50": per_arch_p50,
        "fast": fast,
        "elapsed_s": round(time.time() - t0, 2),
        "n_districts": int(district_states.shape[0]),
        "n_pareto_grid_points": int(pareto["n_grid_points"]),
        "config": {
            "cvar_cfg": cvar_cfg.__dict__,
            "exp_cfg": exp_cfg.__dict__,
        },
    }
    out = REPO / "results" / "distributional_paper_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("=== Done in %.1f s → %s ===", summary["elapsed_s"], out)
    log.info("jointly_improvable_volume=%.3f",
             summary["jointly_improvable_volume"])
    log.info("fraction_districts_risk_sensitive=%.3f",
             summary["fraction_districts_risk_sensitive"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
