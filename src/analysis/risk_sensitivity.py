"""Per-district risk sensitivity score.

For each district state, compare the optimal action under (1) the
expected-value policy and (2) the CVaR policy. Districts where these actions
diverge are "risk-sensitive" — different aggregations of the return
distribution recommend different interventions. Districts where they agree
are "robust" to the choice of risk measure.

Risk sensitivity = ||a_expected - a_cvar||_2 / ||a_expected||_2

Outputs:
  - results/analysis/risk_sensitivity.json
  - figures/risk_sensitivity_histogram.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


def run(fast: bool = False, *,
        expected_agent=None, cvar_agent=None,
        district_states: Optional[torch.Tensor] = None,
        threshold: float = 0.1,
        output_dir: Optional[Path] = None) -> dict:
    if expected_agent is None or cvar_agent is None or district_states is None:
        raise ValueError("risk_sensitivity.run requires `expected_agent`, "
                         "`cvar_agent`, and `district_states`.")
    repo = Path(__file__).resolve().parents[2]
    output_dir = output_dir or (repo / "results" / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = repo / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)

    dev_e = next(expected_agent.actor.parameters()).device
    dev_c = next(cvar_agent.actor.parameters()).device
    s_e = district_states.to(dev_e)
    s_c = district_states.to(dev_c)

    with torch.no_grad():
        a_e = expected_agent.actor.sample_action(s_e, deterministic=True)
        a_c = cvar_agent.actor.sample_action(s_c, deterministic=True)

    a_e_np = a_e.detach().cpu().numpy()
    a_c_np = a_c.detach().cpu().numpy()
    diff = a_e_np - a_c_np
    diff_norm = np.linalg.norm(diff, axis=-1)
    a_e_norm = np.linalg.norm(a_e_np, axis=-1) + 1e-6
    sensitivity = diff_norm / a_e_norm

    fraction = float((sensitivity > threshold).mean())

    out = {
        "threshold": float(threshold),
        "fraction_districts_risk_sensitive": fraction,
        "cvar_vs_expected_action_distance_mean": float(diff_norm.mean()),
        "cvar_vs_expected_action_distance_median": float(np.median(diff_norm)),
        "sensitivity_mean": float(sensitivity.mean()),
        "sensitivity_std": float(sensitivity.std()),
        "sensitivity_p10": float(np.percentile(sensitivity, 10)),
        "sensitivity_p50": float(np.percentile(sensitivity, 50)),
        "sensitivity_p90": float(np.percentile(sensitivity, 90)),
        "per_district": [
            {"idx": i,
             "sensitivity": float(sensitivity[i]),
             "action_distance": float(diff_norm[i]),
             "is_risk_sensitive": bool(sensitivity[i] > threshold)}
            for i in range(sensitivity.shape[0])
        ],
    }

    json_path = output_dir / "risk_sensitivity.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("wrote %s (fraction_risk_sensitive=%.3f)", json_path, fraction)

    try:
        _plot_histogram(sensitivity, threshold,
                        fig_dir / "risk_sensitivity_histogram.pdf")
    except Exception as e:  # noqa: BLE001
        log.warning("histogram failed: %s", e)
    return out


def _plot_histogram(sensitivity: np.ndarray, threshold: float,
                    out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(sensitivity, bins=30, color="#126342", alpha=0.75,
            edgecolor="#0d4a31")
    ax.axvline(threshold, color="#DFBB7F", lw=2,
               label=f"threshold = {threshold}")
    ax.set_xlabel("risk sensitivity  =  ||a_expected − a_cvar|| / ||a_expected||")
    ax.set_ylabel("# districts")
    ax.set_title("Per-district risk sensitivity")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    print("Run via scripts/run_distributional_rl.py — needs both agents.")
