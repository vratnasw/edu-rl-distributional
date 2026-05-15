"""Distributional SAC with CVaR-weighted actor loss (risk_measure=cvar, alpha=0.1).

STATUS: Phase A scaffold. Implementation pending (Phase B).
Reads from:
  - iqn_critic.pt
Outputs: results/checkpoints/distributional_sac.pt
"""
from __future__ import annotations


def run(fast: bool = False) -> dict:
    raise NotImplementedError(
        "Phase A scaffold -- module not yet implemented. "
        "See README.md for the research question this module answers.")


if __name__ == "__main__":
    print("SCAFFOLD ONLY -- module not yet implemented")
