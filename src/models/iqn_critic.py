"""Implicit Quantile Network critic (n_quantiles=32, kappa=1.0).

STATUS: Phase A scaffold. Implementation pending (Phase B).
Reads from:
  - Layer 5 world model
Outputs: results/checkpoints/iqn_critic.pt
"""
from __future__ import annotations


def run(fast: bool = False) -> dict:
    raise NotImplementedError(
        "Phase A scaffold -- module not yet implemented. "
        "See README.md for the research question this module answers.")


if __name__ == "__main__":
    print("SCAFFOLD ONLY -- module not yet implemented")
