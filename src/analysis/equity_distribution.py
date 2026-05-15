"""Return distribution per district archetype.

K-means (k=5) on the Layer-4 district embeddings groups districts into
archetypes. For each archetype, we sample 1000 quantiles of the IQN-predicted
return distribution at the archetype's current state under the trained
CVaR-optimal policy and report summary stats + violin plots.

Outputs:
  - results/analysis/equity_distribution.json
  - figures/return_distributions_by_archetype.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


def _kmeans(embeddings: np.ndarray, k: int = 5,
            n_iter: int = 50, seed: int = 0) -> np.ndarray:
    """Lightweight Lloyd's algorithm; avoids the sklearn dependency."""
    rng = np.random.default_rng(seed)
    n, d = embeddings.shape
    init_idx = rng.choice(n, size=k, replace=False)
    centroids = embeddings[init_idx].copy()
    for _ in range(n_iter):
        dist2 = ((embeddings[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        labels = dist2.argmin(axis=1)
        new_c = np.stack([
            embeddings[labels == j].mean(axis=0) if (labels == j).any()
            else centroids[j]
            for j in range(k)
        ])
        if np.allclose(new_c, centroids, atol=1e-6):
            centroids = new_c
            break
        centroids = new_c
    return labels


def _summary(samples: np.ndarray) -> dict:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "skew": 0.0,
                "p10": 0.0, "p50": 0.0, "p90": 0.0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if std > 1e-9:
        skew = float(((arr - mean) ** 3).mean() / (std ** 3))
    else:
        skew = 0.0
    p10, p50, p90 = (float(x) for x in np.quantile(arr, [0.1, 0.5, 0.9]))
    return {"mean": mean, "std": std, "skew": skew,
            "p10": p10, "p50": p50, "p90": p90}


def run(fast: bool = False, *, agent=None,
        district_states: Optional[torch.Tensor] = None,
        k: int = 5, n_samples: int = 1000,
        output_dir: Optional[Path] = None) -> dict:
    """Compute per-archetype return distributions."""
    if agent is None or district_states is None:
        raise ValueError("equity_distribution.run requires `agent` and "
                         "`district_states`.")
    repo = Path(__file__).resolve().parents[2]
    output_dir = output_dir or (repo / "results" / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = repo / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)

    device = next(agent.actor.parameters()).device
    district_states = district_states.to(device)
    emb = district_states.detach().cpu().numpy()

    labels = _kmeans(emb, k=k, seed=0)
    per_archetype_samples: dict[int, np.ndarray] = {}

    with torch.no_grad():
        a_det = agent.actor.sample_action(district_states, deterministic=True)
    n_quant = agent.cfg.n_quantiles if hasattr(agent, "cfg") else 32
    rounds = max(1, (n_samples + n_quant - 1) // n_quant)
    all_q: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(rounds):
            q = agent.predict_quantiles(district_states, a_det, n=n_quant)
            all_q.append(q.detach().cpu().numpy())
    Q = np.concatenate(all_q, axis=1)[:, :n_samples]   # (B, n_samples)

    archetype_stats: dict[str, dict] = {}
    for j in range(k):
        idx = np.where(labels == j)[0]
        if idx.size == 0:
            archetype_stats[f"archetype_{j}"] = _summary(np.array([]))
            per_archetype_samples[j] = np.array([])
            continue
        samples_j = Q[idx].flatten()
        per_archetype_samples[j] = samples_j
        s = _summary(samples_j)
        s["n_districts"] = int(idx.size)
        archetype_stats[f"archetype_{j}"] = s

    out = {
        "k": int(k),
        "n_samples_per_archetype": int(n_samples),
        "per_archetype": archetype_stats,
        "labels": labels.tolist(),
    }
    json_path = output_dir / "equity_distribution.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("wrote %s", json_path)

    try:
        _plot_violins(per_archetype_samples,
                      fig_dir / "return_distributions_by_archetype.pdf")
    except Exception as e:  # noqa: BLE001
        log.warning("violin plot failed: %s", e)

    return out


def _plot_violins(per_archetype: dict[int, np.ndarray], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted(per_archetype.keys())
    data = [per_archetype[k] for k in keys if per_archetype[k].size > 0]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#126342")
        pc.set_edgecolor("#0d4a31")
        pc.set_alpha(0.6)
    ax.set_xticks(np.arange(1, len(data) + 1))
    ax.set_xticklabels([f"A{k}" for k in keys if per_archetype[k].size > 0])
    ax.set_ylabel("predicted return")
    ax.set_xlabel("district archetype")
    ax.set_title("Return distribution by archetype under CVaR-optimal policy")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    print("Run via scripts/run_distributional_rl.py — needs trained agent.")
