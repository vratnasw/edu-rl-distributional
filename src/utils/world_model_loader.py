"""Load the Layer-5 world-model ensemble from R2 or local disk.

Strategy:
1. Try local cache: ``results/checkpoints/world_model.pt`` (downloaded once).
2. If not present, download from R2 key ``checkpoints/edu-world-model/best.pt``.
3. Reconstruct ``WorldModelEnsemble`` using the ``arch`` dict in the checkpoint
   and load the state dict.

The Layer-5 ensemble code lives at ``c:/Users/visha/Downloads/r_and_d/
edu-world-model/src/models/world_model_ensemble.py``. We add that source dir
to ``sys.path`` so the import is straightforward.

On any failure, this writes ``results/blocker.json`` with a structured error
and re-raises ``WorldModelLoadError`` so the caller can decide whether to
gracefully exit.
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch

log = logging.getLogger(__name__)


class WorldModelLoadError(RuntimeError):
    pass


WORLD_MODEL_R2_KEY = "checkpoints/edu-world-model/best.pt"


def _add_world_model_source_to_path() -> Path:
    """Find the edu-world-model repo (sibling) and add its src/ to sys.path."""
    # repo layout: edu-rl-distributional/ and edu-world-model/ are siblings.
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "edu-world-model" / "src",
        here.parents[2].parent / "edu-world-model" / "src",
        Path("../edu-world-model/src").resolve(),
    ]
    for c in candidates:
        if c.exists():
            sys.path.insert(0, str(c))
            return c
    raise WorldModelLoadError(
        f"edu-world-model/src not found; tried: {[str(c) for c in candidates]}"
    )


def _download_to_cache(cache_path: Path) -> Path:
    """Download the R2 checkpoint to a local cache path."""
    # Make sure the .env is loaded before instantiating the R2 client.
    repo = Path(__file__).resolve().parents[2]
    for env_path in (repo / ".env", repo.parent / ".env",
                     repo.parent / "edu-data-pipeline" / ".env"):
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path, override=False)
            except ImportError:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
            break

    sys.path.insert(0, str(repo / "config"))
    import r2_client as r2  # type: ignore
    c = r2._client()
    obj = c.get_object(Bucket=r2.bucket_name(), Key=WORLD_MODEL_R2_KEY)
    body = obj["Body"].read()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(body)
    log.info("cached world model → %s (%d bytes)", cache_path, len(body))
    return cache_path


def load_world_model(*, device: Optional[torch.device] = None) -> tuple:
    """Returns (world_model, arch_dict). Raises WorldModelLoadError on failure
    (writes results/blocker.json before raising)."""
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    repo = Path(__file__).resolve().parents[2]
    cache = repo / "results" / "checkpoints" / "world_model.pt"

    try:
        if not cache.exists():
            _download_to_cache(cache)
        ck = torch.load(cache, map_location=device, weights_only=False)
        if not isinstance(ck, dict) or "state_dict" not in ck or "arch" not in ck:
            raise WorldModelLoadError(
                f"unexpected checkpoint structure: keys={list(ck.keys()) if isinstance(ck, dict) else type(ck)}"
            )
        arch = ck["arch"]
        _add_world_model_source_to_path()
        from models.ensemble_member import MemberConfig
        from models.world_model_ensemble import WorldModelEnsemble

        member_cfg = MemberConfig(
            state_dim=int(arch["state_dim"]),
            action_dim=int(arch["action_dim"]),
            hidden_dim=int(arch.get("hidden_dim", 256)),
            num_layers=int(arch.get("num_layers", 3)),
        )
        wm = WorldModelEnsemble(
            member_cfg=member_cfg,
            ensemble_size=int(arch.get("ensemble_size", 5)),
        ).to(device)
        wm.load_state_dict(ck["state_dict"])
        wm.eval()
        return wm, arch
    except Exception as e:
        blocker = repo / "results" / "blocker.json"
        blocker.parent.mkdir(parents=True, exist_ok=True)
        blocker.write_text(json.dumps({
            "blocker": "world_model_load_failed",
            "error": f"{type(e).__name__}: {e}",
        }, indent=2), encoding="utf-8")
        log.error("world model load failed (%s); wrote %s",
                  e, blocker)
        raise WorldModelLoadError(str(e))


def load_district_embeddings(*, device: Optional[torch.device] = None) -> torch.Tensor:
    """Load the Layer-4 district embeddings (shape: 202 x T x 128).

    Returns the *latest* timestep slice as the "current state": (202, 128).
    """
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    repo = Path(__file__).resolve().parents[2]
    cache = repo / "results" / "checkpoints" / "district_embeddings.pt"
    if not cache.exists():
        for env_path in (repo / ".env", repo.parent / ".env",
                         repo.parent / "edu-data-pipeline" / ".env"):
            if env_path.exists():
                try:
                    from dotenv import load_dotenv
                    load_dotenv(env_path, override=False)
                except ImportError:
                    for line in env_path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip())
                break
        sys.path.insert(0, str(repo / "config"))
        import r2_client as r2  # type: ignore
        c = r2._client()
        body = c.get_object(Bucket=r2.bucket_name(),
                            Key="embeddings/edu-gnn/district_embeddings.pt"
                            )["Body"].read()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(body)
        log.info("cached district embeddings → %s", cache)
    emb = torch.load(cache, map_location=device, weights_only=False)
    if isinstance(emb, dict) and "embeddings" in emb:
        emb = emb["embeddings"]
    if emb.dim() == 3:
        # (N, T, D) → take latest timestep
        emb = emb[:, -1, :]
    return emb.to(device).float()
