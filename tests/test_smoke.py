"""Smoke tests (Phase B).

After Phase A, the model module run() entry points raised NotImplementedError;
in Phase B they return a structured ok-dict. The analysis modules now require
keyword args and so raise ValueError when called bare — which we test for to
keep the contract explicit.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable without an installed package
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def test_imports():
    from src.models import iqn_critic  # noqa: F401
    from src.models import distributional_sac  # noqa: F401
    from src.analysis import pareto_surface  # noqa: F401
    from src.analysis import equity_distribution  # noqa: F401
    from src.analysis import risk_sensitivity  # noqa: F401


def test_iqn_run_ok():
    from src.models.iqn_critic import run
    out = run(fast=True)
    assert out["ok"] is True
    assert out["q1_shape"] == (4, out["n_quantiles"])
    assert out["q2_shape"] == (4, out["n_quantiles"])


def test_dist_sac_run_ok():
    from src.models.distributional_sac import run
    out = run(fast=True)
    assert out["ok"] is True
    assert "metrics" in out
    assert "q_expected" in out["metrics"]


def test_analysis_requires_kwargs():
    import pytest
    from src.analysis import pareto_surface, equity_distribution, risk_sensitivity
    for mod in (pareto_surface, equity_distribution, risk_sensitivity):
        with pytest.raises(ValueError):
            mod.run(fast=True)


def test_config_loads():
    sys.path.insert(0, str(REPO / "src"))
    from utils.config_loader import load_config
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "data" in cfg
