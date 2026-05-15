"""Smoke test: imports all module stubs and verifies they raise
NotImplementedError at run() time. Phase A scaffold guarantee."""
from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable without an installed package
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def test_imports_and_not_implemented():
    import pytest
    from src.models import iqn_critic as _models_iqn_critic
    from src.models import distributional_sac as _models_distributional_sac
    from src.analysis import pareto_surface as _analysis_pareto_surface
    from src.analysis import equity_distribution as _analysis_equity_distribution
    from src.analysis import risk_sensitivity as _analysis_risk_sensitivity
    mods = [_models_iqn_critic, _models_distributional_sac, _analysis_pareto_surface, _analysis_equity_distribution, _analysis_risk_sensitivity]
    assert len(mods) > 0, "no modules collected"
    for m in mods:
        with pytest.raises(NotImplementedError):
            m.run()


def test_config_loads():
    sys.path.insert(0, str(REPO / "src"))
    from utils.config_loader import load_config
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "data" in cfg
