"""Test the contract on jointly_improvable_volume after a fast run.

This test is skipped unless ``results/distributional_paper_summary.json``
exists (i.e. the fast run has been executed at least once). When present, we
check the headline scalars are in their expected ranges.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "results" / "distributional_paper_summary.json"


@pytest.mark.skipif(not SUMMARY.exists(),
                    reason="run scripts/run_distributional_rl.py --fast first")
def test_summary_ranges():
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    v = data["jointly_improvable_volume"]
    assert isinstance(v, (int, float))
    assert 0.0 <= v <= 1.0

    frac = data["fraction_districts_risk_sensitive"]
    assert 0.0 <= frac <= 1.0

    assert "per_archetype_return_p10" in data
    assert "per_archetype_return_p50" in data
