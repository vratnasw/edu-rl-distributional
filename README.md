# edu-rl-distributional

**Status:** SCAFFOLD (Phase A). Module files raise `NotImplementedError`.
Phase B implementation pending.

**Target journal:** JMLR or AAAI

## Research question

Does optimizing for the lower tail (CVaR at alpha=0.1) of the return distribution -- a Rawlsian objective -- produce more equitable district outcomes than expected-return RL?

## Description

Distributional RL with Implicit Quantile Networks + CVaR for Rawlsian equity optimization.

## Architecture overview

Implicit Quantile Networks with 32 quantiles (kappa=1.0), CVaR risk measure at alpha=0.1, swept across pareto_quantiles [0.1, ..., 0.9] to map the equity / aggregate-utility frontier.

## Layout

```
config/                # config.yaml + canonical r2_client.py
src/                   # module stubs (raise NotImplementedError)
scripts/preflight.py   # infra check (R2 + checkpoints + GPU)
scripts/run_*.py       # orchestrator stub
tests/test_smoke.py    # imports + NotImplementedError assertions
notebooks/             # exploration stub
figures/, results/     # output dirs
```

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in R2 credentials
python scripts/preflight.py
python scripts/run_*.py --fast
pytest tests/
```

## Phase B will produce

- IQN + distributional SAC checkpoints
- Pareto surface (equity vs. expected return)
- Per-district risk-sensitivity diagnostics

## Operational note

The AECF national pull is running in background on this dev machine; do
not attempt parallel heavy compute (training, large embedding jobs) that
might conflict with it.
