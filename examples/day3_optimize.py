"""examples/day3_optimize.py — random portfolio vs equal-weight vs max-Sharpe.

Run:  .venv/bin/python examples/day3_optimize.py
"""
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data
import metrics
import optimizer

data.fetch_prices("3y")
rets = metrics.daily_returns(data.sql_wide_prices())
eff = optimizer.efficient_portfolios(rets)

# One random long-only portfolio, for illustration.
rng = np.random.default_rng(0)
w = rng.random(rets.shape[1])
w /= w.sum()
print("Random portfolio :", metrics.portfolio_stats(w, rets))

print("Equal weight     :", metrics.portfolio_stats(
    np.array([eff["equal"][c] for c in rets.columns]), rets))
print("Max-Sharpe       :", metrics.portfolio_stats(
    np.array([eff["max_sharpe"][c] for c in rets.columns]), rets))
print("Min-vol          :", metrics.portfolio_stats(
    np.array([eff["min_vol"][c] for c in rets.columns]), rets))
