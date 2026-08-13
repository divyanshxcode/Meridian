"""Portfolio optimization: Monte Carlo efficient frontier + analytic max-Sharpe /
min-vol portfolios via SciPy. Also a forward Monte Carlo wealth simulation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from metrics import TRADING_DAYS, portfolio_stats

N_SIM = 4000


def simulate_frontier(rets: pd.DataFrame, n: int = N_SIM, rf: float = 0.0) -> pd.DataFrame:
    """Random long-only weight portfolios -> (ret, vol, sharpe) cloud."""
    n_assets = rets.shape[1]
    rng = np.random.default_rng(7)
    out = np.empty((n, 3))
    for i in range(n):
        w = rng.random(n_assets)
        w /= w.sum()
        s = portfolio_stats(w, rets, rf)
        out[i] = (s["ret"], s["vol"], s["sharpe"])
    return pd.DataFrame(out, columns=["ret", "vol", "sharpe"])


def _neg_sharpe(w, mean_ann, cov_ann, rf):
    port_ret = w @ mean_ann
    port_vol = np.sqrt(w @ cov_ann @ w)
    return -(port_ret - rf) / port_vol


def _port_vol(w, cov_ann):
    return np.sqrt(w @ cov_ann @ w)


def efficient_portfolios(rets: pd.DataFrame, rf: float = 0.0) -> dict:
    """Max-Sharpe and min-variance weights (long-only, fully invested)."""
    n = rets.shape[1]
    mean_ann = rets.mean().values * TRADING_DAYS
    cov_ann = rets.cov().values * TRADING_DAYS
    x0 = np.repeat(1 / n, n)
    bounds = [(0, 1)] * n
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)

    max_sh = minimize(_neg_sharpe, x0, args=(mean_ann, cov_ann, rf),
                      method="SLSQP", bounds=bounds, constraints=cons)
    min_vol = minimize(_port_vol, x0, args=(cov_ann,),
                       method="SLSQP", bounds=bounds, constraints=cons)

    return {
        "max_sharpe": dict(zip(rets.columns, np.clip(max_sh.x, 0, 1))),
        "min_vol": dict(zip(rets.columns, np.clip(min_vol.x, 0, 1))),
        "equal": dict(zip(rets.columns, x0)),
    }


def monte_carlo_wealth(weights: dict, rets: pd.DataFrame,
                       horizon_days: int = 252, n_paths: int = 500,
                       start: float = 100.0) -> pd.DataFrame:
    """Forward-simulate portfolio value using empirical daily-return bootstrap."""
    w = np.array([weights[c] for c in rets.columns])
    port_rets = (rets @ w).dropna().values
    rng = np.random.default_rng(11)
    idx = rng.integers(0, len(port_rets), size=(n_paths, horizon_days))
    draws = port_rets[idx]
    growth = np.cumprod(1 + draws, axis=1) * start
    return pd.DataFrame(growth)
