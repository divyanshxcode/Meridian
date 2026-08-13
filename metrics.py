"""Risk & return analytics. Pure functions over return Series/DataFrames.

Mirrors the metrics an Investments/risk team tracks: annualised return/vol,
Sharpe/Sortino, max drawdown, VaR/CVaR, and beta/alpha vs a benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def annualized_return(rets: pd.Series, rf: float = 0.0) -> float:
    if rf:
        g = (1 + rets).prod() ** (TRADING_DAYS / len(rets)) - 1
        return g - rf
    return (1 + rets.mean()) ** TRADING_DAYS - 1


def annualized_vol(rets: pd.Series) -> float:
    return rets.std() * np.sqrt(TRADING_DAYS)


def sharpe(rets: pd.Series, rf: float = 0.0) -> float:
    excess = rets - rf / TRADING_DAYS
    if excess.std() == 0:
        return 0.0
    return excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)


def sortino(rets: pd.Series, rf: float = 0.0) -> float:
    excess = rets - rf / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return excess.mean() / downside.std() * np.sqrt(TRADING_DAYS)


def max_drawdown(series: pd.Series) -> float:
    """Most negative peak-to-trough return (returned as a positive fraction)."""
    wealth = (1 + series).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    return float(-dd.min())


def var_cvar(rets: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    """Historical 1-day VaR and CVaR at the given confidence (positive losses)."""
    q = np.percentile(rets, (1 - alpha) * 100)
    cvar = rets[rets <= q].mean()
    return float(-q), float(-cvar)


def beta_alpha(rets: pd.Series, bench: pd.Series, rf: float = 0.0) -> tuple[float, float]:
    """CAPM beta and annualised Jensen's alpha."""
    df = pd.concat([rets, bench], axis=1).dropna()
    df.columns = ["a", "b"]
    if df["b"].var() == 0 or len(df) < 3:
        return 0.0, 0.0
    cov = np.cov(df["a"], df["b"])
    beta = cov[0, 1] / cov[1, 1]
    excess_a = df["a"] - rf / TRADING_DAYS
    excess_b = df["b"] - rf / TRADING_DAYS
    alpha_daily = excess_a.mean() - beta * excess_b.mean()
    return float(beta), float(alpha_daily * TRADING_DAYS)


def portfolio_stats(weights: np.ndarray, rets: pd.DataFrame, rf: float = 0.0) -> dict:
    """Return/vol/Sharpe/Sortino for a fixed weight vector over asset returns."""
    w = np.asarray(weights, dtype=float)
    port = rets @ w
    return {
        "ret": annualized_return(port, rf),
        "vol": annualized_vol(port),
        "sharpe": sharpe(port, rf),
        "sortino": sortino(port, rf),
        "mdd": max_drawdown(port),
        "var95": var_cvar(port, 0.95)[0],
        "cvar95": var_cvar(port, 0.95)[1],
    }


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()
