"""Tests for metrics.py — the risk/return foundations.

Each test asserts a textbook fact. If you can explain WHY it is true,
you know the material — and the formulas are provably correct.
"""
import numpy as np
import pandas as pd
import pytest

from metrics import (annualized_vol, beta_alpha, max_drawdown, sharpe,
                     sortino, var_cvar)


def test_sharpe_flat_returns_is_zero():
    """No variation -> no risk premium -> Sharpe 0."""
    r = pd.Series(np.zeros(200))
    assert sharpe(r) == pytest.approx(0.0)


def test_sortino_flat_returns_is_zero():
    r = pd.Series(np.zeros(200))
    assert sortino(r) == pytest.approx(0.0)


def test_max_drawdown_positive_for_declining_series():
    r = pd.Series([-0.01, -0.02, 0.005, -0.03])
    assert max_drawdown(r) > 0


def test_var_le_cvar():
    """Expected shortfall (CVaR) is at least as bad as the VaR quantile."""
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0, 0.02, 2000))
    var, cvar = var_cvar(r, alpha=0.95)
    assert cvar >= var


def test_beta_of_series_against_itself_is_one():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0004, 0.012, 1000))
    beta, alpha = beta_alpha(r, r)
    assert beta == pytest.approx(1.0, abs=1e-6)
    assert alpha == pytest.approx(0.0, abs=1e-6)


def test_annualized_vol_scales_with_sqrt_252():
    rng = np.random.default_rng(9)
    daily = pd.Series(rng.normal(0.0, 0.01, 500))
    # std of daily returns * sqrt(252) must match annualized_vol
    assert annualized_vol(daily) == pytest.approx(daily.std() * np.sqrt(252))
