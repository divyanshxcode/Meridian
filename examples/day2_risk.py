"""examples/day2_risk.py — compute risk metrics from returns.

Run:  .venv/bin/python examples/day2_risk.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data
import metrics

data.fetch_prices("3y")
rets = metrics.daily_returns(data.sql_wide_prices())

for t in ["AAPL", "IVV", "GLD"]:
    r = rets[t]
    print(f"\n{t}:")
    print(f"  ann return : {metrics.annualized_return(r):.2%}")
    print(f"  ann vol     : {metrics.annualized_vol(r):.2%}")
    print(f"  Sharpe      : {metrics.sharpe(r):.2f}")
    print(f"  Sortino     : {metrics.sortino(r):.2f}")
    print(f"  max drawdown: {metrics.max_drawdown(r):.2%}")
    var95, cvar95 = metrics.var_cvar(r)
    print(f"  VaR 95%     : {var95:.2%}   CVaR 95%: {cvar95:.2%}")
