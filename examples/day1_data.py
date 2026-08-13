"""examples/day1_data.py — load SQL, compute daily returns.

Run:  uv run examples/day1_data.py   (or: .venv/bin/python examples/day1_data.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data
import metrics

data.init_schema()
data.seed_assets()
data.fetch_prices("3y")

prices = data.sql_wide_prices()
print("Price matrix:", prices.shape, "(days x assets)")

# Daily returns, computed two ways to show they match.
manual = prices / prices.shift(1) - 1
lib = metrics.daily_returns(prices)
print("Manual vs library returns differ (max abs):",
      float((manual - lib).abs().max().max()))
print("First 3 return rows for AAPL:")
print(lib["AAPL"].head(3).round(5))
