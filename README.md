# Meridian

**Portfolio Analytics & Risk Dashboard** — an end-to-end investment analytics web app
for an Investments team. Ingests market data, lands it in a SQL warehouse, and serves
an interactive dashboard covering portfolio construction, risk measurement, and
forward simulation. A mini "risk engine" in the spirit of how asset managers think
about portfolios.

Stack: **Python · SQL (SQLite) · Streamlit · Plotly · SciPy · yfinance**

---

## Architecture

```
yfinance  ──►  SQLite warehouse (meridian.db)
                  ├─ assets  (ticker, name, sector, class)
                  └─ prices  (ticker, date, close)        ← queried with raw SQL
                        │
                        ▼
              metrics.py      (risk / return math)
              optimizer.py    (efficient frontier, allocation, simulation)
                        │
                        ▼
              app.py  (Streamlit, 4 tabs, Plotly charts)
```

The SQLite database is the single source of truth. Every metric and optimization is
computed from rows pulled by SQL — separating storage from analytics the way a real
investments team would.

## Run

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
streamlit run app.py
```

On first launch the app downloads market data (yfinance, no API key) and builds the
SQLite DB. If the network is unavailable it falls back to a seeded synthetic dataset
and tags the source in the `meta` table, so the dashboard always runs.

## What it does

| Tab | Content |
| --- | --- |
| **Overview** | Asset universe, price history, daily-return correlation heatmap |
| **Risk Metrics** | Per-asset annualized return/vol, Sharpe, Sortino, max drawdown, historical VaR/CVaR (95%), beta & Jensen's alpha vs benchmark |
| **Optimization** | Monte Carlo efficient frontier (4,000 portfolios) + SciPy max-Sharpe / min-vol / equal-weight allocations |
| **Monte Carlo** | Forward wealth simulation (500 paths × 252 days) with percentile outcomes |

## Examples

`examples/` contains runnable scripts that recompute each concept from the SQL data:

```bash
.venv/bin/python examples/day1_data.py     # load SQL, compute returns
.venv/bin/python examples/day2_risk.py     # risk metrics by asset
.venv/bin/python examples/day3_optimize.py # random vs equal-weight vs max-Sharpe
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Asserts textbook facts (Sharpe of flat returns = 0, VaR ≤ CVaR, beta of a series
against itself = 1) so the formulas are provably correct.

## Methodology (stated honestly)

- Risk-free rate assumed **0%** for the demo; wire `RF` in `app.py` to a real rate to
  make Sharpe/Sortino ex-ante.
- Monte Carlo wealth simulation uses an **empirical bootstrap** of historical daily
  returns — no parametric distribution assumed.
- Optimization is **long-only, fully invested** mean-variance (SLSQP). Past returns are
  not a forecast of future performance.
- Synthetic fallback data is deterministic (seeded); it exists only so the app runs
  offline and is clearly flagged in the UI.

## BlackRock JD mapping

| JD requirement | Where Meridian covers it |
| --- | --- |
| Build data & analytics with Python and SQL | `data.py` (yfinance → SQLite), raw SQL queries |
| Work with vast financial data to generate insights | 11-asset universe, risk/return analytics |
| Portfolio & risk management | VaR/CVaR, drawdown, beta/alpha, optimization |
| Interactive dashboards for transparency | Streamlit + Plotly, 4 tabs |
| Drive automation | Monte Carlo simulation + one-click refresh |
| Strong code governance | `tests/` suite, modular `metrics.py` / `optimizer.py` |

## Investment universe

8 blue-chip US equities (AAPL, MSFT, NVDA, AMZN, GOOGL, JPM, XOM, JNJ) + 3 iShares ETFs
(IVV, IUSB, GLD). The iShares selection is a deliberate nod to BlackRock's ETF franchise.

---

*Educational tool — not investment advice.*
