"""Meridian — Portfolio Analytics & Risk Dashboard (Streamlit front-end).

Five tabs: Overview | Risk Metrics | Optimization | Portfolio Builder | Monte Carlo.
All data is read from the SQLite warehouse (data.py); all analytics come from
metrics.py / optimizer.py. Mirrors an Investments team's storage-vs-analytics split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data
import metrics
import optimizer

st.set_page_config(page_title="Meridian", layout="wide", page_icon="M")
RF = 0.0  # annual risk-free (simplify to 0 for demo; wire to a rate later)


_PERIOD_DAYS = {"1y": 252, "3y": 756, "5y": 1260}


@st.cache_data(show_spinner="Loading market data…")
def load_prices(period: str) -> pd.DataFrame:
    data.init_schema()
    data.seed_assets()
    data.fetch_prices(period)
    wide = data.sql_wide_prices()
    # Scope to the selected lookback so the chart/tables actually change with period.
    # The DB may hold more history from a prior longer fetch; tail by trading days.
    return wide.tail(_PERIOD_DAYS.get(period, 756))


def _sidebar() -> tuple[str, str, list[str]]:
    st.sidebar.title("Meridian")
    st.sidebar.caption("Portfolio analytics & risk engine for an investments team.")
    source = data.meta("source")
    if source:
        st.sidebar.info(f"Data source: {source}")
    period = st.sidebar.selectbox("Lookback", ["1y", "3y", "5y"], index=1)
    benchmark = st.sidebar.selectbox("Benchmark", ["IVV", "SPY", "^GSPC"], index=0)
    assets = data.sql_assets()
    classes = sorted(assets["asset_class"].unique())
    chosen = st.sidebar.multiselect("Asset class", classes, default=classes)
    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
    with st.sidebar.expander("Methodology"):
        st.markdown(
            "Risk-free rate assumed 0%. Monte Carlo uses an empirical bootstrap of "
            "historical daily returns. Optimization is long-only, fully invested "
            "mean-variance (SLSQP). Synthetic fallback data is seeded and flagged above "
            "if the live pull fails."
        )
    return period, benchmark, chosen


def main() -> None:
    period, benchmark, chosen = _sidebar()

    prices = load_prices(period)
    assets = data.sql_assets()
    if chosen:
        tickers = assets.loc[assets["asset_class"].isin(chosen), "ticker"].tolist()
        prices = prices[tickers]
    if benchmark not in prices.columns:
        benchmark = prices.columns[0]
    rets = metrics.daily_returns(prices)

    tabs = st.tabs(["Overview", "Risk Metrics", "Optimization",
                   "Portfolio Builder", "Monte Carlo"])
    _overview(tabs[0], prices, rets)
    _risk(tabs[1], prices, rets, benchmark)
    _optimize(tabs[2], prices, rets)
    _builder(tabs[3], rets)
    _mc(tabs[4], rets)


def _overview(tab, prices, rets):
    with tab:
        st.subheader("Universe & price history")
        st.dataframe(data.sql_assets().set_index("ticker"), use_container_width=True)

        # Summary strip — compact, no prose.
        ann = rets.apply(metrics.annualized_return)
        vol = rets.apply(metrics.annualized_vol)
        corr = metrics.correlation_matrix(rets)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best performer (ann. ret)", f"{ann.idxmax()} {ann.max():.1%}")
        c2.metric("Worst performer (ann. ret)", f"{ann.idxmin()} {ann.min():.1%}")
        c3.metric("Most volatile", f"{vol.idxmax()} {vol.max():.1%}")
        c4.metric("Avg pairwise correlation", f"{corr.where(~np.eye(len(corr), dtype=bool)).stack().mean():.2f}")

        fig = px.line(prices, title="Adjusted close")
        fig.update_layout(height=380, xaxis_title="", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                        title="Daily return correlation")
        st.plotly_chart(fig, use_container_width=True)


def _risk(tab, prices, rets, benchmark):
    with tab:
        st.subheader(f"Per-asset risk & return (benchmark: {benchmark})")
        bench = rets[benchmark]
        rows = []
        for t in rets.columns:
            r = rets[t]
            var95, cvar95 = metrics.var_cvar(r)
            beta, alpha = metrics.beta_alpha(r, bench)
            rows.append({
                "Ticker": t,
                "Ann. Return": metrics.annualized_return(r),
                "Ann. Vol": metrics.annualized_vol(r),
                "Sharpe": metrics.sharpe(r),
                "Sortino": metrics.sortino(r),
                "Max DD": metrics.max_drawdown(r),
                "VaR 95%": var95,
                "CVaR 95%": cvar95,
                "Beta": beta,
                "Alpha": alpha,
            })
        df = pd.DataFrame(rows).set_index("Ticker")
        fmt = {c: "{:.2%}" for c in ["Ann. Return", "Ann. Vol", "Max DD", "VaR 95%", "CVaR 95%", "Alpha"]}
        fmt.update({c: "{:.2f}" for c in ["Sharpe", "Sortino", "Beta"]})
        st.dataframe(df.style.format(fmt), use_container_width=True)


def _optimize(tab, prices, rets):
    with tab:
        st.subheader("Efficient frontier & allocations")
        cloud = optimizer.simulate_frontier(rets, rf=RF)
        eff = optimizer.efficient_portfolios(rets, rf=RF)

        fig = px.scatter(cloud, x="vol", y="ret", color="sharpe",
                         color_continuous_scale="Viridis",
                         title="Monte Carlo efficient frontier (4,000 portfolios)",
                         labels={"vol": "Ann. volatility", "ret": "Ann. return"})
        for label, key in [("Max-Sharpe", "max_sharpe"), ("Min-Vol", "min_vol"), ("Equal", "equal")]:
            w = np.array([eff[key][c] for c in rets.columns])
            s = metrics.portfolio_stats(w, rets, RF)
            fig.add_scatter(x=[s["vol"]], y=[s["ret"]], mode="markers+text",
                            text=[label], textposition="top center",
                            marker=dict(size=14, symbol="star"),
                            name=label)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recommended allocations")
        wdf = pd.DataFrame(eff)
        st.dataframe(wdf.style.format("{:.1%}"), use_container_width=True)

        w = np.array([eff["max_sharpe"][c] for c in rets.columns])
        s = metrics.portfolio_stats(w, rets, RF)
        eq = np.array([eff["equal"][c] for c in rets.columns])
        se = metrics.portfolio_stats(eq, rets, RF)
        c1, c2, c3 = st.columns(3)
        c1.metric("Max-Sharpe return", f"{s['ret']:.1%}")
        c2.metric("Max-Sharpe Sharpe", f"{s['sharpe']:.2f}",
                  delta=f"{s['sharpe']-se['sharpe']:.2f} vs equal-weight")
        c3.metric("Max-Sharpe vol", f"{s['vol']:.1%}")


def _builder(tab, rets):
    with tab:
        st.subheader("Portfolio Builder")
        st.caption("Set asset weights with the sliders (normalized to 100%). "
                   "Risk and return metrics recompute as you adjust.")
        eff = optimizer.efficient_portfolios(rets, rf=RF)
        cols = list(rets.columns)

        left, right = tab.columns([1, 1.4])
        with left:
            raw = {c: st.slider(f"{c}", 0, 100,
                                int(round(eff["max_sharpe"][c] * 100)), 1,
                                key=f"w_{c}") for c in cols}
            total = sum(raw.values()) or 1
            w = np.array([raw[c] / total for c in cols])

            s = metrics.portfolio_stats(w, rets, RF)
            m1, m2, m3 = st.columns(3)
            m1.metric("Ann. return", f"{s['ret']:.1%}")
            m2.metric("Ann. vol", f"{s['vol']:.1%}")
            m3.metric("Sharpe", f"{s['sharpe']:.2f}")
            m4, m5, m6 = st.columns(3)
            m4.metric("Sortino", f"{s['sortino']:.2f}")
            m5.metric("Max DD", f"{s['mdd']:.1%}")
            m6.metric("VaR 95%", f"{s['var95']:.2%}")

        with right:
            alloc = pd.DataFrame({"ticker": cols, "weight": w})
            fig = px.bar(alloc, x="ticker", y="weight", title="Your allocation",
                         labels={"weight": "weight"})
            st.plotly_chart(fig, use_container_width=True)

            ms = np.array([eff["max_sharpe"][c] for c in cols])
            sm = metrics.portfolio_stats(ms, rets, RF)
            st.markdown(f"Versus Max-Sharpe recommendation — your Sharpe {s['sharpe']:.2f} "
                        f"vs {sm['sharpe']:.2f}; your volatility {s['vol']:.1%} vs {sm['vol']:.1%}.")
            diff = pd.DataFrame({"ticker": cols, "delta_vs_max_sharpe": w - ms})
            fig = px.bar(diff, x="ticker", y="delta_vs_max_sharpe",
                         title="Weight drift versus Max-Sharpe mix",
                         labels={"delta_vs_max_sharpe": "over / under weight"})
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)


def _mc(tab, rets):
    with tab:
        st.subheader("Forward Monte Carlo wealth simulation")
        eff = optimizer.efficient_portfolios(rets, rf=RF)
        strat = st.selectbox("Allocation", ["max_sharpe", "min_vol", "equal"], index=0)
        w = eff[strat]
        paths = optimizer.monte_carlo_wealth(w, rets, horizon_days=252, n_paths=500)
        final = paths.iloc[:, -1]

        fig = go.Figure()
        for i in range(0, paths.shape[0], 25):
            fig.add_scatter(y=paths.iloc[i].values, mode="lines",
                            line=dict(width=0.5, color="rgba(70,130,180,0.25)"),
                            showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Median terminal", f"${final.median():,.0f}")
        c2.metric("5th pct", f"${final.quantile(0.05):,.0f}")
        c3.metric("95th pct", f"${final.quantile(0.95):,.0f}")
        c4.metric("Prob. profit", f"{(final > 100).mean():.0%}")
        st.caption("Start value $100, 1-year horizon, 500 bootstrapped daily-return paths.")


if __name__ == "__main__":
    main()
