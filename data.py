"""Data layer: fetch market data via yfinance and persist to SQLite.

The SQL layer (SQLite) is the single source of truth for the dashboard. Every
metric/optimization is computed from rows queried with SQL, mirroring how an
investments team would separate storage from analytics.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

DB_PATH = os.path.join(os.path.dirname(__file__), "meridian.db")

# Investment universe: blue-chip US equities + iShares ETFs (BlackRock tie-in).
UNIVERSE = {
    "AAPL": ("Apple", "Technology", "Equity"),
    "MSFT": ("Microsoft", "Technology", "Equity"),
    "NVDA": ("NVIDIA", "Technology", "Equity"),
    "AMZN": ("Amazon", "Consumer", "Equity"),
    "GOOGL": ("Alphabet", "Communication", "Equity"),
    "JPM": ("JPMorgan", "Financials", "Equity"),
    "XOM": ("Exxon Mobil", "Energy", "Equity"),
    "JNJ": ("Johnson & Johnson", "Healthcare", "Equity"),
    "IVV": ("iShares Core S&P 500", "Broad", "ETF"),
    "IUSB": ("iShares Core Total USD Bond", "Fixed Income", "ETF"),
    "GLD": ("SPDR Gold Shares", "Commodity", "ETF"),
}


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_schema() -> None:
    """Create tables if they don't exist."""
    con = _connect()
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS assets (
            ticker      TEXT PRIMARY KEY,
            name        TEXT,
            sector      TEXT,
            asset_class TEXT
        );
        CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT,
            date   TEXT,
            close  REAL,
            PRIMARY KEY (ticker, date)
        );
        CREATE TABLE IF NOT EXISTS meta (
            k TEXT PRIMARY KEY,
            v TEXT
        );
        """
    )
    con.commit()
    con.close()


def seed_assets() -> None:
    con = _connect()
    con.executemany(
        "INSERT OR REPLACE INTO assets(ticker, name, sector, asset_class) VALUES (?,?,?,?)",
        [(t, n, s, a) for t, (n, s, a) in UNIVERSE.items()],
    )
    con.commit()
    con.close()


def fetch_prices(period: str = "3y") -> pd.DataFrame:
    """Download adjusted-close prices and persist to SQLite. Returns wide DataFrame.

    Falls back to deterministic synthetic data if the network/API is unavailable so
    the dashboard always runs; the source is recorded in the meta table.
    """
    tickers = list(UNIVERSE.keys())
    try:
        raw = yf.download(tickers, period=period, interval="1d", auto_adjust=True,
                          progress=False, threads=False)
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
        close = close[tickers].dropna(how="any")
        if len(close) < 50:
            raise ValueError("insufficient rows returned")
        _persist(close)
        _set_meta("source", "live:yfinance")
        return close
    except Exception as exc:  # network/rate-limit -> reproducible synthetic fallback
        print(f"[data] live fetch failed ({exc}); using synthetic fallback")
        close = _synthetic(tickers, period)
        _persist(close)
        _set_meta("source", "synthetic")
        return close


def _persist(close: pd.DataFrame) -> None:
    con = _connect()
    rows = [
        (ticker, str(ts.date()), float(px))
        for ticker, series in close.items()
        for ts, px in series.items()
    ]
    con.executemany("INSERT OR REPLACE INTO prices(ticker, date, close) VALUES (?,?,?)", rows)
    con.commit()
    con.close()


def _synthetic(tickers: list[str], period: str) -> pd.DataFrame:
    days = {"1y": 252, "3y": 756, "5y": 1260}.get(period, 756)
    end = datetime.today()
    idx = pd.bdate_range(end=end, periods=days)
    rng = np.random.default_rng(42)
    out = {}
    for i, t in enumerate(tickers):
        mu = 0.0004 + 0.0002 * (i % 3)          # mild drift differences
        vol = 0.012 + 0.004 * (i % 4)
        rets = rng.normal(mu, vol, days)
        out[t] = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(out, index=idx)


def _set_meta(key: str, value: str) -> None:
    init_schema()
    con = _connect()
    con.execute("INSERT OR REPLACE INTO meta(k, v) VALUES (?,?)", (key, value))
    con.commit()
    con.close()


def meta(key: str) -> str | None:
    init_schema()
    con = _connect()
    row = con.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
    con.close()
    return row[0] if row else None


# ---- SQL query API used by analytics/dashboard ---------------------------------
def sql_prices(ticker: str) -> pd.DataFrame:
    init_schema()
    con = _connect()
    df = pd.read_sql_query(
        "SELECT date, close FROM prices WHERE ticker=? ORDER BY date",
        con, params=(ticker,),
    )
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def sql_wide_prices() -> pd.DataFrame:
    """Wide price matrix (date x ticker) via a raw SQL pivot."""
    init_schema()
    con = _connect()
    long = pd.read_sql_query("SELECT ticker, date, close FROM prices ORDER BY date", con)
    con.close()
    wide = long.pivot(index="date", columns="ticker", values="close").sort_index()
    wide.index = pd.to_datetime(wide.index)
    return wide


def sql_assets() -> pd.DataFrame:
    init_schema()
    con = _connect()
    df = pd.read_sql_query("SELECT * FROM assets", con)
    con.close()
    return df


if __name__ == "__main__":
    init_schema()
    seed_assets()
    px = fetch_prices()
    print(f"rows={len(px)} source={meta('source')}")
    print(sql_wide_prices().tail(3).round(2))
