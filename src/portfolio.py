"""
portfolio.py — Returns, Weights, P&L Engine
===========================================
Builds on top of data_fetcher.build_portfolio_snapshot() to compute:
  - Portfolio-level returns (daily, cumulative)
  - Position weights (current market value)
  - Core / Satellite breakdown
  - Sector / sub_category breakdown
  - Unrealised P&L in KRW (and %)
  - Rebalancing alerts (when a position drifts beyond threshold)
  - Benchmark comparison (SPY as default)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Default thresholds ──────────────────────────────────────────────────────
SATELLITE_MAX = 0.25          # hard ceiling: satellite never exceeds 25 %
REBALANCE_THRESHOLD = 0.05    # alert when a position drifts ±5 pp from target


# ────────────────────────────────────────────────────────────────────────────
# 1. WEIGHTS & P&L
# ────────────────────────────────────────────────────────────────────────────

def compute_weights(snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Add / refresh the 'weight' column (fraction of total portfolio value in KRW).
    Works on the DataFrame returned by build_portfolio_snapshot().
    """
    total_krw = snapshot["value_krw"].sum()
    snapshot = snapshot.copy()
    snapshot["weight"] = snapshot["value_krw"] / total_krw
    return snapshot


def portfolio_summary(snapshot: pd.DataFrame) -> dict:
    """
    High-level portfolio metrics from the latest snapshot.

    Returns
    -------
    dict with keys:
        total_value_krw, total_value_usd, total_cost_krw,
        total_pnl_krw, total_pnl_pct,
        num_positions, usdkrw,
        core_value_krw, satellite_value_krw,
        core_weight, satellite_weight
    """
    snap = compute_weights(snapshot)
    usdkrw = snap["current_price_krw"].iloc[0] / snap["current_price_usd"].iloc[0] \
        if snap["currency"].iloc[0] == "USD" else None

    # Try to infer usdkrw from any USD position
    usd_rows = snap[snap["currency"] == "USD"]
    if not usd_rows.empty and usdkrw is None:
        r = usd_rows.iloc[0]
        usdkrw = r["current_price_krw"] / r["current_price_usd"] if r["current_price_usd"] else None

    total_krw   = snap["value_krw"].sum()
    total_usd   = snap["value_usd"].sum()
    total_cost  = snap["cost_krw"].sum()
    total_pnl   = snap["pnl_krw"].sum()
    total_pnl_pct = total_pnl / total_cost * 100 if total_cost else 0.0

    core_mask = snap["category"] == "core"
    sat_mask  = snap["category"] == "satellite"

    return {
        "total_value_krw":     total_krw,
        "total_value_usd":     total_usd,
        "total_cost_krw":      total_cost,
        "total_pnl_krw":       total_pnl,
        "total_pnl_pct":       total_pnl_pct,
        "num_positions":       len(snap),
        "usdkrw":              usdkrw,
        "core_value_krw":      snap.loc[core_mask, "value_krw"].sum(),
        "satellite_value_krw": snap.loc[sat_mask,  "value_krw"].sum(),
        "core_weight":         snap.loc[core_mask, "weight"].sum(),
        "satellite_weight":    snap.loc[sat_mask,  "weight"].sum(),
    }


def sector_breakdown(snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame grouped by sub_category with:
        value_krw, weight, pnl_krw, pnl_pct (weighted avg), num_positions
    """
    snap = compute_weights(snapshot)
    grouped = snap.groupby("sub_category").agg(
        value_krw   = ("value_krw",  "sum"),
        cost_krw    = ("cost_krw",   "sum"),
        pnl_krw     = ("pnl_krw",    "sum"),
        num_positions = ("ticker",   "count"),
    ).reset_index()
    total_krw = snap["value_krw"].sum()
    grouped["weight"]  = grouped["value_krw"] / total_krw
    grouped["pnl_pct"] = grouped["pnl_krw"] / grouped["cost_krw"] * 100
    return grouped.sort_values("weight", ascending=False).reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────
# 2. DAILY RETURNS  (requires historical prices from data_fetcher)
# ────────────────────────────────────────────────────────────────────────────

def compute_portfolio_returns(
    prices: pd.DataFrame,
    snapshot: pd.DataFrame,
    usdkrw_series: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute daily portfolio returns using current weights as a fixed (buy-and-hold) approximation.

    Parameters
    ----------
    prices        : DataFrame of daily close prices (columns = tickers)
                    from data_fetcher.get_historical_prices()
    snapshot      : latest snapshot with weights
    usdkrx_series : optional daily USD/KRW series — if provided, USD positions
                    are converted to KRW before weighting. If None, FX effect
                    is ignored (returns in USD terms for mixed portfolios).

    Returns
    -------
    pd.Series of daily portfolio returns (decimal), DatetimeIndex
    """
    snap = compute_weights(snapshot)
    weights = snap.set_index("ticker")["weight"]

    # Align: only tickers present in both prices and snapshot
    common = prices.columns.intersection(weights.index)
    if len(common) == 0:
        raise ValueError("No overlapping tickers between prices and snapshot.")

    price_sub  = prices[common].copy()
    weight_sub = weights[common]

    # Normalise weights to sum to 1 over available tickers
    weight_sub = weight_sub / weight_sub.sum()

    # Daily log returns for each position
    daily_rets = price_sub.pct_change().dropna()

    # Weighted sum → portfolio daily return
    port_ret = daily_rets.mul(weight_sub, axis=1).sum(axis=1)
    port_ret.name = "portfolio"
    return port_ret


def compute_cumulative_returns(daily_returns: pd.Series) -> pd.Series:
    """Convert daily returns to cumulative (growth of $1)."""
    return (1 + daily_returns).cumprod()


def get_benchmark_returns(
    ticker: str = "SPY",
    period: str = "2y",
    interval: str = "1d",
) -> pd.Series:
    """Fetch benchmark daily returns (default: SPY)."""
    data = yf.download(ticker, period=period, interval=interval,
                       auto_adjust=True, progress=False)
    if data.empty:
        return pd.Series(name=ticker)
    close = data["Close"].squeeze()
    ret   = close.pct_change().dropna()
    ret.name = ticker
    return ret


# ────────────────────────────────────────────────────────────────────────────
# 3. REBALANCING ALERTS
# ────────────────────────────────────────────────────────────────────────────

def rebalancing_alerts(
    snapshot: pd.DataFrame,
    target_weights: Optional[dict] = None,
    threshold: float = REBALANCE_THRESHOLD,
) -> pd.DataFrame:
    """
    Identify positions that have drifted beyond `threshold` from their targets.

    Parameters
    ----------
    snapshot       : latest snapshot (weights will be recomputed)
    target_weights : dict of {ticker: target_weight} — if None, each position
                     is assumed to target equal weight within its category bucket
    threshold      : drift tolerance (default 5 pp)

    Returns
    -------
    DataFrame with columns: ticker, name, current_weight, target_weight, drift, action
    """
    snap = compute_weights(snapshot)

    if target_weights is None:
        # Default: equal weight within core and satellite buckets
        n_core = (snap["category"] == "core").sum()
        n_sat  = (snap["category"] == "satellite").sum()
        core_share = SATELLITE_MAX if n_sat > 0 else 1.0      # satellite ≤ 25 %
        target_weights = {}
        for _, row in snap.iterrows():
            if row["category"] == "core":
                target_weights[row["ticker"]] = (1 - core_share) / n_core if n_core else 0
            else:
                target_weights[row["ticker"]] = core_share / n_sat if n_sat else 0

    rows = []
    for _, pos in snap.iterrows():
        t = pos["ticker"]
        current = pos["weight"]
        target  = target_weights.get(t, current)   # no target → no alert
        drift   = current - target
        if abs(drift) >= threshold:
            action = "TRIM ↓" if drift > 0 else "ADD ↑"
            rows.append({
                "ticker":         t,
                "name":           pos["name"],
                "current_weight": round(current * 100, 2),
                "target_weight":  round(target  * 100, 2),
                "drift_pp":       round(drift   * 100, 2),
                "action":         action,
            })

    if not rows:
        return pd.DataFrame(columns=["ticker","name","current_weight",
                                     "target_weight","drift_pp","action"])
    return pd.DataFrame(rows).sort_values("drift_pp", key=abs, ascending=False)


def satellite_guard(snapshot: pd.DataFrame) -> dict:
    """
    Enforce the 25 % satellite hard ceiling.

    Returns
    -------
    dict with:
        satellite_weight (float),
        breach (bool),
        excess_krw (float) — how much KRW to trim from satellite to comply
    """
    snap = compute_weights(snapshot)
    sat_w = snap.loc[snap["category"] == "satellite", "weight"].sum()
    total_krw = snap["value_krw"].sum()
    excess = max(0.0, sat_w - SATELLITE_MAX) * total_krw
    return {
        "satellite_weight": sat_w,
        "breach":           sat_w > SATELLITE_MAX,
        "excess_krw":       excess,
    }


# ────────────────────────────────────────────────────────────────────────────
# 4. FULL PERFORMANCE REPORT  (convenience wrapper)
# ────────────────────────────────────────────────────────────────────────────

def performance_report(
    snapshot: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_ticker: str = "SPY",
    period: str = "2y",
) -> dict:
    """
    One-stop performance report combining snapshot + historical returns.

    Returns
    -------
    dict with:
        summary, sector_breakdown, daily_returns (Series),
        cumulative_returns (Series), benchmark_returns (Series),
        rebalancing_alerts (DataFrame), satellite_guard (dict)
    """
    daily   = compute_portfolio_returns(prices, snapshot)
    cumul   = compute_cumulative_returns(daily)
    bench   = get_benchmark_returns(benchmark_ticker, period=period)

    # Align benchmark to same date range
    common_idx = daily.index.intersection(bench.index)
    bench = bench.loc[common_idx]

    return {
        "summary":             portfolio_summary(snapshot),
        "sector_breakdown":    sector_breakdown(snapshot),
        "daily_returns":       daily,
        "cumulative_returns":  cumul,
        "benchmark_returns":   bench,
        "rebalancing_alerts":  rebalancing_alerts(snapshot),
        "satellite_guard":     satellite_guard(snapshot),
    }
