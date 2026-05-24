"""
risk.py — Risk Analytics Engine
================================
Computes professional-grade risk metrics used in buy-side attribution:
  - Parametric VaR / CVaR (Historical + Gaussian)
  - Sharpe Ratio & Sortino Ratio
  - Portfolio Beta (vs benchmark)
  - Maximum Drawdown & Drawdown Duration
  - Rolling Volatility
  - Correlation Matrix

All functions accept pd.Series of daily returns (decimal form, e.g. 0.012 = +1.2%).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Union


# ── Constants ───────────────────────────────────────────────────────────────
TRADING_DAYS = 252
RISK_FREE_ANNUAL = 0.045    # ~4.5% (Fed Funds / SOFR proxy, update periodically)
RISK_FREE_DAILY  = RISK_FREE_ANNUAL / TRADING_DAYS


# ────────────────────────────────────────────────────────────────────────────
# 1. VaR & CVaR
# ────────────────────────────────────────────────────────────────────────────

def historical_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    """
    Historical (non-parametric) Value at Risk.

    Parameters
    ----------
    returns    : daily return series
    confidence : e.g. 0.95 → 95% VaR
    horizon    : holding period in days (scales by √horizon)

    Returns
    -------
    VaR as a positive decimal (loss), e.g. 0.023 means −2.3%
    """
    q = np.percentile(returns.dropna(), (1 - confidence) * 100)
    return abs(q) * np.sqrt(horizon)


def historical_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    """
    Historical CVaR (Expected Shortfall) — average loss beyond VaR.
    More conservative and coherent than VaR; preferred by risk managers.
    """
    r = returns.dropna()
    var_threshold = np.percentile(r, (1 - confidence) * 100)
    tail = r[r <= var_threshold]
    cvar = abs(tail.mean()) * np.sqrt(horizon)
    return cvar


def parametric_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    """
    Parametric (Gaussian) VaR. Assumes normal distribution.
    Use alongside historical VaR — if they diverge, the distribution has fat tails.
    """
    mu    = returns.mean()
    sigma = returns.std()
    z     = stats.norm.ppf(1 - confidence)
    var   = -(mu + z * sigma) * np.sqrt(horizon)
    return max(var, 0.0)


def var_summary(
    returns: pd.Series,
    portfolio_value_krw: float,
    confidence: float = 0.95,
) -> dict:
    """
    Full VaR/CVaR report at both 1-day and 10-day horizons.
    Converts to KRW monetary amounts for intuition.
    """
    h_var_1d   = historical_var(returns, confidence, horizon=1)
    h_cvar_1d  = historical_cvar(returns, confidence, horizon=1)
    h_var_10d  = historical_var(returns, confidence, horizon=10)
    h_cvar_10d = historical_cvar(returns, confidence, horizon=10)
    p_var_1d   = parametric_var(returns, confidence, horizon=1)

    return {
        "confidence":         confidence,
        # 1-day
        "hist_var_1d_pct":    round(h_var_1d   * 100, 3),
        "hist_cvar_1d_pct":   round(h_cvar_1d  * 100, 3),
        "param_var_1d_pct":   round(p_var_1d   * 100, 3),
        "hist_var_1d_krw":    round(h_var_1d   * portfolio_value_krw),
        "hist_cvar_1d_krw":   round(h_cvar_1d  * portfolio_value_krw),
        # 10-day (Basel / regulatory standard)
        "hist_var_10d_pct":   round(h_var_10d  * 100, 3),
        "hist_cvar_10d_pct":  round(h_cvar_10d * 100, 3),
        "hist_var_10d_krw":   round(h_var_10d  * portfolio_value_krw),
    }


# ────────────────────────────────────────────────────────────────────────────
# 2. SHARPE & SORTINO
# ────────────────────────────────────────────────────────────────────────────

def sharpe_ratio(
    returns: pd.Series,
    risk_free_daily: float = RISK_FREE_DAILY,
    annualise: bool = True,
) -> float:
    """
    Sharpe Ratio = (E[R] - Rf) / σ(R)
    Annualised by default (multiplied by √252).
    """
    excess = returns - risk_free_daily
    if excess.std() == 0:
        return np.nan
    ratio = excess.mean() / excess.std()
    return ratio * np.sqrt(TRADING_DAYS) if annualise else ratio


def sortino_ratio(
    returns: pd.Series,
    risk_free_daily: float = RISK_FREE_DAILY,
    annualise: bool = True,
) -> float:
    """
    Sortino Ratio = (E[R] - Rf) / σ_downside(R)
    Only penalises downside volatility — more appropriate for asymmetric return profiles.
    """
    excess     = returns - risk_free_daily
    downside   = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0:
        return np.nan
    ratio = excess.mean() / downside_std
    return ratio * np.sqrt(TRADING_DAYS) if annualise else ratio


def calmar_ratio(
    returns: pd.Series,
    annualise: bool = True,
) -> float:
    """
    Calmar Ratio = Annualised Return / Maximum Drawdown.
    Used in hedge fund contexts; high Calmar → good risk-adjusted momentum.
    """
    ann_return = (1 + returns.mean()) ** TRADING_DAYS - 1
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    return ann_return / mdd


# ────────────────────────────────────────────────────────────────────────────
# 3. BETA & TRACKING ERROR
# ────────────────────────────────────────────────────────────────────────────

def portfolio_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Portfolio Beta = Cov(Rp, Rm) / Var(Rm)
    Beta > 1 → amplified market moves; β < 1 → defensive.
    """
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    p = portfolio_returns.loc[common].dropna()
    b = benchmark_returns.loc[common].dropna()
    common2 = p.index.intersection(b.index)
    p, b = p.loc[common2], b.loc[common2]
    if len(p) < 10:
        return np.nan
    cov    = np.cov(p, b)[0, 1]
    var_b  = np.var(b)
    return cov / var_b if var_b != 0 else np.nan


def tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    annualise: bool = True,
) -> float:
    """
    Tracking Error = std(Rp - Rb).
    Low TE → portfolio moves like benchmark (index-like).
    High TE → active management / concentrated bets.
    """
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    active = portfolio_returns.loc[common] - benchmark_returns.loc[common]
    te = active.std()
    return te * np.sqrt(TRADING_DAYS) if annualise else te


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Information Ratio = Mean(Active Return) / Tracking Error.
    IR > 0.5 is considered good; > 1.0 is exceptional.
    """
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    active = portfolio_returns.loc[common] - benchmark_returns.loc[common]
    te = tracking_error(portfolio_returns, benchmark_returns, annualise=True)
    if te == 0:
        return np.nan
    ann_active = active.mean() * TRADING_DAYS
    return ann_active / te


# ────────────────────────────────────────────────────────────────────────────
# 4. DRAWDOWN
# ────────────────────────────────────────────────────────────────────────────

def drawdown_series(returns: pd.Series) -> pd.Series:
    """Returns the drawdown series (fraction from running peak)."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd


def max_drawdown(returns: pd.Series) -> float:
    """Maximum Drawdown as a positive decimal (e.g. 0.15 = −15%)."""
    dd = drawdown_series(returns)
    return abs(dd.min())


def drawdown_duration(returns: pd.Series) -> dict:
    """
    Returns:
        max_duration_days : longest time spent below peak
        current_duration_days : days since last all-time-high
        in_drawdown : whether portfolio is currently below peak
    """
    cum  = (1 + returns).cumprod()
    peak = cum.cummax()
    underwater = (cum < peak)

    # Compute run-lengths of underwater periods
    changes = underwater.astype(int).diff().fillna(0)
    starts  = changes[changes == 1].index
    ends    = changes[changes == -1].index

    durations = []
    for s in starts:
        later_ends = ends[ends > s]
        e = later_ends[0] if len(later_ends) else returns.index[-1]
        durations.append((e - s).days)

    max_dur = max(durations) if durations else 0
    cur_dur = (returns.index[-1] - peak.idxmax()).days if underwater.iloc[-1] else 0

    return {
        "max_duration_days":     max_dur,
        "current_duration_days": cur_dur,
        "in_drawdown":           bool(underwater.iloc[-1]),
    }


# ────────────────────────────────────────────────────────────────────────────
# 5. ROLLING METRICS
# ────────────────────────────────────────────────────────────────────────────

def rolling_volatility(
    returns: pd.Series,
    window: int = 21,
    annualise: bool = True,
) -> pd.Series:
    """Rolling annualised volatility (21-day default = ~1 month)."""
    rv = returns.rolling(window).std()
    return rv * np.sqrt(TRADING_DAYS) if annualise else rv


def rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    risk_free_daily: float = RISK_FREE_DAILY,
) -> pd.Series:
    """Rolling Sharpe ratio (63-day default = ~3 months)."""
    excess = returns - risk_free_daily
    roll_mean = excess.rolling(window).mean()
    roll_std  = excess.rolling(window).std()
    return (roll_mean / roll_std) * np.sqrt(TRADING_DAYS)


def rolling_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Rolling Beta over a trailing window."""
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    p = portfolio_returns.loc[common]
    b = benchmark_returns.loc[common]

    def beta_w(p_w, b_w):
        if len(p_w) < 5:
            return np.nan
        cov   = np.cov(p_w, b_w)[0, 1]
        var_b = np.var(b_w)
        return cov / var_b if var_b != 0 else np.nan

    betas = []
    for i in range(window, len(p) + 1):
        betas.append(beta_w(p.iloc[i-window:i].values, b.iloc[i-window:i].values))

    return pd.Series(betas, index=p.index[window-1:], name="rolling_beta")


# ────────────────────────────────────────────────────────────────────────────
# 6. CORRELATION & FULL RISK REPORT
# ────────────────────────────────────────────────────────────────────────────

def correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of daily returns. Drop tickers with insufficient data."""
    rets = prices.pct_change().dropna(how="all")
    return rets.corr()


def full_risk_report(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    portfolio_value_krw: float,
    confidence: float = 0.95,
) -> dict:
    """
    One-stop risk report — everything a risk manager or interviewer would ask for.

    Returns
    -------
    dict with all key metrics
    """
    ann_vol = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
    ann_ret = (1 + portfolio_returns.mean()) ** TRADING_DAYS - 1
    mdd     = max_drawdown(portfolio_returns)

    return {
        # Return / risk
        "annualised_return_pct":  round(ann_ret * 100, 2),
        "annualised_vol_pct":     round(ann_vol * 100, 2),
        "sharpe_ratio":           round(sharpe_ratio(portfolio_returns), 3),
        "sortino_ratio":          round(sortino_ratio(portfolio_returns), 3),
        "calmar_ratio":           round(calmar_ratio(portfolio_returns), 3),

        # Drawdown
        "max_drawdown_pct":       round(mdd * 100, 2),
        "drawdown_duration":      drawdown_duration(portfolio_returns),

        # Market sensitivity
        "beta":                   round(portfolio_beta(portfolio_returns, benchmark_returns), 3),
        "tracking_error_pct":     round(tracking_error(portfolio_returns, benchmark_returns) * 100, 2),
        "information_ratio":      round(information_ratio(portfolio_returns, benchmark_returns), 3),

        # VaR / CVaR
        "var":                    var_summary(portfolio_returns, portfolio_value_krw, confidence),
    }
