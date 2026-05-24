"""
factors.py — Fama-French 5 Factor Model
========================================
Implements professional-grade factor attribution identical to buy-side risk
decomposition:

    R_p - R_f = α + β₁·(Mkt-RF) + β₂·SMB + β₃·HML + β₄·RMW + β₅·CMA + ε

Factor definitions (Kenneth French):
  Mkt-RF : Market excess return (systematic market risk)
  SMB    : Small Minus Big (size premium — small-cap tilt)
  HML    : High Minus Low (value premium — B/M ratio tilt)
  RMW    : Robust Minus Weak (profitability premium)
  CMA    : Conservative Minus Aggressive (investment premium)

Alpha (α): Risk-adjusted excess return NOT explained by factor exposures.
           This is what active managers are paid to generate.

Data source: Kenneth French Data Library (free, academic standard)
  URL: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
  Via: pandas_datareader (ff.FamaFrench5Factor())
"""

from __future__ import annotations

import io
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import yfinance as yf

warnings.filterwarnings("ignore")

TRADING_DAYS = 252
FF5_DATASET  = "F-F_Research_Data_5_Factors_2x3_daily"


# ────────────────────────────────────────────────────────────────────────────
# 1. FACTOR DATA — Kenneth French Library
# ────────────────────────────────────────────────────────────────────────────

def fetch_ff5_factors(
    start: str,
    end: str,
    frequency: str = "daily",
) -> pd.DataFrame:
    """
    Download Fama-French 5 Factors from Kenneth French's Data Library.

    Returns DataFrame with columns: [Mkt-RF, SMB, HML, RMW, CMA, RF]
    Values are in decimal form (already divided by 100).

    Parameters
    ----------
    start / end  : 'YYYY-MM-DD'
    frequency    : 'daily' (default) or 'monthly'
    """
    dataset = FF5_DATASET if frequency == "daily" else "F-F_Research_Data_5_Factors_2x3"
    try:
        ff = web.DataReader(dataset, "famafrench", start=start, end=end)[0]
        ff = ff / 100  # French library stores as percentages
        ff.index = pd.to_datetime(ff.index)
        ff.columns = [c.strip() for c in ff.columns]
        # Rename for consistency
        rename = {"Mkt-RF": "Mkt-RF", "SMB": "SMB", "HML": "HML",
                  "RMW": "RMW", "CMA": "CMA", "RF": "RF"}
        ff = ff.rename(columns=rename)
        return ff
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch FF5 factors. Check internet connection.\n"
            f"pandas_datareader error: {e}"
        )


# ────────────────────────────────────────────────────────────────────────────
# 2. OLS REGRESSION — Full Factor Model
# ────────────────────────────────────────────────────────────────────────────

def run_factor_regression(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    add_momentum: bool = False,
) -> dict:
    """
    Run OLS regression of portfolio excess returns on FF5 factors.

    Parameters
    ----------
    portfolio_returns : daily portfolio return series (decimal)
    factors           : DataFrame from fetch_ff5_factors() with RF column
    add_momentum      : if True, add UMD (Up-Minus-Down) momentum factor
                        via a separate yfinance proxy (AQR-style extension)

    Returns
    -------
    dict with:
        alpha_daily, alpha_annual, alpha_tstat, alpha_pval,
        betas (dict: factor → loading),
        tstats (dict: factor → t-stat),
        pvalues (dict: factor → p-value),
        r_squared, adj_r_squared, n_obs,
        residuals (Series), fitted (Series),
        model (statsmodels OLS result object — for further analysis)
    """
    # Align dates
    common = portfolio_returns.index.intersection(factors.index)
    if len(common) < 60:
        raise ValueError(
            f"Only {len(common)} overlapping observations. Need ≥60 days."
        )

    p  = portfolio_returns.loc[common]
    ff = factors.loc[common]

    # Excess portfolio return
    y = p - ff["RF"]
    y.name = "excess_return"

    # Factor matrix
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    X = ff[factor_cols].copy()

    # Optional: add momentum proxy (MOM)
    if add_momentum and "MOM" in ff.columns:
        factor_cols.append("MOM")
        X = ff[factor_cols]

    X = sm.add_constant(X)   # adds alpha intercept

    # OLS estimation
    model  = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    params = model.params
    tstats = model.tvalues
    pvals  = model.pvalues

    alpha_daily  = float(params["const"])
    alpha_annual = (1 + alpha_daily) ** TRADING_DAYS - 1

    betas = {f: float(params[f]) for f in factor_cols}

    return {
        "alpha_daily":    alpha_daily,
        "alpha_annual":   alpha_annual,
        "alpha_annual_pct": round(alpha_annual * 100, 3),
        "alpha_tstat":    float(tstats["const"]),
        "alpha_pval":     float(pvals["const"]),
        "betas":          betas,
        "tstats":         {f: float(tstats[f]) for f in factor_cols},
        "pvalues":        {f: float(pvals[f])  for f in factor_cols},
        "r_squared":      round(model.rsquared, 4),
        "adj_r_squared":  round(model.rsquared_adj, 4),
        "n_obs":          int(model.nobs),
        "residuals":      model.resid,
        "fitted":         model.fittedvalues,
        "model":          model,   # keep full object for custom analysis
    }


# ────────────────────────────────────────────────────────────────────────────
# 3. RETURN DECOMPOSITION
# ────────────────────────────────────────────────────────────────────────────

def decompose_returns(
    regression_result: dict,
    factors: pd.DataFrame,
    portfolio_returns: pd.Series,
) -> pd.DataFrame:
    """
    Decompose total portfolio return into factor contributions + alpha.

    Returns a DataFrame with one column per component:
        alpha, Mkt-RF, SMB, HML, RMW, CMA, (residual)
    Each column shows the daily return contribution of that component.

    Cumulative sum gives "waterfall" attribution used in CFA / risk reports.
    """
    common = portfolio_returns.index.intersection(factors.index)
    ff     = factors.loc[common]
    rf     = ff["RF"]
    factor_cols = list(regression_result["betas"].keys())
    betas       = regression_result["betas"]
    alpha_d     = regression_result["alpha_daily"]

    decomp = pd.DataFrame(index=common)
    decomp["rf"]     = rf.values
    decomp["alpha"]  = alpha_d

    for f in factor_cols:
        decomp[f] = betas[f] * ff[f].values

    # Residual (idiosyncratic)
    residuals = regression_result["residuals"]
    decomp["residual"] = residuals.reindex(common).values

    return decomp


def attribution_summary(decomp: pd.DataFrame) -> pd.DataFrame:
    """
    Annualised contribution of each factor to total portfolio return.

    Returns DataFrame with columns: component, daily_contrib, annual_contrib_pct
    """
    rows = []
    for col in decomp.columns:
        daily = decomp[col].mean()
        annual = (1 + daily) ** TRADING_DAYS - 1
        rows.append({
            "component":          col,
            "daily_contrib_bps":  round(daily * 10000, 2),
            "annual_contrib_pct": round(annual * 100, 3),
        })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# 4. INDIVIDUAL POSITION FACTOR SCORES (optional enrichment)
# ────────────────────────────────────────────────────────────────────────────

def run_position_betas(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run FF3 regression on each individual position to show which factor each
    stock is loading on. Useful for identifying where your factor risks live.

    Returns DataFrame: ticker × [alpha_annual, beta_Mkt, beta_SMB, beta_HML, R²]
    """
    rets   = prices.pct_change().dropna()
    common = rets.index.intersection(factors.index)
    rets   = rets.loc[common]
    ff     = factors.loc[common]
    rf     = ff["RF"]

    factor_cols = ["Mkt-RF", "SMB", "HML"]   # FF3 for individual stocks
    X = sm.add_constant(ff[factor_cols])

    results = []
    for ticker in rets.columns:
        y = rets[ticker] - rf
        try:
            mdl = OLS(y, X).fit()
            p   = mdl.params
            results.append({
                "ticker":       ticker,
                "alpha_annual_pct": round(((1 + p["const"]) ** TRADING_DAYS - 1) * 100, 2),
                "beta_Mkt":     round(float(p["Mkt-RF"]), 3),
                "beta_SMB":     round(float(p["SMB"]), 3),
                "beta_HML":     round(float(p["HML"]), 3),
                "r_squared":    round(mdl.rsquared, 3),
            })
        except Exception:
            continue

    return pd.DataFrame(results).sort_values("alpha_annual_pct", ascending=False)


# ────────────────────────────────────────────────────────────────────────────
# 5. ROLLING FACTOR EXPOSURES
# ────────────────────────────────────────────────────────────────────────────

def rolling_factor_betas(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    window: int = 126,
) -> pd.DataFrame:
    """
    Rolling OLS betas over a trailing window (default: 126 days = ~6 months).
    Shows whether factor exposures are stable or drifting over time.

    Returns DataFrame: DatetimeIndex × [Mkt-RF, SMB, HML, RMW, CMA]
    """
    common = portfolio_returns.index.intersection(factors.index)
    p  = portfolio_returns.loc[common]
    ff = factors.loc[common]
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    results = {f: [] for f in factor_cols}
    results["alpha"] = []
    idx = []

    for i in range(window, len(p) + 1):
        p_w   = p.iloc[i-window:i]
        ff_w  = ff.iloc[i-window:i]
        y = p_w.values - ff_w["RF"].values
        X = sm.add_constant(ff_w[factor_cols].values)
        try:
            mdl = OLS(y, X).fit()
            params = mdl.params
            results["alpha"].append(params[0])
            for j, f in enumerate(factor_cols):
                results[f].append(params[j + 1])
        except Exception:
            results["alpha"].append(np.nan)
            for f in factor_cols:
                results[f].append(np.nan)
        idx.append(p.index[i - 1])

    df = pd.DataFrame(results, index=idx)
    df["alpha_annual_pct"] = ((1 + df["alpha"]) ** TRADING_DAYS - 1) * 100
    return df


# ────────────────────────────────────────────────────────────────────────────
# 6. FULL FACTOR REPORT (convenience wrapper)
# ────────────────────────────────────────────────────────────────────────────

def full_factor_report(
    portfolio_returns: pd.Series,
    prices: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """
    End-to-end Fama-French 5 Factor report.

    Steps:
      1. Fetch FF5 factors from French library
      2. Run OLS regression
      3. Decompose returns into factor contributions
      4. Compute individual position betas (FF3)
      5. Compute rolling 6-month betas (regime detection)

    Returns
    -------
    dict with: factors, regression, decomposition, attribution,
               position_betas, rolling_betas
    """
    if start is None:
        start = str(portfolio_returns.index.min().date())
    if end is None:
        end   = str(portfolio_returns.index.max().date())

    print("  [1/4] Fetching FF5 factors from Kenneth French Library …")
    factors = fetch_ff5_factors(start, end)

    print("  [2/4] Running OLS regression …")
    reg = run_factor_regression(portfolio_returns, factors)

    print("  [3/4] Decomposing returns …")
    decomp    = decompose_returns(reg, factors, portfolio_returns)
    attrib    = attribution_summary(decomp)

    print("  [4/4] Computing position betas & rolling exposures …")
    pos_betas = run_position_betas(prices, factors)
    roll_betas = rolling_factor_betas(portfolio_returns, factors)

    return {
        "factors":        factors,
        "regression":     reg,
        "decomposition":  decomp,
        "attribution":    attrib,
        "position_betas": pos_betas,
        "rolling_betas":  roll_betas,
    }
