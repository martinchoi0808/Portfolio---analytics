# Portfolio Analytics

Personal quantitative research platform for portfolio analysis and factor modelling.

## Project Structure

```
portfolio-analytics/
├── data/
│   └── holdings.csv          # Portfolio positions (update here after trades)
├── src/
│   ├── data_fetcher.py       # Price data: yfinance + pykrx + Google Sheets
│   ├── portfolio.py          # Returns, weights, P&L engine
│   ├── risk.py               # VaR, Sharpe, Beta, Drawdown
│   └── factors.py            # Fama-French 5 Factor Model
├── research/
│   ├── 01_portfolio_overview.ipynb
│   ├── 02_risk_analytics.ipynb
│   ├── 03_factor_model.ipynb
│   ├── 04_fixed_income.ipynb
│   └── experiments/
├── .github/workflows/
│   └── daily_data.yml        # Auto price update via GitHub Actions
├── requirements.txt
└── .env.example
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/portfolio-analytics.git
cd portfolio-analytics
pip install -r requirements.txt
cp .env.example .env
# Fill in .env with your Google Sheets credentials
```

## Updating Holdings

**Option A (Recommended): Google Sheets**
- Open Google Sheets link → change `shares` column → done
- Python automatically reads latest data

**Option B: Direct CSV**
- Edit `data/holdings.csv` directly on GitHub web UI
- Change `shares` column only

## Research Notebooks

| Notebook | Contents |
|---|---|
| 01_portfolio_overview | P&L, weights, sector breakdown, rebalancing alerts |
| 02_risk_analytics | VaR, CVaR, Sharpe, Sortino, correlation heatmap |
| 03_factor_model | Fama-French 5 Factor, alpha decomposition |
| 04_fixed_income | Duration, yield curve, spread analysis |

## Factor Model

Uses Kenneth French Data Library (free) for factor returns.
Runs OLS regression:

```
R_p - R_f = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + β₄(RMW) + β₅(CMA) + ε
```

Output: alpha, factor loadings, R², t-statistics — identical to buy-side attribution analysis.
