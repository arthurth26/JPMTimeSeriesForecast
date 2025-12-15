# Balance Sheet Monte Carlo Forecaster

A lightweight, live-data-driven Python tool for projecting multi-year balance sheets using Monte Carlo simulation on Pretax Income growth, with automatic data fetching from Yahoo Finance.

No manual CSV downloads required — just call the main function with a ticker!

## Features

- **Live data fetching** via `yfinance` — always uses the latest annual income statement and balance sheet.
- **Monte Carlo forecasting** driven by historical Pretax Income growth.
- **Smart weighting options**:
  - Unweighted, harmonic, or **heavy recent** (exponential decay) to capture momentum in growth stocks like AMD or NVDA.
- **Direct equity scaling** ensures perfect balance sheet integrity (Assets = Liabilities + Equity).
- **Configurable scenarios**: base (50th percentile), conservative (lower), aggressive (higher).
- **No clipping by default** — preserves full upside for high-growth companies.
- Optional raw data export to `out/` folder.
- Clean, aligned console summary table including the latest actual year (Year 0).

## Installation

This project uses **uv** for fast, reliable, and reproducible dependency management.

### 1. Install uv (if you don't have it yet)

```bash
# Recommended: official installer
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv

# Clone the repo first (if you haven't already)
git clone https://github.com/arthurth26/JPMTimeSeriesForecast
cd JPMTimeSeriesForecast

# Install dependencies exactly as locked
uv sync

# Simple base case with momentum bias (great for growth stocks)
forecast_balance_sheet_multi_year('AMD', years=3, percentile=50.0, heavy_recent=True)

# Conservative scenario
forecast_balance_sheet_multi_year('INTC', years=2, percentile=25.0, heavy_recent=False)

# Aggressive growth + save raw data
forecast_balance_sheet_multi_year(
    'NVDA',
    years=3,
    percentile=75.0,
    heavy_recent=True,
    download_data=True,
    clip_outlier_growths=0.0  # no clipping
)

===============================================================================================================
MULTI-YEAR BALANCE SHEET FORECAST - AMD (incl. Year 0 Actual)
Scenario: 50.0th percentile | Mean growth: +38.42% (±42.17%)
Weighting: Heavy recent emphasis (exponential decay)
===============================================================================================================
Period           Year     Growth       Total Assets         Liabilities              Equity
---------------------------------------------------------------------------------------------------------------
Y0 (2025) Actual 2025      N/A       68,421,000,000      28,834,000,000      39,587,000,000
Y1 (2026) Proj.  2026     +38.4%    94,712,000,000      39,954,000,000      54,758,000,000
Y2 (2027) Proj.  2027     +41.2%   133,756,000,000      56,414,000,000      77,342,000,000
Y3 (2028) Proj.  2028     +36.9%   183,124,000,000      77,240,000,000     105,884,000,000
===============================================================================================================