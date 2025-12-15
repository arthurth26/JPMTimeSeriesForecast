# Balance Sheet Monte Carlo Forecaster

A lightweight, live-data-driven Python tool for projecting multi-year balance sheets using Monte Carlo simulation on Pretax Income growth, with automatic data fetching from Yahoo Finance.

No manual CSV downloads required — just call the main function with a ticker!

## Features

- **Live data fetching** via `yfinance` — always uses the latest annual income statement and balance sheet.
- **Monte Carlo forecasting** driven by historical Pretax Income growth.
- **Smart weighting options**:
  - Unweighted, harmonic, or **heavy recent** (exponential decay) to capture momentum in growth stocks like AMD or NVDA.
- **Direct equity scaling** ensures perfect balance sheet integrity (Assets = Liabilities + Equity).
- **Configurable outputs** Can output as a csv or txt file (keyword: output_mode) for further analysis
- **Configurable scenarios**: set the percentile (1-100) for aggresive or conservative forecasting.
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
```

# Example usage in file

```python
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
    clip_outlier_growths=0.0
)
```
# Command Line Usage

### Optional Flags

| Flag                  | Description                                                                                   | Default              |
|-----------------------|-----------------------------------------------------------------------------------------------|----------------------|
| `--years`             | Number of future years to forecast (must be 1, 2, or 3)                                        | `3`                  |
| `--percentile`        | Percentile of Monte Carlo paths to use (10 = conservative, 50 = base, 90 = aggressive)         | `50.0`               |
| `--heavy-recent`      | Enable strong exponential weighting (1, ½, ¼, ⅛…) — heavily favors recent performance (great for momentum stocks like NVDA or AMD) | Disabled             |
| `--download-data`     | Save raw income statement and balance sheet downloaded from Yahoo Finance to the `out/` folder | Disabled             |
| `--clip`              | **Enable** or set custom growth-rate clipping. Use `--clip 0.3` for ±30%, `--clip 0.1` for ±10%, etc. (overrides default ±20%) | 0       |
| `--seed`              | Random seed for the Monte Carlo simulations. Use the same seed for fully reproducible results across runs. | `0`                  |

```bash
# Default clipping (±20%)
python cli.py NVDA --heavy-recent

# Custom clipping (±30%)
python cli.py NVDA --heavy-recent --clip 0.3

# No clipping at all
python cli.py NVDA --heavy-recent --no-clip
```

# Sample Output
```bash
================================================================================
PROJECTED BALANCE SHEET SUMMARY (Year 2025) - NVDA
================================================================================
TOTAL ASSETS: MATCH (within ±1) (balanced via Equity)
   Projected (Liab + Equity): 129,919,756,118
   Reported scaled:           129,919,756,118
   Difference:                0

   --- Asset Components ---
   Current Assets:                    93,278,289,430
      ├── Cash & Equivalents:         50,302,709,311
      ├── Receivables:                26,851,006,486
      ├── Inventory:                  11,734,582,501
      └── Other Current:              4,389,991,132
   Non-Current Assets:                34,211,895,887
      ├── Net PPE:                     9,401,635,742
      ├── Goodwill & Intangibles:      6,979,049,811
      ├── Investments & Advances:      3,942,959,418
      └── Other Non-Current:           13,888,250,916

   --- Liability Components ---
   Current Liabilities:                      21,009,326,428
      ├── Payables & Accrued Expenses:       17,835,866,914
      └── Other Current Liabilities:         3,173,459,514
   Non-Current Liabilities:                  16,562,292,186
      ├── LT Debt & Capital Leases:          11,620,496,282
      ├── Deferred Credits:                  2,259,605,618
      └── Other Non-Current Liabilities:     2,682,190,286

TOTAL LIABILITIES:                         37,571,618,614
PROJECTED SHAREHOLDERS' EQUITY:            92,348,137,504
BALANCE CHECK (A vs L + E):                 0 (should be 0)
================================================================================

================================================================================
PROJECTED BALANCE SHEET SUMMARY (Year 2026) - NVDA
================================================================================
TOTAL ASSETS: MATCH (within ±1) (balanced via Equity)
   Projected (Liab + Equity): 146,661,465,005
   Reported scaled:           146,661,465,004
   Difference:                1

   --- Asset Components ---
   Current Assets:                    105,298,308,661
      ├── Cash & Equivalents:         56,784,812,885
      ├── Receivables:                30,311,078,667
      ├── Inventory:                  13,246,723,302
      └── Other Current:              4,955,693,807
   Non-Current Assets:                38,620,506,390
      ├── Net PPE:                     10,613,148,551
      ├── Goodwill & Intangibles:      7,878,383,551
      ├── Investments & Advances:      4,451,056,729
      └── Other Non-Current:           15,677,917,559

   --- Liability Components ---
   Current Liabilities:                      23,716,628,515
      ├── Payables & Accrued Expenses:       20,134,230,924
      └── Other Current Liabilities:         3,582,397,591
   Non-Current Liabilities:                  18,696,540,916
      ├── LT Debt & Capital Leases:          13,117,935,715
      ├── Deferred Credits:                  2,550,782,732
      └── Other Non-Current Liabilities:     3,027,822,469

TOTAL LIABILITIES:                         42,413,169,431
PROJECTED SHAREHOLDERS' EQUITY:            104,248,295,574
BALANCE CHECK (A vs L + E):                 0 (should be 0)
================================================================================

================================================================================
PROJECTED BALANCE SHEET SUMMARY (Year 2027) - NVDA
================================================================================
TOTAL ASSETS: MATCH (within ±1) (balanced via Equity)
   Projected (Liab + Equity): 168,610,149,702
   Reported scaled:           168,610,149,702
   Difference:                0

   --- Asset Components ---
   Current Assets:                    121,056,772,386
      ├── Cash & Equivalents:         65,282,968,510
      ├── Receivables:                34,847,296,197
      ├── Inventory:                  15,229,167,382
      └── Other Current:              5,697,340,297
   Non-Current Assets:                44,400,274,904
      ├── Net PPE:                     12,201,463,867
      ├── Goodwill & Intangibles:      9,057,426,434
      ├── Investments & Advances:      5,117,181,540
      └── Other Non-Current:           18,024,203,063

   --- Liability Components ---
   Current Liabilities:                      27,265,950,769
      ├── Payables & Accrued Expenses:       23,147,427,923
      └── Other Current Liabilities:         4,118,522,846
   Non-Current Liabilities:                  21,494,579,795
      ├── LT Debt & Capital Leases:          15,081,106,032
      ├── Deferred Credits:                  2,932,521,219
      └── Other Non-Current Liabilities:     3,480,952,544

TOTAL LIABILITIES:                         48,760,530,564
PROJECTED SHAREHOLDERS' EQUITY:            119,849,619,138
BALANCE CHECK (A vs L + E):                 0 (should be 0)
================================================================================

==============================================================================================================
MULTI-YEAR BALANCE SHEET FORECAST - NVDA (incl. Year 0 Actual)
Scenario: 50.0th percentile | Mean growth: +14.29% (±14.00%)
==============================================================================================================
Period           Year         Growth         Total Assets          Liabilities               Equity
--------------------------------------------------------------------------------------------------------------
Y0 (2025) Actual 2025            N/A      111,601,000,000       32,274,000,000       79,327,000,000
Y1 (2026) Proj.  2026         +16.4%      129,919,756,118       37,571,618,614       92,348,137,504
Y2 (2027) Proj.  2027         +31.4%      146,661,465,005       42,413,169,431      104,248,295,574
Y3 (2028) Proj.  2028         +51.1%      168,610,149,702       48,760,530,564      119,849,619,138
==============================================================================================================
Notes:
  • Year 0 = Latest reported actuals
  • Future years = Projected using selected Pretax Income growth percentile
  • Assets = Liabilities + Equity enforced | Negative equity capped at 0
==============================================================================================================
```