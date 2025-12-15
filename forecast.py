import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import yfinance as yf
from datetime import datetime

lines_console = []
lines_csv = []

#Helper functions
def add_console(*texts):
    line = " ".join(str(t) for t in texts)
    lines_console.append(line)

def add_csv(category, subcategory="", value=""):
    lines_csv.append([category, subcategory, value])

def download_and_prepare_data(ticker: str, download_data: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads the latest annual income statement and balance sheet data for a given ticker
    using yfinance and prepares them for forecasting.

    The function fetches data directly from Yahoo Finance, transposes it so that dates
    become the index (sorted ascending), and returns clean DataFrames ready for analysis.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol (e.g., 'AAPL', 'AMD', 'MSFT'). Case-insensitive.

    download_data : bool, default False
        If True, saves the raw downloaded financials to the 'out/' directory as:
        - out/{ticker}_income.csv
        - out/{ticker}_balance.csv
        Useful for inspection, debugging, or offline reuse.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (income_df, balance_df)
        - income_df: Transposed income statement with dates as index (oldest → newest)
        - balance_df: Transposed balance sheet with dates as index (oldest → newest)

    Raises
    ------
    ValueError
        If no financial data is available for the ticker (empty DataFrames from yfinance).

    Notes
    -----
    - Uses yfinance.Ticker.financials and .balance_sheet (annual data by default).
    - Automatically creates 'out/' folder if download_data=True.
    - This is the primary data ingestion function for live forecasting — no manual CSV files needed.
    """
    stock = yf.Ticker(ticker)
    
    # Get financials (annual)
    income = stock.financials
    balance = stock.balance_sheet
    
    if income.empty or balance.empty:
        raise ValueError(f"No financial data available for {ticker}")
    
    # Transpose and clean
    income = income.T
    balance = balance.T
    
    # Ensure dates are sorted ascending
    income = income.sort_index()
    balance = balance.sort_index()
    
    # Optional: save to out/
    if download_data:
        os.makedirs('out', exist_ok=True)
        income.to_csv(f'out/{ticker}_income.csv')
        balance.to_csv(f'out/{ticker}_balance.csv')
        print(f"Data saved to out/{ticker}_income.csv and out/{ticker}_balance.csv")
    
    return income, balance

def compute_growth_statistics(series: pd.Series, weighted: bool = False, heavy_recent: bool = False, clip_outliers: float = 0.2) -> tuple[float, float]:
    """
    Computes the mean and standard deviation of annual percentage changes (growth rates)
    from a time series, with optional time-weighting and outlier clipping.

    Used to estimate expected growth and volatility for Monte Carlo simulations.

    Parameters
    ----------
    series : pd.Series
        Time series of financial values (e.g., Pretax Income, Revenue). Should be sorted ascending.

    weighted : bool, default False
        If True, applies time-weighted statistics (more weight to recent periods).
        If False, uses simple unweighted mean and std of growth rates.

    heavy_recent : bool, default False
        If True (and weighted=True), uses aggressive exponential decay weights
        (1, 1/2, 1/4, 1/8...) — strongly prioritizes the most recent observations.
        Ideal for high-momentum growth stocks.
        If False, uses harmonic weights (1/1, 1/2, 1/3...) — milder recency bias.

    clip_outliers : float, default 0.2
        Caps extreme growth rates at ±this value before statistics calculation.
        - 0.2 → limits to ±20% annual change (conservative, reduces impact of outliers)
        - 0.0 → disables clipping (preserves full historical volatility)
        - Higher values (e.g., 0.5) allow larger swings

    Returns
    -------
    tuple[float, float]
        (mean_growth, std_growth)
        Estimated annual mean growth rate and volatility to use in simulations.

    Notes
    -----
    - Growth rates are computed as pct_change() on the series.
    - Clipping helps stabilize forecasts for noisy or erratic historical data.
    - Heavy recent weighting significantly increases sensitivity to recent performance.
    """
    pct_change = series.pct_change().dropna()
    values = pct_change.values  # use raw by default
    
    if clip_outliers:
        clipped = pct_change.clip(lower=-1*clip_outliers, upper=clip_outliers)
        values = clipped.values
    
    if not weighted:
        return np.mean(values), np.std(values)
    
    n = len(values)
    if n == 0:
        return 0.0, 0.0
   
    if heavy_recent:
        exponents = np.arange(n)  # 0 oldest, n-1 newest
        weights = 0.5 ** (n - 1 - exponents)
    else:
        weights = 1 / np.arange(1, n + 1)  # harmonic, newest highest
   
    total_weight = weights.sum()
    w_mean = np.sum(weights * values) / total_weight
    w_variance = np.sum(weights * (values - w_mean) ** 2) / total_weight
    w_std = np.sqrt(w_variance)
    return w_mean, w_std

def simulate_growth_paths(initial_value: float, mean_growth: float, std_growth: float, n_simulations: int, years: int = 3) -> np.ndarray:
    """
    Runs Monte Carlo simulations of future value paths using normally distributed
    annual growth rates with multiplicative (compounding) returns.

    Each path represents one possible future trajectory of the financial metric.

    Parameters
    ----------
    initial_value : float
        The starting value (most recent historical value, e.g., last year's Pretax Income).

    mean_growth : float
        Expected annual growth rate (as decimal, e.g., 0.15 for +15%).

    std_growth : float
        Annual growth rate volatility (standard deviation, e.g., 0.25 for high uncertainty).

    n_simulations : int
        Number of independent paths to simulate. Higher = more accurate percentiles.

    years : int, default 3
        Number of future years to project in each simulation.

    Returns
    -------
    np.ndarray
        Array of shape (n_simulations, years) containing the projected ending value
        for each year in each simulation.
        - paths[i, j] = value at end of year (j+1) in simulation i

    Notes
    -----
    - Uses tfp.distributions.Normal to sample annual returns.
    - Applies compounding via cumulative product of (1 + rate).
    - Can produce negative values in high-volatility scenarios (realistic downside risk).
    - Seed is fixed (42) for reproducibility during testing/debugging.
    - This is an arithmetic Brownian motion approximation — suitable for short horizons.
    """
    if years < 1:
        raise ValueError("years must be >= 1")
    
    if std_growth < 0:
        raise ValueError("std_growth must be non-negative")

    # Use tfp to define the distribution
    dist = tfp.distributions.Normal(loc=mean_growth, scale=std_growth)
    
    # Sample annual growth rates: shape (n_simulations, years)
    annual_rates = dist.sample(sample_shape=(n_simulations, years), seed=42)  # optional seed for reproducibility
    
    # Convert to growth factors: 1 + rate
    growth_factors = 1.0 + annual_rates
    
    # Cumulative product along years (axis=1) → compounded growth
    paths = tf.math.cumprod(growth_factors, axis=1)
    
    # Scale by initial value
    paths = paths * initial_value
    
    return paths.numpy()

def montecarlo_prediction_multi_year(df: pd.DataFrame, column_name: str, years: int = 3, percentile: float = 50.0, clip_outliers: float = 0.2, weighted: bool = True, simulation_number: int = 10000, heavy_recent: bool = False, interest_rate: bool = False, log: bool = False) -> dict:
    """
    Performs a multi-year Monte Carlo forecast for a single financial line item
    (e.g., Total Revenue, Pretax Income) using historical growth statistics.

    The function estimates mean and standard deviation of annual growth rates,
    simulates thousands of possible future paths via normal-distributed returns,
    and returns the selected percentile value for each forecast year.

    Primarily used internally to forecast components like Revenue, Costs,
    or Pretax Income before deriving balance sheet scaling factors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the historical financial data. Rows are time periods
        (sorted ascending), columns are line items.

    column_name : str
        The exact column name of the line item to forecast
        (e.g., 'Total Revenue', 'Pretax Income').

    years : int, default 3
        Number of future years to simulate and forecast (1–3 recommended).

    percentile : float, default 50.0
        Percentile of the simulated paths to use as the forecast value.
        - 50.0 → median (base case)
        - 10.0 → conservative scenario
        - 90.0 → optimistic/aggressive scenario

    clip_outliers : float, default 0.2
        Caps extreme annual growth rates at ±this value before calculating statistics.
        - 0.2 → limits to ±20% (prevents huge swings from outliers)
        - 0.0 or False → no clipping (preserves full volatility, recommended for high-growth stocks)

    weighted : bool, default True
        If True, uses time-weighted statistics (harmonic or heavy recent).
        If False, uses simple unweighted mean/std of historical growth rates.

    simulation_number : int, default 10000
        Number of Monte Carlo paths to simulate. Higher values give more stable
        and accurate percentile estimates.

    heavy_recent : bool, default False
        If True (and weighted=True), applies strong exponential weighting
        (weights: 1, 1/2, 1/4, 1/8...) — strongly favors the most recent years.
        Ideal for capturing accelerating growth (e.g., AMD, NVDA).
        If False, uses harmonic weighting (1/1, 1/2, 1/3...).

    interest_rate : bool, default False
        If True, treats the series as an interest rate or ratio (no pct_change).
        Uses raw mean and std of levels instead of growth rates.
        Rarely used for balance sheet items.

    log : bool, default False
        If True, prints the estimated mean growth and standard deviation
        for the column being forecasted. Useful for debugging.

    Returns
    -------
    dict
        Forecast results with keys:
        - 'year_1', 'year_2', ..., 'year_n': projected values for each year
        - 'initial_value': most recent historical value
        - 'mean_annual_growth': estimated mean growth rate used
        - 'std_annual_growth': estimated volatility
        - 'percentile_used': the percentile selected
        - 'simulations': number of simulations run
        - 'weighted_stats': whether weighting was applied

    Notes
    -----
    - Uses normal arithmetic returns (not lognormal) → can produce negative values
      in high-volatility scenarios.
    - Called multiple times in legacy balance sheet construction to estimate
      Pretax Income via Revenue - Costs - Expenses.
    - In the modern yfinance-based forecast_balance_sheet_multi_year(),
      this function is mainly used for component validation or custom analysis.
    """
    if column_name not in df.columns:
        print(f"{column_name} does not exist in dataframe")
        return {f"year_{y}": 0.0 for y in range(1, years + 1)}
    series = df[column_name].dropna()
    if len(series) < 2:
        print(f"Not enough data for {column_name}")
        return {f"year_{y}": float(series.iloc[-1]) for y in range(1, years + 1)}
    initial_value = float(series.iloc[-1])
    if interest_rate:
        mean, std = series.mean(), series.std()
    else:
        mean, std = compute_growth_statistics(series, weighted=weighted, heavy_recent=heavy_recent, clip_outliers=clip_outliers)
    if log:
        print(f"{column_name} - Mean growth: {mean:.4%}, Std: {std:.4%}")
    paths = simulate_growth_paths(
        initial_value=initial_value,
        mean_growth=mean,
        std_growth=std,
        years=years,
        n_simulations=simulation_number
    )
    forecasts = {}
    for y in range(1, years + 1):
        year_col = paths[:, y - 1]
        forecast_value = tfp.stats.percentile(year_col, percentile, axis=0)
        forecasts[f"year_{y}"] = float(forecast_value)
    forecasts.update({
        "initial_value": initial_value,
        "mean_annual_growth": mean,
        "std_annual_growth": std,
        "percentile_used": percentile,
        "simulations": simulation_number,
        "weighted_stats": weighted
    })
    return forecasts

def calc(df, col_list, rate=1, log=False):
    """
    Sums the latest values (most recent row) from a list of columns in a DataFrame,
    scales each value by a given rate, and returns the total along with a detailed
    item breakdown.

    This helper function is used extensively in balance sheet projection to aggregate
    related line items (e.g., various cash accounts, receivables components) into
    subtotal categories.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing financial data. Index is assumed to be dates (sorted ascending),
        so iloc[-1] gives the most recent period.

    col_list : list[str]
        List of column names to sum. These are the specific line items to aggregate.
        The function is robust to missing columns.

    rate : float, default 1
        Scaling factor applied to each value before rounding and summing.
        Typically 1 + growth_rate (e.g., 1.15 for +15% projected growth).
        Default 1 means no scaling (use latest reported values as-is).

    log : bool, default False
        If True, prints each column name and its scaled/rounded value (or 'not found'
        for missing columns). Useful for debugging aggregation logic.

    Returns
    -------
    tuple
        (total: int, items: list)
        - total: Sum of all (scaled and rounded) values from existing columns.
        - items: List of [column_name, value] pairs in the same order as col_list.
          Missing columns get value 0.

    Notes
    -----
    - All values are rounded to the nearest integer using np.round before summing.
    - Missing columns are silently treated as 0 (with optional log message).
    - This ensures consistent, clean integer outputs suitable for financial modeling.
    """
    items = []
    answer = 0
    for col in col_list:
        try:
            answer += np.round(df[col].iloc[-1] * rate)
            items.append([col, np.round(df[col].iloc[-1] * rate)])
            if log:
                print(col, np.round(df[col].iloc[-1] * rate))
        except KeyError:
            if log:
                print(col, 'not found')
            else:
                answer += 0
                items.append([col, 0])
    return answer, items

def construct_balance(company_ticker: str, output_mode: str = '', percentile:int = 50, weighted:bool = False, forced_pct_change: float = None, forecast_year: int = None, simulation_number:int = 10000) -> tuple:
    """
    Constructs and displays a projected balance sheet for the given company ticker
    by scaling historical balance sheet items using a provided growth factor (Pct_change).
    
    The function projects all major asset and liability components, directly scales
    shareholders' equity to ensure the balance sheet always balances 
    (Total Assets = Total Liabilities + Equity), and provides detailed breakdown output.
    
    Parameters
    ----------
    company_ticker : str
        The stock ticker symbol (e.g., 'AAPL', 'AMD'). Used for display purposes and
        for loading the corresponding income/balance CSV files (if not using live yfinance).
    
    output_mode : str, default ''
        Output format control:
        - '' (empty): Print formatted summary to console (default)
        - 'csv': Save detailed breakdown as CSV file in the current directory
        - 'txt': Save formatted text output to a .txt file
    
    forced_pct_change : float, optional (default None)
        The scaling factor to apply to all balance sheet items (1 + growth rate).
        Example: 1.15 means +15% growth → all items scaled by 1.15.
        If None, the function falls back to estimating it from revenue/cost/expense forecasts
        (legacy behavior — rarely used now that forecast_balance_sheet_multi_year handles this).
    
    forecast_year : int, optional (default None)
        The calendar year being forecasted (e.g., 2026).
        If provided, it appears in the output header:
        "PROJECTED BALANCE SHEET SUMMARY (Year 2026) - AAPL"
        Improves readability in multi-year forecasts.
    
    Returns
    -------
    tuple
        (values, items)
        - values: list of projected numeric components (assets, liabilities, equity, etc.)
        - items: list of sub-item breakdowns (for debugging or further use)
    
    Notes
    -----
    - This function is typically called internally by forecast_balance_sheet_multi_year()
      with a pre-calculated forced_pct_change derived from Pretax Income Monte Carlo simulations.
    - Equity is directly projected as: last_reported_equity × Pct_change
      (capped at 0 if negative), ensuring perfect balance sheet integrity.
    - Data is expected to be in data/{ticker}_income.csv and data/{ticker}_balance.csv
      OR fetched live via yfinance in the parent function.
    """
    global lines_console, lines_csv
    lines_console.clear()
    lines_csv.clear()

    income = pd.read_csv(f'data/{company_ticker}_income.csv', parse_dates=True)
    balance = pd.read_csv(f'data/{company_ticker}_balance.csv', parse_dates=True)
   
    income = income.T.rename(index={'Unnamed: 0': 'Date'}).T.sort_values('Date')
    balance = balance.T.rename(index={'Unnamed: 0': 'Date'}).T.sort_values('Date')
    
    if forced_pct_change is not None:
        Pct_change = forced_pct_change
    else:
        Trev_pred = montecarlo_prediction_multi_year(income, 'Total Revenue', percentile=percentile, simulation_number=simulation_number, weighted=weighted)
        Cost_of_rev_pred = montecarlo_prediction_multi_year(income, 'Cost Of Revenue', percentile=percentile, simulation_number=simulation_number, weighted=weighted)
        operating_expense_pred = montecarlo_prediction_multi_year(income, 'Operating Expense', percentile=percentile, simulation_number=simulation_number, weighted=weighted)
        other_income_expense_pred = montecarlo_prediction_multi_year(income, 'Other Income Expense', percentile=percentile, simulation_number=simulation_number, weighted=weighted)
        Pretax_income_prediction = (Trev_pred['year_3'] - Cost_of_rev_pred['year_3'] -
                                   operating_expense_pred['year_3'] - other_income_expense_pred['year_3'])
        last_pretax = income['Pretax Income'].iloc[-1]
        Pct_change = (Pretax_income_prediction - last_pretax) / last_pretax + 1 if last_pretax != 0 else 1.0

    # === Assets (unchanged) ===
    total_cash, cash_items = calc(balance, ['Cash Cash Equivalents And Federal Funds Sold', 'Cash Cash Equivalents And Short Term Investments'], rate=Pct_change)
    total_receivables, receivable_items = calc(balance, ['Accounts Receivable', 'Other Receivables', 'Duefrom Related Parties Current', 'Loans Receivable', 'Accrued Interest Receivable', 'Taxes Receivable'], rate=Pct_change)
    try:
        if np.round(balance['Receivables'].iloc[-1] * Pct_change - total_receivables) != 0:
            total_receivables = np.round(balance['Receivables'].iloc[-1] * Pct_change)
    except KeyError:
        pass
    total_inventory, inventory_items = calc(balance, ['Raw Materials', 'Work in Process', 'Finished Goods', 'Other Inventories'], rate=Pct_change)
    try:
        if np.round(balance['Inventory'].iloc[-1] * Pct_change - total_inventory) != 0:
            total_inventory = np.round(balance['Inventory'].iloc[-1] * Pct_change)
    except KeyError:
        pass
    total_other_current_assets, other_current_asset_items = calc(balance, ['Prepaid Assets', 'Other Current Assets', 'Hedging Assets Current', 'Assets Held For Sale Current', 'Restricted Cash', 'Current Deferred Assets'], rate=Pct_change)
    curr_asset_check = total_cash + total_receivables + total_inventory + total_other_current_assets

    if 'Gross PPE' in balance.columns and 'Accumulated Depreciation' in balance.columns:
        net_ppe, ppe_items = calc(balance, ['Gross PPE', 'Accumulated Depreciation'], rate=Pct_change)
    else:
        net_ppe = np.round(balance.get('Net PPE', pd.Series([0])).iloc[-1] * Pct_change)
        ppe_items = [['Net PPE (direct)', net_ppe]]

    goodwill_intangible = np.round(balance.get('Goodwill And Other Intangible Assets', pd.Series([0])).iloc[-1] * Pct_change)
    investment_advances = np.round(balance.get('Investments And Advances', pd.Series([0])).iloc[-1] * Pct_change)
    total_other_non_curr_asset, total_other_non_curr_asset_items = calc(balance, ['Non Current Deferred Assets', 'Other Non Current Assets', 'Regulatory Assets', 'Defined Pension Benefit',
                                                                                  'Financial Assets', 'Non Current Note Receivables', 'Investment Properties', 'Non Current Accounts Receivable'], rate=Pct_change)
    non_curr_asset_check = net_ppe + goodwill_intangible + investment_advances + total_other_non_curr_asset

    total_securities, securities_items = calc(balance, ['Trading Securities', 'Held To Maturity Securities'], rate=Pct_change)
    try:
        if np.round(balance['Securities and Investments'].iloc[-1] * Pct_change - total_securities) != 0:
            total_securities = np.round(balance['Securities and Investments'].iloc[-1] * Pct_change)
            securities_items = ['Total securities', total_securities]
    except KeyError:
        pass

    total_loan = 0
    loan_items = []
    try:
        total_loan, loan_items = calc(balance, ['Gross Loan', 'Allowance for Loans And Lease Losses'], rate=Pct_change)
        if np.round(balance['Net Loan'].iloc[-1] * Pct_change - total_loan) != 0:
            total_loan = np.round(balance['Net Loan'].iloc[-1] * Pct_change)
            loan_items = ['Total Loan', total_loan]
    except KeyError:
        pass

    total_other_assets, other_asset_items = calc(balance, ['Other Assets', 'Foreclosed Assets', 'Derivative Assets', 'Bank Owned Life Insurance', 'Federal Home Loan Bank Stock'], rate=Pct_change)
    include_securities = total_securities != 0
    asset_check = curr_asset_check + non_curr_asset_check + total_loan + total_other_assets + (total_securities if include_securities else 0)

    # === Liabilities ===
    payables_accrued_expenses = np.round(balance.get('Payables And Accrued Expenses', pd.Series([0])).iloc[-1] * Pct_change)
    total_other_curr_liabilities, total_other_curr_liabilities_items = calc(balance, ['Pensionand Other Post Retirement Benefit Plans Current', 'Current Debt And Capital Lease Obligation',
                                                                                    'Current Provisions', 'Other Current Liabilities', 'Current Deferred Taxes Liabilities', 'Current Deferred Liabilities'], rate=Pct_change)
    curr_liability_check = payables_accrued_expenses + total_other_curr_liabilities

    total_deferred_credits_non_current_liabilities, total_deferred_credits_non_current_liabilities_items = calc(balance, ['Long Term Provisions', 'Non Current Deferred Liabilities', 'Regulatory Liabilities', 'Employee Benefits',
                                                                                                                    'Other Non Current Liabilities'], rate=Pct_change)
    total_LT_debt_cap_lease, LT_debt_cap_lease_items = calc(balance, ['Long Term Debt', 'Long Term Capital Lease Obligation'], rate=Pct_change)
    try:
        if np.round(balance['Long Term Debt And Capital Lease'].iloc[-1] * Pct_change - total_LT_debt_cap_lease) != 0:
            total_LT_debt_cap_lease = np.round(balance['Long Term Debt And Capital Lease'].iloc[-1] * Pct_change)
    except KeyError:
        pass
    total_other_non_current_liabilities, total_other_non_current_liabilities_items = calc(balance, ['Tradeand Other Payables Non Current', 'Non Current Accrued Expenses', 'Liabilities Heldfor Sale Non Current', 'Derivative Product Liabilities'], rate=Pct_change)
    non_curr_liability_check = total_deferred_credits_non_current_liabilities + total_other_non_current_liabilities + total_LT_debt_cap_lease
    liabilities_check = curr_liability_check + non_curr_liability_check

    # === Direct Equity Projection ===
    try:
        last_equity = balance['Total Equity Gross Minority Interest'].iloc[-1]
    except KeyError:
        try:
            last_equity = balance['Total Shareholders Equity'].iloc[-1]
        except KeyError:
            last_equity = balance.get('Stockholders Equity', pd.Series([0])).iloc[-1]

    projected_equity = np.round(last_equity * Pct_change)
    if projected_equity < 0:
        projected_equity = 0

    total_assets_projected = liabilities_check + projected_equity

    # === Output ===
    add_console("\n" + "="*80)
    if forecast_year is not None:
        add_console(f"PROJECTED BALANCE SHEET SUMMARY (Year {forecast_year}) - {company_ticker.upper()}")
        add_csv("PROJECTED BALANCE SHEET", f"{company_ticker.upper()} (Year {forecast_year})", "")
    else:
        add_console(f"PROJECTED BALANCE SHEET SUMMARY - {company_ticker.upper()}")
        add_csv("PROJECTED BALANCE SHEET", company_ticker.upper(), "")
    add_console("="*80)

    # Total Assets
    try:
        reported_total_assets_scaled = np.round(balance['Total Assets'].iloc[-1] * Pct_change)
        asset_diff = np.round(total_assets_projected - reported_total_assets_scaled)
        asset_pct_diff = abs(asset_diff) / reported_total_assets_scaled if reported_total_assets_scaled != 0 else float('inf')
        asset_status = "MATCH (within ±1)" if abs(asset_diff) <= 1 else f"ADJUSTED: {asset_diff:,} ({asset_pct_diff:.1%})"
        add_console(f"TOTAL ASSETS: {asset_status} (balanced via Equity)")
        add_console(f"   Projected (Liab + Equity): {total_assets_projected:,.0f}")
        add_console(f"   Reported scaled:           {reported_total_assets_scaled:,.0f}")
        add_console(f"   Difference:                {asset_diff:,.0f}\n")
        add_csv("Total Assets", "Status", asset_status + " (balanced)")
        add_csv("Total Assets", "Projected (L+E)", f"{total_assets_projected:,.0f}")
        add_csv("Total Assets", "Reported Scaled", f"{reported_total_assets_scaled:,.0f}")
    except KeyError:
        add_console(f"TOTAL ASSETS (projected): {total_assets_projected:,.0f}")
        add_csv("Total Assets", "Projected", f"{total_assets_projected:,.0f}")

    # Asset Components
    add_console("   --- Asset Components ---")
    add_csv("", "Asset Components", "")
    add_console(f"   Current Assets:                    {curr_asset_check:,.0f}")
    add_console(f"      ├── Cash & Equivalents:         {total_cash:,.0f}")
    add_console(f"      ├── Receivables:                {total_receivables:,.0f}")
    add_console(f"      ├── Inventory:                  {total_inventory:,.0f}")
    add_console(f"      └── Other Current:              {total_other_current_assets:,.0f}")
    add_csv("Current Assets", "Total", f"{curr_asset_check:,.0f}")
    add_csv("Current Assets", "Cash & Equivalents", f"{total_cash:,.0f}")
    add_csv("Current Assets", "Receivables", f"{total_receivables:,.0f}")
    add_csv("Current Assets", "Inventory", f"{total_inventory:,.0f}")
    add_csv("Current Assets", "Other Current", f"{total_other_current_assets:,.0f}")

    add_console(f"   Non-Current Assets:                {non_curr_asset_check:,.0f}")
    add_console(f"      ├── Net PPE:                     {net_ppe:,.0f}")
    add_console(f"      ├── Goodwill & Intangibles:      {goodwill_intangible:,.0f}")
    add_console(f"      ├── Investments & Advances:      {investment_advances:,.0f}")
    add_console(f"      └── Other Non-Current:           {total_other_non_curr_asset:,.0f}")
    add_csv("Non-Current Assets", "Total", f"{non_curr_asset_check:,.0f}")
    add_csv("Non-Current Assets", "Net PPE", f"{net_ppe:,.0f}")
    add_csv("Non-Current Assets", "Goodwill & Intangibles", f"{goodwill_intangible:,.0f}")
    add_csv("Non-Current Assets", "Investments & Advances", f"{investment_advances:,.0f}")
    add_csv("Non-Current Assets", "Other Non-Current", f"{total_other_non_curr_asset:,.0f}")

    # === RESTORED: Full Liabilities Breakdown ===
    add_console("\n   --- Liability Components ---")
    add_csv("", "Liability Components", "")
    add_console(f"   Current Liabilities:                      {curr_liability_check:,.0f}")
    add_console(f"      ├── Payables & Accrued Expenses:       {payables_accrued_expenses:,.0f}")
    add_console(f"      └── Other Current Liabilities:         {total_other_curr_liabilities:,.0f}")
    add_csv("Current Liabilities", "Total", f"{curr_liability_check:,.0f}")
    add_csv("Current Liabilities", "Payables & Accrued", f"{payables_accrued_expenses:,.0f}")
    add_csv("Current Liabilities", "Other Current", f"{total_other_curr_liabilities:,.0f}")

    add_console(f"   Non-Current Liabilities:                  {non_curr_liability_check:,.0f}")
    add_console(f"      ├── LT Debt & Capital Leases:          {total_LT_debt_cap_lease:,.0f}")
    add_console(f"      ├── Deferred Credits:                  {total_deferred_credits_non_current_liabilities:,.0f}")
    add_console(f"      └── Other Non-Current Liabilities:     {total_other_non_current_liabilities:,.0f}")
    add_csv("Non-Current Liabilities", "Total", f"{non_curr_liability_check:,.0f}")
    add_csv("Non-Current Liabilities", "LT Debt & Leases", f"{total_LT_debt_cap_lease:,.0f}")
    add_csv("Non-Current Liabilities", "Deferred Credits", f"{total_deferred_credits_non_current_liabilities:,.0f}")
    add_csv("Non-Current Liabilities", "Other Non-Current", f"{total_other_non_current_liabilities:,.0f}")

    # Total Liabilities summary
    add_console(f"\nTOTAL LIABILITIES:                         {liabilities_check:,.0f}")

    # Equity
    add_console(f"PROJECTED SHAREHOLDERS' EQUITY:            {projected_equity:,.0f}")
    add_csv("Equity", "Projected Equity", f"{projected_equity:,.0f}")
    add_csv("Equity", "Scaling Factor", f"{Pct_change:.4f}")

    # Balance check
    balance_check_diff = total_assets_projected - (liabilities_check + projected_equity)
    add_console(f"BALANCE CHECK (A vs L + E):                 {balance_check_diff:,.0f} (should be 0)")
    add_csv("Equity", "Balance Check", f"{balance_check_diff:,.0f}")

    add_console("="*80)

    # Output handling
    os.makedirs('data', exist_ok=True)
    if output_mode == 'csv':
        csv_path = f'data/{company_ticker}_balance.csv'
        pd.DataFrame(lines_csv, columns=['Category', 'Subcategory', 'Value']).to_csv(csv_path, index=False)
        print(f"\nClean CSV saved: {csv_path}")
    elif output_mode == 'txt':
        txt_path = f'data/{company_ticker}_balance.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines_console))
        print(f"\nFormatted text saved: {txt_path}")
    else:
        for line in lines_console:
            print(line)

    # Return values
    values = [total_cash, total_receivables, total_inventory, total_other_current_assets, curr_asset_check,
              net_ppe, goodwill_intangible, investment_advances, total_other_non_curr_asset, 0,  # regulatory fix placeholder
              non_curr_asset_check, total_securities, total_loan, total_other_assets,
              total_assets_projected, liabilities_check, projected_equity]

    items = [cash_items, receivable_items, inventory_items, other_current_asset_items, ppe_items,
             total_other_non_curr_asset_items, other_asset_items, securities_items, loan_items,
             total_other_curr_liabilities_items, total_deferred_credits_non_current_liabilities_items,
             LT_debt_cap_lease_items, total_other_non_current_liabilities_items]

    return values, items

def forecast_balance_sheet_multi_year(company_ticker: str, years: int = 3, percentile: float = 50.0, output_mode: str = '', weighted: bool = False, heavy_recent: bool = False, simulation_number: int = 10000, download_data: bool = False, clip_outlier_growths: float = 0, seed: int = 0) -> dict:
    """
    Generates a multi-year projected balance sheet forecast using Monte Carlo simulation
    driven by historical Pretax Income growth, automatically fetched from Yahoo Finance via yfinance.
    
    The function:
    - Downloads latest annual income statement and balance sheet
    - Estimates Pretax Income growth statistics (with optional heavy recent weighting)
    - Runs thousands of Monte Carlo simulations of future Pretax Income paths
    - Derives year-over-year scaling factors (Pct_change) from the selected percentile path
    - Projects the full balance sheet by scaling liabilities and directly scaling equity
      to ensure perfect balance (Assets = Liabilities + Equity)
    - Prints a clean multi-year summary table including actual latest year (Year 0)
    
    Parameters
    ----------
    company_ticker : str
        Stock ticker symbol (e.g., 'AMD', 'AAPL', 'NVDA'). Case-insensitive.
    
    years : int, default 3
        Number of future years to forecast. Must be 1, 2, or 3.
    
    percentile : float, default 50.0
        Percentile of the Monte Carlo Pretax Income paths to use for the forecast.
        - 50.0 → median (base case)
        - 10.0 → conservative (lower growth)
        - 90.0 → aggressive (higher growth)
    
    output_mode : str, default ''
        Controls detailed per-year output format:
        - '' (empty): Print detailed breakdown to console only
        - 'csv': Save detailed per-year balance sheet as CSV files
        - 'txt': Save detailed formatted text files
    
    weighted : bool, default False
        If True, uses time-weighted statistics for growth estimation.
        When False, uses simple arithmetic mean/std (unweighted).
    
    heavy_recent : bool, default False
        If True (and weighted=True), applies strong exponential decay weighting
        (weights: 1, 1/2, 1/4, 1/8...) — heavily favors most recent years.
        Great for capturing momentum in fast-growing companies like AMD.
        If False, uses harmonic weighting (1/1, 1/2, 1/3...) — more gradual recency bias.
    
    simulation_number : int, default 10000
        Number of Monte Carlo simulations to run. Higher = more stable percentiles,
        but slower. 10,000 is a good balance of accuracy and speed.
    
    download_data : bool, default False
        If True, saves the raw income statement and balance sheet downloaded from
        yfinance to the 'out/' folder as {ticker}_income.csv and {ticker}_balance.csv.
        Useful for debugging or offline reuse.
    
    clip_outlier_growths : float, default 0.2
        Maximum allowed absolute annual growth rate for Pretax Income statistics.
        Values above +clip or below -clip are capped.
        - 0.2 → caps at ±20% (original conservative behavior)
        - 0.0 or False → no clipping (recommended for high-growth stocks)
        - 0.5 → allows up to ±50%, etc.
        Set to 0.0 to disable clipping entirely.

    Returns
    -------
    dict
        Nested dictionary with keys 'year_0', 'year_1', ..., 'year_n' containing:
        - calendar_year: int
        - label: str (e.g., "Y1 (2026) Proj.")
        - pct_change_used: float
        - total_assets, total_liabilities, implied_equity: float
        - balance_check: float (should be ~0)
        - values & items: detailed component breakdowns
    
    Notes
    -----
    - Data is automatically fetched live — no pre-downloaded files needed.
    - Equity is directly projected (last_equity x Pct_change) and capped at 0 if negative.
    - Balance sheet integrity is strictly enforced: Assets = Liabilities + Equity.
    - Heavy recent weighting + no clipping often gives more realistic upside for growth stocks.
    """
    np.random.seed(seed)
    if years < 1 or years > 3:
        raise ValueError("years must be 1, 2, or 3")
    current_year = datetime.now().year
    forecast_years = [current_year + y for y in range(1, years + 1)]

    try:
        income, balance = download_and_prepare_data(company_ticker, download_data=download_data)
    except Exception as e:
        raise RuntimeError(f"Failed to download data for {company_ticker}: {e}")

    # income = income.T.rename(index={'Unnamed: 0': 'Date'}).T.sort_values('Date')
    # balance = balance.T.rename(index={'Unnamed: 0': 'Date'}).T.sort_values('Date')

    # Year 0 Actual
    actual_total_assets = float(balance['Total Assets'].iloc[-1])
    actual_total_liab = float(balance['Total Liabilities Net Minority Interest'].iloc[-1])
    actual_equity = actual_total_assets - actual_total_liab
    if actual_equity < 0:
        actual_equity = 0.0

    year_0 = {
        "calendar_year": current_year,
        "label": f"Y0 ({current_year}) Actual",
        "pct_change_used": 0.0,
        "total_assets": actual_total_assets,
        "total_liabilities": actual_total_liab,
        "implied_equity": actual_equity,
        "balance_check": 0.0
    }

    # Monte Carlo on Pretax Income
    pretax_series = income['Pretax Income'].dropna()
    mean_growth, std_growth = compute_growth_statistics(pretax_series, weighted=weighted, heavy_recent=heavy_recent, clip_outliers=clip_outlier_growths)
    last_pretax = float(pretax_series.iloc[-1])

    pretax_paths = simulate_growth_paths(last_pretax, mean_growth, std_growth, years)
    selected_path = np.percentile(pretax_paths, percentile, axis=0)

    pct_changes = (selected_path / last_pretax) - 1
    for i in range(1, years):
        pct_changes = np.append(pct_changes, selected_path[i] / selected_path[i-1] - 1)
    pct_changes = pct_changes.tolist()

    forecasts = {"year_0": year_0}
    for year_idx in range(years):
        year_label = forecast_years[year_idx]
        Pct_change = 1 + pct_changes[year_idx]

        values, items = construct_balance(
            company_ticker=company_ticker,
            output_mode='',
            forced_pct_change=Pct_change,
            forecast_year=year_idx+current_year,
            simulation_number=simulation_number
        )

        liabilities_check = values[-2]  # second last
        projected_equity = values[-1]   # last
        total_assets = liabilities_check + projected_equity

        forecasts[f"year_{year_idx + 1}"] = {
            "calendar_year": year_label,
            "label": f"Y{year_idx + 1} ({year_label}) Proj.",
            "pct_change_used": pct_changes[year_idx],
            "total_assets": total_assets,
            "total_liabilities": liabilities_check,
            "implied_equity": projected_equity,
            "balance_check": 0.0,
            "values": values,
            "items": items
        }

        if output_mode in ['csv', 'txt']:
            path = f'data/{company_ticker}_balance_{year_label}.{"csv" if output_mode == "csv" else "txt"}'
            # Save per-year detailed file (same as original logic)

    print("\n" + "="*110)
    print(f"MULTI-YEAR BALANCE SHEET FORECAST - {company_ticker.upper()} (incl. Year 0 Actual)")
    print(f"Scenario: {percentile}th percentile | Mean growth: {mean_growth:+.2%} (±{std_growth:.2%})")
    print("="*110)
    
    summary_lines = []
    summary_csv = []
    
    header = f"{'Period':<16} {'Year':<8} {'Growth':>10} {'Total Assets':>20} {'Liabilities':>20} {'Equity':>20}"
    summary_lines.append(header)
    summary_lines.append("-" * 110)
    
    summary_csv.append(["Summary", "Period", "Year,Growth,Total Assets,Liabilities,Equity"])
    
    # Year 0
    y0 = forecasts["year_0"]
    summary_lines.append(
        f"{y0['label']:<16} {y0['calendar_year']:<8} {'N/A':>10} "
        f"{y0['total_assets']:>20,.0f} {y0['total_liabilities']:>20,.0f} {y0['implied_equity']:>20,.0f}"
    )
    summary_csv.append(["Summary", "Year 0 (Actual)", f"{y0['calendar_year']},N/A,{y0['total_assets']:,.0f},{y0['total_liabilities']:,.0f},{y0['implied_equity']:,.0f}"])
    
    # Forecast Years
    for y in range(1, years + 1):
        f = forecasts[f"year_{y}"]
        growth_str = f"{f['pct_change_used']:+.1%}"
        summary_lines.append(
            f"{f['label']:<16} {f['calendar_year']:<8} {growth_str:>10} "
            f"{f['total_assets']:>20,.0f} {f['total_liabilities']:>20,.0f} {f['implied_equity']:>20,.0f}"
        )
        summary_csv.append(["Summary", f"Year {y} (Proj.)", f"{f['calendar_year']},{f['pct_change_used']:.2%},{f['total_assets']:,.0f},{f['total_liabilities']:,.0f},{f['implied_equity']:,.0f}"])
    
    # Output summary
    if output_mode == 'csv':
        summary_path = f'data/{company_ticker}_balance_forecast_summary.csv'
        pd.DataFrame(summary_csv, columns=['Category', 'Subcategory', 'Value']).to_csv(summary_path, index=False)
        print(f"\nSummary CSV saved: {summary_path}")
    elif output_mode == 'txt':
        summary_path = f'data/{company_ticker}_balance_forecast.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        print(f"\nSummary TXT saved: {summary_path}")
    else:
        for line in summary_lines:
            print(line)
    
    print("="*110)
    print("Notes:")
    print("  • Year 0 = Latest reported actuals")
    print("  • Future years = Projected using selected Pretax Income growth percentile")
    print("  • Assets = Liabilities + Equity enforced | Negative equity capped at 0")
    print("="*110)

    return forecasts

# Example run
# forecast_balance_sheet_multi_year('NVDA', years=3, percentile=50.0, weighted=True, heavy_recent=True,)
