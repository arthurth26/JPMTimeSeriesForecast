# cli.py

import argparse
from forecast import forecast_balance_sheet_multi_year

def main():
    parser = argparse.ArgumentParser(
        description="Run multi-year balance sheet Monte Carlo forecast for a stock ticker."
    )
    
    parser.add_argument(
        'ticker',
        type=str,
        help="Stock ticker symbol (e.g., NVDA, AMD, AAPL)"
    )
    
    parser.add_argument(
        '--years',
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Number of years to forecast (1-3)"
    )
    
    parser.add_argument(
        '--percentile',
        type=float,
        default=50.0,
        help="Percentile scenario: 10=conservative, 50=base, 90=aggressive"
    )
    
    parser.add_argument(
        '--heavy-recent',
        action='store_true',
        help="Use strong recent weighting (exponential decay) — great for momentum stocks"
    )
    
    parser.add_argument(
        '--download-data',
        action='store_true',
        help="Save raw income/balance data from yfinance to out/"
    )
    
    parser.add_argument(
        '--no-clip',
        dest='clip_outlier_growths',
        action='store_const',
        const=0.0,
        help="Disable growth rate clipping (±20%) — recommended for high-growth stocks"
    )
    
    args = parser.parse_args()
    
    print(f"Running forecast for {args.ticker.upper()}...\n")
    
    forecast_balance_sheet_multi_year(
        company_ticker=args.ticker,
        years=args.years,
        percentile=args.percentile,
        heavy_recent=args.heavy_recent,
        download_data=args.download_data,
        clip_outlier_growths=args.clip_outlier_growths or 0.0 
    )

if __name__ == "__main__":
    main()