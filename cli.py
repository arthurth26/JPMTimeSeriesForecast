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
        help="Number of years to forecast (1–3, default: 3)"
    )
    
    parser.add_argument(
        '--percentile',
        type=float,
        default=50.0,
        help="Percentile scenario: 10=conservative, 50=base, 90=aggressive (default: 50.0)"
    )
    
    parser.add_argument(
        '--heavy-recent',
        action='store_true',
        help="Use strong recent weighting (exponential decay) — ideal for momentum/growth stocks"
    )
    
    parser.add_argument(
        '--download-data',
        action='store_true',
        help="Save raw income and balance sheet data from yfinance to the 'out/' folder"
    )
    
    parser.add_argument(
        '--clip',
        type=float,
        default=0,
        metavar='VALUE',
        help="Enable and set growth-rate clipping threshold (e.g., --clip 0.2 for ±20%% cap, --clip 0.3 for ±30%%)."
             "Default: 0.0 (no clipping)"
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help="Random seed for Monte Carlo simulations (default: 0). Use same seed for reproducible results."
    )
    
    args = parser.parse_args()
    
    print(f"Running forecast for {args.ticker.upper()}...\n")
    print(f"Configuration:")
    print(f"  Years: {args.years}")
    print(f"  Percentile: {args.percentile}")
    print(f"  Heavy recent weighting: {'Yes' if args.heavy_recent else 'No'}")
    print(f"  Download raw data: {'Yes' if args.download_data else 'No'}")
    print(f"  Growth clipping: {'Disabled' if args.clip == 0.0 else f'±{args.clip:.0%}'}")
    print(f"  Random seed: {args.seed}\n")
    
    forecast_balance_sheet_multi_year(
        company_ticker=args.ticker,
        years=args.years,
        percentile=args.percentile,
        heavy_recent=args.heavy_recent,
        download_data=args.download_data,
        clip_outlier_growths = args.clip,
        seed=args.seed
    )

if __name__ == "__main__":
    main()