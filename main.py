# main.py
import argparse
from src import extractingesg, utilityfunc, logicopt

def main(tickers, start_date, end_date, use_mock=True, seed=2025):
    print(f"\n🌿 ESG-based portfolio optimization for: {tickers}\n")

    # 1. ESG Scores
    if use_mock:
        raw_esg = extractingesg.build_mock_raw(tickers, seed=seed)
    else:
        raw_esg = extractingesg.download_and_extract(tickers)

    esg_results = extractingesg.run_esg_analysis(raw_esg)
    print("ESG Scores:", esg_results)

    # 2. Price Data
    price_df = utilityfunc.download_price_data(tickers, start_date, end_date)
    if price_df is None or price_df.empty:
        print("⚠️ No price data found.")
        return
    print(f"Retrieved {len(price_df)} rows of stock price data.")

    # 3. Optimize Portfolio
    try:
        weights = logicopt.optimize_portfolio(price_df, esg_scores=esg_results)
    except Exception as e:
        print(f"⚠️ Max Sharpe failed ({e}), falling back to min volatility...")
        weights = logicopt.optimize_portfolio(price_df)

    print("\nOptimized Portfolio Weights:")
    print(weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESG Portfolio Optimization CLI")
    parser.add_argument("--tickers", nargs="+", required=True, help="List of stock tickers")
    parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--live", action="store_true", help="Use live ESG data instead of mock")

    args = parser.parse_args()
    main(args.tickers, args.start_date, args.end_date, use_mock=not args.live)
