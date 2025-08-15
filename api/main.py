from src import extractingesg, scorecomputation, utilityfunc, logicopt, visualize

def main(tickers, start_date, end_date, use_mock=False):
    print(f"\nESG-based portfolio optimization for: {tickers}\n")

    # Step 1: Collect raw ESG sources (mock or real)
    if use_mock:
        raw_sources = extractingesg.build_mock_raw(tickers, seed=2025)  # 👈 varied, deterministic
        print("[Info] Using MOCK ESG sources for testing.")
    else:
        raw_sources = extractingesg.download_and_extract(tickers)

    # Step 2: ESG analysis → E/S/G scores (0..1)
    esg_results = extractingesg.run_esg_analysis(raw_sources)
    print("ESG Scores:", esg_results)

    # Step 3: Prices
    price_df = utilityfunc.download_price_data(tickers, start_date, end_date)
    print(f"Retrieved {len(price_df)} rows of stock price data.")

    # Step 4: Optimize (with ESG influence)
    try:
        weights = logicopt.optimize_portfolio(price_df, esg_scores=esg_results)
    except Exception as e:
        print(f"⚠️ Max Sharpe failed ({e}), falling back to minimum volatility.")
        weights = logicopt.optimize_portfolio(price_df)

    print("Optimized Portfolio Weights:", weights)
    visualize.plot_portfolio_weights(weights)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ESG-based Portfolio Optimization CLI")
    parser.add_argument("--tickers", nargs='+', required=True, help="List of tickers (e.g. AAPL MSFT TSLA)")
    parser.add_argument("--start_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--mock_esg", action="store_true", help="Use varied mock ESG inputs for testing")
    args = parser.parse_args()

    main(args.tickers, args.start_date, args.end_date, use_mock=args.mock_esg)