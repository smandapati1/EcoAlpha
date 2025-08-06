from src import extractingesg, scorecomputation, utilityfunc, logicopt, visualize

def main(tickers, start_date, end_date):
    print(f"\nESG-based portfolio optimization for: {tickers}\n")

    # Step 1: Download & extract ESG-related text
    raw_texts = extractingesg.download_and_extract(tickers)

    # Step 2: Run ESG analysis (convert text → ESG scores)
    esg_results = extractingesg.run_esg_analysis(raw_texts)
    print("ESG Scores:", esg_results)

    # Step 3: Download historical price data
    price_df = utilityfunc.download_price_data(tickers, start_date, end_date)
    print(f"Retrieved {len(price_df)} rows of stock price data.")

    # Step 4: Optimize portfolio with ESG integration
    try:
        weights = logicopt.optimize_portfolio(price_df, esg_scores=esg_results)
    except Exception as e:
        print(f"⚠️ Max Sharpe failed ({e}), falling back to minimum volatility.")
        weights = logicopt.optimize_portfolio(price_df)

    print("Optimized Portfolio Weights:", weights)

    # Step 5: Visualize portfolio allocation
    visualize.plot_portfolio_weights(weights)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ESG-based Portfolio Optimization CLI")
    parser.add_argument("--tickers", nargs='+', required=True, help="List of tickers (e.g. AAPL MSFT TSLA)")
    parser.add_argument("--start_date", required=True, help="Start date in YYYY-MM-DD")
    parser.add_argument("--end_date", required=True, help="End date in YYYY-MM-DD")
    args = parser.parse_args()

    main(args.tickers, args.start_date, args.end_date)
