import argparse
from src import extractingesg, scorecomputation, utilityfunc, logicopt

def main(tickers, start_date, end_date):
    print(f"\n ESG-based portfolio optimization for: {tickers} \n")

    # Step 1: Load ESG reports
    raw_texts = extractingesg.download_and_extract(tickers)

    # Step 2: Extract ESG scores from reports
    esg_results = scorecomputation.extractingesg(raw_texts)
    print("ESG Scores:", esg_results)

    # Step 3: Fetch stock price data
    price_df = utilityfunc.download_price_data(tickers, start_date, end_date)
    print(f"\nRetrieved {len(price_df)} rows of stock price data.")

    # Step 4: Run portfolio optimization
    weights = logicopt.optimize_portfolio(price_df, esg_scores=esg_results)
    print("\nOptimal Portfolio Weights (ESG-adjusted):")
    for ticker, weight in weights.items():
        print(f"{ticker}: {weight:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ESG-based Portfolio Optimization")
    parser.add_argument("--tickers", nargs="+", required=True, help="List of stock tickers")
    parser.add_argument("--start_date", required=True, help="Start date in YYYY-MM-DD")
    parser.add_argument("--end_date", required=True, help="End date in YYYY-MM-DD")

    args = parser.parse_args()
    main(args.tickers, args.start_date, args.end_date)

