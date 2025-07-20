import argparse

from src import esg_extraction, scoring, optimizer

def main(tickers, start_date, end_date):
    print(f"\n ESG-based portfolio optimization for: {tickers} ")

    raw_texts = esg_extraction.download_and_extract(tickers)

    esg_results = esg_extraction.run_esg_analysis(raw_texts)

    esg_scores = scoring.compute_esg_scores(esg_results)

    weights = optimizer.optimize_portfolio(tickers, esg_scores, start_date, end_date)

    print("\n The portfolio weights are: ")

    for ticker, weight in weights.items():
        print(f"{ticker}: {weight: .2%}")
   
if __name__ == "__main__":
      parser = argparse.ArgumentParser(description="ESG-Based Portfolio Optimizer CLI")
      
      parser.add_argument('--tickers', nargs='+', required=True, help="List of stock tickers (e.g., AAPL MSFT NVDA)")
      
      parser.add_argument('--start_date', type=str, required=True, help="Start date for price data (e.g., 2023-01-01)")
      
      parser.add_argument('--end_date', type=str, required=True, help="End date for price data (e.g., 2023-12-31)")

      args = parser.parse_args()
      
      main(args.tickers, args.start_date, args.end_date)
