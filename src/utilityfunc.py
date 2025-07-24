import yfinance as yf
import pandas as pd
import os
import json

def download_price_data(tickers, start_date, end_date):
    """
    Download historical price data using yfinance.
    Returns a DataFrame with adjusted close prices.
    """
    price_data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]

    # Save raw price data (optional)
    os.makedirs("data/processed", exist_ok=True)
    price_data.to_csv("data/processed/price_data.csv")

    return price_data

def save_portfolio_weights(weights, output_path="data/processed/optimized_weights.json"):
    """
    Save portfolio weights as a JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(weights, f, indent=2)