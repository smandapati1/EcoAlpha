import yfinance as yf
import pandas as pd

def download_price_data(tickers, start_date, end_date):
    """
    Downloads adjusted close prices for the given tickers and date range.
    Handles both single and multiple tickers.

    Returns:
        pd.DataFrame: DataFrame of adjusted close prices
    """
    # Always download with auto_adjust=True to avoid confusion
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    # If only one ticker, return as a DataFrame
    if isinstance(tickers, str) or (isinstance(tickers, list) and len(tickers) == 1):
        return data[['Close']].rename(columns={'Close': tickers[0] if isinstance(tickers, list) else tickers})
    else:
        return data['Close']
