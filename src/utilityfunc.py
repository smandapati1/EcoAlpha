import yfinance as yf
import pandas as pd

def download_price_data(tickers, start_date, end_date):
    """
    Downloads historical adjusted close price data for the given tickers
    between start_date and end_date using yfinance.

    Parameters:
        tickers (list or str): Ticker symbols
        start_date (str): Start date in 'YYYY-MM-DD'
        end_date (str): End date in 'YYYY-MM-DD'

    Returns:
        pd.DataFrame: DataFrame with adjusted close prices
    """
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=False)

    if isinstance(tickers, str) or len(tickers) == 1:
        # Single ticker case
        return data["Adj Close"].to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)
    else:
        # Multiple ticker case (MultiIndex columns)
        return data["Adj Close"]
