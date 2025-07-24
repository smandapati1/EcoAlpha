

import pandas as pd
import json
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


def load_data(price_path, esg_path):
   
    prices = pd.read_csv(price_path, index_col="Date", parse_dates=True)
    with open(esg_path, "r") as f:
        esg_data = json.load(f)

    esg_scores = {company: info["score"] for company, info in esg_data.items()}
    esg_df = pd.DataFrame.from_dict(esg_scores, orient="index", columns=["ESG"])
    return prices, esg_df


def optimize_portfolio(prices, esg_df, min_avg_esg=0.6):
   
    tickers = [ticker for ticker in prices.columns if ticker in esg_df.index]

    prices = prices[tickers]
    esg_df = esg_df.loc[tickers]

    mu = mean_historical_return(prices)
    S = CovarianceShrinkage(prices).ledoit_wolf()

    ef = EfficientFrontier(mu, S)

    adjusted_mu = mu.copy()
    for ticker in tickers:
        if esg_df.loc[ticker, "ESG"] < min_avg_esg:
            adjusted_mu[ticker] *= 0.8  

    ef = EfficientFrontier(adjusted_mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    performance = ef.portfolio_performance(verbose=True)

    return cleaned_weights, performance


def allocate_discrete(weights, prices, total_portfolio_value=10_000):
   
    latest_prices = get_latest_prices(prices)
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio_value)
    allocation, leftover = da.lp_portfolio()
    return allocation, leftover
