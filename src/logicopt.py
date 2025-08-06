import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

def compute_esg_weight_adjustments(esg_scores):
    """
    Normalize ESG scores and compute weight adjustments.
    """
    esg_df = pd.DataFrame(esg_scores).T
    # Normalize each component
    normalized = esg_df / esg_df.max()
    # Average the components to get overall ESG score
    normalized['ESG'] = normalized.mean(axis=1)
    # Normalize final ESG scores
    final_weights = normalized['ESG'] / normalized['ESG'].sum()
    return final_weights.to_dict()

def optimize_portfolio(price_data, esg_scores=None):
    """
    Optimize a portfolio optionally incorporating ESG scores into the objective.
    """
    mu = mean_historical_return(price_data)
    S = sample_cov(price_data)

    ef = EfficientFrontier(mu, S)

    if esg_scores:
        esg_weights = compute_esg_weight_adjustments(esg_scores)
        # Adjust expected returns using ESG scores
        for ticker in mu.index:
            if ticker in esg_weights:
                mu[ticker] *= esg_weights[ticker]

        ef = EfficientFrontier(mu, S)

    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    return cleaned_weights
