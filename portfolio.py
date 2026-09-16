import numpy as np
import math


def calculate_portfolio_return(annual_returns, weights):
    portfolio_return = (annual_returns * weights).sum()
    return portfolio_return


def calculate_portfolio_variance(weights, covariance_matrix):
    horizontal_weights = weights[np.newaxis, :]
    vertical_weights = weights[:, np.newaxis]

    portfolio_variance = (
        horizontal_weights @ covariance_matrix @ vertical_weights
    ).to_numpy().item()

    return float(portfolio_variance)


def calculate_sharpe_ratio(portfolio_return, portfolio_variance, risk_free_rate):
    portfolio_std_dev = math.sqrt(portfolio_variance)

    if portfolio_std_dev == 0:
        return 0.0

    portfolio_sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev

    return portfolio_sharpe_ratio
