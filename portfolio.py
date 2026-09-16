import numpy as np


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
