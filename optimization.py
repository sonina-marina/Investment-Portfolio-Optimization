import numpy as np
from scipy.optimize import _constraints, minimize
from scipy.sparse.linalg import qmr

from portfolio import (
    calculate_portfolio_return,
    calculate_portfolio_variance,
    calculate_sharpe_ratio,
)


def objective_function_variance(weights, covariance_matrix):
    return calculate_portfolio_variance(weights, covariance_matrix)


def optimize_min_variance(covariance_matrix, num_assets):
    initial_weights = np.ones(num_assets) / num_assets

    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    bounds = [(0,1) for _ in range(num_assets)]

    result = minimize(
        objective_function_variance,
        initial_weights,
        args=(covariance_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result


def objective_function_sharpe(
    weights, 
    annual_returns,
    covariance_matrix, 
    risk_free_rate
):

    portfolio_return = calculate_portfolio_return(annual_returns, weights)
    portfolio_variance = calculate_portfolio_variance(weights, covariance_matrix)

    return -calculate_sharpe_ratio(portfolio_return, portfolio_variance, risk_free_rate)

def optimize_max_sharpe(
    annual_returns,
    covariance_matrix,
    risk_free_rate,
    num_assets
):
    initial_weights = np.ones(num_assets) / num_assets

    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    bounds = [(0, 1) for _ in range(num_assets)]

    result = minimize(
        objective_function_sharpe,
        initial_weights,
        args=(annual_returns, covariance_matrix, risk_free_rate),
        bounds=bounds,
        constraints=constraints
    )

    return result
