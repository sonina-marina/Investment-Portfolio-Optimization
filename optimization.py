import numpy as np
from scipy.optimize import _constraints, minimize

from portfolio import (
    calculate_portfolio_return,
    calculate_portfolio_variance,
    calculate_sharpe_ratio,
)


def objective_function(weights, covariance_matrix):
    return calculate_portfolio_variance(weights, covariance_matrix)


def optimize_min_variance(covariance_matrix, num_assets):
    initial_weights = np.ones(num_assets) / num_assets

    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    bounds = [(0,1) for _ in range(num_assets)]

    result = minimize(
        objective_function,
        initial_weights,
        args=(covariance_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result

