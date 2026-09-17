import numpy as np
import math

from data import download_data, calculate_returns
from optimization import optimize_min_variance, optimize_max_sharpe


def main():

    tickers = ["AAPL", "MSFT", "NVDA"]

    df = download_data(tickers)
    annual_returns, covariance_matrix = calculate_returns(df)

    num_assets = len(tickers)
    risk_free_rate = 0.04

    min_variance_result = optimize_min_variance(covariance_matrix, num_assets)
    risk_pct = math.sqrt(min_variance_result.fun) * 100
    mvp_weights_pct = [f"{w * 100:.2f}%" for w in min_variance_result.x]

    print("Minimum Variance Portfolio")
    print("Weights:", mvp_weights_pct)
    print("Variance:", round(min_variance_result.fun, 4))
    print(f"Standard Deviation: {round(risk_pct, 2)}%")
    print("Success:", min_variance_result.success)

    max_sharpe_result = optimize_max_sharpe(
        annual_returns,
        covariance_matrix,
        risk_free_rate,
        num_assets
    )

    ms_weights_pct = [f"{w * 100:.2f}%" for w in max_sharpe_result.x]

    print("-" * 20)
    print("Maximum Sharpe Portfolio")
    print("Weights:", ms_weights_pct)
    print("Sharpe Ratio", round(-max_sharpe_result.fun, 4))
    print("Success:", max_sharpe_result.success)


if __name__ == "__main__":
    main()
