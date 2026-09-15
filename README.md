# Portfolio Optimization with Python

An independent learning project focused on understanding investment portfolio optimization through both financial theory and Python implementation.

## About the Project

This project was created as a personal learning project to better understand financial markets, investment theory, and quantitative approaches to portfolio management.

Before starting the implementation, I focused on the theoretical foundations of portfolio management. I studied relevant chapters of William F. Sharpe’s Investments, as well as additional materials and online resources covering portfolio theory, risk, return, diversification, and optimization.

I also started learning how Python can be used for financial data analysis, working with libraries such as Pandas and NumPy, and using Python for Finance as one of the learning resources.

I am currently moving from the theoretical part to the practical implementation of the project.

## What I Want to Implement

The main goal of the project is to build a Python-based tool that can analyze historical market data and find portfolios with different risk-return characteristics.

The planned workflow includes:

1. Collect historical market data
    * Download historical prices for selected assets.
    * Store and process the data using Pandas DataFrames.
2. Calculate asset returns
    * Calculate daily returns from historical prices.
    * Analyze the relationship between assets.
3. Analyze portfolio characteristics
    * Calculate expected portfolio return.
    * Calculate portfolio variance and volatility.
    * Calculate the covariance matrix of asset returns.
4. Portfolio optimization
    * Find the Minimum Variance Portfolio (MVP).
    * Optimize portfolio weights under constraints such as no short selling.
    * Explore different target return levels.
5. Efficient Frontier
    * Generate a set of optimal portfolios.
    * Calculate their expected return and risk.
    * Visualize the Efficient Frontier.
6. Sharpe Ratio
    * Calculate the Sharpe ratio for portfolios.
    * Find the portfolio with the highest risk-adjusted return.
7. Visualization and analysis
    * Visualize asset and portfolio characteristics.
    * Compare different optimized portfolios.
    * Use the results to better understand the relationship between risk and return.

## Financial Theory

The theoretical part of the project is based on concepts from modern portfolio theory, including:

* expected return;
* variance and standard deviation;
* covariance and correlation;
* portfolio return;
* portfolio variance;
* diversification;
* risk-return trade-off;
* Minimum Variance Portfolio;
* Efficient Frontier;
* Sharpe Ratio;
* portfolio weights and constraints;
* no-short-selling portfolios.

For a portfolio with weights $w_1, w_2, …, w_n$, the expected return can be represented as:

$$ 
E(R_p) = \sum_{i=1}^{n} w_i E(R_i) 
$$

Portfolio variance is calculated using the covariance matrix:

$$
\sigma_p^2 = w^T \Sigma w
$$

where $w$ is the vector of portfolio weights and $\Sigma$ is the covariance matrix of asset returns.

Sharpe Ratio

The Sharpe Ratio measures the risk-adjusted return of a portfolio by comparing its excess return over the risk-free rate with its volatility.

$$
S_p = \frac{E(R_p)-R_f}{\sigma_p}
$$

where:
* $E(R_p)$ is the expected portfolio return;
* $R_f$ is the risk-free rate;
* $\sigma_p$ is the portfolio standard deviation $volatility$.

A higher Sharpe Ratio indicates a higher return relative to the amount of risk taken.

---

The project will use these theoretical concepts as the foundation for the Python implementation.

**Technologies**

* Python
* Pandas - data processing and analysis
* NumPy - numerical computations and matrix operations
* SciPy - numerical optimization
* yfinance - obtaining historical market data
* Matplotlib - data visualization

**Learning Resources**

The project is based on a combination of financial and programming resources, including:

* William F. Sharpe - Investments
* Python for Finance
* Additional articles, documentation, and online educational resources

The resources are used not only to implement the algorithms, but also to understand the financial meaning behind the calculations.

**Project Status - In progress** 

The theoretical foundation has been studied, and the project is now moving into the implementation stage.

The next steps are to implement the data pipeline, return and risk calculations, portfolio optimization, and Efficient Frontier visualization.

As the project develops, I also plan to improve the analysis and make the results easier to interpret from both a programming and financial perspective.
