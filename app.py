import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
import math

from data import download_data, calculate_returns
from portfolio import (
    calculate_portfolio_return,
    calculate_portfolio_variance,
    calculate_sharpe_ratio
)
from optimization import (
    optimize_min_variance, 
    optimize_max_sharpe,
    calculate_efficient_frontier
)

st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem !important; /* Было около 6rem, сделали аккуратные 2rem */
    }
    h1 {
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }
    h2 {
        margin-top: 15px !important;
        padding-top: 0px !important;
    }rder: 1px solid #e0e4ec;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Portfolio Optimization",
    layout="wide"
)

st.title("Portfolio Optimization")

AVAILABLE_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "JPM",
    "V",
    "MA",
    "WMT",
    "KO",
    "PEP",
    "JNJ",
    "XOM",
    "CVX",
    "SPY",
    "QQQ"
]

def main():

    #--- Left side ---
    left_column, spacer, right_column = st.columns([0.3, 0.15, 0.55])

    with left_column:

        st.header("Portfolio Settings")

        tickers = st.multiselect(
            "Select assets",
            AVAILABLE_TICKERS
        )

        risk_free_rate_percent = st.number_input(
            "Risk-free rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=4.0,
            step=0.5,
            format="%.2f"
        )

        risk_free_rate = risk_free_rate_percent / 100

        calculate_button = st.button(
            "Calculate",
            type="primary",
            use_container_width=True
        )

    #--- Calculations ---
    if calculate_button:

        if len(tickers) < 2:
            st.warning("Please select at least two assets")
            return

        with st.spinner("Downloading market data and calculating..."):

            df = download_data(tickers)

            annual_returns, covariance_matrix = calculate_returns(df)

            num_assets = len(tickers)

            #-esults-- Minimum Variance Portfolio
            min_variance_result = optimize_min_variance(
                covariance_matrix,
                num_assets
            )
            
            #--- Maximum Sharpe Retio ---
            max_sharpe_result = optimize_max_sharpe(
                annual_returns,
                covariance_matrix,
                risk_free_rate,
                num_assets
            )

            #--- Efficient Frontier ---
            frontier_returns, frontier_volatilities, frontier_weights = (
                calculate_efficient_frontier(
                    annual_returns,
                    covariance_matrix,
                    num_assets
                )
            )

        #--- Results ---

        with left_column:

            st.subheader("Results")

            min_weights = min_variance_result.x
            min_variance = min_variance_result.fun
            min_volatility = math.sqrt(min_variance)


            st.markdown("**Minimum Variance Portfolio**")
                
            min_portfolio_text = ""
            for ticker, weight in zip(tickers, min_weights):
                if weight > 0.001:  # выводим только то, у чего вес больше нуля
                    min_portfolio_text += f"{ticker}: {weight:.2%}  \n"
            
            min_portfolio_text += f"**Volatility**: {min_volatility:.2%}"

            st.markdown(min_portfolio_text)

            st.markdown("<hr style='margin: 5px 0;'>", unsafe_allow_html=True)

            st.markdown("**Maximum Sharpe Portfolio**")

            max_weights = max_sharpe_result.x
            max_sharpe = -max_sharpe_result.fun
            
            max_sharpe_text = ""
            for ticker, weight in zip(tickers, max_weights):
                max_sharpe_text += f"{ticker}: {weight:.2%}  \n"

            max_sharpe_text += f"**Sharpe Ratio**: {max_sharpe:.2f}"
            st.markdown(max_sharpe_text)

        #--- Right side ---

        with right_column:

            st.header("Efficient Frontier")

            fig, ax = plt.subplots(figsize=(7, 5))

            #--- Minimum Variance Portfolio ---
            min_return = calculate_portfolio_return(
                annual_returns,
                min_weights
            )

            efficient_mask = frontier_returns >= min_return
            efficient_returns = frontier_returns[efficient_mask]
            efficient_volatilities = frontier_volatilities[efficient_mask]

            ax.plot(
                efficient_volatilities,
                efficient_returns,
                label="Efficient Frontier",
                color="red"
            )

            ax.scatter(
                min_volatility,
                min_return,
                marker="o",
                s=100,
                label="Minimum Variance",
                color="red"
            )

            #--- Maximum Sharpe Portfolio ---
            max_variance = calculate_portfolio_variance(
                max_weights,
                covariance_matrix
            )

            max_volatility = math.sqrt(max_variance)

            max_return = calculate_portfolio_return(
                annual_returns,
                max_weights
            )

            ax.scatter(
                max_volatility,
                max_return,
                marker="*",
                s=150,
                label="Maximum Sharpe"
            )

            ax.set_xlabel("Volatility")
            ax.set_ylabel("Expected Return")
            ax.set_title("Efficient Frontier")

            ax.legend()
            ax.grid(True)

            st.pyplot(fig)


if __name__ == "__main__":
    main()
