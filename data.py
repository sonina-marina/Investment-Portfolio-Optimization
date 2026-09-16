import yfinance as yf
import pandas as pd
import numpy as np


def download_data(tickers):
    df = yf.download(
        tickers,
        period="2y",
        auto_adjust=True
    )

    df = df["Close"]
    return df


def calculate_returns(df):
    daily_returns = df.pct_change().dropna()
    mean_returns = daily_returns.mean()
    annual_returns = mean_returns * 252

    covariance_matrix = daily_returns.cov() * 252

    return annual_returns, covariance_matrix
