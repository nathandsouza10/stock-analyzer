import numpy as np
import yfinance as yf
from datetime import datetime
import pandas as pd


def get_daily_stock_data(tickers):
    today = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(tickers, start="1800-01-01", end=today)
    return data['Close']


def get_modern_portfolio(stocks_list):
    cycles = 252  # number of stock trading days in a year

    daily_stock_data = get_daily_stock_data(stocks_list)
    closing_price_df = pd.DataFrame({stock: daily_stock_data[stock] for stock in stocks_list})

    log_returns = np.log(closing_price_df / closing_price_df.shift(1))

    num_portfolios = 1000
    num_stocks = len(stocks_list)

    # Generate random weights for all portfolios at once
    weights = np.random.rand(num_portfolios, num_stocks)
    weights = weights / np.sum(weights, axis=1)[:, np.newaxis]

    # Calculate expected returns and volatilities for all portfolios
    expected_returns = np.sum(weights * log_returns.mean().values[np.newaxis, :], axis=1) * cycles
    cov_matrix = log_returns.cov().values
    volatilities = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix * cycles, weights))

    portfolios = pd.DataFrame({'Return': expected_returns, 'Volatility': volatilities})
    return portfolios, weights


def split_data(stock, lookback, test_size=0.2):
    data_raw = stock  # convert to numpy array
    data = []

    # create all possible sequences of length lookback
    for index in range(len(data_raw) - lookback):
        sequence = data_raw[index: index + lookback]

        # Check for NaN values in the sequence
        if not np.any(np.isnan(sequence)):
            data.append(sequence)

    data = np.array(data)
    test_set_size = int(np.round(test_size * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    return x_train, y_train, x_test, y_test
