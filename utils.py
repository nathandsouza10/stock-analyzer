import numpy as np
import yfinance as yf
from datetime import datetime
import pandas as pd


def get_daily_stock_data(tickers):
    today = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(tickers, start="1980-01-01", end=today)
    return data['Close']


def get_modern_portfolio(stocks_list):
    cycles = 252  # number of stock trading days in a year
    closing_price_df = pd.DataFrame()
    for stock in stocks_list:
        closing_price_df[stock] = get_daily_stock_data(stocks_list)[stock]

    log_returns = np.log(closing_price_df / closing_price_df.shift(1))
    portfolio_returns = []
    portfolio_volatilities = []
    weights_list = []
    for x in range(1000):
        weights = np.random.random(len(stocks_list))
        weights /= np.sum(weights)
        weights_list.append(weights)
        portfolio_returns.append(np.sum(weights * log_returns.mean()) * cycles)
        portfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * cycles, weights))))

    portfolio_returns = np.array(portfolio_returns)
    portfolio_volatilities = np.array(portfolio_volatilities)

    portfolios = pd.DataFrame({'Return': portfolio_returns, 'Volatility': portfolio_volatilities})
    return portfolios, weights_list


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
