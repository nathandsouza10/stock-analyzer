import numpy as np
import yfinance as yf
from datetime import datetime
import pandas as pd
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml  # Importing this to check if it's installed and available
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


def get_daily_stock_data(tickers):
    today = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(tickers, start="2007-01-01", end=today)
    return data['Close'].dropna()


def get_monthly_stock_data(tickers):
    today = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(tickers, start="2007-01-01", end=today, interval='1mo')
    return data['Close'].dropna()


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


def get_model_performance(X_train, y_train, X_test, y_test, models):
    best_model = None
    best_mse = float('inf')

    for name, model, param_grid in models:
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_grid = grid_search.best_estimator_

        y_pred = best_grid.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        if mse < best_mse:
            best_mse = mse
            best_model = best_grid

    return best_model
