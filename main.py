import alpaca.data

import pandas
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random

import streamlit as st
from models import LSTM
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
from utils import split_data
import altair as alt

from sklearn.preprocessing import MinMaxScaler

device = ('cuda' if torch.cuda.is_available() else 'cpu')

st.set_page_config(layout="wide")

st.write("running on :", device)
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

st.title("Stock Analyzer")
with st.expander("Menu"):
    STOCKS = st.multiselect(
        'Choose stock option(s)',
        ["BTC/USD", "ETH/USD", "LINK/USD", "SHIB/USD", "BCH/USD"],
        ["BTC/USD"]
    )
    TIMEFRAME = st.selectbox("Choose timeframe:", [TimeFrame.Hour, TimeFrame.Day])

with st.spinner("Loading data from alpaca"):
    client = CryptoHistoricalDataClient()
    request_params = CryptoBarsRequest(
        symbol_or_symbols=STOCKS,
        timeframe=TIMEFRAME,
        start=datetime(1600, 7, 1),  # ridiculous datetime chosen to get earliest and latest possible stock price
        end=datetime(3030, 9, 1)
    )

    bars = client.get_crypto_bars(request_params)
st.success("Completed: Loading Data")

st.subheader("Stock Evaluation")

model_eval_df = pd.DataFrame()
with st.spinner("performing model evaluation"):
    for option in STOCKS:
        data = bars.df.loc[option]
        price = data[['close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        price['close'] = scaler.fit_transform(price['close'].values.reshape(-1, 1))
        lookback = 10
        x_train, y_train, x_test, y_test = split_data(price, lookback, test_size=0.25)
        x_train = torch.from_numpy(x_train).type(torch.Tensor).to(device)
        x_test = torch.from_numpy(x_test).type(torch.Tensor).to(device)
        y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor).to(device)
        y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor).to(device)

        num_epochs = 1000
        model = LSTM(input_dim=1, hidden_dim=32, output_dim=1, num_layers=2)
        model = model.to(device)
        criterion = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        hist = np.zeros(num_epochs)
        for t in range(num_epochs):
            y_train_pred = model(x_train)
            loss = torch.sqrt(criterion(y_train_pred, y_train_lstm))  # RMSE
            hist[t] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        with torch.no_grad():
            y_test_pred = model(x_test)

        model_eval_df[option] = [torch.sqrt(criterion(y_test_pred, y_test_lstm)).item(),
                                 torch.sqrt(criterion(y_train_pred, y_train_lstm)).item()]

st.dataframe(model_eval_df, use_container_width=True)


def get_modern_portfolio():
    closing_price_df = pd.DataFrame()
    for stock in STOCKS:
        closing_price_df[stock] = bars.df.loc[stock]['close']

    log_returns = np.log(closing_price_df / closing_price_df.shift(1))
    portfolio_returns = []
    portfolio_volatilities = []
    weights_list = []
    for x in range(2000):
        weights = np.random.random(len(STOCKS))
        weights /= np.sum(weights)
        weights_list.append(weights)
        portfolio_returns.append(np.sum(weights * log_returns.mean()) * 365)
        portfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 365, weights))))

    portfolio_returns = np.array(portfolio_returns)
    portfolio_volatilities = np.array(portfolio_volatilities)

    portfolios = pd.DataFrame({'Return': portfolio_returns, 'Volatility': portfolio_volatilities})
    return portfolios, weights_list


st.subheader("Modern Portfolio Theory")
portfolios, weights_list = get_modern_portfolio()

col1, col2 = st.columns(2)
with col1:
    st.write("Minimum volatility portfolio")
    min_volatility_df = portfolios[portfolios['Volatility'] == min(portfolios['Volatility'])]
    st.dataframe(min_volatility_df, use_container_width=True)
    optimal_weightings_df = pd.DataFrame()
    for i, stock in enumerate(STOCKS):
        optimal_weightings_df[stock] = [weights_list[min_volatility_df.index[0]][i]]
    st.dataframe(optimal_weightings_df, use_container_width=True)

with col2:
    st.write("Portfolio Graph")
    chart = alt.Chart(portfolios).mark_circle().encode(
        x='Volatility',
        y='Return'
    ).interactive()
    st.altair_chart(chart, theme="streamlit", use_container_width=True)
