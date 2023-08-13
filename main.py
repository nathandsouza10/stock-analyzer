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
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
from utils import split_data
import altair as alt
import os
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

load_dotenv()
device = ('cuda' if torch.cuda.is_available() else 'cpu')

st.set_page_config(layout="wide")

st.write("running on :", device)
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

st.title("Stock Analyzer")
with st.expander("Menu", expanded=True):
    STOCKS = st.multiselect(
        'Choose stock option(s)',
        ["AAPL", "MSFT", "TSLA"],
        ["AAPL"]
    )
    LOOKBACK = st.slider("lookback(days):", 2, 20)
    TEST_SIZE = st.slider("Choose test size ratio", 0.01, 0.99)

# get crypto data
client = StockHistoricalDataClient(os.getenv("api_key"), os.getenv("secret"))
request = StockBarsRequest(
    symbol_or_symbols=STOCKS,
    timeframe=TimeFrame.Day,
    start=datetime(1800, 10, 1),
    end=datetime(2023, 6, 1),
)

bars = client.get_stock_bars(request)
st.subheader("LSTM stock evaluation")
model_evaluations = {}
with st.spinner("performing LSTM model evaluation"):
    for option in STOCKS:
        eval_df = pd.DataFrame()
        data = bars.df.loc[option]
        price = data[['close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        price['close'] = scaler.fit_transform(price['close'].values.reshape(-1, 1))
        x_train, y_train, x_test, y_test = split_data(price, LOOKBACK, test_size=TEST_SIZE)
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
        for epoch in range(num_epochs):
            y_train_pred = model(x_train)
            loss = torch.sqrt(criterion(y_train_pred, y_train_lstm))  # RMSE
            hist[epoch] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        with torch.no_grad():
            y_test_pred = model(x_test)

        eval_df['y_pred'] = y_test_pred.numpy().squeeze()
        eval_df['y_test'] = y_test_lstm.numpy().squeeze()
        model_evaluations[option] = eval_df

cols = st.columns(len(model_evaluations))

for index, stock_option in enumerate(model_evaluations):
    eval_df = model_evaluations[stock_option]
    with cols[index]:
        st.write(f"{stock_option}")
        st.line_chart(eval_df)


def get_modern_portfolio():
    cycles = 252  # number of stock trading days in a year
    closing_price_df = pd.DataFrame()
    for stock in STOCKS:
        closing_price_df[stock] = bars.df.loc[stock]['close']

    log_returns = np.log(closing_price_df / closing_price_df.shift(1))
    portfolio_returns = []
    portfolio_volatilities = []
    weights_list = []
    for x in range(5000):
        weights = np.random.random(len(STOCKS))
        weights /= np.sum(weights)
        weights_list.append(weights)
        portfolio_returns.append(np.sum(weights * log_returns.mean()) * cycles)
        portfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * cycles, weights))))

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
