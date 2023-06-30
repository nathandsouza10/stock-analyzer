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

st.title("Trading Bot (Day-to-Day)")

client = CryptoHistoricalDataClient()
STOCKS = ["BTC/USD", "ETH/USD", "LINK/USD", "SHIB/USD"]
request_params = CryptoBarsRequest(
    symbol_or_symbols=STOCKS,
    timeframe=TimeFrame.Day,
    start=datetime(1600, 7, 1),  # ridiculous datetime chosen to get earliest and latest possible stock price
    end=datetime(2030, 9, 1)
)

bars = client.get_crypto_bars(request_params)
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

option = st.selectbox(
    "Symbol",
    STOCKS,
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
)
with st.spinner("gathering data..."):
    data = bars.df.loc[option]
    price = data[['close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    price['close'] = scaler.fit_transform(price['close'].values.reshape(-1, 1))
    st.line_chart(price)

lookback = 10
x_train, y_train, x_test, y_test = split_data(price, lookback, test_size=0.25)
x_train = torch.from_numpy(x_train).type(torch.Tensor).to(device)
x_test = torch.from_numpy(x_test).type(torch.Tensor).to(device)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor).to(device)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor).to(device)

st.subheader("Models")
with st.expander("LSTM Evaluation", expanded=True):
    with st.spinner("calculating trends..."):
        num_epochs = 1000
        model = LSTM(input_dim=1, hidden_dim=32, output_dim=1, num_layers=3)
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

        st.write(f"RMSE Loss against epochs (last loss: {round(hist[-1], 6)})")
        loss_chart = pd.DataFrame()
        loss_chart['loss'] = hist
        loss_chart.reset_index(drop=True)
        st.line_chart(loss_chart, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Train")
            chart_data = pd.DataFrame()
            chart_data['y_train'] = np.squeeze(y_train)
            chart_data['y_train_pred'] = torch.squeeze(y_train_pred.cpu().detach())
            st.line_chart(chart_data, use_container_width=True)
        with col2:
            st.write("Test")
            chart_data = pd.DataFrame()
            chart_data['y_test'] = np.squeeze(y_test)
            chart_data['y_test_pred'] = torch.squeeze(y_test_pred.cpu().detach())
            st.line_chart(chart_data, use_container_width=True)


@st.cache_data
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
    st.dataframe( min_volatility_df, use_container_width=True)
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
