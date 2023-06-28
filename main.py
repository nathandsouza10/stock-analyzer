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

from sklearn.preprocessing import MinMaxScaler

device = ('cuda' if torch.cuda.is_available() else 'cpu')

st.set_page_config(layout="wide")

st.write("running on :", device)
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

st.title("Trading Bot (Day-to-Day)")

client = CryptoHistoricalDataClient()
request_params = CryptoBarsRequest(
                        symbol_or_symbols=["BTC/USD", "ETH/USD", "DAI/USD", "GRT/USD", "BTC/USD", "LINK/USD", "SHIB/USD"],
                        timeframe=TimeFrame.Day,
                        start=datetime(1600, 7, 1), #ridiculous datetime chosen to get earliest possible stock price
                        end=datetime(2022, 9, 1)
                 )

bars = client.get_crypto_bars(request_params)
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

option = st.selectbox(
        "Symbol",
        ("BTC/USD", "ETH/USD", "DAI/USD", "GRT/USD", "LINK/USD"),
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

with st.expander("LSTM Evaluation"):
    with st.spinner("calculating trends..."):
        num_epochs = 1500
        model = LSTM(input_dim=1, hidden_dim=32, output_dim=1, num_layers=3)
        model = model.to(device)
        criterion = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        hist = np.zeros(num_epochs)
        for t in range(num_epochs):
            y_train_pred = model(x_train)
            loss = torch.sqrt(criterion(y_train_pred, y_train_lstm)) #RMSE
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
