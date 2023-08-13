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
from datetime import datetime, timedelta
from utils import split_data
import altair as alt
import os
# from dotenv import load_dotenv
# load_dotenv()
# api_key = os.getenv("api_key")
# secret = os.getenv("secret")

from sklearn.preprocessing import MinMaxScaler


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
        ["AAPL", "MSFT", "TSLA", "AMZN", "WMT", "PFE", "KO"],
        ["AAPL", "MSFT", "TSLA", "AMZN", "WMT", "PFE", "KO"]
    )
    LOOKBACK = st.slider("lookback(days):", 2, 20)
    TEST_SIZE = st.slider("Choose test size ratio", 0.01, 0.99)

# get crypto data
client = StockHistoricalDataClient(st.secrets("api_key"), st.secrets("secret"))
request = StockBarsRequest(
    symbol_or_symbols=STOCKS,
    timeframe=TimeFrame.Day,
    start=datetime(1800, 10, 1),
    end=datetime.now() - timedelta(days=1),
)

bars = client.get_stock_bars(request)


def get_modern_portfolio():
    cycles = 252  # number of stock trading days in a year
    closing_price_df = pd.DataFrame()
    for stock in STOCKS:
        closing_price_df[stock] = bars.df.loc[stock]['close']

    log_returns = np.log(closing_price_df / closing_price_df.shift(1))
    portfolio_returns = []
    portfolio_volatilities = []
    weights_list = []
    for x in range(1000):
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
with st.spinner("Loading..."):
    portfolios, weights_list = get_modern_portfolio()
    col1, col2 = st.columns(2)
    with col1:
        st.write("Minimum volatility portfolio")
        min_volatility_df = portfolios[portfolios['Volatility'] == min(portfolios['Volatility'])]
        st.dataframe(min_volatility_df, use_container_width=True)
        optimal_weightings_df = pd.DataFrame()
        for i, stock in enumerate(STOCKS):
            optimal_weightings_df[stock] = [weights_list[min_volatility_df.index[0]][i]]
        st.write("Minimum volatility portfolio ratio")
        st.dataframe(optimal_weightings_df, use_container_width=True)

    with col2:
        st.write("Portfolio Graph")
        chart = alt.Chart(portfolios).mark_circle().encode(
            x='Volatility',
            y='Return'
        ).interactive()
        st.altair_chart(chart, theme="streamlit", use_container_width=True)

st.subheader("Moving Average analysis")
with st.spinner("Loading..."):
    risk_df = pd.DataFrame()
    for option in STOCKS:
        data = bars.df.loc[option]
        price = data[['close']].copy()  # Create a copy to avoid modifying the original dataframe
        price['200day(MA)'] = price.rolling(window=200).mean()
        price['buy/sell'] = 'Hold'  # default action is to hold
        cash = 1000
        shares_owned = 0
        already_bought = False  # flag to check if you've already bought shares
        already_sold = False  # flag to check if you've already sold shares

        for i, row in price.iterrows():
            # Check if the close price is greater than the 200-day MA, you haven't bought any shares yet,
            # and you haven't already sold.
            if row['close'] > row['200day(MA)'] and cash > 0 and row['close'] > 0 and not already_bought and not already_sold:
                shares_bought = cash // row['close']  # Buy as many shares as you can
                cash -= shares_bought * row['close']  # Update your cash after buying
                shares_owned += shares_bought  # Update your total shares owned
                price.at[i, 'buy/sell'] = 'Buy'
                already_bought = True  # set the flag to true after buying

            # If the close price falls below the 200-day MA, you own shares, and you haven't already sold.
            elif row['close'] < row['200day(MA)'] and shares_owned > 0 and not already_sold:
                cash += shares_owned * row['close']  # Sell all your shares
                shares_owned = 0  # Reset shares owned to zero
                price.at[i, 'buy/sell'] = 'Sell'
                already_sold = True  # set the flag to true after selling

        # Assuming you sell all the shares at the last available price if you still have any
        final_portfolio_value = shares_owned * price.iloc[-1]['close'] + cash
        st.write(final_portfolio_value)
        st.write(price)

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
