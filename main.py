import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import streamlit as st
from models import LSTM
from datetime import datetime, timedelta
from utils import split_data, get_daily_stock_data, get_modern_portfolio
import altair as alt
from sklearn.preprocessing import MinMaxScaler
import os

api_key, secret = st.secrets["api_key"], st.secrets["secret"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(layout="wide")

st.write("running on :", device)
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

st.title("Stock Analyzer")
with st.expander("Menu", expanded=True):
    STOCKS = st.multiselect(
        'Please Choose stocks:',
        ['AAPL', 'MSFT', 'GOOGL'],
        ['AAPL', 'MSFT', 'GOOGL'])
    LOOKBACK = st.slider("lookback(days):", 2, 20)
    TEST_SIZE = st.slider("Choose test size ratio", 0.01, 0.99)

bars = get_daily_stock_data(STOCKS)

st.subheader("Modern Portfolio Theory")
with st.spinner("Loading..."):
    portfolios, weights_list = get_modern_portfolio(STOCKS)
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

st.subheader("Moving Average analysis (200 day)")

cash_init = st.number_input("Enter Cash Invested", value=1000)

with st.spinner("Loading..."):
    risk_df = pd.DataFrame(index=STOCKS, columns=['final_portfolio_value', 'profit/loss'])
    for option in STOCKS:
        data = bars[option]
        price = data.reset_index(drop=True).to_frame(name='close')
        price['200day(MA)'] = price.rolling(window=200).mean()
        price['buy/sell'] = 'Hold'  # default action is to hold
        cash = cash_init
        shares_owned = 0
        already_bought = False  # flag to check if you've already bought shares
        already_sold = False  # flag to check if you've already sold shares

        for i, row in price.iterrows():
            # Check if the close price is greater than the 200-day MA, you haven't bought any shares yet,
            # and you haven't already sold.
            if row['close'] > row['200day(MA)'] and cash > 0 and row[
                'close'] > 0 and not already_bought and not already_sold:
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
        profit_loss = final_portfolio_value - cash_init
        percentage_increase = ((final_portfolio_value - cash_init) / cash_init) * 100

        risk_df.at[option, 'final_portfolio_value'] = final_portfolio_value
        risk_df.at[option, 'profit/loss'] = profit_loss
        risk_df.at[option, 'percentage_change'] = percentage_increase

    risk_df = risk_df.sort_values(by='final_portfolio_value', ascending=False)
    st.dataframe(risk_df, use_container_width=True, column_config={
        "percentage_increase": st.column_config.NumberColumn(format="%d%%")
    })

st.subheader("LSTM stock evaluation")
model_evaluations = {}
with st.spinner("performing LSTM model evaluation"):
    for option in STOCKS:
        eval_df = pd.DataFrame()
        price = bars[option]
        scaler = MinMaxScaler(feature_range=(0, 1))
        price = scaler.fit_transform(price.values.reshape(-1, 1))
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
