import random
from utils import *
import altair as alt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import streamlit as st
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from datetime import timedelta
import pandas as pd

device = get_device()

st.set_page_config(layout="wide")

st.write("running on :", device)
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

st.title("Stock Analyzer")
# Create three columns
col1, col2 = st.columns(2)

# Left column for the line chart
with col1:
    STOCKS = st.multiselect(
        'Please Choose stocks:',
        ['AAPL', 'MSFT', 'GOOGL', 'NFLX'],
        ['AAPL', 'MSFT', 'GOOGL', 'NFLX'])

# Middle column for the future prices DataFrame
with col2:
    st.write("Market Cap. ratio")
    # Fetch the market cap for each ticker
    market_caps = {}
    for ticker in STOCKS:
        stock = yf.Ticker(ticker)
        market_caps[ticker] = stock.info["marketCap"]

    # Convert the market_caps dictionary to a pandas DataFrame
    df = pd.DataFrame(list(market_caps.items()), columns=['Ticker', 'Market Cap'])

    # Create a pie chart using altair
    pie_chart = alt.Chart(df).mark_arc(innerRadius=50, outerRadius=150).encode(
        theta='Market Cap:Q',
        color='Ticker:N',
        tooltip=['Ticker', 'Market Cap']
    )

    # Display the pie chart using streamlit
    st.altair_chart(pie_chart)

bars = get_daily_stock_data(STOCKS)

st.subheader("Modern Portfolio Theory")
with st.spinner("Loading..."):
    portfolios, weights_list = get_modern_portfolio(STOCKS)
    col1, col2 = st.columns(2)

    # Efficiently find the index of the minimum volatility
    min_vol_idx = portfolios['Volatility'].idxmin()
    min_volatility_df = portfolios.iloc[[min_vol_idx]]

    with col1:
        st.write("Minimum volatility portfolio")
        st.dataframe(min_volatility_df, use_container_width=True)

        # Efficiently create the optimal weightings DataFrame
        optimal_weightings_df = pd.DataFrame([weights_list[min_vol_idx]], columns=STOCKS)

        st.write("Minimum volatility portfolio ratio")
        st.dataframe(optimal_weightings_df, use_container_width=True)

    with col2:
        st.write("Portfolio Graph")
        chart = alt.Chart(portfolios).mark_circle().encode(
            x='Volatility',
            y='Return'
        ).interactive()
        st.altair_chart(chart, theme="streamlit", use_container_width=True)

st.subheader("Moving Average Analysis (200 day)")
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
                shares_owned += shares_bought  # Update your total shares aowned
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

st.subheader("Stock Time Series Analysis")

# Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],

}

# Gradient Boosting
param_grid_gb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [3, 4, 5]
}

# Linear Regression
param_grid_lr = {
    'fit_intercept': [True, False],
}

# Ridge Regression
param_grid_ridge = {
    'alpha': [0.001, 0.1, 1, 10],
    'fit_intercept': [True, False],
}

# Lasso Regression
param_grid_lasso = {
    'alpha': [0.001, 0.1, 1, 10],
    'fit_intercept': [True, False],
}

# Elastic Net
param_grid_en = {
    'alpha': [0.001, 0.1, 1],
    'l1_ratio': [0.2, 0.5, 0.7],
    'fit_intercept': [True, False],
}

# Support Vector Regression
param_grid_svr = {
    'C': [0.2, 0.5, 2],
    'epsilon': [0.05, 0.1, 0.5],
    'kernel': ['linear', 'poly', 'rbf']
}

# K-Nearest Neighbors
param_grid_knn = {
    'n_neighbors': [2, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Decision Tree
param_grid_dt = {
    'criterion': ['mse', 'friedman_mse', 'mae'],
    'splitter': ['best', 'random'],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

models = [
    ('RandomForest', RandomForestRegressor(), param_grid_rf),
    ('GradientBoosting', GradientBoostingRegressor(), param_grid_gb),
    ('LinearRegression', LinearRegression(), param_grid_lr),
    ('Ridge', Ridge(), param_grid_ridge),
    ('Lasso', Lasso(), param_grid_lasso),
    ('ElasticNet', ElasticNet(), param_grid_en),
    ('SVR', SVR(), param_grid_svr),
    ('KNN', KNeighborsRegressor(), param_grid_knn),
    ('DecisionTree', DecisionTreeRegressor(), param_grid_dt)
]

model_performance_dict = {}
for ticker in STOCKS:
    with st.spinner(f"Performing time series analysis for: {ticker}"):
        series = get_monthly_stock_data(ticker)
        df = series.reset_index()
        df.columns = ['Date', 'Price']
        df['Date'] = pd.to_datetime(df['Date'])

        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day_Delta'] = (df['Date'] - df['Date'].min()).dt.days

        X = df[['Year', 'Month', 'Day_Delta']].values
        y = df['Price'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        best_model = get_model_performance(X_train, y_train, X_test, y_test, models)
        model_performance_dict[ticker] = {
            'Best Model': best_model.__class__.__name__,
            'MSE': mean_squared_error(y_test, best_model.predict(X_test))
        }


        def estimate_price(input_date):
            year = input_date.year
            month = input_date.month
            day_delta = (input_date - df['Date'].min()).days
            return best_model.predict([[year, month, day_delta]])[0]


        future_dates = [df['Date'].iloc[-1]] + [df['Date'].iloc[-1] + timedelta(days=i * 30) for i in range(1, 12)]
        future_prices = [estimate_price(date) for date in future_dates]

        future_df = pd.DataFrame({
            'Date': future_dates,
            'Estimated Prices': future_prices
        })

        combined_df = pd.DataFrame({
            'Historical': df.set_index('Date')['Price'],
            'Future': future_df.set_index('Date')['Estimated Prices']
        })

        st.write(f"{ticker}")

        # Create three columns
        col1, col2, col3 = st.columns(3)

        # Left column for the line chart
        with col1:
            st.line_chart(combined_df)

        # Middle column for the future prices DataFrame
        with col2:
            st.dataframe(future_df, use_container_width=True)

        # Right column for model performance
        with col3:
            st.write(f"Best Model: {best_model.__class__.__name__}")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, best_model.predict(X_test))}")
