import random
import altair as alt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import streamlit as st
from parameters import *
from PytorchWrapper import PyTorchWrapper
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from datetime import timedelta
from utils import *

device = get_device()

st.set_page_config(layout="wide")

st.write("running on :", device)
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

st.title("Stock Analyzer")

# Create three columns
col1, col2, col3 = st.columns(3)

with col1:
    STOCKS = st.multiselect(
        'Please Choose stocks:',
        ['AAPL', 'MSFT', 'GOOGL', 'NFLX'],
        ['AAPL', 'MSFT', 'GOOGL', 'NFLX'])

# Check if any stocks are selected
if not STOCKS:
    st.warning("Please select at least one stock to proceed.")
else:
    if len(STOCKS) == 1:
        st.warning("You've selected only one stock. Some visualizations might not be as informative without a "
                   "comparison.")
    with col2:
        st.write("Market Cap. ratio")
        market_caps = {}
        for ticker in STOCKS:
            stock = yf.Ticker(ticker)
            market_caps[ticker] = stock.info["marketCap"]
        df = pd.DataFrame(list(market_caps.items()), columns=['Ticker', 'Market Cap'])
        pie_chart = alt.Chart(df).mark_arc(innerRadius=50, outerRadius=150).encode(
            theta='Market Cap:Q',
            color='Ticker:N',
            tooltip=['Ticker', 'Market Cap']
        )
        st.altair_chart(pie_chart)

    with col3:
        # Call the get_pe_ratios function
        pe_values = get_pe_ratios(STOCKS)
        pe_df = pd.DataFrame(list(pe_values.items()), columns=['Ticker', 'P/E Ratio'])
        st.write("P/E Ratios")
        horizontal_bar_chart = alt.Chart(pe_df).mark_bar().encode(
            y='Ticker:N',  # Set y-axis to Ticker
            x='P/E Ratio:Q',  # Set x-axis to P/E Ratio
            tooltip=['Ticker', 'P/E Ratio'],
            color='Ticker:N'
        ).properties(title="P/E Ratios Comparison")
        st.altair_chart(horizontal_bar_chart, use_container_width=True)

    bars = get_daily_stock_data(STOCKS)

    st.subheader("Modern Portfolio Theory")
    if len(STOCKS) == 1:
        st.warning("Modern Portfolio Theory could not be processed as only one stock is selected")
    else:
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
            if len(STOCKS) == 1:
                data = bars
            else:
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

    models = [
        ('RandomForest', RandomForestRegressor(), param_grid_rf),
        ('GradientBoosting', GradientBoostingRegressor(), param_grid_gb),
        ('LinearRegression', LinearRegression(), param_grid_lr),
        ('Ridge', Ridge(), param_grid_ridge),
        ('Lasso', Lasso(), param_grid_lasso),
        ('ElasticNet', ElasticNet(), param_grid_en),
        ('SVR', SVR(), param_grid_svr),
        ('KNN', KNeighborsRegressor(), param_grid_knn),
        ('DecisionTree', DecisionTreeRegressor(), param_grid_dt),
        ("LSTM", PyTorchWrapper(input_dim=3, hidden_dim=64, num_layers=2, output_dim=1), param_grid_lstm)
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

            col1, col2, col3 = st.columns(3)
            with col1:
                st.line_chart(combined_df)
            with col2:
                st.dataframe(future_df, use_container_width=True)
            with col3:
                st.write(f"Best Model: {best_model.__class__.__name__}")
                st.write(f"Mean Squared Error: {mean_squared_error(y_test, best_model.predict(X_test))}")
