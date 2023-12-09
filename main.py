import torch
import torch.nn as nn
import random
import altair as alt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from utils import *

device = get_device()

st.set_page_config(layout="wide")

st.write("running on :", device)
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

st.title("Stock Analyzer")

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
            market_caps[ticker] = stock.fast_info["marketCap"]
        df = pd.DataFrame(list(market_caps.items()), columns=['Ticker', 'Market Cap'])
        pie_chart = alt.Chart(df).mark_arc(innerRadius=50, outerRadius=150).encode(
            theta='Market Cap:Q',
            color='Ticker:N',
            tooltip=['Ticker', 'Market Cap']
        )
        st.altair_chart(pie_chart)

    with col3:
        # Call the get_fiftyDayAverage function
        fifty_day_avg_values = get_fiftyDayAverage(STOCKS)
        fifty_day_avg_df = pd.DataFrame(list(fifty_day_avg_values.items()), columns=['Ticker', '50 Day Avg'])
        st.write("50 Day Moving Averages")
        horizontal_bar_chart = alt.Chart(fifty_day_avg_df).mark_bar().encode(
            y='Ticker:N',  # Set y-axis to Ticker
            x='50 Day Avg:Q',  # Set x-axis to 50 Day Avg
            tooltip=['Ticker', '50 Day Avg'],
            color='Ticker:N'
        ).properties(title="50 Day Moving Average Comparison")
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


    st.subheader("Stock Time Series Analysis")
    st.subheader("LSTM")
    from LSTM import LSTM

    # Function to create sequences
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)


    # Function for predicting future values
    def predict_next_days(model, last_sequence, days=5):
        future_predictions = []
        for _ in range(days):
            with torch.no_grad():
                last_sequence_tensor = torch.tensor(last_sequence.reshape(1, seq_length, -1), dtype=torch.float32).to(
                    device)
                pred = model(last_sequence_tensor).cpu().numpy()
                future_predictions.append(pred[0][0])
                last_sequence = np.append(last_sequence[1:], pred[0])
        return future_predictions

    input_dim = 1
    output_dim = 1
    with st.form(key='my_form'):
        st.subheader("Set Hyperparameters")
        seq_length = st.number_input('Sequence Length', min_value=1, value=40)
        hidden_dim = st.number_input('Hidden Dimension', min_value=1, value=50)
        num_layers = st.number_input('Number of Layers', min_value=1, value=4)
        num_epochs = st.number_input('Number of Epochs', min_value=1, value=500)
        learning_rate = st.number_input('Learning Rate', min_value=0.00001, max_value=0.01, value=0.001, step=0.00001,
                                        format="%.5f")

        submit_button = st.form_submit_button(label='Submit')

    # Iterate through each stock
    for stock in bars:
        stock_data = bars[stock]

        # Data preparation
        data = stock_data.values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        X, y = create_sequences(data_scaled, seq_length)
        X_train, y_train = torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(
            device)

        # Model initialization
        model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
        model.to(device)
        model = nn.DataParallel(model)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # List to store loss values
        epoch_losses = []

        # Training the model
        for epoch in range(num_epochs):
            model.train()
            outputs = model(X_train)
            optimizer.zero_grad()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            if epoch % 10 == 0:
                print(f'Stock: {stock}, Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')

        # Predict the next 5 days
        last_sequence = data_scaled[-seq_length:]
        predictions_scaled = predict_next_days(model, last_sequence, days=5)
        predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

        # Preparing data for plotting
        stock_data.index = pd.to_datetime(stock_data.index)
        last_date = pd.to_datetime(stock_data.index[-1])
        start_date = last_date + pd.Timedelta(days=1)
        prediction_dates = pd.date_range(start=start_date, periods=5)
        predictions_series = pd.Series(predictions, index=prediction_dates)

        # Combine actual and predicted data
        combined_data = pd.concat([stock_data[-120:], predictions_series.rename('Prediction')], axis=1)

        # Create two columns for plots
        col1, col2, col3 = st.columns(3)

        # Plot using Streamlit in the first column
        with col1:
            st.write(f'Stock Data and Predictions for {stock}')
            st.line_chart(combined_data)

        # Plot the loss per epoch in the second column
        with col2:
            st.write('Training Loss per Epoch')
            st.line_chart(pd.Series(epoch_losses, name='Loss'))
        # Plot the loss per epoch in the second column
        with col3:
            st.write('prediction series')
            st.write(predictions_series)

