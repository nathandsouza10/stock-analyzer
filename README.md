# Stock Analyzer

## Description

This Streamlit application serves as a comprehensive tool for stock analysis. It offers several functionalities, including portfolio optimization based on Modern Portfolio Theory, Moving Average Analysis for individual stocks, and time series analysis for future price prediction using machine learning models.

## Features

1. **Stock Selection:** Choose from a list of available stocks (AAPL, MSFT, GOOGL, NFLX).
2. **Modern Portfolio Theory Analysis:** Get the most efficient portfolio with minimum volatility.
3. **Moving Average Analysis:** Buy/Sell signal generation based on 200-day moving average.
4. **Stock Time Series Analysis:** Future price predictions based on multiple machine learning models like Random Forest, Gradient Boosting, Linear Regression, and more.

## Installation

1. Clone the repository.
    git clone https://github.com/your-repo/stock-analyzer.git
2. Navigate to the project directory.
    cd stock-analyzer
3. Install the required packages.
    pip install -r requirements.txt

## Usage

Run the Streamlit application.
streamlit run app.py
Then navigate to http://localhost:8501/ in your browser.

## How it works

### Modern Portfolio Theory

1. **Portfolio Graph:** Plots return against volatility for a collection of portfolios containing the chosen stocks.
2. **Minimum Volatility Portfolio:** Shows the portfolio with the minimum volatility and the respective weightage of each stock in that portfolio.

### Moving Average Analysis

1. Compares the closing price to its 200-day moving average.
2. Generates Buy or Sell signals based on this comparison.

### Stock Time Series Analysis

1. Splits the historical price data into training and test sets.
2. Grid search is performed to find the best hyperparameters for each model.
3. Future prices are predicted using the best-performing model.
4. Displays the future price predictions in a line chart along with the historical prices.
