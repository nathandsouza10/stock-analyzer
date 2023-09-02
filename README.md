
# Stock Analyzer

This script uses the Yahoo Finance API to retrieve historical data for various stocks and performs analysis using an LSTM model. The script also includes a Streamlit interface for user interaction.

## Dependencies

- yfinance
- pandas
- numpy
- torch
- streamlit
- altair
- sklearn

## Installation

To install the required dependencies, you can use pip:

```
pip install -r requirements.txt
```

## Usage

1. Install the required dependencies.
2. Run the script using Streamlit: `streamlit run main.py`
3. Use the Streamlit interface to select stock options and timeframe.
4. The script will retrieve historical data for the selected stocks and perform analysis using an LSTM model.
5. The results of the analysis will be displayed in the Streamlit interface.

## How it works

The script retrieves historical data for the selected stocks using the Yahoo Finance API. The data is preprocessed and split into training and test sets. An LSTM model is then trained on the training data to predict stock prices. The trained model is used to make predictions on the test data and the results are evaluated.

The script also includes a Streamlit interface for user interaction. The user can select stock options and timeframe using the interface. The results of the analysis are displayed in the Streamlit interface.

The dashboard application can be found at: https://stockreviewer.streamlit.app/


![image](https://github.com/nathandsouza10/stock-analyzer/assets/85251596/5a0a2cfa-2c67-453c-bc90-14cf635f063d)
![image](https://github.com/nathandsouza10/stock-analyzer/assets/85251596/2bc00592-dc82-4ca2-ba67-d7560a758719)

