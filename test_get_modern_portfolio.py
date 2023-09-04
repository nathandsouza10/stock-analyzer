import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from utils import get_modern_portfolio  # Import the function from your actual module


# Mock daily stock data function
def mock_get_daily_stock_data(stocks_list):
    # Simulate daily stock data with random numbers.
    # In a real-world scenario, you'd pull this data from a database or API.
    data = {stock: np.random.rand(100) for stock in stocks_list}
    return data


class TestGetModernPortfolio(unittest.TestCase):

    @patch('utils.get_daily_stock_data', side_effect=mock_get_daily_stock_data)
    def test_get_modern_portfolio_shape(self, mock_get_data):
        stocks_list = ['AAPL', 'GOOGL', 'AMZN']
        portfolios, weights = get_modern_portfolio(stocks_list)

        # Check if portfolios DataFrame has the correct shape
        self.assertEqual(portfolios.shape, (1000, 2))

        # Check if weights array has the correct shape
        self.assertEqual(weights.shape, (1000, len(stocks_list)))

    @patch('utils.get_daily_stock_data', side_effect=mock_get_daily_stock_data)
    def test_get_modern_portfolio_content(self, mock_get_data):
        stocks_list = ['AAPL', 'GOOGL', 'AMZN']
        portfolios, weights = get_modern_portfolio(stocks_list)

        # Check if portfolios DataFrame has 'Return' and 'Volatility' columns
        self.assertIn('Return', portfolios.columns)
        self.assertIn('Volatility', portfolios.columns)

        # Check if the values are in a reasonable range.
        # These ranges would depend on your specific case.
        self.assertTrue((portfolios['Return'] >= -1).all() or (portfolios['Return'] <= 1).all())
        self.assertTrue((portfolios['Volatility'] >= 0).all())

        # Check if weights sum up to 1 for each portfolio
        self.assertTrue(np.allclose(weights.sum(axis=1), 1))


if __name__ == '__main__':
    unittest.main()
