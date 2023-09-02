import unittest
from unittest.mock import patch, Mock
import numpy as np
from utils import get_model_performance


class TestGetModelPerformance(unittest.TestCase):

    @patch('utils.GridSearchCV')  # Replace with the actual import path
    @patch('utils.mean_squared_error')  # Replace with the actual import path
    def test_get_model_performance(self, MockMSE, MockGridSearchCV):
        # Create some mock data
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([1, 2, 3])
        X_test = np.array([[7, 8], [9, 10]])
        y_test = np.array([4, 5])

        # Mock models and their parameters
        mock_model1 = Mock()
        mock_model2 = Mock()
        param_grid1 = {'param1': [1, 2, 3]}
        param_grid2 = {'param2': [1, 2, 3]}

        # Models to test
        models = [
            ('Model1', mock_model1, param_grid1),
            ('Model2', mock_model2, param_grid2)
        ]

        # Mock behavior of GridSearchCV
        mock_best_estimator1 = Mock()
        mock_best_estimator2 = Mock()
        MockGridSearchCV.return_value.best_estimator_ = mock_best_estimator1  # For the first call
        mock_best_estimator1.predict.return_value = np.array([4, 5])

        MockGridSearchCV.return_value.best_estimator_ = mock_best_estimator2  # For the second call
        mock_best_estimator2.predict.return_value = np.array([4.1, 5.1])

        # Mock behavior of mean_squared_error
        MockMSE.side_effect = [0.2, 0.1]  # MSE for mock_model1 and mock_model2 respectively

        # Test get_model_performance
        best_model = get_model_performance(X_train, y_train, X_test, y_test, models)

        # Verify that the model with the lowest MSE is returned
        self.assertEqual(best_model, mock_best_estimator2)
