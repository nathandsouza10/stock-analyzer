import pytest
import numpy as np
from utils import split_data  # Replace `your_module` with the actual module name where `split_data` resides.


def test_split_data():
    # Create a synthetic stock dataset: 10 time steps, 3 features
    stock_data = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
        [22, 23, 24],
        [25, 26, 27],
        [28, 29, 30]
    ])

    # Define the lookback and test_size
    lookback = 3
    test_size = 0.2

    x_train, y_train, x_test, y_test = split_data(stock_data, lookback, test_size)

    # Check shapes
    assert x_train.shape == (6, 2, 3)
    assert y_train.shape == (6, 3)
    assert x_test.shape == (1, 2, 3)
    assert y_test.shape == (1, 3)

    # Check values
    assert np.all(x_train[0] == np.array([[1, 2, 3], [4, 5, 6]]))
    assert np.all(y_train[0] == np.array([7, 8, 9]))
    assert np.all(x_test[0] == np.array([[19, 20, 21], [22, 23, 24]]))
    assert np.all(y_test[0] == np.array([25, 26, 27]))
