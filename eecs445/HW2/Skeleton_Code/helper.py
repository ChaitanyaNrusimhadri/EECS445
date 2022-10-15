"""
EECS 445 - Introduction to Machine Learning
HW1 helper
"""

import pandas as pd

def load_data(fname):
    """
    Loads the data in file specified by `fname`. The file specified should be a csv with n rows and (d+1) columns,
    with the first column being label/output

    Returns X: an nxd array, where n is the number of examples and d is the dimensionality.
            y: an nx1 array, where n is the number of examples
    """
    data = pd.read_csv(fname).values
    X = data[:, 1:]
    y = data[:, 0]
    return X, y
