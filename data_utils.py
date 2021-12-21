import numpy as np
import os


def load_data(X, y):
    """
    Loads data from data directory.
    """

    return X, y.reshape(-1, 1)
