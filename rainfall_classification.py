"""
Predict whether the rainfall is larger than 20 or not
"""
import numpy as np


def change_y_to_cat(training_y_rainfall):
    """
    change the rainfall number to category:
    1 - larger than 20
    0 - smaller than 20
    """
    threshold = np.ones(shape=(len(training_y_rainfall), 1)) * 20
    return training_y_rainfall >= threshold
