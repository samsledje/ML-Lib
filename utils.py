"""
Utility Functions for SamML
"""

# Imports
import numpy as np

def sigmoid(x):
    return (1 / (1 + np.exp(x)))

def dev_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)
