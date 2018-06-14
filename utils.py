"""
Utility Functions for ML-Lib
"""
from sklearn.datasets import load_breast_cancer
import random
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# I/O Functions
def print_header(s):
    S = len(s)
    bar = "-" * (S + 4)
    header = "| " + s + " |"
    print('\n')
    print(bar)
    print(header)
    print(bar)

def backspace(l):
    sys.stdout.write('\b'*l + ' '*l + '\b' * l)

def write(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()

def writeln(msg):
    sys.stdout.write(msg)
    sys.stdout.write('\n')
    sys.stdout.flush()

def read_data(txt):
    with open(txt) as f:
        return np.loadtxt(f)

# Utility Functions
def train_test_split(x, y, p_train):
    """Splits records into train and test set, with {p_train} of the records in the train set, and the rest in the test set

    Arguments:
        x {np.ndarray} -- features
        y {np.ndarray} -- labels
        p_train {float} -- proportion of data to be in train set, [0,1]

    Returns:
        np.ndarray, np.ndarray, np.ndarray, np.ndarray -- train_x, train_y, test_x, test_y
    """

    train_x = test_x = np.empty((0, x.shape[1]))
    train_y = test_y = np.empty((0, x.shape[0]))

    for i in range(len(x)):
        if random.random() <= p_train:
            train_x = np.vstack((train_x, x[i]))
            train_y = np.append(train_y, y[i])
        else:
            test_x = np.vstack((test_x, x[i]))
            test_y = np.append(test_y, y[i])

    return train_x, train_y, test_x, test_y