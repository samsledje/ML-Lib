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

def train_test_split(x, y, p_train):
    """Splits records into train and test set, with {p_train} of the records in the train set, and the rest in the test set

    Arguments:
        x {np.ndarray} -- features
        y {np.ndarray} -- labels
        p_train {float} -- proportion of data to be in train set, [0,1]

    Returns:
        np.ndarray, np.ndarray, np.ndarray, np.ndarray -- train_x, train_y, test_x, test_y
    """

    p_test = 1 - p_train
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

def plot_roc(title, label, proba):
    """Plots ROC curve

    Arguments:
        label {np.ndarray} -- true label
        proba {np.ndarray} -- predicted probabilites of class membership
    """
    rates = []

    pair = pd.concat((pd.DataFrame(proba), pd.DataFrame(label)), axis=1)
    pair.columns = ("proba", "target")
    pair_sorted = pair.sort_values(pair.columns[0], ascending=False)
    matrix = pair_sorted.as_matrix()
    for i in matrix:
        thresh = i[0]
        rates.append(_calc_rates(matrix, thresh))

    r = pd.DataFrame(rates)
    plt.plot(r[1].values, r[0].values)
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0,1.1, 0.1))
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive')
    plt.show()

def print_header(s):
    S = len(s)
    bar = "-" * (S + 4)
    header = "| " + s + " |"
    print('\n')
    print(bar)
    print(header)
    print(bar)

def read_data(txt):
    with open(txt) as f:
        return np.loadtxt(f)

def calculate_sse(clusters, centroids):
        
    k = len(centroids) 
    sse = 0

    # Calculate SSE
    for c in range(k):
        centroid = centroids[c]
        for d in clusters[c]:
            sse += _point_distance(d, centroid)**2

    return sse

# Helper Functions

def _calc_rates(matrix, thresh):
    """Calculates true positive rate and false positive rate for a given threshold

    Arguments:
        matrix {np.ndarray} -- true labels and predicted probabilities
        thresh {float} -- threshold for a round of ROC construction

    Returns:
        [float, float] -- true positive rate, false positive rate
    """

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    n = len(matrix)
    for i in matrix:
        pred = 1 if i[0] >= thresh else 0
        if pred == 1:
            if i[1] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if i[1] == 0:
                tn += 1
            else:
                fn += 1
    tpr = tp / (tp+fn)
    fpr = fp / (fp + tn)
    return tpr, fpr

def _sigmoid(x):
    return (1 / (1 + np.exp(x)))

def _dev_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def _point_distance(X1, X2):
    
    return np.linalg.norm(X1-X2)

def _cluster_distance(C1, C2, similarity_measure):

    if similarity_measure == "min":
        min_dist = np.Infinity
        for i in C1:
            for j in C2:
                dist = _point_distance(i, j)
                if (dist < min_dist):
                    min_dist = dist
        return min_dist
                
    elif similarity_measure == "max":
        max_dist = 0
        for i in C1:
            for j in C2:
                dist = _point_distance(i, j)
                if (dist > max_dist):
                    min_dist = dist
        return max_dist

    elif similarity_measure == "average":
        I = len(C1)
        J = len(C2)
        d_mat = np.ndarray((I, J))
        for i in range(I):
            for j in range(J):
                d_mat[i][j] = point_distance(C1[i], C2[j])
        return d_mat.sum() / (I*J)
        
    elif similarity_measure == "centroids":
        c1_centroid = np.sum(C1, axis=0) / len(C1)
        c2_centroid = np.sum(C2, axis=0) / len(C2)
        return _point_distance(c1_centroid, c2_centroid)

    else:
        print("Invalid similarity measurement.")
        sys.exit(1)
    return _point_distance(C1[0], C2[0])